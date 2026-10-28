"""Train additive M4/M5 runs using one validated augmented Train table.

The default policy requires genuinely approved human-review records. The
explicit ai_exploratory policy permits the user-authorized, clearly marked
AI-generated file without claiming human review or confirmed course approval.
Neither policy fits on Dev/Test or overwrites existing baseline artifacts.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import warnings
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from src.common import config, digest, labels, normalize, read_questions, text_key
from src.models import build_model
from scripts.prepare_training_review import AUDIT, METADATA, WORKSHEET, check_protection, validate_worksheet


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def signature(model):
    """Feature/classifier settings, excluding the searched classifier C."""
    features = model.named_steps["features"]
    vectorizers = features.transformer_list if hasattr(features, "transformer_list") else [("word", features)]
    keys = ["analyzer", "lowercase", "ngram_range", "min_df", "max_df", "norm", "sublinear_tf",
            "use_idf", "smooth_idf", "stop_words", "strip_accents", "token_pattern"]
    vectorizer_settings = {name: {k: estimator.get_params()[k] for k in keys} for name, estimator in vectorizers}
    classifier = model.named_steps["classifier"].get_params().copy()
    classifier.pop("C")
    return {"features": vectorizer_settings, "classifier": classifier}


def eligible_rows(rows, frame, policy):
    """Select augmentation under an explicit, auditable review policy."""
    validate_worksheet(rows, frame)
    if policy == "human_reviewed":
        selected = rows[rows.review_status == "approved"].copy()
        origin = "human_reviewed_paraphrase"
    elif policy == "ai_exploratory":
        if not {"origin", "human_review_status", "generation_method"}.issubset(rows.columns):
            raise ValueError("AI exploratory input is missing origin records.")
        if not rows.origin.eq("ai_generated").all() or not rows.human_review_status.eq("not_reviewed").all():
            raise ValueError("AI exploratory origin/review records are inconsistent.")
        if not rows.review_status.eq("pending").all() or not rows.reviewer.eq("").all():
            raise ValueError("Unreviewed AI input must retain pending status and no claimed human reviewer.")
        selected = rows.copy()
        origin = "ai_generated_unreviewed"
    else:
        raise ValueError("Unknown augmentation policy.")
    if selected.empty or set(selected.label) != set(labels()):
        raise ValueError("Eligible paraphrases must be nonempty and cover all six classes.")
    if not selected.paraphrase_text.str.strip().ne("").all():
        raise ValueError("Every selected paraphrase must contain text.")
    return selected, origin


def augmented_table(frame, selected, origin):
    """Keep original IDs and inherit each paraphrase's full parent provenance."""
    training = frame[frame.split == "train"].copy()
    training["parent_question_id"] = training.question_id
    training["row_origin"] = "original"
    for field in ["paraphrase_author", "paraphrase_reviewer", "paraphrase_reviewed_at_utc", "paraphrase_review_notes"]:
        training[field] = ""
    parents = frame.set_index("question_id")
    added = []
    for row in selected.itertuples():
        parent = parents.loc[row.parent_question_id]
        if parent.split != "train":
            raise ValueError("An augmentation parent belongs to a held-out split.")
        record = parent.to_dict()
        record.update(question_id="augmentation::" + row.parent_question_id,
                      parent_question_id=row.parent_question_id, row_origin=origin,
                      question_text=normalize(row.paraphrase_text), text_key=text_key(row.paraphrase_text),
                      question_length_words=len(normalize(row.paraphrase_text).split()),
                      parent_original_text=parent.question_text, analysis_template="",
                      paraphrase_author=row.author, paraphrase_reviewer=row.reviewer,
                      paraphrase_reviewed_at_utc=row.reviewed_at_utc, paraphrase_review_notes=row.review_notes)
        added.append(record)
    combined = pd.concat([training, pd.DataFrame(added)], ignore_index=True).fillna("")
    if combined.question_id.duplicated().any() or combined.text_key.duplicated().any():
        raise ValueError("Duplicate training IDs or normalized text.")
    if set(combined.split) != {"train"}:
        raise ValueError("Augmented table contains a held-out split.")
    if set(combined.text_key).intersection(frame.loc[frame.split != "train", "text_key"]):
        raise ValueError("Augmented training text overlaps Dev/Test.")
    if not combined.qtype.map(config()["label_map"]).eq(combined.label).all():
        raise ValueError("Augmented labels do not match the fixed mapping.")
    if combined.loc[combined.row_origin == "original", "question_id"].tolist() != training.question_id.tolist():
        raise ValueError("Original training IDs changed.")
    return combined


def train(run_id, worksheet=None, policy="human_reviewed", check_only=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("Use a short run ID containing letters, numbers, underscores or hyphens.")
    if (ROOT / "results/final/freeze.json").exists():
        raise ValueError("Final protocol is frozen; preserve its history before any new training.")
    worksheet = Path(worksheet) if worksheet else (ROOT / "data/paraphrases/train_paraphrases_ai_generated.csv" if policy == "ai_exploratory" else WORKSHEET)
    if not worksheet.is_absolute():
        worksheet = ROOT / worksheet
    frame = read_questions()
    rows = pd.read_csv(worksheet, keep_default_na=False)
    record = json.loads((AUDIT / "selection.json").read_text())
    if rows.parent_question_id.tolist() != record["selected_parent_ids"] or digest(METADATA) != record["metadata_sha256"]:
        raise ValueError("Augmentation parents/provenance differ from the frozen selection.")
    protection = json.loads((AUDIT / "baseline_snapshot.json").read_text())
    extra = [WORKSHEET, METADATA, ROOT / "scripts/prepare_training_review.py", ROOT / "tests/test_training_review.py"]
    extra += [p for p in AUDIT.glob("*") if p.is_file()]
    protection.update({str(p.relative_to(ROOT)): digest(p) for p in extra})
    check_protection(protection)
    baseline_run = json.loads((ROOT / "results/dev/run.json").read_text())
    for relative, expected in baseline_run["inputs"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Baseline input/source no longer matches: " + relative)
    generation = None
    if policy == "ai_exploratory":
        generation = json.loads((ROOT / "results/ai_augmentation_generation.json").read_text())
        if generation["generated_csv_sha256"] != digest(worksheet) or generation["human_reviewed"] is not False:
            raise ValueError("AI generation record does not match the training file.")
        if generation["permitted_use"] != "exploratory_ai_unreviewed_augmentation":
            raise ValueError("AI input is not authorized for exploratory use.")
    selected, origin = eligible_rows(rows, frame, policy)
    augmented = augmented_table(frame, selected, origin)
    dev = frame[frame.split == "dev"]
    signatures = {}
    for original, extension in [("M2", "M4"), ("M3", "M5")]:
        path = ROOT / ("models/" + original + ".joblib")
        if digest(path) != baseline_run["models"][original]["model_sha256"]:
            raise ValueError("Baseline model checksum mismatch: " + original)
        saved = joblib.load(path)
        signatures[extension] = signature(build_model(extension))
        if signature(saved) != signatures[extension]:
            raise ValueError("Architecture differs from the original baseline: " + extension)
    readiness = {"run_id": run_id, "augmentation_policy": policy, "original_train_n": int((frame.split == "train").sum()),
                 "paraphrase_n": len(selected), "augmented_train_n": len(augmented), "dev_n": len(dev),
                 "class_counts": selected.label.value_counts().to_dict(), "same_augmented_table_for_M4_M5": True,
                 "human_reviewed_protocol_satisfied": policy == "human_reviewed", "test_evaluated": False}
    print(json.dumps(readiness, indent=2), flush=True)
    if check_only:
        return readiness
    data_dir = ROOT / "data/processed/augmentation_runs" / run_id
    model_dir = ROOT / "models/augmentation_runs" / run_id
    result_dir = ROOT / "results/augmentation_dev" / run_id
    if any(p.exists() for p in [data_dir, model_dir, result_dir]):
        raise ValueError("Run output already exists; use a new run ID to preserve it.")
    for directory in [data_dir, model_dir, result_dir]:
        directory.mkdir(parents=True, exist_ok=False)
    write_json(result_dir / "baseline_protection.json", protection)
    write_json(result_dir / "run_started.json", {**readiness, "started_at_utc": datetime.now(timezone.utc).isoformat()})
    # Lazy plotting import keeps policy validation independent of graphics caches.
    from src.metrics import detailed_outputs, score
    augmented_path = data_dir / "train_augmented.csv"
    augmented.to_csv(augmented_path, index=False)
    rows.to_csv(data_dir / "augmentation_source.csv", index=False)
    augmented[["question_id", "parent_question_id", "row_origin", "label", "disease_group", "split", "source_file", "source_url"]].to_csv(result_dir / "training_ids.csv", index=False)
    train_hash = digest(augmented_path)
    metrics_rows, trials, predictions, models = [], [], [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for model_id in ["M4", "M5"]:
            best, best_c, best_f1 = None, None, -1.0
            for c in config()["c_grid"]:
                estimator = build_model(model_id, c)
                estimator.fit(augmented.question_text.tolist(), augmented.label.tolist())
                pred = estimator.predict(dev.question_text.tolist())
                result = score(dev.label, pred)
                trials.append({"model_id": model_id, "C": c, **result})
                if result["macro_f1"] > best_f1:
                    best, best_c, best_f1 = estimator, c, result["macro_f1"]
            if digest(augmented_path) != train_hash:
                raise ValueError("The augmented training table changed between model fits.")
            model_path = model_dir / (model_id + ".joblib")
            joblib.dump(best, model_path, compress=3)
            loaded = joblib.load(model_path)
            pred = loaded.predict(dev.question_text.tolist())
            if not (pred == best.predict(dev.question_text.tolist())).all():
                raise ValueError("Reloaded model predictions differ.")
            result = {"model_id": model_id, "C": best_c, "train_n": len(augmented), **score(dev.label, pred)}
            metrics_rows.append(result)
            detail = dev[["question_id", "question_text", "label", "disease_group", "split"]].copy()
            detail["model_id"], detail["prediction"], detail["augmentation_policy"] = model_id, pred, policy
            predictions.append(detail)
            detailed_outputs(dev.label, pred, model_id, result_dir, model_id)
            models[model_id] = {"C": best_c, "path": str(model_path.relative_to(ROOT)), "sha256": digest(model_path),
                                "augmented_train_sha256": train_hash, "architecture": signatures[model_id], "reload_predictions_match": True}
            print(json.dumps(result), flush=True)
    pd.DataFrame(metrics_rows).to_csv(result_dir / "metrics.csv", index=False)
    pd.DataFrame(trials).to_csv(result_dir / "tuning.csv", index=False)
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_csv(result_dir / "predictions.csv", index=False)
    errors = all_predictions[all_predictions.label != all_predictions.prediction].copy()
    errors["human_error_category"], errors["human_notes"] = "", ""
    errors.to_csv(result_dir / "error_analysis_pending.csv", index=False)
    baseline_metrics = pd.read_csv(ROOT / "results/dev/metrics.csv")
    baseline_metrics = baseline_metrics[baseline_metrics.model_id.isin(["M2", "M3"])].copy()
    baseline_metrics["augmentation_policy"] = "original_train_only"
    new_metrics = pd.DataFrame(metrics_rows)
    new_metrics["augmentation_policy"] = policy
    pd.concat([baseline_metrics, new_metrics], ignore_index=True).to_csv(result_dir / "comparison_M2_M3_M4_M5.csv", index=False)
    paths = [worksheet, ROOT / "config.json", ROOT / "data/processed/questions_clean.csv", ROOT / "data/splits/split_ids.csv", METADATA,
             ROOT / "results/dev/run.json", ROOT / "results/dev/metrics.csv", Path(__file__), ROOT / "scripts/prepare_training_review.py"]
    paths += list((ROOT / "src").glob("*.py"))
    if generation is not None:
        paths += [ROOT / "results/ai_augmentation_generation.json", ROOT / "scripts/generate_ai_paraphrases.py"]
    inputs = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): digest(path) for path in paths}
    check_protection(protection)
    write_json(result_dir / "run.json", {**readiness, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]},
               "inputs": inputs, "augmented_train_path": str(augmented_path.relative_to(ROOT)), "augmented_train_sha256": train_hash,
               "models": models, "baseline_files_preserved": len(protection), "human_error_analysis": "pending",
               "course_permission": "pending_user_confirmation", "test_evaluated": False,
               "interpretation": "Exploratory AI-generated, human-unreviewed augmentation" if policy == "ai_exploratory" else "Human-reviewed training augmentation"})
    return pd.DataFrame(metrics_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worksheet", help="Project-relative path to the source worksheet.")
    parser.add_argument("--augmentation-policy", choices=["human_reviewed", "ai_exploratory"], default="human_reviewed")
    parser.add_argument("--check", action="store_true", help="Validate and summarize inputs without creating training outputs.")
    args = parser.parse_args()
    train(args.run_id, args.worksheet, args.augmentation_policy, args.check)
