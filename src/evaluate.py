"""Freeze the complete protocol before evaluating held-out test data.

Do not run freeze until human-written paraphrases and their real reviews are
complete. No final test evaluation is executed by the default training flow.
"""
import argparse
import json
from datetime import datetime, timezone
import joblib
import pandas as pd
from .common import ROOT, digest, read_questions, save_json, snapshot
from .metrics import detailed_outputs, score
from .variants import reviewed_variants

MODEL_IDS = ["M0", "M1", "M2", "M3", "M4", "M5"]


def protocol_paths():
    paths = [ROOT / "config.json", ROOT / "data/processed/questions_clean.csv", ROOT / "data/splits/split_ids.csv",
             ROOT / "data/paraphrases/train_paraphrases.csv", ROOT / "data/paraphrases/test_variants.csv",
             ROOT / "results/dev/run.json", ROOT / "models/selected_model.json"]
    return paths + list((ROOT / "src").glob("*.py")) + [ROOT / ("models/" + m + ".joblib") for m in MODEL_IDS]


def validate_ready():
    frame = read_questions()
    train_variants = reviewed_variants(ROOT / "data/paraphrases/train_paraphrases.csv", "train", frame)
    variants = reviewed_variants(ROOT / "data/paraphrases/test_variants.csv", "test", frame, require_all=True)
    if train_variants.empty or variants.empty:
        raise ValueError("Real human-written and reviewed paraphrases are still required.")
    run = json.loads((ROOT / "results/dev/run.json").read_text())
    if set(run["models"]) != set(MODEL_IDS):
        raise ValueError("Train all M0-M5 together using python -m src.train --models M0 M1 M2 M3 M4 M5.")
    for path, expected in run["inputs"].items():
        if digest(ROOT / path) != expected:
            raise ValueError("Training inputs/source changed since fitting: " + path)
    augmentation_hash = digest(ROOT / "data/paraphrases/train_paraphrases.csv")
    for m in MODEL_IDS:
        if digest(ROOT / ("models/" + m + ".joblib")) != run["models"][m]["model_sha256"]:
            raise ValueError("Saved estimator changed: " + m)
    for m in ["M4", "M5"]:
        if run["models"][m]["training_paraphrases_sha256"] != augmentation_hash:
            raise ValueError("M4 and M5 must use the same unchanged reviewed augmentation file.")
    complete = variants.groupby("parent_question_id").variant_type.agg(set)
    paired_ids = set(complete[complete.map(lambda s: s == {"paraphrase", "spelling"})].index)
    paired = frame[frame.question_id.isin(paired_ids)]
    if paired.empty or set(paired.label) != set(frame.label):
        raise ValueError("Paired original/paraphrase/spelling subset must cover all fixed classes.")
    return frame, variants, paired


def freeze():
    destination = ROOT / "results/final/freeze.json"
    if destination.exists():
        raise ValueError("Protocol already frozen. Preserve the existing freeze and test history.")
    frame, variants, paired = validate_ready()
    record = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "inputs": snapshot(protocol_paths()),
              "models": MODEL_IDS, "full_test_n": int((frame.split == "test").sum()),
              "paired_parent_ids": sorted(paired.question_id), "paired_n": len(paired),
              "approved_test_variant_n": len(variants), "test_evaluated": False}
    save_json(destination, record)
    print("Protocol frozen. Now run python -m src.evaluate run once.")


def evaluate():
    path = ROOT / "results/final/freeze.json"
    if not path.exists():
        raise ValueError("Final test remains locked. Complete human reviews, train all models, then freeze.")
    record = json.loads(path.read_text())
    for relative, expected in record["inputs"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Frozen input changed: " + relative)
    directory = ROOT / "results/final"
    if (directory / "evaluation_complete.json").exists():
        print("Identical frozen evaluation already completed; preserved existing results.")
        return pd.read_csv(directory / "metrics.csv")
    frame, variants, paired = validate_ready()
    test = frame[frame.split == "test"]
    rows, predictions, paired_changes = [], [], []
    for model_id in MODEL_IDS:
        model = joblib.load(ROOT / ("models/" + model_id + ".joblib"))
        views = {
            "full_original": test.rename(columns={"question_id": "parent_question_id", "question_text": "text"}),
            "paired_original": paired.rename(columns={"question_id": "parent_question_id", "question_text": "text"})
        }
        for kind in ["paraphrase", "spelling"]:
            views["paired_" + kind] = variants[(variants.variant_type == kind) & variants.parent_question_id.isin(paired.question_id)]
        paired_predictions = {}
        for version, view in views.items():
            view = view.sort_values("parent_question_id")
            pred = model.predict(view.text.tolist())
            rows.append({"model_id": model_id, "version": version, **score(view.label, pred)})
            detail = view[["parent_question_id", "text", "label", "disease_group"]].copy()
            detail["variant_id"] = view.variant_id.values if "variant_id" in view else view.parent_question_id.values
            detail["model_id"], detail["version"], detail["prediction"] = model_id, version, pred
            predictions.append(detail)
            detailed_outputs(view.label, pred, model_id, directory / "confusion_matrices", model_id + "_" + version)
            if version.startswith("paired_"):
                paired_predictions[version] = detail.set_index("parent_question_id")
        original = paired_predictions["paired_original"]
        for kind in ["paraphrase", "spelling"]:
            variant = paired_predictions["paired_" + kind].reindex(original.index)
            paired_changes.append(pd.DataFrame({
                "parent_question_id": original.index, "model_id": model_id, "variant_type": kind,
                "label": original.label.values, "original_prediction": original.prediction.values,
                "variant_prediction": variant.prediction.values,
                "prediction_changed": original.prediction.values != variant.prediction.values,
                "original_correct": original.prediction.values == original.label.values,
                "variant_correct": variant.prediction.values == original.label.values
            }))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(directory / "metrics.csv", index=False)
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_csv(directory / "predictions.csv", index=False)
    changes = pd.concat(paired_changes, ignore_index=True)
    changes.to_csv(directory / "paired_changes.csv", index=False)
    changes.groupby(["model_id", "variant_type"]).agg(n=("parent_question_id", "size"), changed_fraction=("prediction_changed", "mean"), original_accuracy=("original_correct", "mean"), variant_accuracy=("variant_correct", "mean")).to_csv(directory / "paired_summary.csv")
    errors = all_predictions[all_predictions.label != all_predictions.prediction].copy()
    errors["human_error_category"], errors["human_notes"] = "", ""
    errors.to_csv(directory / "error_analysis.csv", index=False)
    save_json(directory / "evaluation_complete.json", {"freeze_sha256": digest(path), "test_evaluated": True,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(), "human_error_analysis": "pending"})
    print(metrics.to_string(index=False))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze() if args.action == "freeze" else evaluate()
