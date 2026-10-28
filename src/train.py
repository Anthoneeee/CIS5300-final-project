"""Fit M0-M3 on train, choose C on dev; M4-M5 need approved train paraphrases."""
import argparse
import json
import platform
import warnings
import importlib.metadata
import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from .common import ROOT, config, digest, read_questions, save_json, snapshot
from .metrics import detailed_outputs, score
from .models import RULES, build_model
from .variants import reviewed_variants


def training_inputs():
    paths = [ROOT / "config.json", ROOT / "data/processed/questions_clean.csv", ROOT / "data/splits/split_ids.csv"]
    paths += list((ROOT / "src").glob("*.py"))
    return paths


def train(model_ids=None):
    if (ROOT / "results/final/freeze.json").exists():
        raise ValueError("Final protocol is frozen. Do not retrain after reading test results.")
    model_ids = model_ids or ["M0", "M1", "M2", "M3"]
    frame = read_questions()
    training, dev = frame[frame.split == "train"], frame[frame.split == "dev"]
    augmented = None
    if any(m in {"M4", "M5"} for m in model_ids):
        path = ROOT / "data/paraphrases/train_paraphrases.csv"
        approved = reviewed_variants(path, "train", frame)
        if approved.empty or set(approved.label) != set(config()["label_map"].values()):
            raise ValueError("M4/M5 require real approved training paraphrases covering every class.")
        augmented = pd.concat([training[["question_text", "label"]], approved[["text", "label"]].rename(columns={"text": "question_text"})], ignore_index=True)
    directory = ROOT / "results/dev"
    directory.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    trials, best_rows, predictions, model_meta = [], [], [], {}
    # Treat incomplete convergence as failure, not an unnoticed warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for model_id in model_ids:
            data = augmented if model_id in {"M4", "M5"} else training
            best, selected, best_score = None, None, -1.0
            for c in ([None] if model_id in {"M0", "M1"} else config()["c_grid"]):
                model = build_model(model_id, c or 1.0)
                model.fit(data.question_text.tolist(), data.label.tolist())
                pred = model.predict(dev.question_text.tolist())
                metrics = score(dev.label, pred)
                trials.append({"model_id": model_id, "C": c, **metrics})
                # Stable tie-break: lower C because grid is ascending.
                if metrics["macro_f1"] > best_score:
                    best, selected, best_score = model, c, metrics["macro_f1"]
            model_path = ROOT / ("models/" + model_id + ".joblib")
            joblib.dump(best, model_path, compress=3)
            loaded = joblib.load(model_path)
            pred = loaded.predict(dev.question_text.tolist())
            assert (pred == best.predict(dev.question_text.tolist())).all(), "Reload prediction mismatch"
            row = {"model_id": model_id, "C": selected, "train_n": len(data), **score(dev.label, pred)}
            best_rows.append(row)
            detail = dev[["question_id", "disease_group", "label"]].copy()
            detail["model_id"], detail["prediction"] = model_id, pred
            predictions.append(detail)
            detailed_outputs(dev.label, pred, model_id, directory, model_id)
            model_meta[model_id] = {"C": selected, "train_n": len(data), "model_sha256": digest(model_path),
                                    "training_paraphrases_sha256": digest(ROOT / "data/paraphrases/train_paraphrases.csv") if model_id in {"M4", "M5"} else None}
            print(json.dumps(row), flush=True)
    pd.DataFrame(trials).to_csv(directory / "tuning.csv", index=False)
    summary = pd.DataFrame(best_rows)
    summary.to_csv(directory / "metrics.csv", index=False)
    pd.concat(predictions).to_csv(directory / "predictions.csv", index=False)
    selected = max(best_rows, key=lambda x: (x["macro_f1"], x["accuracy"], -int(x["model_id"][1:])))
    save_json(ROOT / "models/selected_model.json", {"model_id": selected["model_id"], "selection_split": "dev", "selection_metric": "macro_f1 (accuracy, then simpler model for ties)"})
    save_json(directory / "rules.json", {"priority_order": RULES, "fallback": "training majority", "source": "intent definitions; fixed before final testing"})
    save_json(directory / "run.json", {"python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]},
                                        "inputs": snapshot(training_inputs()), "models": model_meta, "test_evaluated": False})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["M0", "M1", "M2", "M3", "M4", "M5"], default=["M0", "M1", "M2", "M3"])
    train(parser.parse_args().models)
