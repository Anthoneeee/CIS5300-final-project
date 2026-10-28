"""Load a saved estimator for intent-only inference; never fit during demo."""
import argparse
import json
import joblib
from .common import ROOT, normalize

EXPLANATIONS = {
    "Basic information": "Asks what the disease or condition is.",
    "Symptoms": "Asks about signs or symptoms.",
    "Causes": "Asks why the disease occurs or what causes it.",
    "Diagnosis and tests": "Asks about diagnosis, examinations or testing.",
    "Treatment": "Asks about treatment or management options.",
    "Prevention": "Asks how to prevent the disease or reduce its risk."
}


def predict(text, model_id=None):
    if not normalize(text):
        raise ValueError("Enter a non-empty English medical question.")
    if model_id is None:
        model_id = json.loads((ROOT / "models/selected_model.json").read_text())["model_id"]
    estimator = joblib.load(ROOT / ("models/" + model_id + ".joblib"))
    label = str(estimator.predict([normalize(text)])[0])
    return {"question": normalize(text), "model_id": model_id, "intent": label, "meaning": EXPLANATIONS[label]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--model", choices=["M0", "M1", "M2", "M3", "M4", "M5"])
    args = parser.parse_args()
    print(json.dumps(predict(args.question, args.model), ensure_ascii=False, indent=2))
