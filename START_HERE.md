# Start here: MedQuAD project review package

This package includes the Python code, five executed notebooks, processed data,
frozen split IDs, saved M0-M5 pipelines, development results, and provenance.
The current README contains the completed M4/M5 development scores.

Use Python 3.13 to match the tested environment. From the extracted
medquad_project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.predict "What are the symptoms of asthma?"
```

The command above loads the saved baseline model. To inspect the saved M4/M5
models without retraining:

```bash
python - <<'PYCODE'
import json
from pathlib import Path
import joblib
run = json.loads(Path("results/augmentation_dev/ai_unreviewed_001/run.json").read_text())
question = "What treatment options are available for asthma?"
for model_id in ["M4", "M5"]:
    model = joblib.load(run["models"][model_id]["path"])
    print(model_id, model.predict([question])[0])
PYCODE
```

You can also open notebooks/05_augmented_models.ipynb in a notebook editor.
Review README.md and docs/AI_AUGMENTATION_RUN.md for the study design and limits.
M4/M5 use AI-generated paraphrases that have not been human-reviewed. Final
Test and paired robustness evaluation remain pending. All reported numerical
results are development-set results.

Raw XML is excluded to keep the package smaller. The pinned download command,
upstream license, provenance, and processed tables are included. Running the
saved-model demos does not require downloading raw XML or retraining.

The earlier code ZIP in dist/ is retained as an immutable historical snapshot.
Do not replace existing model, result, or split files when reviewing this copy.
FILE_MANIFEST.json contains SHA-256 checksums of the included project files.
