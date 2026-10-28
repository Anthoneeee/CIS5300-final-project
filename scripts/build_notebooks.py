"""Build readable course notebooks. Execute them separately to verify outputs."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = '''from pathlib import Path
import sys
ROOT = Path.cwd()
if not (ROOT / "config.json").exists():
    ROOT = ROOT.parent
assert (ROOT / "config.json").exists(), "Open this notebook inside the project."
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)
import pandas as pd
'''


def write(name, cells):
    notebook = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text) for kind, text in cells])
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python"}
    nbf.write(notebook, ROOT / "notebooks" / name)


def build():
    (ROOT / "notebooks").mkdir(exist_ok=True)
    write("01_data_preparation.ipynb", [
        ("md", "# 01 · MedQuAD data preparation\n\nCIS 5300: Medical Question Intent Classification. The original XML comes from MedQuAD by Ben Abacha and Demner-Fushman (CC BY 4.0). This notebook fixes the dataset version, extracts questions and metadata, selects six disease-related intent classes, and freezes an approximately 80/10/10 split by disease group. Answers never enter the classifier.\n\nBefore running: install dependencies from the project root as described in the README. The student must confirm dataset approval and permission for AI assistance with the teaching team. This code was prepared with Codex assistance."),
        ("code", SETUP),
        ("md", "## Parse, clean, and freeze\n\nThe parser also supports legacy XML with lowercase tags. Disease scope follows the original annotations. Conflicting labels are quarantined, and identical questions with the same label are deduplicated. Existing frozen splits cannot be silently changed."),
        ("code", "from src.prepare_data import prepare\nquestions = prepare()\nquestions[['question_id', 'question_text', 'label', 'focus', 'split']].head()"),
        ("code", "pd.read_csv('results/class_counts.csv')"),
        ("code", "pd.read_csv('results/split_class_counts.csv')"),
        ("code", "from IPython.display import Markdown, display\ndisplay(Markdown(Path('results/data_audit.md').read_text()))"),
        ("md", "## Stage A checks\n\nCheck unique IDs, disjoint disease groups, disjoint normalized questions, and coverage of every class in each split. Grouping is limited by the completeness of synonym annotations. Shared CUIs alone do not establish disease equivalence; unresolved cross-split CUI overlap is recorded in the audit. Test labels are used here only to check split coverage; no classifier test scores are computed."),
        ("code", "from src.prepare_data import validate_splits\nvalidate_splits(questions)\nprint('Data checks passed; no classifier test scores computed.')")
    ])
    write("02_models_and_extensions.ipynb", [
        ("md", "# 02 · Baselines and controlled extensions\n\nM0 predicts the training-set majority class. M1 uses fixed-priority rules. M2 uses word TF-IDF + LinearSVC, and M3 combines word and character TF-IDF + LinearSVC. Only C is tuned, using the candidates {0.1, 1, 10} and development Macro-F1. Ties select the smaller C. Vocabulary, IDF, and the classifier are fitted only on training data."),
        ("code", SETUP),
        ("code", "from src.train import train\ndev_metrics = train()\ndev_metrics"),
        ("code", "pd.read_csv('results/dev/tuning.csv')"),
        ("md", "## Human materials required for M4/M5\n\nThe worksheets contain 180 training parent questions and 120 test parent questions, selected after splitting without consulting predictions. Training and test paraphrases cannot share parents. Manually fill in each paraphrase's text, author, reviewer, and review_notes; mark it approved only after a real review. Spelling variants are generated from the original question, independently of paraphrases, using fixed rules, and also require review.\n\nThe following cell creates blank review worksheets. It does not write or approve paraphrases."),
        ("code", "from src.variants import make_worksheets, reviewed_variants\nmake_worksheets()\nworksheet = pd.read_csv('data/paraphrases/train_paraphrases.csv', keep_default_na=False)\nprint(worksheet.review_status.value_counts())\nworksheet.head(3)"),
        ("code", "approved = reviewed_variants('data/paraphrases/train_paraphrases.csv', 'train')\nif approved.empty:\n    print('M4/M5 pending: no human-approved training paraphrases.')\nelse:\n    print('Reviewed candidates:', len(approved))\n    print('After all reviews, explicitly train all six together:')\n    print('python -m src.train --models M0 M1 M2 M3 M4 M5')"),
        ("md", "The development and test splits remain fixed. Do not use final test scores to revise rules or choose parameters. Near-perfect scores may reflect repeated question templates; the independent expression-variation experiment remains necessary.")
    ])
    write("03_evaluation_and_errors.ipynb", [
        ("md", "# 03 · Frozen evaluation and error analysis\n\nBy default, this notebook displays development results and final-evaluation status. Once real human paraphrases and reviews are complete and all M0–M5 models have been trained, explicitly run `python -m src.evaluate freeze` in the terminal. The freeze records SHA-256 hashes of code, inputs, models, and variant files. Final evaluation becomes available only after this step.\n\nEvery model uses the same fixed class order, Macro-F1, accuracy, per-class precision/recall, and confusion matrices. Expression variation is compared on the same parent questions with complete original, paraphrase, and spelling versions."),
        ("code", SETUP),
        ("code", "dev_metrics = pd.read_csv('results/dev/metrics.csv')\ndev_metrics"),
        ("code", "from IPython.display import Image, display\ndisplay(Image(filename='results/dev/M2_confusion.png'))"),
        ("code", "freeze_path = Path('results/final/freeze.json')\nif freeze_path.exists():\n    from src.evaluate import evaluate\n    final_metrics = evaluate()\n    display(final_metrics)\nelse:\n    print('FINAL TEST NOT EVALUATED. Human paraphrases/reviews and protocol freeze are pending.')"),
        ("md", "## Development errors for inspection\n\nThe human analysis of final test errors will be saved in `results/final/error_analysis.csv`; that file has not yet been produced. The following cell displays actual development-set errors. If no errors exist, the table remains empty. These examples are part of development analysis."),
        ("code", "from src.common import read_questions\npred = pd.read_csv('results/dev/predictions.csv')\nerrors = pred[(pred.model_id == 'M2') & (pred.label != pred.prediction)]\nerrors.merge(read_questions()[['question_id', 'question_text']], on='question_id')")
    ])
    write("04_demo.ipynb", [
        ("md", "# 04 · Intent classification demo\n\nEnter an English medical question to obtain a predicted intent and an explanation of the class. This demo reloads the fixed development-stage model without refitting it. SVM decision scores are not probabilities. The project classifies intent and does not generate medical answers. The example questions below are demonstration inputs; they are not experimental results."),
        ("code", SETUP),
        ("code", "from src.predict import predict\nquestion = 'What treatment options are available for asthma?'\npredict(question)"),
        ("code", "examples = [\n    'What is asthma?',\n    'What are the symptoms of asthma?',\n    'What causes asthma?',\n    'How do doctors diagnose asthma?',\n    'What are the treatments for asthma?',\n    'How can asthma be prevented?'\n]\npd.DataFrame([predict(text) for text in examples])"),
        ("md", "## An actual development-set failure\n\nRetrieve an error from the saved development predictions. The model is not adjusted for this demonstration. High development scores do not establish an ability to understand real patient language."),
        ("code", "import json\nfrom src.common import read_questions\nmodel_id = json.loads(Path('models/selected_model.json').read_text())['model_id']\npredictions = pd.read_csv('results/dev/predictions.csv')\nfailed = predictions[(predictions.model_id == model_id) & (predictions.label != predictions.prediction)]\nif len(failed):\n    example = failed.merge(read_questions()[['question_id', 'question_text']], on='question_id').iloc[0]\n    print('True label:', example.label)\n    display(predict(example.question_text))\nelse:\n    print('No error in this development split. Do not invent a failure case.')")
    ])


if __name__ == "__main__":
    build()
