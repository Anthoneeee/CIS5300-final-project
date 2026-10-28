# M4/M5 exploratory AI augmentation run

Run ID: `ai_unreviewed_001`.

The user explicitly requested generation of the 180 paraphrases and their direct use for training. This run therefore uses **AI-generated, human-unreviewed augmentation**. It does not satisfy the original manually checked augmentation protocol. Course permission for this use remains unconfirmed. No record has been marked human-approved, and the original blank review worksheet is preserved.

## Data and training

- Original Train: 14,158 questions from the existing fixed split.
- Added training paraphrases: 180, exactly 30 for each of the six existing classes.
- Augmented Train: 14,338 rows, used identically by M4 and M5.
- Development: the unchanged 1,880-question Dev split.
- Final Test: the original 1,807-question holdout has not been evaluated in this run.

The paraphrases use AI-authored templates, deterministic realization with the parent focus, and source-specific overrides for rare questions. This method has limited template diversity and has not undergone human semantic review. A source question mixing HPS diagnosis and treatment remains flagged as ambiguous; its inherited label is preserved.

Automated checks verify parent membership in Train, inherited labels and source metadata, retained disease focus, unique IDs, and absence of exact normalized duplicates against all original Train/Dev/Test questions and other generated paraphrases. These checks do not establish semantic equivalence or clinical correctness.

M4 uses the existing M2 word (1, 2)-gram TF-IDF + LinearSVC architecture. M5 uses the existing M3 word plus char_wb (3, 5)-gram TF-IDF + LinearSVC architecture. Both search C in [0.1, 1, 10], selecting on fixed Dev Macro-F1 and preferring the smaller C on a tie. The actual saved baseline pipelines were checked for architecture equivalence before fitting. Both vectorizers and classifiers were fitted anew on the same augmented Train table. Dev/Test rows were not fitted.

## Actual development results

| Model | Training regime | Selected C | Dev accuracy | Dev Macro-F1 |
|---|---|---:|---:|---:|
| M2 | Original Train | 1.0 | 0.999468 | 0.999208 |
| M3 | Original Train | 0.1 | 0.999468 | 0.999208 |
| M4 | Original Train + 180 unreviewed AI paraphrases | 0.1 | 0.999468 | 0.999208 |
| M5 | Original Train + the same 180 AI paraphrases | 0.1 | 0.999468 | 0.999208 |

All three searched C values tied for each augmented model. M4 and M5 show no measured improvement over M2/M3 on the original Dev set. MedQuAD's repeated templates and the existing near-ceiling baseline scores limit the interpretation. No expression-robustness improvement has been established.

Both augmented models make one Dev error on the question `What is (are) How to prevent frostbite and hypothermia ?`: the original label is Basic information, and the prediction is Prevention. An AI interpretation is that the prevention phrase inside the topic title competes with the generic definition-question wrapper; the source-label boundary also warrants human inspection. This is not a completed human error analysis. The inherited label was not changed, and the error was not used to revise these models.

## Saved files

- `data/paraphrases/train_paraphrases_ai_generated.csv`: all 180 generated examples, AI-origin records, and pending human-review status.
- `results/ai_augmentation_generation.json`: generation method, authorization scope, checksums, and pending course-permission status.
- `data/processed/augmentation_runs/ai_unreviewed_001/train_augmented.csv`: complete shared training table with original IDs, new augmentation IDs, parent IDs, source files/URLs, labels, groups, and origin.
- `models/augmentation_runs/ai_unreviewed_001/M4.joblib` and `M5.joblib`: saved complete pipelines.
- `results/augmentation_dev/ai_unreviewed_001/`: tuning trials, per-class precision/recall/F1, confusion matrices, predictions, all training IDs, comparison table, pending human error-analysis worksheet, run configuration, and validation evidence.
- `notebooks/05_augmented_models.ipynb`: inspect the saved run and perform inference without refitting.

The original M0–M3 pipelines, selected-model record, configuration, data split, processed tables, source implementation, development results, review files, and earlier code ZIP remain unchanged. All extension artifacts use separate new paths.

## Reproduction

From the project root, with the pinned dependencies installed:

```bash
# On a checkout where generation outputs do not yet exist:
python scripts/generate_ai_paraphrases.py

# Validate without fitting or creating training outputs:
python scripts/train_augmented_models.py --run-id ai_unreviewed_001 --augmentation-policy ai_exploratory --check

# On a checkout without this run's output directories:
python scripts/train_augmented_models.py --run-id ai_unreviewed_001 --augmentation-policy ai_exploratory

# When run outputs already exist, choose a fresh run ID rather than overwrite them.
python -m unittest discover -s tests -p 'test_augmented_training.py' -v
```

The default training policy remains `human_reviewed` and rejects unapproved AI examples. The `ai_exploratory` policy must be chosen explicitly and verifies the generated file against its provenance record. It preserves pending review status and does not claim course approval.

Final Test comparison and paired paraphrase/spelling evaluation remain pending. The existing Test variant worksheet has no human-approved, frozen paraphrase set. Future evaluation should freeze its protocol separately and preserve the current run history.
