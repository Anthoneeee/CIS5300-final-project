# Training paraphrase review: current stopping point

The current phase inspects existing baselines and prepares a human-writing and review worksheet. M4/M5 training, augmented-data construction, protocol freezing, and final evaluation have not been performed.

## Files to use

- `data/paraphrases/train_paraphrase_candidates.csv`: the new worksheet to complete. It contains 180 parents from the frozen Train split, 30 per class. Every row starts as `pending`, with an empty `paraphrase_text`.
- `data/paraphrases/train_paraphrase_candidate_metadata.csv`: immutable parent provenance, including the original question ID, source file/URL, focus, qtype, split, and analysis template.
- `results/training_paraphrase_review/coverage.csv`: actual class, disease-group, and template coverage.
- `results/training_paraphrase_review/selection.json`: deterministic selection protocol, fixed parent IDs, checksums, and pending-stage flags.
- `results/training_paraphrase_review/baseline_snapshot.json`: checksums protecting the pre-existing data, models, results, code, notebooks, and code ZIP.

The older `train_paraphrases.csv` and `test_variants.csv` are preserved. Use the new candidate file for this phase; the current legacy training command reads the older file, which has a different schema. No automatic conversion or training occurs.

## What can be reused

The actual project root contains `src/`, `scripts/`, `data/processed/`, `data/splits/`, `data/paraphrases/`, `models/`, `results/dev/`, and four notebooks. The following existing files were inspected directly:

| Existing artifact | Reuse |
|---|---|
| `src/models.py` | `build_model('M4')` uses the M2 word-feature architecture; `build_model('M5')` uses the M3 word-plus-character architecture. Both return complete pipelines. |
| `config.json` | Original six-class mapping, seed 5220, word (1, 2)-grams, character (3, 5)-grams, and C grid [0.1, 1, 10]. |
| `data/processed/questions_clean.csv` | Existing questions and provenance; no parsing or cleaning rerun is required. |
| `data/splits/split_ids.csv` | The frozen Train/Dev/Test manifest. It matches the processed records. |
| `models/M2.joblib`, `models/M3.joblib` | Saved original baselines for later comparison; checksums match `results/dev/run.json`. |
| `src/metrics.py` | Shared fixed-label Macro-F1, accuracy, per-class precision/recall/F1, and confusion-matrix code. |
| `results/dev/metrics.csv`, `tuning.csv`, `predictions.csv`, and `run.json` | Existing development evidence and provenance; preserve these files. |

The fixed split has 14,158 Train questions in 4,990 groups, 1,880 Dev questions in 624 groups, and 1,807 Test questions in 624 groups. Every split covers all six classes. Group and normalized-question overlap checks pass. The existing audit still records 471 CUI identifiers spanning splits; complete medical concept isolation is not established.

The saved M2 uses C=1.0. The saved M3 uses C=0.1. Both have development accuracy 0.999468 and Macro-F1 0.999208. Word features use lowercase TF-IDF with min_df=1 and L2 normalization. M3 adds char_wb character features through FeatureUnion. LinearSVC uses class_weight=None, dual='auto', max_iter=10000, tol=1e-4, and random_state=5220.

All input/source hashes recorded in the baseline development run matched at inspection. No baseline was refitted or evaluated on Test during this phase.

The unchanged label mapping is `information` → Basic information, `symptoms` → Symptoms, `causes` → Causes, `exams and tests` → Diagnosis and tests, `treatment` → Treatment, and `prevention` → Prevention. Every processed row was checked against this mapping.

## Selection and actual coverage

Within each class, selection first prefers an unseen focus-masked template, then an unseen disease group, then lower template/group counts. A seed-dependent SHA-256 ordering resolves ties reproducibly. Selection uses only Train rows and does not consult model predictions or errors.

| Class | Train questions available | Selected parents | Disease groups selected | Templates selected / available |
|---|---:|---:|---:|---:|
| Basic information | 4,220 | 30 | 30 | 8 / 8 |
| Symptoms | 3,233 | 30 | 30 | 3 / 3 |
| Causes | 1,758 | 30 | 30 | 1 / 1 |
| Diagnosis and tests | 1,485 | 30 | 30 | 4 / 4 |
| Treatment | 2,539 | 30 | 30 | 6 / 6 |
| Prevention | 923 | 30 | 28 | 7 / 7 |

This is a diversity-focused pilot sample: rare templates are deliberately oversampled. It is not proportional to their frequency in Train. Prevention repeats two groups to retain all seven available templates. The Causes class has only one observed template; no additional original-template diversity is invented. Template identities come from the existing focus-masking audit and are not a medically validated template taxonomy.

## Manual writing and review

1. Read `original_text`, `label`, and `disease_group`. Check source provenance when needed.
2. Write an English `paraphrase_text` that preserves the disease and a single original information intent. Avoid adding another disease, negation, diagnosis request, or treatment request that changes the task. Do not prefix the question with its target label.
3. Personally check the wording and label. Some rare source questions may be ambiguous or combine intents. Reject an unsuitable candidate rather than changing its inherited label or frozen parent metadata.
4. Record actual `author`, `reviewer`, `reviewed_at_utc` (for example, an actual ISO 8601 UTC timestamp), and substantive `review_notes`. If the same person writes and reviews a row, record that honestly.
5. Set `review_status` to `approved` only after the real review. Leave unfinished rows `pending`; mark unsuitable rows `rejected` and explain why.

The approved status is a human attestation. Programmatic checks cannot establish that a person performed a review. No AI-written paraphrases have been generated in this phase. AI-generated candidate wording requires confirmation that the actual course policy permits it, and still requires your real review before approval.

Do not edit the parent IDs, original texts, labels, disease groups, provenance file, or frozen split. Exact duplicate checks run on every nonblank candidate, including pending rows. They cover all original Train/Dev/Test questions and other candidate paraphrases. They do not certify semantic uniqueness or intent preservation; those require human judgment.

## Commands and verified boundaries

From the project root, using the existing Python environment:

```bash
python scripts/prepare_training_review.py
python scripts/prepare_training_review.py --check
python -m unittest discover -s tests -p 'test_training_review.py' -v
```

The first command creates the new review bundle once. Subsequent runs preserve existing files and validate them. The second command checks the edited worksheet, immutable parent order/provenance, and protected baseline checksums without writing files, building training data, or fitting models.

After you finish reviewing, provide the completed worksheet and ask to continue. Only approved rows may then be merged with original Train, with preserved IDs and source records, to create `train_augmented.csv`. M4 and M5 must consume the exact same augmented table and tune only on the fixed Dev split.

## Required adaptation after review

Do not invoke the current `src.train` entry point for M4/M5 yet: although its feature builders are reusable, it writes shared `results/dev/metrics.csv`, `predictions.csv`, `tuning.csv`, and `run.json`, and replaces the selected-model record. A separate additive training run is needed to preserve the existing baseline evidence.

The current `src.evaluate` also expects all M0–M5 and complete reviewed test variants. After review, the requested M2/M3/M4/M5 comparison needs a preservation-safe protocol that reuses existing baselines and treats robustness as optional and pending when reviewed, frozen variants are unavailable.

The existing test worksheet currently has 220 pending rows and 20 skipped rows. No reviewed/frozen Test paraphrase-and-spelling set is available. Final Test evaluation and robustness evaluation remain pending. `train_augmented.csv`, M4, and M5 have not been created.
