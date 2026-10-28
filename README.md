# Medical Question Intent Classification with MedQuAD

This CIS 5300 project classifies an English medical question into one of six intents: **Basic information, Symptoms, Causes, Diagnosis and tests, Treatment, or Prevention**. Only question text enters the classifier. Original `qtype` annotations supply the labels; answers, disease topics, sources, and concept IDs are excluded from model features.

## Current status

Data preparation, M0–M3 baseline development, and an exploratory M4/M5 augmentation run are complete. The repository includes saved models, five executed notebooks, audit records, and actual development metrics. **M4/M5 used 180 AI-generated training paraphrases without human review. Final testing and paired expression-variation evaluation remain pending.** The originally planned human-reviewed augmentation protocol and remaining course deliverables still need to be completed.

- Original questions: 47,457. After scope filtering and deduplication: 17,845 questions in 6,238 groups supported by the available disease annotations.
- Train / development / test questions: 14,158 / 1,880 / 1,807, using the fixed seed `5220`.
- Disease groups and normalized duplicate questions do not cross splits. Every split covers all six classes.
- A total of 471 CUI identifiers still occur across splits. Grouping follows the available focus and synonym annotations; complete medical concept isolation is not established. See the [data audit](results/data_audit.md).

| Model | Method | Development Macro-F1 | Development accuracy |
|---|---|---:|---:|
| M0 | Training-set majority class | 0.074701 | 0.288830 |
| M1 | Fixed keyword and regular-expression rules | 0.986207 | 0.989362 |
| M2 | Word TF-IDF + linear SVM | 0.999208 | 0.999468 |
| M3 | Word + character TF-IDF + linear SVM | 0.999208 | 0.999468 |
| M4 | M2 + 180 AI-generated training paraphrases | 0.999208 | 0.999468 |
| M5 | M3 + the same 180 AI-generated training paraphrases | 0.999208 | 0.999468 |

These values come from actual runs: [baseline metrics](results/dev/metrics.csv) and the [M2–M5 development comparison](results/augmentation_dev/ai_unreviewed_001/comparison_M2_M3_M4_M5.csv). M2–M5 have identical development metrics; this augmentation run shows no measured improvement on the original Dev set. M4 and M5 each used the same 14,338-row augmented Train table and selected C=0.1. Their paraphrases retain pending human-review status; see the [run details and limitations](docs/AI_AUGMENTATION_RUN.md). The baseline demo continues to use M2, selected before augmentation under the predefined preference for the simpler model. MedQuAD contains many repeated question templates, limiting what these high scores establish about language variation. These development scores do not establish performance on real patient expressions. Final testing remains pending. This implementation does not claim to reproduce the original experiments in Joachims (1998).

## Installation and execution

The tested environment uses Python **3.13.9** and the exact package versions in `requirements.txt`. Use Python 3.13 to reproduce this environment; compatibility with every Python 3.7+ environment has not been established. Git is required to obtain the fixed dataset version. Run these commands from the repository root; all project paths are relative:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.download_data
python -m src.prepare_data
python -m src.train
python -m src.variants
python -m src.predict "What treatment options are available for asthma?"
python -m unittest discover -s tests -v
```

The repository includes trained M0–M3 models and separate [M4/M5 pipelines](models/augmentation_runs/ai_unreviewed_001). After installing dependencies, you can directly run `python -m src.predict "What are the symptoms of asthma?"` with the baseline model. Use Notebook 05 for saved M4/M5 inference. Rebuild the original processed tables with the data-preparation commands above. The separate AI augmentation ZIP includes the augmented training table and all saved models; the earlier code ZIP preserves the original project snapshot.

Raw XML is obtained from the authors' repository. `download_data` checks out commit **577bd37b96c02d1833b2c9eed2de9f96964e96cb**, records each XML file's checksum, and saves the upstream license and provenance. It refuses to overwrite an existing checkout containing a different version or local changes.

Run the notebooks in this order:

1. [01_data_preparation.ipynb](notebooks/01_data_preparation.ipynb): compatible XML parsing, disease scope, six-class mapping, cleaning, grouping, and split freezing.
2. [02_models_and_extensions.ipynb](notebooks/02_models_and_extensions.ipynb): M0–M3, development-set parameter selection, and augmentation review worksheets.
3. [03_evaluation_and_errors.ipynb](notebooks/03_evaluation_and_errors.ipynb): development results and final-evaluation status. Final evaluation remains unavailable until the protocol is frozen.
4. [04_demo.ipynb](notebooks/04_demo.ipynb): load the saved model, explain predicted intents, and show an actual development-set error.
5. [05_augmented_models.ipynb](notebooks/05_augmented_models.ipynb): inspect the completed exploratory M4/M5 run, development comparison, and saved-pipeline inference without refitting.

Open the notebooks in VS Code or Jupyter and select a Python kernel containing the required dependencies. The notebooks and CLI share the same functions and evaluation code. After the final protocol is frozen, Notebook 02 refuses to retrain models to preserve the evaluation history.

## Data and experiment design

**Disease scope.** In the four mixed MedlinePlus collections, retain only records originally annotated as `Category=Disease`. Other collections follow the disease scope described in the upstream readme. Missing questions or topics are not inferred from keywords. Excluded question types and their exclusion reasons are recorded. Missing answers do not cause exclusion. Identical normalized questions with the same label are deduplicated; conflicting labels are quarantined for review. The parser supports four legacy NINDS files with lowercase tags and one CDC file with a `DiseaseFile` root, and records files without question nodes.

**Grouping and splitting.** Normalize focus names and group identical names together. Connect different names only through explicit, unambiguous synonym-to-focus annotations; do not merge names whose annotated CUI sets are disjoint. Before deduplication, identical questions connect their split groups. Sharing an arbitrary CUI alone does not establish disease equivalence. All links and unresolved ambiguities are recorded. `GroupShuffleSplit` holds out 20% of groups and divides those groups equally between development and test. Question proportions can differ from group proportions. A single predefined seed is used; test scores do not influence split selection.

**Controlled feature comparison.** M2 uses word (1, 2)-gram TF-IDF. M3 adds `char_wb` (3, 5)-gram TF-IDF through `FeatureUnion`. Both search `C` values of 0.1, 1, and 10 using development Macro-F1; ties select the smaller `C`. Vocabulary, IDF, and classifiers are fitted only on each model's training inputs. Development examples are not added to training. See `config.json` for settings and `results/dev/run.json` for the actual environment and input/model checksums.

## Human paraphrases and final evaluation

[train_paraphrase_candidates.csv](data/paraphrases/train_paraphrase_candidates.csv) is the preserved blank human-review worksheet with 30 training parent questions per class, totaling 180 rows. The completed exploratory run uses the separate [AI-generated paraphrase file](data/paraphrases/train_paraphrases_ai_generated.csv); its rows remain pending human review. The older [train_paraphrases.csv](data/paraphrases/train_paraphrases.csv) is retained as a legacy worksheet. [test_variants.csv](data/paraphrases/test_variants.csv) contains 20 fixed test parent questions per class, totaling 120 parents, each with separate paraphrase and spelling rows. Selection does not use model predictions. Training and test paraphrases have different parents.

For a future human-reviewed augmentation run, fill in `paraphrase_text`, actual `author`, `reviewer`, `reviewed_at_utc`, and `review_notes` in the candidate worksheet. Set `review_status=approved` only after a real review. If one person both writes and reviews an example, record that honestly. Training augmentation may use intent-preserving paraphrases that have been manually checked and comply with course policy. Test paraphrases must be manually written and label-checked. The program checks fields and provenance; it cannot certify whether a person actually performed the review.

Spelling variants use a fixed list of safe question-cue words and apply one internal adjacent-character transposition. Disease names, negations, and numbers are protected. Examples without a safe edit are marked `skipped`; remaining spelling variants still require review. Paraphrases and spelling variants are created independently from the original question.

After the required training reviews are complete, use the additive trainer with a fresh run ID:

```bash
python scripts/train_augmented_models.py --run-id human_reviewed_001 --augmentation-policy human_reviewed
```

The default human-reviewed policy uses only approved training paraphrases and requires all six classes. The completed exploratory run explicitly selected the separate `ai_exploratory` policy under the user's direct-use instruction; it does not claim human review or confirmed course permission. Both policies use the exact same augmented table for M4 and M5 and save outputs separately from the original baselines. See the [run guide](docs/AI_AUGMENTATION_RUN.md) for the exploratory command and [review instructions](docs/TRAINING_PARAPHRASE_REVIEW.md) for the original protocol.

Final evaluation needs a separately frozen protocol for the preserved M2/M3 and the selected M4/M5 run. The legacy `src.evaluate` entry point expects the earlier all-model training layout and a complete reviewed test-variant set; it has not been run for the exploratory extension. After final test errors have been inspected, model and rule adjustments fall outside the frozen evaluation. Any necessary implementation repair must retain its history and identify the results it replaces.

Final outputs include full disease-held-out test metrics, absolute original/paraphrase/spelling scores for the same paired parents, individual predictions, prediction changes, per-class metrics, confusion matrices, and an error-analysis worksheet for human annotation. Paired comparisons use only parents with all three versions. Comparing full-test scores with a smaller paraphrase subset cannot by itself measure the effect of paraphrasing.

## Sources, licensing, and pending confirmations

- Dataset: [abachaa/MedQuAD](https://github.com/abachaa/MedQuAD), CC BY 4.0. The upstream license and readme are saved in `data/provenance`. Answers removed by the authors remain absent; this project does not collect answers. See the [saved license](data/provenance/MedQuAD_LICENSE.txt).
- Dataset paper: Ben Abacha A, Demner-Fushman D. A Question-Entailment Approach to Question Answering. BMC Bioinformatics. 2019;20:511. [DOI](https://doi.org/10.1186/s12859-019-3119-4).
- Method APIs: [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html), and [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).
- The research scope and experiment plan follow the user-provided MedQuAD project handoff. The local course `Final_Projects.ipynb` was checked for code and deliverable requirements.
- **The implementation, comments, and documentation were prepared with Codex assistance. The student must understand the code and disclose and cite assistance according to course policy. Instructor approval of the dataset, cross-course reuse, and generative-AI assistance remains pending confirmation. No approval has been assumed.**
- The 8–10-slide presentation for December 1, AMIA report for December 15, and 10-minute code demonstration remain pending. The report page-limit conflict identified in the handoff requires clarification from the teaching team. Results must come from actual experiments.

## Packaging

```bash
python -m src.package_submission
```

This creates `dist/CIS5300-medquad-intent-code.zip` containing processed data, trained models, notebooks, code, and results, with exact instructions to obtain the larger raw XML files. The report, slides, and the student's recorded demonstration remain separate deliverables.
