"""Prepare Train-only human paraphrase worksheets without fitting any model.

Run from the project root with `python scripts/prepare_training_review.py`.
Use --check to validate a worksheet after editing it. Existing worksheets and
baseline artifacts are never overwritten. No paraphrases are generated here.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.common import config, digest, labels, read_questions, text_key

WORKSHEET = ROOT / "data/paraphrases/train_paraphrase_candidates.csv"
METADATA = ROOT / "data/paraphrases/train_paraphrase_candidate_metadata.csv"
AUDIT = ROOT / "results/training_paraphrase_review"
FIELDS = ["parent_question_id", "original_text", "paraphrase_text", "label",
          "disease_group", "review_status", "review_notes", "author", "reviewer", "reviewed_at_utc"]


def select_candidates(frame, per_class=30, seed=5220):
    """Prefer unseen templates, then unseen groups, using seeded hash ties.

    Template diversity is deliberately oversampled for this small pilot;
    this is not a prevalence-proportional sample. All selection inputs must
    come from Train. Development/test predictions are never consulted.
    """
    if set(frame.split) != {"train"}:
        raise ValueError("Candidate selection requires a nonempty Train-only frame.")
    if frame.question_id.duplicated().any():
        raise ValueError("Duplicate candidate parent IDs.")
    if per_class < 1:
        raise ValueError("per_class must be positive.")
    selected = []
    for label in labels():
        pool = frame[frame.label == label].copy()
        if pool.empty:
            raise ValueError("Missing training class: " + label)
        pool["template_key"] = pool.analysis_template.map(text_key)
        pool["tie_rank"] = [hashlib.sha256((str(seed) + "|" + qid).encode()).hexdigest() for qid in pool.question_id]
        pool = pool.sort_values(["tie_rank", "question_id"])
        group_counts, template_counts = Counter(), Counter()
        remaining = list(pool.to_dict("records"))
        for _ in range(min(per_class, len(remaining))):
            # Stable min() tie-break follows the seeded hash ordering above.
            best = min(range(len(remaining)), key=lambda i: (
                template_counts[remaining[i]["template_key"]] > 0,
                group_counts[remaining[i]["disease_group"]] > 0,
                template_counts[remaining[i]["template_key"]],
                group_counts[remaining[i]["disease_group"]]
            ))
            row = remaining.pop(best)
            selected.append(row)
            template_counts[row["template_key"]] += 1
            group_counts[row["disease_group"]] += 1
    return pd.DataFrame(selected).drop(columns="tie_rank")


def protected_files():
    """Protect existing data, models, results, notebooks and source by hash."""
    files = [ROOT / p for p in ["README.md", "config.json", "requirements.txt", ".gitignore"]]
    for folder in ["src", "models", "results", "data/processed", "data/splits", "data/paraphrases", "data/provenance", "notebooks", "dist"]:
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file())
    files.extend(ROOT / "scripts" / p for p in ["build_notebooks.py", "execute_notebooks.py"])
    files.append(ROOT / "tests/test_integrity.py")
    return sorted(p for p in set(files) if p.exists() and p not in {WORKSHEET, METADATA}
                  and AUDIT not in p.parents and "__pycache__" not in p.parts and p.name != ".DS_Store")


def check_protection(snapshot):
    for relative, expected in snapshot.items():
        path = ROOT / relative
        if not path.exists() or digest(path) != expected:
            raise ValueError("Protected baseline changed: " + relative)


def validate_worksheet(rows, frame):
    """Validate every parent and every nonblank candidate, including pending."""
    if not set(FIELDS).issubset(rows.columns):
        raise ValueError("Review worksheet is missing required columns.")
    if rows.empty or rows.parent_question_id.duplicated().any():
        raise ValueError("Worksheet must have unique parent IDs and at least one row.")
    if not rows.review_status.isin(["pending", "approved", "rejected"]).all():
        raise ValueError("review_status must be pending, approved, or rejected.")
    parents = frame.set_index("question_id")
    if not rows.parent_question_id.isin(parents.index).all():
        raise ValueError("Worksheet includes an unknown parent ID.")
    matched = parents.loc[rows.parent_question_id].reset_index(drop=True)
    rows = rows.reset_index(drop=True)
    if not (matched.split == "train").all():
        raise ValueError("Every parent must belong to the frozen Train split.")
    for column, source in [("original_text", "question_text"), ("label", "label"), ("disease_group", "disease_group")]:
        if not (rows[column] == matched[source]).all():
            raise ValueError("Parent metadata mismatch: " + column)
    original_texts = set(frame.question_text.map(text_key))
    forbidden = set(frame.loc[frame.split != "train", "question_text"].map(text_key))
    seen = set()
    for row in rows.itertuples():
        key = text_key(row.paraphrase_text)
        if key:
            if key in forbidden:
                raise ValueError("Paraphrase exactly duplicates Dev/Test text.")
            if key in original_texts or key in seen:
                raise ValueError("Paraphrase duplicates an original question or another candidate.")
            seen.add(key)
        if row.review_status == "approved":
            if not all(str(value).strip() for value in [row.paraphrase_text, row.review_notes, row.author, row.reviewer, row.reviewed_at_utc]):
                raise ValueError("Approved rows require text, real authorship/review records, notes and a review timestamp.")
            stamp = pd.to_datetime(row.reviewed_at_utc, utc=True, errors="coerce")
            if pd.isna(stamp):
                raise ValueError("Invalid review timestamp.")
    return {"rows": len(rows), "review_status": rows.review_status.value_counts().to_dict(),
            "nonblank_paraphrases": int(rows.paraphrase_text.str.strip().ne("").sum()),
            "parents_in_frozen_train": True, "metadata_matches_parents": True,
            "exact_cross_split_candidate_duplicates": 0}


def write_new(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(text)


def check_existing(frame):
    required = [WORKSHEET, METADATA, AUDIT / "selection.json", AUDIT / "baseline_snapshot.json"]
    if not all(p.exists() for p in required):
        raise ValueError("Incomplete existing review bundle. Preserve its files and investigate before creating anything.")
    before = json.loads((AUDIT / "baseline_snapshot.json").read_text())
    check_protection(before)
    rows = pd.read_csv(WORKSHEET, keep_default_na=False)
    record = json.loads((AUDIT / "selection.json").read_text())
    if rows.parent_question_id.tolist() != record["selected_parent_ids"]:
        raise ValueError("Frozen candidate parent IDs/order changed.")
    if digest(METADATA) != record["metadata_sha256"]:
        raise ValueError("Candidate provenance metadata changed.")
    checks = validate_worksheet(rows, frame)
    print(json.dumps({"existing_files_preserved": True, "protected_baseline_files": len(before), **checks}, indent=2))
    return checks


def prepare(check_only=False):
    frame = read_questions()
    cfg = config()
    if check_only or WORKSHEET.exists() or METADATA.exists() or AUDIT.exists():
        return check_existing(frame)
    baseline = {str(p.relative_to(ROOT)): digest(p) for p in protected_files()}
    original_run = json.loads((ROOT / "results/dev/run.json").read_text())
    for relative, expected in original_run["inputs"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Saved development provenance does not match current input: " + relative)
    for name, entry in original_run["models"].items():
        if digest(ROOT / ("models/" + name + ".joblib")) != entry["model_sha256"]:
            raise ValueError("Saved baseline model checksum mismatch: " + name)
    train = frame[frame.split == "train"].copy()
    selected = select_candidates(train, cfg["train_parents_per_class"], cfg["seed"])
    rows = pd.DataFrame({"parent_question_id": selected.question_id,
                         "original_text": selected.question_text, "paraphrase_text": "",
                         "label": selected.label, "disease_group": selected.disease_group,
                         "review_status": "pending", "review_notes": "", "author": "",
                         "reviewer": "", "reviewed_at_utc": ""})[FIELDS]
    checks = validate_worksheet(rows, frame)
    columns = ["question_id", "original_qid", "source", "source_file", "source_url", "focus",
               "qtype", "label", "disease_group", "split", "analysis_template", "template_key"]
    metadata = selected[columns].rename(columns={"question_id": "parent_question_id"})
    coverage = []
    for label in labels():
        all_rows = train[train.label == label]
        chosen = selected[selected.label == label]
        coverage.append({"label": label, "available_train_questions": len(all_rows),
                         "selected_questions": len(chosen), "selected_disease_groups": int(chosen.disease_group.nunique()),
                         "available_templates": int(all_rows.analysis_template.map(text_key).nunique()),
                         "selected_templates": int(chosen.template_key.nunique())})
    write_new(WORKSHEET, rows.to_csv(index=False))
    write_new(METADATA, metadata.to_csv(index=False))
    record = {"seed": cfg["seed"], "requested_per_class": cfg["train_parents_per_class"],
              "selection_rule": "within each class: unseen template, unseen disease group, lower counts; deterministic seeded hash tie-break",
              "sampling_interpretation": "diversity-focused pilot; rare templates oversampled, not population-proportional",
              "source_split": "train", "ai_generated_paraphrases": 0,
              "paraphrase_text_status": "blank_for_human_authoring_pending_course_policy",
              "coverage": coverage, "selected_parent_ids": rows.parent_question_id.tolist(),
              "worksheet_initial_sha256": digest(WORKSHEET), "metadata_sha256": digest(METADATA),
              "frozen_split_sha256": digest(ROOT / "data/splits/split_ids.csv"),
              "protected_baseline_file_count": len(baseline), "checks": checks,
              "train_augmented_created": False, "M4_M5_trained": False, "final_test_evaluated": False,
              "legacy_training_caveat": "src/train.py writes shared development outputs; do not invoke it for this extension until an additive run path is implemented after review",
              "legacy_evaluation_caveat": "src/evaluate.py requires all M0-M5 and complete reviewed test variants; the later M2-M5 comparison with optional robustness needs a separate preservation-safe path"}
    write_new(AUDIT / "baseline_snapshot.json", json.dumps(baseline, indent=2) + "\n")
    write_new(AUDIT / "selection.json", json.dumps(record, indent=2) + "\n")
    write_new(AUDIT / "coverage.csv", pd.DataFrame(coverage).to_csv(index=False))
    check_protection(baseline)
    print(pd.DataFrame(coverage).to_string(index=False))
    print(json.dumps(checks, indent=2))
    print("STOP: human authoring/review required. No augmented dataset, fitting, or final evaluation performed.")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate existing review records without writing files or training models.")
    prepare(parser.parse_args().check)
