"""Create blank human-writing worksheets and validate real review records.

No generated paraphrase is presented as human written or approved. Spelling
variants operate on predeclared non-medical cue words, never disease names.
"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, config, labels, read_questions, text_key

FIELDS = ["variant_id", "parent_question_id", "variant_type", "original_text", "text", "label",
          "disease_group", "split", "review_status", "author", "reviewer", "review_notes", "change_record"]
SPELLING_WORDS = {"symptoms", "treatments", "treatment", "diagnose", "prevention", "prevent", "causes", "information"}


def select_parents(frame, per_class):
    rng = np.random.default_rng(config()["seed"])
    selected = []
    for label in labels():
        candidates = frame[frame.label == label].sort_values("question_id")
        # Shuffle once; prefer different disease groups, then fill if necessary.
        candidates = candidates.iloc[rng.permutation(len(candidates))]
        diverse = candidates.drop_duplicates("disease_group").head(per_class)
        if len(diverse) < per_class:
            diverse = pd.concat([diverse, candidates[~candidates.question_id.isin(diverse.question_id)].head(per_class - len(diverse))])
        selected.append(diverse)
    return pd.concat(selected)


def spelling_variant(text, focus):
    protected = {m.group().casefold() for m in re.finditer(r"[A-Za-z]+", focus)}
    for m in re.finditer(r"[A-Za-z]+", text):
        word = m.group()
        if word.casefold() not in SPELLING_WORDS or word.casefold() in protected:
            continue
        # One adjacent internal-character transposition; no negations or digits.
        for k in range(1, len(word) - 2):
            if word[k] != word[k + 1]:
                changed = word[:k] + word[k + 1] + word[k] + word[k + 2:]
                record = {"operation": "adjacent_internal_transposition", "position": m.start() + k,
                          "before": word, "after": changed}
                return text[:m.start()] + changed + text[m.end():], json.dumps(record)
    return "", "skipped_no_safe_cue_word"


def make_worksheets():
    frame = read_questions()
    directory = ROOT / "data/paraphrases"
    directory.mkdir(parents=True, exist_ok=True)
    cfg = config()
    for split, n, filename in [("train", cfg["train_parents_per_class"], "train_paraphrases.csv"),
                               ("test", cfg["test_parents_per_class"], "test_variants.csv")]:
        rows = []
        for row in select_parents(frame[frame.split == split], n).itertuples():
            types = ["paraphrase"] if split == "train" else ["paraphrase", "spelling"]
            for kind in types:
                text, change = ("", "") if kind == "paraphrase" else spelling_variant(row.question_text, row.focus)
                rows.append({"variant_id": split + "::" + row.question_id + "::" + kind,
                             "parent_question_id": row.question_id, "variant_type": kind,
                             "original_text": row.question_text, "text": text, "label": row.label,
                             "disease_group": row.disease_group, "split": split,
                             "review_status": "pending" if kind == "paraphrase" or text else "skipped",
                             "author": "" if kind == "paraphrase" else "deterministic_spelling_script",
                             "reviewer": "", "review_notes": "", "change_record": change})
        path = directory / filename
        if path.exists():
            print("Preserved existing human worksheet:", filename)
        else:
            pd.DataFrame(rows, columns=FIELDS).to_csv(path, index=False)
            print("Created {} with {} pending/skipped rows".format(filename, len(rows)))


def reviewed_variants(path, split, frame=None, require_all=False):
    frame = read_questions() if frame is None else frame
    rows = pd.read_csv(path, keep_default_na=False)
    if not set(FIELDS).issubset(rows.columns):
        raise ValueError("Variant worksheet is missing required columns.")
    if rows.variant_id.duplicated().any():
        raise ValueError("Duplicate variant IDs.")
    if rows.duplicated(["parent_question_id", "variant_type"]).any():
        raise ValueError("Only one variant of each type per parent is allowed.")
    if not rows.review_status.isin(["pending", "approved", "rejected", "skipped"]).all():
        raise ValueError("Unknown review status.")
    parents = frame.set_index("question_id")
    approved = rows[rows.review_status == "approved"].copy()
    if require_all and not rows.review_status.isin(["approved", "skipped"]).all():
        raise ValueError("All final-test rows must be approved or explicitly skipped before freezing.")
    seen = set()
    forbidden = set(frame.loc[frame.split != split, "question_text"].map(text_key))
    training_originals = set(frame.loc[frame.split == "train", "text_key"])
    for row in approved.itertuples():
        if row.parent_question_id not in parents.index:
            raise ValueError("Unknown parent: " + row.parent_question_id)
        parent = parents.loc[row.parent_question_id]
        if parent.split != split or row.split != split or row.label != parent.label or row.disease_group != parent.disease_group:
            raise ValueError("Variant split, label or group does not match its parent.")
        if row.original_text != parent.question_text:
            raise ValueError("Original text does not match frozen parent.")
        if row.variant_type not in (["paraphrase"] if split == "train" else ["paraphrase", "spelling"]):
            raise ValueError("Invalid variant type.")
        if not row.text.strip() or not row.author.strip() or not row.reviewer.strip() or not row.review_notes.strip():
            raise ValueError("Approved variants require text and real author/reviewer/review notes.")
        if text_key(row.text) == text_key(row.original_text):
            raise ValueError("Variant duplicates its own original.")
        if text_key(row.text) in forbidden or text_key(row.text) in seen:
            raise ValueError("Duplicate variant or exact cross-split text overlap.")
        seen.add(text_key(row.text))
        if split == "train" and text_key(row.text) in training_originals:
            raise ValueError("Training paraphrase duplicates an original training question.")
    return approved


if __name__ == "__main__":
    make_worksheets()
