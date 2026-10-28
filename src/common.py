"""Shared paths, serialization and label order. All paths are project relative."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def labels():
    return list(config()["label_map"].values())


def normalize(text):
    """Minimal input cleaning; do not remove question words or medical terms."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip()


def text_key(text):
    return normalize(text).casefold()


def name_key(text):
    return re.sub(r"[^\w]+", " ", text_key(text)).strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def snapshot(paths):
    """Bind runs to exact files, rather than only the random seed."""
    return {str(Path(p).relative_to(ROOT)): digest(p) for p in sorted(paths)}


def read_questions():
    import pandas as pd
    path = ROOT / "data/processed/questions_clean.csv"
    if not path.exists():
        raise ValueError("Run python -m src.prepare_data first.")
    frame = pd.read_csv(path, keep_default_na=False)
    frozen_path = ROOT / "data/splits/split_ids.csv"
    if not frozen_path.exists():
        raise ValueError("Missing frozen split IDs; run data preparation first.")
    frozen = pd.read_csv(frozen_path, keep_default_na=False)
    columns = ["question_id", "disease_group", "split"]
    actual = frame[columns].sort_values("question_id").reset_index(drop=True)
    expected = frozen[columns].sort_values("question_id").reset_index(drop=True)
    if not actual.equals(expected) or frame.question_id.duplicated().any():
        raise ValueError("Processed rows do not match the frozen split manifest.")
    if not frame.split.isin(["train", "dev", "test"]).all():
        raise ValueError("Invalid split name.")
    if (frame.groupby("disease_group").split.nunique() > 1).any():
        raise ValueError("Disease group crosses splits.")
    if not (frame.question_text.map(text_key) == frame.text_key).all():
        raise ValueError("Question text and normalized duplicate key disagree.")
    if frame.text_key.duplicated().any():
        raise ValueError("Processed data contains duplicate question text.")
    for split in ["train", "dev", "test"]:
        if set(frame.loc[frame.split == split, "label"]) != set(labels()):
            raise ValueError("Split class coverage does not match the fixed labels.")
    return frame
