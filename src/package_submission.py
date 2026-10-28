"""Create a portable code ZIP, retaining local models and processed data.

Raw XML is obtainable using the pinned download command in README; its clone
and Git metadata are excluded. This ZIP is not a completed report/video.
"""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from .common import ROOT


def package():
    target = ROOT / "dist/CIS5300-medquad-intent-code.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    allowed = [ROOT / p for p in ["src", "notebooks", "scripts", "tests", "models", "results", "docs", "data"]]
    files = [ROOT / p for p in ["README.md", "requirements.txt", "config.json", ".gitignore"]]
    for directory in allowed:
        files += [p for p in directory.rglob("*") if p.is_file() and not {"__pycache__", ".ipynb_checkpoints", ".git", "raw"}.intersection(p.relative_to(ROOT).parts)]
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, Path("medquad_project") / path.relative_to(ROOT))
    print(target.relative_to(ROOT), target.stat().st_size, "bytes")
    return target


if __name__ == "__main__":
    package()
