"""Download the authors' XML at a pinned Git commit; never overwrite local data."""
import argparse
import shutil
import subprocess
from datetime import datetime, timezone
from .common import ROOT, config, digest, save_json


def download():
    cfg = config()
    target = ROOT / "data/raw/MedQuAD"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", cfg["upstream_url"], str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", cfg["upstream_commit"]], check=True)
    commit = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if commit != cfg["upstream_commit"]:
        raise ValueError("Existing raw checkout is a different version; preserve it and resolve explicitly.")
    dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True)
    if dirty:
        raise ValueError("Raw checkout has local changes; refusing to claim pristine provenance.")
    provenance = ROOT / "data/provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    for source, dest in [("readme.txt", "upstream_readme.txt"), ("LICENSE.txt", "MedQuAD_LICENSE.txt")]:
        shutil.copyfile(target / source, provenance / dest)
    manifest = {str(p.relative_to(target)): digest(p) for p in sorted(target.glob("*/*.xml"))}
    save_json(provenance / "raw_manifest.json", manifest)
    # Preserve the first download/verification timestamp on repeat runs.
    metadata = provenance / "source.json"
    if not metadata.exists():
        save_json(metadata, {
            "upstream_url": cfg["upstream_url"], "commit": commit,
            "download_verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "xml_file_count": len(manifest), "license": "CC BY 4.0 (see saved upstream license)",
            "dataset_approval": "pending_user_confirmation",
            "cross_course_reuse_permission": "pending_user_confirmation",
            "generative_ai_permission": "pending_user_confirmation"
        })
    print("Verified {} XML files at {}".format(len(manifest), commit))
    return target


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    download()
