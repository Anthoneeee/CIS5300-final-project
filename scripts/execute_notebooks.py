"""Run each course notebook from a fresh kernel, stopping on any error."""
import json
import os
import tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def execute():
    # Notebook runtime/cache files are disposable and not part of the project.
    with tempfile.TemporaryDirectory(prefix="medquad-notebook-") as runtime:
        for name in ["MPLCONFIGDIR", "XDG_CACHE_HOME", "JUPYTER_RUNTIME_DIR", "IPYTHONDIR"]:
            destination = Path(runtime) / name.lower()
            destination.mkdir()
            os.environ[name] = str(destination)
        completed = []
        for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            client = NotebookClient(notebook, timeout=240, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
            client.execute()
            nbformat.write(notebook, path)
            completed.append({"notebook": path.name, "code_cells": sum(c.cell_type == "code" for c in notebook.cells), "fresh_kernel": True, "errors": 0})
            print("PASS", path.name, flush=True)
        (ROOT / "results/notebook_execution.json").write_text(json.dumps(completed, indent=2) + "\n")


if __name__ == "__main__":
    execute()
