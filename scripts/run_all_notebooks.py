#!/usr/bin/env python3
"""Execute all six from-scratch notebooks in numerical order.

The trajectory-generation notebook is resumable: with its default FORCE=False,
existing trajectories are verified and skipped. Delete/move data/raw or set
FORCE=True in that notebook to regenerate the complete campaign.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks" / "from_scratch"
NAMES = [
    "00_environment_and_smoke_test.ipynb",
    "01_generate_paper_trajectories.ipynb",
    "02_analyze_ensembles.ipynb",
    "03_fit_reduced_models.ipynb",
    "04_generate_paper_figures.ipynb",
    "05_validate_and_export.ipynb",
]
for name in NAMES:
    path = NB_DIR / name
    print(f"\n=== {name} ===", flush=True)
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
        "--execute", "--inplace", str(path), "--ExecutePreprocessor.timeout=-1",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
