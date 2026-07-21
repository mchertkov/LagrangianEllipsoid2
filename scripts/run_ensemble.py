#!/usr/bin/env python3
"""Run one named ensemble from configs/ensembles.json.

The default behavior skips files that already exist. Expensive production runs are
serial by default; use --workers to opt into multiprocessing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command(cfg: dict, seed: int) -> list[str]:
    runner = "run_one_seed.py" if cfg.get("mode", "full") == "full" else "run_shape_control.py"
    return [
        sys.executable, str(ROOT / "scripts" / runner),
        "--name", str(cfg["name"]), "--seed", str(seed),
        "--outdir", str(ROOT / cfg["outdir"]),
        "--zeta", str(cfg["zeta"]), "--kmax", str(cfg["kmax"]),
        "--ngrid", str(cfg["ngrid"]), "--dt", str(cfg["dt"]),
        "--T", str(cfg["T"]), "--N", str(cfg["N"]),
        "--subsets", str(cfg["subsets"]), "--r0-uv", str(cfg["r0_uv"]),
        "--stride", str(cfg["stride"]), "--tau-model", str(cfg["tau_model"]),
        "--mee-tol", str(cfg["mee_tol"]),
    ]


def run_one(cfg: dict, seed: int, force: bool) -> None:
    out = ROOT / cfg["outdir"] / f"{cfg['name']}_seed{seed:02d}.npz"
    if out.exists() and not force:
        print(f"skip {out.name}")
        return
    subprocess.run(command(cfg, seed), cwd=ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ensemble")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    configs = json.loads((ROOT / "configs" / "ensembles.json").read_text())
    if args.ensemble not in configs:
        raise SystemExit(f"Unknown ensemble {args.ensemble!r}; choose from {sorted(configs)}")
    cfg = configs[args.ensemble]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        list(ex.map(lambda s: run_one(cfg, int(s), args.force), cfg["seeds"]))


if __name__ == "__main__":
    main()
