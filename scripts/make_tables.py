#!/usr/bin/env python
"""Aggregates final summaries into CSV tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories to include.")
    parser.add_argument("--tables-dir", default="results/tables", help="Output directory for CSV tables.")
    return parser.parse_args()


def load_row(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "final_summary.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    return {
        "run_dir": str(run_dir),
        "method": summary["method"],
        "dataset_name": summary["dataset_name"],
        "model_name": summary["model_name"],
        "rank": config["lora"]["rank"],
        "projection_rank_k": config["lora"]["projection_rank_k"],
        "best_eval_metric": summary["best_eval_metric"],
        "notes": config["experiment"].get("notes", ""),
    }


def main() -> None:
    args = parse_args()
    rows = [load_row(Path(run_dir)) for run_dir in args.run_dirs]
    frame = pd.DataFrame(rows)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    main_methods = {"lora", "oplora", "lora_nsc", "pc_lora", "pc_lora_weighted"}
    ablation_mask = frame["notes"].fillna("").str.contains("ablation", case=False)
    main_mask = frame["method"].isin(main_methods) & ~ablation_mask
    frame[main_mask].to_csv(tables_dir / "main_results.csv", index=False)
    frame[ablation_mask | ~frame["method"].isin(main_methods)].to_csv(tables_dir / "ablation.csv", index=False)


if __name__ == "__main__":
    main()
