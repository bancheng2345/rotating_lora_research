#!/usr/bin/env python
"""Plots rank sweep results from completed runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_rotating_lora_research")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories to include.")
    parser.add_argument("--output", default="results/figures/rank_sweep.png", help="Output figure path.")
    return parser.parse_args()


def load_run(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "final_summary.json").read_text(encoding="utf-8"))
    config = json.loads(json.dumps({}))
    import yaml

    config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    return {
        "method": summary["method"],
        "best_eval_metric": summary["best_eval_metric"],
        "rank": config["lora"]["rank"],
    }


def main() -> None:
    args = parse_args()
    rows = [load_run(Path(run_dir)) for run_dir in args.run_dirs]
    frame = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5))
    for method, method_df in frame.groupby("method"):
        method_df = method_df.sort_values("rank")
        plt.plot(method_df["rank"], method_df["best_eval_metric"], marker="o", label=method)
    plt.xlabel("LoRA rank")
    plt.ylabel("Best eval metric")
    plt.title("Rank Sweep")
    plt.legend()
    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


if __name__ == "__main__":
    main()

