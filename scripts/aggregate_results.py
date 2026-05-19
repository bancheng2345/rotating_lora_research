#!/usr/bin/env python
"""Aggregates run summaries and mechanism metrics into CSV tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.run_summary import build_final_summary, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root directory containing run subdirectories.")
    parser.add_argument("--out", default="results/tables", help="Output table directory.")
    return parser.parse_args()


def iter_run_dirs(root: Path):
    for path in sorted(root.rglob("config_resolved.yaml")):
        yield path.parent


def mechanism_summary_row(run_dir: Path) -> dict:
    records = load_jsonl(run_dir / "mechanism_metrics.jsonl")
    if not records:
        return {"run_dir": str(run_dir)}
    final_step = max(record["step"] for record in records)
    final_records = [record for record in records if record["step"] == final_step]
    return {
        "run_dir": str(run_dir),
        "step": final_step,
        "rho_k_mean": pd.Series([record["rho_k"] for record in final_records if record.get("rho_k") is not None]).mean(),
        "omega_full_mean": pd.Series([record["omega_full_mean"] for record in final_records if record.get("omega_full_mean") is not None]).mean(),
        "omega_residual_mean": pd.Series([record["omega_residual_mean"] for record in final_records if record.get("omega_residual_mean") is not None]).mean(),
        "orth_error_A_mean": pd.Series([record["orth_error_A"] for record in final_records if record.get("orth_error_A") is not None]).mean(),
        "delta_w_norm_mean": pd.Series([record["delta_w_norm"] for record in final_records if record.get("delta_w_norm") is not None]).mean(),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    mechanism_rows = []
    for run_dir in iter_run_dirs(root):
        summary_path = run_dir / "final_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = build_final_summary(run_dir)
        config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
        summary_rows.append(
            {
                **summary,
                "run_dir": str(run_dir),
                "notes": config.get("experiment", {}).get("notes", ""),
                "method": config["lora"]["method"],
            }
        )
        mechanism_rows.append(mechanism_summary_row(run_dir))

    summary_frame = pd.DataFrame(summary_rows)
    mechanism_frame = pd.DataFrame(mechanism_rows)
    main_methods = {"lora", "oplora", "lora_nsc", "pc_lora", "pc_lora_weighted"}
    ablation_mask = summary_frame["notes"].fillna("").str.contains("ablation", case=False)
    summary_frame[summary_frame["method"].isin(main_methods) & ~ablation_mask].to_csv(
        out_dir / "main_results.csv",
        index=False,
    )
    summary_frame[ablation_mask | ~summary_frame["method"].isin(main_methods)].to_csv(
        out_dir / "ablation_results.csv",
        index=False,
    )
    mechanism_frame.to_csv(out_dir / "mechanism_summary.csv", index=False)


if __name__ == "__main__":
    main()
