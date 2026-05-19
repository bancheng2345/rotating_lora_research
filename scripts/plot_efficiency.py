#!/usr/bin/env python
"""Plots efficiency metrics for one or more runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_rotating_lora_research")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache_rotating_lora_research")

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=None, help="Single run directory.")
    parser.add_argument("--run-dirs", nargs="+", default=None, help="Explicit run directories.")
    parser.add_argument("--root", default=None, help="Root directory containing nested run directories.")
    parser.add_argument("--compare_methods", default=None, help="Comma-separated method filter when using --root.")
    parser.add_argument("--output-dir", default="results/figures", help="Figure output directory.")
    return parser.parse_args()


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def discover_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        return [Path(args.run_dir)]
    if args.run_dirs:
        return [Path(path) for path in args.run_dirs]
    if not args.root:
        raise ValueError("Provide --run_dir, --run-dirs, or --root.")
    methods = None
    if args.compare_methods:
        methods = {method.strip() for method in args.compare_methods.split(",") if method.strip()}
    run_dirs = []
    for config_path in Path(args.root).rglob("config_resolved.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if methods is None or config["lora"]["method"] in methods:
            run_dirs.append(config_path.parent)
    return sorted(run_dirs)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for run_dir in discover_run_dirs(args):
        frame = load_jsonl(run_dir / "efficiency_metrics.jsonl")
        if frame.empty:
            continue
        config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
        frame["method"] = config["lora"]["method"]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No efficiency_metrics.jsonl files found.")

    efficiency = pd.concat(frames, ignore_index=True)
    grouped = efficiency.groupby(["method", "step"], as_index=False).mean(numeric_only=True)

    plt.figure(figsize=(8, 5))
    for method, method_frame in grouped.groupby("method"):
        plt.plot(method_frame["step"], method_frame["step_time_sec"], label=f"{method} step_time")
    plt.xlabel("Step")
    plt.ylabel("Step Time (sec)")
    plt.title("Efficiency: Step Time")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "efficiency_step_time.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for method, method_frame in grouped.groupby("method"):
        memory_column = None
        for candidate in ["peak_gpu_memory_gb", "current_gpu_memory_gb", "gpu_memory_allocated_mb"]:
            if candidate in method_frame.columns:
                memory_column = candidate
                break
        if memory_column is None:
            continue
        plt.plot(method_frame["step"], method_frame[memory_column], label=f"{method} {memory_column}")
    plt.xlabel("Step")
    plt.ylabel("Peak GPU Memory (GB)")
    plt.title("Efficiency: GPU Memory")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "efficiency_gpu_memory.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for method, method_frame in grouped.groupby("method"):
        plt.plot(method_frame["step"], method_frame["capture_metric_time_sec"], label=f"{method} capture")
        plt.plot(method_frame["step"], method_frame["rho_metric_time_sec"], linestyle="--", label=f"{method} rho")
    plt.xlabel("Step")
    plt.ylabel("Metric Overhead (sec)")
    plt.title("Efficiency: Metric Overhead")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "efficiency_metric_overhead.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
