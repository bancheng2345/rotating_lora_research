#!/usr/bin/env python
"""Plots mechanism and efficiency curves for one or more runs."""

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
    parser.add_argument("--compare_methods", default=None, help="Comma-separated methods when using --root.")
    parser.add_argument("--output-dir", default=None, help="Optional figure output directory override.")
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
        run_dir = config_path.parent
        if methods is None:
            run_dirs.append(run_dir)
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config["lora"]["method"] in methods:
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def default_output_dir(args: argparse.Namespace, run_dirs: list[Path]) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    if args.run_dir and len(run_dirs) == 1:
        return run_dirs[0] / "figures"
    return Path("results/figures")


def mechanism_frame_for_run(run_dir: Path) -> pd.DataFrame:
    frame = load_jsonl(run_dir / "mechanism_metrics.jsonl")
    if frame.empty:
        return frame
    config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    frame["run_dir"] = str(run_dir)
    frame["method"] = config["lora"]["method"]
    frame["projection_rank_k_cfg"] = int(config["lora"].get("projection_rank_k", 0))
    return frame


def efficiency_frame_for_run(run_dir: Path) -> pd.DataFrame:
    frame = load_jsonl(run_dir / "efficiency_metrics.jsonl")
    if frame.empty:
        return frame
    config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    frame["run_dir"] = str(run_dir)
    frame["method"] = config["lora"]["method"]
    return frame


def _save_grouped_line_plot(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    group_key: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(8, 5))
    grouped = frame.groupby([group_key, x], as_index=False).mean(numeric_only=True)
    for key, key_frame in grouped.groupby(group_key):
        if y not in key_frame.columns:
            continue
        plt.plot(key_frame[x], key_frame[y], label=str(key))
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_single_run(run_dir: Path, output_dir: Path) -> None:
    mechanism = mechanism_frame_for_run(run_dir)
    efficiency = efficiency_frame_for_run(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if mechanism.empty:
        raise FileNotFoundError(f"No mechanism metrics found under {run_dir}")

    mechanism_grouped = mechanism.groupby("step", as_index=False).mean(numeric_only=True)
    plt.figure(figsize=(8, 5))
    plt.plot(mechanism_grouped["step"], mechanism_grouped["omega_full_mean"], label="omega_full")
    plt.plot(mechanism_grouped["step"], mechanism_grouped["omega_residual_mean"], label="omega_residual")
    plt.xlabel("Step")
    plt.ylabel("Omega")
    plt.title("Full vs Residual Capture")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "omega_full_vs_residual.png", dpi=200)
    plt.close()

    rho_frame = mechanism[mechanism["rho_k"].notna()].copy()
    plt.figure(figsize=(8, 5))
    if rho_frame["k"].notna().any():
        rho_grouped = rho_frame.groupby(["step", "k"], as_index=False).mean(numeric_only=True)
        for k, k_frame in rho_grouped.groupby("k"):
            plt.plot(k_frame["step"], k_frame["rho_k"], label=f"k={int(k)}")
    else:
        rho_grouped = rho_frame.groupby("step", as_index=False).mean(numeric_only=True)
        plt.plot(rho_grouped["step"], rho_grouped["rho_k"], label="rho_k")
    plt.xlabel("Step")
    plt.ylabel("rho_k")
    plt.title("Subspace Interference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "rho_k_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(mechanism_grouped["step"], mechanism_grouped["orth_error_A"], label="orth_error_A")
    plt.xlabel("Step")
    plt.ylabel("Orth Error")
    plt.title("A Orthogonality Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "orth_error_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(mechanism_grouped["step"], mechanism_grouped["cap_loss"], label="cap_loss")
    plt.xlabel("Step")
    plt.ylabel("Capture Loss")
    plt.title("Capture Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cap_loss_curve.png", dpi=200)
    plt.close()

    if not efficiency.empty:
        efficiency_grouped = efficiency.groupby("step", as_index=False).mean(numeric_only=True)
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(efficiency_grouped["step"], efficiency_grouped["step_time_sec"], label="step_time_sec", color="tab:blue")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Step Time (sec)", color="tab:blue")
        ax2 = ax1.twinx()
        memory_column = None
        for candidate in ["peak_gpu_memory_gb", "current_gpu_memory_gb", "gpu_memory_allocated_mb"]:
            if candidate in efficiency_grouped.columns:
                memory_column = candidate
                break
        if memory_column is not None:
            ax2.plot(efficiency_grouped["step"], efficiency_grouped[memory_column], label=memory_column, color="tab:orange")
            ax2.set_ylabel(memory_column, color="tab:orange")
        fig.tight_layout()
        plt.title("Efficiency Curve")
        plt.savefig(output_dir / "efficiency_curve.png", dpi=200)
        plt.close(fig)


def plot_multi_run(run_dirs: list[Path], output_dir: Path) -> None:
    mechanism_frames = [mechanism_frame_for_run(run_dir) for run_dir in run_dirs]
    mechanism = pd.concat([frame for frame in mechanism_frames if not frame.empty], ignore_index=True)
    if mechanism.empty:
        raise FileNotFoundError("No mechanism metrics found for the provided runs.")

    output_dir.mkdir(parents=True, exist_ok=True)
    mechanism_compare = mechanism.copy()
    mechanism_compare = mechanism_compare[
        mechanism_compare["k"].isna()
        | (mechanism_compare["k"] == mechanism_compare["projection_rank_k_cfg"])
    ]
    _save_grouped_line_plot(
        mechanism_compare,
        x="step",
        y="omega_residual_mean",
        group_key="method",
        output_path=output_dir / "omega_curve.png",
        title="Residual Capture by Method",
        ylabel="omega_residual_mean",
    )
    _save_grouped_line_plot(
        mechanism_compare[mechanism_compare["rho_k"].notna()],
        x="step",
        y="rho_k",
        group_key="method",
        output_path=output_dir / "rho_curve.png",
        title="rho_k by Method",
        ylabel="rho_k",
    )
    _save_grouped_line_plot(
        mechanism_compare,
        x="step",
        y="orth_error_A",
        group_key="method",
        output_path=output_dir / "orth_error_curve.png",
        title="Orthogonality Error by Method",
        ylabel="orth_error_A",
    )
    _save_grouped_line_plot(
        mechanism_compare,
        x="step",
        y="cap_loss",
        group_key="method",
        output_path=output_dir / "cap_loss_curve.png",
        title="Capture Loss by Method",
        ylabel="cap_loss",
    )

    efficiency_frames = [efficiency_frame_for_run(run_dir) for run_dir in run_dirs]
    efficiency = pd.concat([frame for frame in efficiency_frames if not frame.empty], ignore_index=True)
    if not efficiency.empty:
        _save_grouped_line_plot(
            efficiency,
            x="step",
            y="step_time_sec",
            group_key="method",
            output_path=output_dir / "efficiency_curve.png",
            title="Step Time by Method",
            ylabel="step_time_sec",
        )


def main() -> None:
    args = parse_args()
    run_dirs = discover_run_dirs(args)
    output_dir = default_output_dir(args, run_dirs)
    if len(run_dirs) == 1 and args.run_dir:
        plot_single_run(run_dirs[0], output_dir)
        return
    plot_multi_run(run_dirs, output_dir)


if __name__ == "__main__":
    main()
