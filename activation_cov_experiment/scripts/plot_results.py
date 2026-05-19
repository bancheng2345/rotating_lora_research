#!/usr/bin/env python
"""Plots activation-covariance vs SVD comparison results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/activation_cov_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/activation_cov_cache")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from activation_cov_experiment.src.io_utils import direction_run_dir, figures_run_dir, load_config_stack, metrics_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument(
        "--base-config",
        default="activation_cov_experiment/configs/activation_cov_base.yaml",
    )
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_stack(args.config, base_config_path=args.base_config, overrides=args.override)
    metrics_dir = metrics_run_dir(config)
    directions_dir = direction_run_dir(config) / "activation_cov"
    figures_dir = figures_run_dir(config)
    figures_dir.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(metrics_dir / "direction_comparison.csv")
    actcov_summary = json.loads((directions_dir / "summary.json").read_text(encoding="utf-8"))

    plt.figure(figsize=(7, 4))
    for target in actcov_summary["targets"]:
        ks = sorted(int(key) for key in target["cumulative_energy_at_k"].keys())
        values = [target["cumulative_energy_at_k"][str(k)] for k in ks]
        label = f"L{target['layer_index']}:{target['module_name'].split('.')[-1]}"
        plt.plot(ks, values, marker="o", label=label)
    plt.xlabel("k")
    plt.ylabel("Cumulative explained energy")
    plt.title("Activation covariance explained variance")
    if len(actcov_summary["targets"]) <= 8:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "explained_variance_curve.png", dpi=180)
    plt.close()

    agg = comparison.groupby("k", as_index=False).agg(
        capture_actcov_mean=("capture_actcov_mean", "mean"),
        capture_svd_mean=("capture_svd_mean", "mean"),
        capture_random_mean=("capture_random_mean", "mean"),
        residual_actcov_mean=("residual_actcov_mean", "mean"),
        residual_svd_mean=("residual_svd_mean", "mean"),
        residual_random_mean=("residual_random_mean", "mean"),
    )

    plt.figure(figsize=(7, 4))
    plt.plot(agg["k"], agg["capture_actcov_mean"], marker="o", label="ActivationCov")
    plt.plot(agg["k"], agg["capture_svd_mean"], marker="s", label="WeightSVD")
    plt.plot(agg["k"], agg["capture_random_mean"], marker="^", label="Random")
    plt.xlabel("k")
    plt.ylabel("Captured activation energy")
    plt.title("Capture energy vs k")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "capture_energy_vs_k.png", dpi=180)
    plt.close()

    principal = comparison.copy()
    principal["label"] = principal.apply(
        lambda row: f"L{int(row['layer_index'])}:{str(row['module_name']).split('.')[-1]}",
        axis=1,
    )
    principal = principal.groupby("label", as_index=False)["mean_angle_deg"].mean()
    plt.figure(figsize=(max(6, 0.6 * len(principal)), 4))
    plt.bar(principal["label"], principal["mean_angle_deg"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean principal angle (deg)")
    plt.title("Principal angles: ActivationCov vs WeightSVD")
    plt.tight_layout()
    plt.savefig(figures_dir / "principal_angles.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(agg["k"], agg["residual_actcov_mean"], marker="o", label="ActivationCov residual")
    plt.plot(agg["k"], agg["residual_svd_mean"], marker="s", label="WeightSVD residual")
    plt.plot(agg["k"], agg["residual_random_mean"], marker="^", label="Random residual")
    plt.xlabel("k")
    plt.ylabel("Residual activation energy")
    plt.title("Residual energy vs k")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "residual_energy_vs_k.png", dpi=180)
    plt.close()

    max_k = comparison["k"].max()
    heatmap_frame = comparison[comparison["k"] == max_k].copy()
    heatmap_frame["layer_index"] = heatmap_frame["layer_index"].astype(int)
    heatmap_frame["module"] = heatmap_frame["module_name"].map(lambda name: str(name).split(".")[-1])
    pivot = heatmap_frame.pivot(index="layer_index", columns="module", values="capture_gain_actcov_minus_svd")
    plt.figure(figsize=(max(5, 1.3 * len(pivot.columns)), max(4, 0.7 * len(pivot.index))))
    im = plt.imshow(pivot.values, aspect="auto", cmap="coolwarm")
    plt.colorbar(im, label="ActivationCov capture gain over SVD")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Module")
    plt.ylabel("Layer")
    plt.title(f"Capture gain heatmap @ k={max_k}")
    plt.tight_layout()
    plt.savefig(figures_dir / "layer_module_heatmap.png", dpi=180)
    plt.close()

    print(json.dumps({"figures_dir": str(figures_dir)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
