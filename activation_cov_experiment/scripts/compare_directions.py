#!/usr/bin/env python
"""Compares activation-covariance and weight-SVD directions on real activations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from activation_cov_experiment.src.io_utils import activation_run_dir, direction_run_dir, load_activation_metadata, load_config_stack, metrics_run_dir, save_csv_rows, save_json
from activation_cov_experiment.src.subspace_metrics import (
    capture_energy,
    mean_std,
    orthogonality_error,
    principal_angle_stats,
    random_orthonormal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument(
        "--base-config",
        default="activation_cov_experiment/configs/activation_cov_base.yaml",
    )
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def iter_chunk_tensors(target_record: Dict[str, Any], *, activation_root: Path, max_tokens: int | None):
    """Yields activation chunks truncated to the configured token budget."""

    consumed = 0
    target_dir = activation_root / str(target_record["module_key"])
    for chunk_name in target_record["chunk_files"]:
        tensor = torch.load(target_dir / chunk_name, map_location="cpu")
        if max_tokens is not None:
            remaining = int(max_tokens) - consumed
            if remaining <= 0:
                break
            tensor = tensor[:remaining]
        if tensor.numel() == 0:
            continue
        consumed += int(tensor.shape[0])
        yield tensor.to(dtype=torch.float32)
        if max_tokens is not None and consumed >= int(max_tokens):
            break


def capture_stats_for_basis(chunks: List[torch.Tensor], basis: torch.Tensor) -> Dict[str, float]:
    """Computes mean/std capture-energy statistics for one orthonormal basis."""

    ratios: List[torch.Tensor] = []
    for chunk in chunks:
        ratios.append(capture_energy(chunk, basis))
    merged = torch.cat(ratios, dim=0)
    return {
        "mean": float(merged.mean().detach().cpu().item()),
        "std": float(merged.std(unbiased=False).detach().cpu().item()),
    }


def main() -> None:
    args = parse_args()
    config = load_config_stack(args.config, base_config_path=args.base_config, overrides=args.override)
    activation_dir = activation_run_dir(config)
    direction_dir = direction_run_dir(config)
    metrics_dir = metrics_run_dir(config)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_activation_metadata(activation_dir / "metadata.json")
    top_k_values = sorted({int(value) for value in config["covariance"].get("top_k_values", [4, 8, 16, 32])})
    max_tokens = config["covariance"].get("max_tokens")
    random_trials = int(config["metrics"].get("random_trials", 3))

    csv_rows: List[Dict[str, Any]] = []
    json_rows: List[Dict[str, Any]] = []

    for target in metadata["targets"]:
        act_payload = torch.load(direction_dir / "activation_cov" / f"{target['module_key']}.pt", map_location="cpu")
        svd_payload = torch.load(direction_dir / "weight_svd" / f"{target['module_key']}.pt", map_location="cpu")
        activation_chunks = list(iter_chunk_tensors(target, activation_root=activation_dir, max_tokens=max_tokens))
        if not activation_chunks:
            continue
        hidden_dim = int(target["hidden_dim"])
        max_supported_k = min(hidden_dim, act_payload["eigenvectors"].shape[1], svd_payload["input_vectors"].shape[1])

        for k in top_k_values:
            if k > max_supported_k:
                continue
            q_act = act_payload["eigenvectors"][:, :k]
            q_svd = svd_payload["input_vectors"][:, :k]
            angle_stats = principal_angle_stats(q_act, q_svd)
            act_stats = capture_stats_for_basis(activation_chunks, q_act)
            svd_stats = capture_stats_for_basis(activation_chunks, q_svd)

            random_means: List[float] = []
            for trial in range(random_trials):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(17_000 + 1_003 * trial + 97 * int(target["layer_index"] or 0) + k)
                q_random = random_orthonormal(hidden_dim, k, generator=generator)
                random_means.append(capture_stats_for_basis(activation_chunks, q_random)["mean"])
            random_mean, random_std = mean_std(random_means)

            row = {
                "layer_index": target["layer_index"],
                "module_name": target["module_name"],
                "module_key": target["module_key"],
                "k": k,
                "orthogonality_error_actcov": orthogonality_error(q_act),
                "orthogonality_error_svd": orthogonality_error(q_svd),
                **angle_stats,
                "capture_actcov_mean": act_stats["mean"],
                "capture_actcov_std": act_stats["std"],
                "capture_svd_mean": svd_stats["mean"],
                "capture_svd_std": svd_stats["std"],
                "capture_random_mean": random_mean,
                "capture_random_std": random_std,
                "residual_actcov_mean": 1.0 - act_stats["mean"],
                "residual_svd_mean": 1.0 - svd_stats["mean"],
                "residual_random_mean": (None if random_mean is None else 1.0 - random_mean),
                "capture_gain_actcov_minus_svd": act_stats["mean"] - svd_stats["mean"],
                "num_tokens_used": int(sum(chunk.shape[0] for chunk in activation_chunks)),
            }
            csv_rows.append(row)
            json_rows.append(row)

    save_csv_rows(metrics_dir / "direction_comparison.csv", csv_rows)
    save_json(
        metrics_dir / "direction_comparison.json",
        {
            "run_name": str(config["output"]["run_name"]),
            "rows": json_rows,
            "top_k_values": top_k_values,
            "random_trials": random_trials,
        },
    )
    print(json.dumps({"metrics_dir": str(metrics_dir), "num_rows": len(csv_rows)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
