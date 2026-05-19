#!/usr/bin/env python
"""Compares ActivationCov, SVD, and M_WA directions with input/output capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from activation_cov_experiment.src.io_utils import (
    activation_run_dir,
    build_model_and_tokenizer,
    direction_run_dir,
    load_activation_metadata,
    load_config_stack,
    metrics_run_dir,
    save_csv_rows,
    save_json,
)
from activation_cov_experiment.src.subspace_metrics import (
    EPS,
    capture_energy,
    mean_std,
    orthogonality_error,
    principal_angle_stats,
    random_orthonormal,
)
from activation_cov_experiment.src.svd_utils import operation_matrix


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
    """Computes mean/std input capture-energy statistics."""

    ratios: List[torch.Tensor] = []
    for chunk in chunks:
        ratios.append(capture_energy(chunk, basis))
    merged = torch.cat(ratios, dim=0)
    return {
        "mean": float(merged.mean().detach().cpu().item()),
        "std": float(merged.std(unbiased=False).detach().cpu().item()),
    }


def output_capture_energy(values: torch.Tensor, basis: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Returns per-token output-energy capture ratios after input projection.

    OutputCapture(Q, z) = || W0 Q Q^T z ||^2 / || W0 z ||^2.
    """

    values_f = values.to(dtype=torch.float32)
    basis_f = basis.to(dtype=torch.float32)
    weight_f = weight.to(dtype=torch.float32)
    projected_inputs = (values_f @ basis_f) @ basis_f.T
    full_outputs = values_f @ weight_f.T
    projected_outputs = projected_inputs @ weight_f.T
    numerator = (projected_outputs * projected_outputs).sum(dim=-1)
    denominator = (full_outputs * full_outputs).sum(dim=-1).clamp_min(EPS)
    return (numerator / denominator).clamp(0.0, 1.0)


def output_capture_stats_for_basis(
    chunks: List[torch.Tensor],
    basis: torch.Tensor,
    weight: torch.Tensor,
) -> Dict[str, float]:
    """Computes mean/std output capture-energy statistics."""

    ratios: List[torch.Tensor] = []
    for chunk in chunks:
        ratios.append(output_capture_energy(chunk, basis, weight))
    merged = torch.cat(ratios, dim=0)
    return {
        "mean": float(merged.mean().detach().cpu().item()),
        "std": float(merged.std(unbiased=False).detach().cpu().item()),
    }


def flatten_method_stats(prefix: str, input_stats: Dict[str, float], output_stats: Dict[str, float]) -> Dict[str, float]:
    """Flattens capture stats for one direction method into CSV-friendly keys."""

    return {
        f"input_capture_{prefix}_mean": input_stats["mean"],
        f"input_capture_{prefix}_std": input_stats["std"],
        f"output_capture_{prefix}_mean": output_stats["mean"],
        f"output_capture_{prefix}_std": output_stats["std"],
        f"input_residual_{prefix}_mean": 1.0 - input_stats["mean"],
        f"output_residual_{prefix}_mean": 1.0 - output_stats["mean"],
    }


def load_basis_payloads(direction_dir: Path, module_key: str) -> Dict[str, torch.Tensor]:
    """Loads direction payloads for one target module."""

    act_payload = torch.load(direction_dir / "activation_cov" / f"{module_key}.pt", map_location="cpu")
    svd_payload = torch.load(direction_dir / "weight_svd" / f"{module_key}.pt", map_location="cpu")
    mwa_payload = torch.load(direction_dir / "mwa" / f"{module_key}.pt", map_location="cpu")
    return {
        "actcov": act_payload["eigenvectors"],
        "svd": svd_payload["input_vectors"],
        "mwa": mwa_payload["input_vectors"],
        "mwa_mapped": mwa_payload["c_sqrt_mapped_vectors"],
    }


def main() -> None:
    args = parse_args()
    config = load_config_stack(args.config, base_config_path=args.base_config, overrides=args.override)
    activation_dir = activation_run_dir(config)
    direction_dir = direction_run_dir(config)
    metrics_dir = metrics_run_dir(config)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_activation_metadata(activation_dir / "metadata.json")
    top_k_values = sorted(
        {
            int(value)
            for value in (
                config.get("joint", {}).get("top_k_values")
                or config["covariance"].get("top_k_values")
                or config["svd"].get("top_k_values", [4, 8, 16, 32])
            )
        }
    )
    max_tokens = config["covariance"].get("max_tokens")
    random_trials = int(config["metrics"].get("random_trials", 3))

    device = torch.device("cpu")
    model, _, _ = build_model_and_tokenizer(config, device=device)
    model.eval()
    modules = dict(model.named_modules())

    csv_rows: List[Dict[str, Any]] = []
    json_rows: List[Dict[str, Any]] = []

    for target in metadata["targets"]:
        activation_chunks = list(iter_chunk_tensors(target, activation_root=activation_dir, max_tokens=max_tokens))
        if not activation_chunks:
            continue
        module_key = str(target["module_key"])
        bases = load_basis_payloads(direction_dir, module_key)
        weight = operation_matrix(modules[target["module_name"]])
        hidden_dim = int(target["hidden_dim"])
        max_supported_k = min(hidden_dim, *(basis.shape[1] for basis in bases.values()))

        for k in top_k_values:
            if k > max_supported_k:
                continue
            q_act = bases["actcov"][:, :k]
            q_svd = bases["svd"][:, :k]
            q_mwa = bases["mwa"][:, :k]
            q_mwa_mapped = bases["mwa_mapped"][:, :k]
            method_bases = {
                "actcov": q_act,
                "svd": q_svd,
                "mwa": q_mwa,
                "mwa_mapped": q_mwa_mapped,
            }

            row: Dict[str, Any] = {
                "layer_index": target["layer_index"],
                "module_name": target["module_name"],
                "module_key": module_key,
                "k": k,
                "num_tokens_used": int(sum(chunk.shape[0] for chunk in activation_chunks)),
                "orthogonality_error_actcov": orthogonality_error(q_act),
                "orthogonality_error_svd": orthogonality_error(q_svd),
                "orthogonality_error_mwa": orthogonality_error(q_mwa),
                "orthogonality_error_mwa_mapped": orthogonality_error(q_mwa_mapped),
            }
            row.update({f"actcov_vs_svd_{key}": value for key, value in principal_angle_stats(q_act, q_svd).items()})
            row.update({f"mwa_vs_svd_{key}": value for key, value in principal_angle_stats(q_mwa, q_svd).items()})
            row.update({f"mwa_vs_actcov_{key}": value for key, value in principal_angle_stats(q_mwa, q_act).items()})

            for name, basis in method_bases.items():
                row.update(
                    flatten_method_stats(
                        name,
                        capture_stats_for_basis(activation_chunks, basis),
                        output_capture_stats_for_basis(activation_chunks, basis, weight),
                    )
                )

            random_input_means: List[float] = []
            random_output_means: List[float] = []
            for trial in range(random_trials):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(31_000 + 997 * trial + 101 * int(target["layer_index"] or 0) + k)
                q_random = random_orthonormal(hidden_dim, k, generator=generator)
                random_input_means.append(capture_stats_for_basis(activation_chunks, q_random)["mean"])
                random_output_means.append(output_capture_stats_for_basis(activation_chunks, q_random, weight)["mean"])
            random_input_mean, random_input_std = mean_std(random_input_means)
            random_output_mean, random_output_std = mean_std(random_output_means)
            row.update(
                {
                    "input_capture_random_mean": random_input_mean,
                    "input_capture_random_std": random_input_std,
                    "output_capture_random_mean": random_output_mean,
                    "output_capture_random_std": random_output_std,
                    "input_gain_mwa_minus_svd": row["input_capture_mwa_mean"] - row["input_capture_svd_mean"],
                    "output_gain_mwa_minus_svd": row["output_capture_mwa_mean"] - row["output_capture_svd_mean"],
                    "input_gain_mwa_minus_actcov": row["input_capture_mwa_mean"] - row["input_capture_actcov_mean"],
                    "output_gain_mwa_minus_actcov": row["output_capture_mwa_mean"] - row["output_capture_actcov_mean"],
                }
            )
            csv_rows.append(row)
            json_rows.append(row)

    save_csv_rows(metrics_dir / "mwa_direction_comparison.csv", csv_rows)
    save_json(
        metrics_dir / "mwa_direction_comparison.json",
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
