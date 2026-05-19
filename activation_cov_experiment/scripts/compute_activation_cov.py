#!/usr/bin/env python
"""Computes activation covariance top-k eigen-directions from collected activations."""

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

from activation_cov_experiment.src.covariance import StreamingCovariance, top_eigenpairs
from activation_cov_experiment.src.io_utils import activation_run_dir, direction_run_dir, ensure_dir, load_activation_metadata, load_config_stack, save_json, save_yaml
from activation_cov_experiment.src.subspace_metrics import orthogonality_error


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
    """Yields activation chunk tensors truncated to the requested token budget."""

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
        yield tensor
        if max_tokens is not None and consumed >= int(max_tokens):
            break


def main() -> None:
    args = parse_args()
    config = load_config_stack(args.config, base_config_path=args.base_config, overrides=args.override)
    activations_dir = activation_run_dir(config)
    metadata = load_activation_metadata(activations_dir / "metadata.json")
    output_dir = ensure_dir(direction_run_dir(config) / "activation_cov")
    save_yaml(output_dir / "config_resolved.yaml", config)

    top_k_values = sorted({int(value) for value in config["covariance"].get("top_k_values", [4, 8, 16, 32])})
    max_tokens = config["covariance"].get("max_tokens")
    dtype_name = str(config["covariance"].get("dtype_for_cov", "float32"))
    dtype = getattr(torch, dtype_name)
    centered = bool(config["covariance"].get("centered", False))

    summary_rows: List[Dict[str, Any]] = []
    for target in metadata["targets"]:
        streamer = StreamingCovariance(int(target["hidden_dim"]), dtype=dtype, device="cpu")
        for tensor in iter_chunk_tensors(target, activation_root=activations_dir, max_tokens=max_tokens):
            streamer.update(tensor)
        state = streamer.finalize(centered=centered)
        max_k = min(max(top_k_values), int(target["hidden_dim"]))
        eig = top_eigenpairs(state.matrix, max_k=max_k)
        eigenvalues = eig["eigenvalues"].clamp_min(0.0)
        eigenvectors = eig["eigenvectors"]
        total_energy = max(state.trace, 1e-8)
        cumulative = torch.cumsum(eigenvalues, dim=0) / total_energy
        explained = eigenvalues / total_energy
        metrics = {
            str(k): float(cumulative[min(k, cumulative.shape[0]) - 1].detach().cpu().item())
            for k in top_k_values
            if k <= cumulative.shape[0]
        }
        explained_at_k = {
            str(k): float(explained[: min(k, explained.shape[0])].sum().detach().cpu().item())
            for k in top_k_values
            if k <= explained.shape[0]
        }
        payload = {
            "module_name": target["module_name"],
            "module_key": target["module_key"],
            "layer_index": target["layer_index"],
            "count": state.count,
            "centered": centered,
            "mean": state.mean,
            "matrix_trace": state.trace,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "top_k_values": top_k_values,
            "explained_variance_ratio_at_k": explained_at_k,
            "cumulative_energy_at_k": metrics,
            "orthogonality_error": orthogonality_error(eigenvectors),
        }
        torch.save(payload, output_dir / f"{target['module_key']}.pt")
        summary_rows.append(
            {
                "module_name": target["module_name"],
                "module_key": target["module_key"],
                "layer_index": target["layer_index"],
                "num_tokens": state.count,
                "hidden_dim": target["hidden_dim"],
                "centered": centered,
                "matrix_trace": state.trace,
                "top_eigenvalues": [float(value) for value in eigenvalues[: min(8, len(eigenvalues))].tolist()],
                "explained_variance_ratio_at_k": explained_at_k,
                "cumulative_energy_at_k": metrics,
                "orthogonality_error": orthogonality_error(eigenvectors),
            }
        )

    save_json(output_dir / "summary.json", {"targets": summary_rows, "top_k_values": top_k_values, "centered": centered})
    print(json.dumps({"output_dir": str(output_dir), "targets": summary_rows}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
