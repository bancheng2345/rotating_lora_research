#!/usr/bin/env python
"""Computes data-weight joint directions M_WA = C_z^{1/2} W0^T W0 C_z^{1/2}."""

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

from activation_cov_experiment.src.io_utils import (
    activation_run_dir,
    build_model_and_tokenizer,
    direction_run_dir,
    ensure_dir,
    load_activation_metadata,
    load_config_stack,
    save_json,
    save_yaml,
)
from activation_cov_experiment.src.subspace_metrics import orthogonality_error, orthonormalize_columns
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


def compute_joint_basis(
    *,
    covariance_eigenvectors: torch.Tensor,
    covariance_eigenvalues: torch.Tensor,
    weight_matrix: torch.Tensor,
    max_k: int,
    eps: float,
) -> Dict[str, torch.Tensor]:
    """Computes top joint directions from a low-rank C_z approximation.

    With C_z ~= U diag(lambda) U^T, the non-zero part of
    M_WA = C_z^{1/2} W0^T W0 C_z^{1/2} is:

        U [diag(sqrt(lambda)) U^T W0^T W0 U diag(sqrt(lambda))] U^T.

    We solve the bracketed small eigenproblem and map eigenvectors back to
    input space as U b. A second C_z^{1/2}-mapped basis is also stored for
    diagnostics, but `input_vectors` is the direct M_WA eigenbasis.
    """

    u = covariance_eigenvectors.to(dtype=torch.float32, device="cpu")
    eigenvalues = covariance_eigenvalues.to(dtype=torch.float32, device="cpu").clamp_min(0.0)
    keep = eigenvalues > float(eps)
    if not bool(keep.any()):
        raise ValueError("Activation covariance eigenvalues are all zero; cannot compute M_WA.")
    u = u[:, keep]
    eigenvalues = eigenvalues[keep]
    rank = min(int(max_k), u.shape[1])

    sqrt_lambda = torch.sqrt(eigenvalues).clamp_min(float(eps))
    c_sqrt_basis = u * sqrt_lambda.unsqueeze(0)
    weighted_basis = weight_matrix.to(dtype=torch.float32, device="cpu") @ c_sqrt_basis
    small_mwa = weighted_basis.T @ weighted_basis
    small_mwa = 0.5 * (small_mwa + small_mwa.T)
    eigvals, eigvecs = torch.linalg.eigh(small_mwa)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(0.0)
    eigvecs = eigvecs[:, order]

    # Direct eigenvectors of M_WA in the original input coordinate subspace.
    input_vectors = orthonormalize_columns(u @ eigvecs[:, :rank])
    # C_z^{1/2}-mapped variant for diagnostics.
    mapped_vectors = orthonormalize_columns(c_sqrt_basis @ eigvecs[:, :rank])
    return {
        "joint_eigenvalues": eigvals[:rank],
        "input_vectors": input_vectors,
        "c_sqrt_mapped_vectors": mapped_vectors,
        "covariance_rank_used": torch.tensor(int(u.shape[1])),
        "small_mwa_trace": torch.trace(small_mwa),
    }


def main() -> None:
    args = parse_args()
    config = load_config_stack(args.config, base_config_path=args.base_config, overrides=args.override)
    activations_dir = activation_run_dir(config)
    metadata = load_activation_metadata(activations_dir / "metadata.json")
    direction_dir = direction_run_dir(config)
    activation_cov_dir = direction_dir / "activation_cov"
    output_dir = ensure_dir(direction_dir / "mwa")
    save_yaml(output_dir / "config_resolved.yaml", config)

    device = torch.device("cpu")
    model, _, model_name = build_model_and_tokenizer(config, device=device)
    model.eval()
    modules = dict(model.named_modules())

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
    max_k = max(top_k_values)
    eps = float(config.get("joint", {}).get("eps", 1e-8))

    summary_rows: List[Dict[str, Any]] = []
    for target in metadata["targets"]:
        act_path = activation_cov_dir / f"{target['module_key']}.pt"
        if not act_path.exists():
            raise FileNotFoundError(f"Missing activation covariance directions: {act_path}")
        act_payload = torch.load(act_path, map_location="cpu")
        module = modules[target["module_name"]]
        weight_matrix = operation_matrix(module)
        max_supported_k = min(
            max_k,
            int(target["hidden_dim"]),
            int(act_payload["eigenvectors"].shape[1]),
            int(weight_matrix.shape[1]),
        )
        joint_payload = compute_joint_basis(
            covariance_eigenvectors=act_payload["eigenvectors"][:, :max_supported_k],
            covariance_eigenvalues=act_payload["eigenvalues"][:max_supported_k],
            weight_matrix=weight_matrix,
            max_k=max_supported_k,
            eps=eps,
        )
        payload = {
            "module_name": target["module_name"],
            "module_key": target["module_key"],
            "layer_index": target["layer_index"],
            "model_name_or_path": model_name,
            "top_k_values": top_k_values,
            "matrix_shape": list(weight_matrix.shape),
            "input_vectors": joint_payload["input_vectors"],
            "c_sqrt_mapped_vectors": joint_payload["c_sqrt_mapped_vectors"],
            "joint_eigenvalues": joint_payload["joint_eigenvalues"],
            "covariance_rank_used": int(joint_payload["covariance_rank_used"].item()),
            "small_mwa_trace": float(joint_payload["small_mwa_trace"].item()),
            "orthogonality_error_input": orthogonality_error(joint_payload["input_vectors"]),
            "orthogonality_error_c_sqrt_mapped": orthogonality_error(joint_payload["c_sqrt_mapped_vectors"]),
        }
        torch.save(payload, output_dir / f"{target['module_key']}.pt")
        summary_rows.append(
            {
                "module_name": target["module_name"],
                "module_key": target["module_key"],
                "layer_index": target["layer_index"],
                "matrix_shape": list(weight_matrix.shape),
                "covariance_rank_used": payload["covariance_rank_used"],
                "top_joint_eigenvalues": [
                    float(value)
                    for value in joint_payload["joint_eigenvalues"][: min(8, len(joint_payload["joint_eigenvalues"]))].tolist()
                ],
                "small_mwa_trace": payload["small_mwa_trace"],
                "orthogonality_error_input": payload["orthogonality_error_input"],
                "orthogonality_error_c_sqrt_mapped": payload["orthogonality_error_c_sqrt_mapped"],
            }
        )

    save_json(output_dir / "summary.json", {"targets": summary_rows, "top_k_values": top_k_values, "eps": eps})
    print(json.dumps({"output_dir": str(output_dir), "targets": summary_rows}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
