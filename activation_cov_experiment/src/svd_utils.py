"""Weight SVD helpers for comparing activation-aware and weight-only directions."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def operation_matrix(module: nn.Module) -> torch.Tensor:
    """Returns the effective [d_out, d_in] matrix used by a linear-like module."""

    if not hasattr(module, "weight"):
        raise ValueError(f"Module {module} has no weight.")
    weight = module.weight.detach().to(dtype=torch.float32, device="cpu")
    if isinstance(module, nn.Linear):
        return weight
    if hasattr(module, "nf"):  # GPT-2 Conv1D stores [d_in, d_out]
        return weight.T
    if weight.shape[0] <= weight.shape[1]:
        return weight
    return weight


def compute_weight_svd(
    module: nn.Module,
    *,
    max_k: int,
    use_lowrank: bool,
) -> Dict[str, torch.Tensor]:
    """Computes weight SVD and returns input/output singular subspaces."""

    matrix = operation_matrix(module)
    max_rank = min(matrix.shape)
    k = min(int(max_k), max_rank)
    if use_lowrank and k < max_rank:
        q = min(max(k + 8, k), max_rank)
        u, singular_values, v = torch.svd_lowrank(matrix, q=q)
        vh = v.T
    else:
        u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    return {
        "matrix": matrix,
        "singular_values": singular_values[:k],
        "input_vectors": vh[:k].T,
        "output_vectors": u[:, :k],
    }
