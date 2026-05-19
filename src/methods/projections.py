"""Low-rank projection utilities used by OPLoRA and PC-LoRA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ProjectionBasis:
    """Stores truncated singular-vector bases for left/right projections."""

    left_basis: Optional[torch.Tensor]
    right_basis: Optional[torch.Tensor]
    rank_k: int
    left_source: str = "svd"
    right_source: str = "svd"


def validate_projection_rank(
    rank_k: int,
    out_features: int,
    in_features: int,
    mode: str = "clip",
) -> int:
    """Validates or clips projection rank against a matrix shape."""

    max_rank = min(out_features, in_features)
    if rank_k < 0:
        raise ValueError(f"projection_rank_k must be >= 0, got {rank_k}")
    if rank_k <= max_rank:
        return rank_k
    if mode == "clip":
        return max_rank
    if mode == "error":
        raise ValueError(
            f"projection_rank_k={rank_k} exceeds min(d_out, d_in)={max_rank}"
        )
    raise ValueError(f"Unknown projection rank mode: {mode}")


def random_orthonormal_basis(
    dim: int,
    rank_k: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generates a random orthonormal basis with QR decomposition."""

    if rank_k == 0:
        return torch.empty(dim, 0, device=device, dtype=dtype)
    matrix = torch.randn(dim, rank_k, device=device, dtype=dtype, generator=generator)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q[:, :rank_k]


def _move_basis_like(basis: Optional[torch.Tensor], tensor: torch.Tensor) -> Optional[torch.Tensor]:
    if basis is None or basis.numel() == 0:
        return None
    return basis.to(device=tensor.device, dtype=tensor.dtype)


def apply_right_projection(
    inputs: torch.Tensor,
    right_basis: Optional[torch.Tensor],
) -> torch.Tensor:
    """Applies P_R x = x - V_k (V_k^T x) without materializing P_R."""

    basis = _move_basis_like(right_basis, inputs)
    if basis is None:
        return inputs
    coeff = torch.matmul(inputs, basis)
    return inputs - torch.matmul(coeff, basis.transpose(0, 1))


def apply_left_projection(
    outputs: torch.Tensor,
    left_basis: Optional[torch.Tensor],
) -> torch.Tensor:
    """Applies P_L y = y - U_k (U_k^T y) without materializing P_L."""

    basis = _move_basis_like(left_basis, outputs)
    if basis is None:
        return outputs
    coeff = torch.matmul(outputs, basis)
    return outputs - torch.matmul(coeff, basis.transpose(0, 1))


def left_project_weight(
    weight: torch.Tensor,
    left_basis: Optional[torch.Tensor],
) -> torch.Tensor:
    """Applies P_L W = W - U_k (U_k^T W) using low-rank structure."""

    basis = _move_basis_like(left_basis, weight)
    if basis is None:
        return weight
    return weight - torch.matmul(basis, torch.matmul(basis.transpose(0, 1), weight))


def right_project_weight(
    weight: torch.Tensor,
    right_basis: Optional[torch.Tensor],
) -> torch.Tensor:
    """Applies W P_R = W - (W V_k) V_k^T using low-rank structure."""

    basis = _move_basis_like(right_basis, weight)
    if basis is None:
        return weight
    return weight - torch.matmul(torch.matmul(weight, basis), basis.transpose(0, 1))

