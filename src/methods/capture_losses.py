"""Capture and orthogonality regularizers for PC-LoRA."""

from __future__ import annotations

from typing import Optional

import torch

from src.utils.metrics import (
    compute_omega_values,
    orthogonality_error_value,
    orthogonality_loss_value,
)


def compute_omega(
    a_matrix: torch.Tensor,
    activations: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes omega_A(z) = sqrt(1 - captured / ||z||^2) for a batch of activations."""

    return compute_omega_values(a_matrix, activations, eps=eps)


def reduce_capture_loss(
    omega: torch.Tensor,
    *,
    weighting: str = "none",
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduces omega values into a scalar capture loss."""

    if omega.numel() == 0:
        return omega.new_tensor(0.0)
    loss = omega.mean()
    if weighting == "none":
        return loss
    if weighting == "loss":
        if weight is None:
            raise ValueError("Loss-weighted capture requires a detached loss weight.")
        return loss * weight
    if weighting == "grad_norm":
        raise NotImplementedError("grad_norm weighting is reserved for a later revision.")
    raise ValueError(f"Unknown capture weighting: {weighting}")


def orthogonality_loss(a_matrix: torch.Tensor) -> torch.Tensor:
    """Computes ||A A^T - I||_F^2 for a LoRA A matrix."""

    return orthogonality_loss_value(a_matrix)


def orthogonality_error(a_matrix: torch.Tensor) -> torch.Tensor:
    """Returns the Frobenius error used as a mechanism metric."""

    return orthogonality_error_value(a_matrix).detach()
