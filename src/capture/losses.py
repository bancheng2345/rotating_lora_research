"""Capture-side losses for task/data direction guidance."""

from __future__ import annotations

from typing import Dict

import torch

from src.utils.metrics import rowspace_alignment_stats


def capture_alignment_stats(
    a_matrix: torch.Tensor,
    capture_basis: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Measures LoRA A row-space alignment with capture-only directions.

    The optimized loss is 1 - normalized ||Q_A^T Q_capture||_F^2. This is
    equivalent to maximizing the alignment objective up to a constant, while
    keeping the scalar non-negative and easy to log.
    """

    return rowspace_alignment_stats(a_matrix, capture_basis, eps=eps)

