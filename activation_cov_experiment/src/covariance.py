"""Streaming activation covariance utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class CovarianceState:
    """Summary of a streaming covariance computation."""

    count: int
    mean: torch.Tensor
    matrix: torch.Tensor
    trace: float


class StreamingCovariance:
    """Accumulates second-moment statistics without storing all activations."""

    def __init__(self, dim: int, *, dtype: torch.dtype = torch.float32, device: str = "cpu"):
        self.dim = int(dim)
        self.dtype = dtype
        self.device = torch.device(device)
        self.count = 0
        self.sum_vector = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
        self.sum_outer = torch.zeros((self.dim, self.dim), dtype=self.dtype, device=self.device)

    def update(self, values: torch.Tensor) -> None:
        """Adds a [tokens, hidden] tensor into the streaming state."""

        if values.numel() == 0:
            return
        if values.ndim != 2 or values.shape[-1] != self.dim:
            raise ValueError(
                f"Expected values with shape [tokens, {self.dim}], got {tuple(values.shape)}."
            )
        matrix = values.to(device=self.device, dtype=self.dtype)
        self.count += int(matrix.shape[0])
        self.sum_vector += matrix.sum(dim=0)
        self.sum_outer += matrix.T @ matrix

    def finalize(self, *, centered: bool) -> CovarianceState:
        """Returns the mean and second-moment/covariance matrix."""

        if self.count == 0:
            raise ValueError("Cannot finalize covariance with zero samples.")
        count_float = float(self.count)
        mean = self.sum_vector / count_float
        second_moment = self.sum_outer / count_float
        if centered:
            matrix = second_moment - torch.outer(mean, mean)
        else:
            matrix = second_moment
        matrix = 0.5 * (matrix + matrix.T)
        trace = float(torch.trace(matrix).detach().cpu().item())
        return CovarianceState(count=self.count, mean=mean, matrix=matrix, trace=trace)


def top_eigenpairs(
    matrix: torch.Tensor,
    *,
    max_k: int,
) -> Dict[str, torch.Tensor]:
    """Computes top-k eigenpairs for a symmetric covariance matrix."""

    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    k = min(int(max_k), eigenvectors.shape[1])
    return {
        "eigenvalues": eigenvalues[:k],
        "eigenvectors": eigenvectors[:, :k],
        "all_eigenvalues": eigenvalues,
    }
