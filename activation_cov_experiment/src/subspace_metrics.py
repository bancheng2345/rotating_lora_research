"""Subspace comparison and activation energy metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch


EPS = 1e-8


@dataclass
class RunningStats:
    """Streaming scalar mean/std tracker."""

    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        tensor = values.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
        self.count += int(tensor.numel())
        self.total += float(tensor.sum().item())
        self.total_sq += float((tensor * tensor).sum().item())
        self.minimum = min(self.minimum, float(tensor.min().item()))
        self.maximum = max(self.maximum, float(tensor.max().item()))

    def to_dict(self, prefix: str) -> Dict[str, float | None]:
        if self.count == 0:
            return {
                f"{prefix}_mean": None,
                f"{prefix}_std": None,
                f"{prefix}_min": None,
                f"{prefix}_max": None,
            }
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        return {
            f"{prefix}_mean": mean,
            f"{prefix}_std": math.sqrt(variance),
            f"{prefix}_min": self.minimum,
            f"{prefix}_max": self.maximum,
        }


def orthogonality_error(q: torch.Tensor) -> float:
    """Returns ||Q^T Q - I||_F for a basis matrix with column vectors."""

    if q.numel() == 0:
        return 0.0
    gram = q.T @ q
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return float(torch.linalg.norm(gram - identity, ord="fro").detach().cpu().item())


def orthonormalize_columns(matrix: torch.Tensor) -> torch.Tensor:
    """Returns a column-orthonormal basis via reduced QR."""

    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q


def random_orthonormal(dim: int, k: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Samples a random Gaussian matrix and orthonormalizes its columns."""

    matrix = torch.randn(dim, k, generator=generator, dtype=torch.float32)
    return orthonormalize_columns(matrix)


def project_onto_basis(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Projects [tokens, hidden] values onto an orthonormal basis without forming QQ^T."""

    return (values @ basis) @ basis.T


def capture_energy(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Returns per-token capture energy ratios in [0, 1]."""

    values_f = values.to(dtype=torch.float32)
    basis_f = basis.to(dtype=torch.float32)
    projected = project_onto_basis(values_f, basis_f)
    numerator = (projected * projected).sum(dim=-1)
    denominator = (values_f * values_f).sum(dim=-1).clamp_min(EPS)
    return (numerator / denominator).clamp(0.0, 1.0)


def principal_angle_stats(q_a: torch.Tensor, q_b: torch.Tensor) -> Dict[str, float]:
    """Computes principal-angle summary statistics between two subspaces."""

    overlap = q_a.T @ q_b
    singular_values = torch.linalg.svdvals(overlap).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.arccos(singular_values))
    k = max(min(q_a.shape[1], q_b.shape[1]), 1)
    similarity = float((overlap.pow(2).sum().detach().cpu().item()) / float(k))
    return {
        "mean_angle_deg": float(angles.mean().detach().cpu().item()),
        "max_angle_deg": float(angles.max().detach().cpu().item()),
        "min_angle_deg": float(angles.min().detach().cpu().item()),
        "subspace_similarity": similarity,
    }


def capture_energy_summary_from_tensors(
    tensors: Iterable[torch.Tensor],
    basis: torch.Tensor,
) -> Dict[str, float | None]:
    """Streams capture-energy stats across activation chunks."""

    stats = RunningStats()
    for tensor in tensors:
        stats.update(capture_energy(tensor, basis))
    return stats.to_dict("capture")


def mean_std(values: List[float]) -> tuple[float | None, float | None]:
    """Returns mean/std for a list of python floats."""

    if not values:
        return None, None
    mean_value = sum(values) / len(values)
    variance = max(sum((value - mean_value) ** 2 for value in values) / len(values), 0.0)
    return mean_value, math.sqrt(variance)
