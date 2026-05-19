"""Metric helpers shared across training, evaluation, and analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch


def detach_scalar(value: torch.Tensor | float | int | None) -> float | None:
    """Converts tensors to Python floats without gradients."""

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().to(dtype=torch.float32).cpu().item())
    return float(value)


def perplexity_from_loss(loss: float | None) -> float | None:
    """Safely converts average loss to perplexity."""

    if loss is None or not math.isfinite(loss):
        return None
    return float(math.exp(min(loss, 20.0)))


def safe_ratio(numerator: float | torch.Tensor, denominator: float | torch.Tensor, *, eps: float = 1e-12) -> float:
    """Returns a numerically stable scalar ratio."""

    numerator_value = float(numerator) if not isinstance(numerator, torch.Tensor) else float(numerator.detach().cpu().item())
    denominator_value = float(denominator) if not isinstance(denominator, torch.Tensor) else float(denominator.detach().cpu().item())
    if abs(denominator_value) <= eps:
        return 0.0
    return numerator_value / denominator_value


@dataclass
class RunningScalarStats:
    """Running summary statistics for streamed scalar tensors."""

    count: int = 0
    sum: float = 0.0
    sumsq: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")

    def update(self, values: torch.Tensor | Iterable[float] | None) -> None:
        """Adds a tensor or iterable of values to the running statistics."""

        if values is None:
            return
        if isinstance(values, torch.Tensor):
            flat = values.detach().to(dtype=torch.float32).reshape(-1).cpu()
            if flat.numel() == 0:
                return
            self.count += int(flat.numel())
            self.sum += float(flat.sum().item())
            self.sumsq += float((flat * flat).sum().item())
            self.min = min(self.min, float(flat.min().item()))
            self.max = max(self.max, float(flat.max().item()))
            return

        values_list = [float(value) for value in values]
        if not values_list:
            return
        tensor = torch.tensor(values_list, dtype=torch.float32)
        self.update(tensor)

    def to_dict(self, prefix: str) -> Dict[str, float | None]:
        """Returns mean/std/min/max fields using a consistent prefix."""

        if self.count == 0:
            return {
                f"{prefix}_mean": None,
                f"{prefix}_std": None,
                f"{prefix}_min": None,
                f"{prefix}_max": None,
            }
        mean = self.sum / self.count
        variance = max(self.sumsq / self.count - mean * mean, 0.0)
        std = math.sqrt(variance)
        return {
            f"{prefix}_mean": float(mean),
            f"{prefix}_std": float(std),
            f"{prefix}_min": float(self.min),
            f"{prefix}_max": float(self.max),
        }


def safe_mean_std(values: torch.Tensor | Iterable[float] | None) -> Dict[str, float | None]:
    """Returns mean/std/min/max for a tensor or iterable without raising on empties."""

    stats = RunningScalarStats()
    stats.update(values)
    return stats.to_dict("value")


def _compute_gram_pinv(a_matrix: torch.Tensor, *, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    compute_dtype = torch.float32 if a_matrix.dtype != torch.float64 else torch.float64
    a_matrix = a_matrix.to(dtype=compute_dtype)
    gram = torch.matmul(a_matrix, a_matrix.transpose(0, 1))
    identity = torch.eye(
        gram.shape[0],
        device=gram.device,
        dtype=gram.dtype,
    )
    gram = gram + eps * identity
    gram_pinv = torch.linalg.pinv(gram)
    return a_matrix, gram_pinv


def compute_capture_projection_stats(
    a_matrix: torch.Tensor,
    activations: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Computes omega values plus captured/total energy terms for a batch of activations."""

    if activations.numel() == 0:
        empty = torch.zeros(0, device=activations.device, dtype=torch.float32)
        return {
            "omega": empty,
            "captured_energy": empty,
            "total_energy": empty,
        }

    flat_activations = activations.reshape(-1, activations.shape[-1]).to(dtype=torch.float32)
    a_matrix, gram_pinv = _compute_gram_pinv(a_matrix, eps=eps)
    az = torch.matmul(flat_activations, a_matrix.transpose(0, 1))
    captured = torch.sum(torch.matmul(az, gram_pinv) * az, dim=-1)
    captured = torch.clamp(captured, min=0.0)
    total = torch.sum(flat_activations * flat_activations, dim=-1).clamp_min(eps)
    ratio = torch.clamp(captured / total, min=0.0, max=1.0)
    omega = torch.sqrt(torch.clamp(1.0 - ratio, min=0.0, max=1.0))
    return {
        "omega": omega,
        "captured_energy": captured,
        "total_energy": total,
    }


def compute_omega_values(
    a_matrix: torch.Tensor,
    activations: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes omega_A(z) for a batch of activations."""

    return compute_capture_projection_stats(a_matrix, activations, eps=eps)["omega"]


def orthogonality_loss_value(a_matrix: torch.Tensor) -> torch.Tensor:
    """Computes ||A A^T - I||_F^2 with float32 accumulation."""

    a_matrix = a_matrix.to(dtype=torch.float32)
    gram = torch.matmul(a_matrix, a_matrix.transpose(0, 1))
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    diff = gram - identity
    return torch.sum(diff * diff)


def orthogonality_error_value(
    a_matrix: torch.Tensor,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    """Computes ||A A^T - I||_F, optionally normalized by rank."""

    fro_error = torch.sqrt(torch.clamp(orthogonality_loss_value(a_matrix), min=0.0))
    if normalize and a_matrix.shape[0] > 0:
        return fro_error / float(a_matrix.shape[0])
    return fro_error


def rowspace_alignment_stats(
    a_matrix: torch.Tensor,
    target_basis: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Measures how well rowspace(A) aligns with a target input-direction basis.

    A has shape [r, d]. The row-space basis is obtained from QR(A^T), so it
    lives in the same d-dimensional input space as target_basis [d, k].
    The score is ||Q_A^T Q_target||_F^2 / min(rank(A), k), in [0, 1].
    """

    if target_basis.numel() == 0:
        zero = a_matrix.new_tensor(0.0, dtype=torch.float32)
        return {"alignment_score": zero, "alignment_loss": zero}

    compute_dtype = torch.float32 if a_matrix.dtype != torch.float64 else torch.float64
    a_matrix = a_matrix.to(dtype=compute_dtype)
    target_basis = target_basis.to(device=a_matrix.device, dtype=compute_dtype)
    if target_basis.ndim != 2:
        raise ValueError(f"Expected 2D target_basis, got shape {tuple(target_basis.shape)}.")
    if target_basis.shape[0] != a_matrix.shape[1]:
        raise ValueError(
            "ActivationCov target basis dimension does not match LoRA A input dimension: "
            f"target={target_basis.shape[0]}, A={a_matrix.shape[1]}."
        )

    q_a, _ = torch.linalg.qr(a_matrix.transpose(0, 1), mode="reduced")
    q_target, _ = torch.linalg.qr(target_basis, mode="reduced")
    denom = min(q_a.shape[1], q_target.shape[1])
    if denom <= 0:
        zero = a_matrix.new_tensor(0.0, dtype=compute_dtype)
        return {"alignment_score": zero, "alignment_loss": zero}

    overlap = torch.matmul(q_a.transpose(0, 1), q_target)
    score = torch.sum(overlap * overlap) / max(float(denom), eps)
    score = torch.clamp(score, min=0.0, max=1.0)
    return {
        "alignment_score": score,
        "alignment_loss": torch.clamp(1.0 - score, min=0.0, max=1.0),
    }


def compute_rho_k_from_basis(
    delta_weight: torch.Tensor,
    left_basis: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Computes rho_k and the projected update norm from a left singular basis."""

    if left_basis.numel() == 0:
        delta_norm = float(torch.norm(delta_weight).detach().cpu().item())
        return {
            "rho_k": 0.0,
            "delta_w_norm": delta_norm,
            "q_delta_w_norm": 0.0,
        }

    delta_weight = delta_weight.to(dtype=torch.float32)
    basis = left_basis.to(device=delta_weight.device, dtype=delta_weight.dtype)
    delta_norm_tensor = torch.norm(delta_weight)
    delta_norm = float(delta_norm_tensor.detach().cpu().item())
    if delta_norm <= eps:
        return {
            "rho_k": 0.0,
            "delta_w_norm": 0.0,
            "q_delta_w_norm": 0.0,
        }

    projected_coeff = torch.matmul(basis.transpose(0, 1), delta_weight)
    q_delta_w_norm_tensor = torch.norm(projected_coeff)
    rho = float((q_delta_w_norm_tensor.pow(2) / delta_norm_tensor.pow(2)).detach().cpu().item())
    return {
        "rho_k": rho,
        "delta_w_norm": delta_norm,
        "q_delta_w_norm": float(q_delta_w_norm_tensor.detach().cpu().item()),
    }


def summarize_metric_records(records: List[Dict[str, float | None]], key: str) -> float | None:
    """Averages a numeric key across JSON-like records while skipping nulls."""

    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def gpu_memory_stats() -> Dict[str, float | None]:
    """Returns current/peak CUDA memory in MiB and GiB when CUDA is available."""

    if not torch.cuda.is_available():
        return {
            "current_gpu_memory_mb": None,
            "peak_gpu_memory_mb": None,
            "current_gpu_memory_gb": None,
            "peak_gpu_memory_gb": None,
            "gpu_memory_allocated_mb": None,
            "gpu_memory_reserved_mb": None,
        }

    allocated = float(torch.cuda.memory_allocated() / (1024 ** 2))
    reserved = float(torch.cuda.memory_reserved() / (1024 ** 2))
    peak = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return {
        "current_gpu_memory_mb": allocated,
        "peak_gpu_memory_mb": peak,
        "current_gpu_memory_gb": allocated / 1024.0,
        "peak_gpu_memory_gb": peak / 1024.0,
        "gpu_memory_allocated_mb": allocated,
        "gpu_memory_reserved_mb": reserved,
    }
