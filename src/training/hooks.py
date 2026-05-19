"""Training-time helpers for adapter losses, metric hooks, and logging."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import nn

from src.methods.lora_base import BaseLoRALinear, iter_lora_layers
from src.utils.metrics import RunningScalarStats, orthogonality_loss_value
from src.utils.svd import build_projection_basis


@dataclass
class MechanismCollectionResult:
    """Per-step mechanism records plus metric overhead summaries."""

    records: List[Dict[str, object]]
    capture_metric_time_sec: float
    rho_metric_time_sec: float
    svd_time_sec: float
    svd_cache_hits: int
    svd_cache_misses: int


def adapter_layers(model: nn.Module) -> List[BaseLoRALinear]:
    """Returns all injected LoRA-family layers in traversal order."""

    return list(iter_lora_layers(model))


def _zero_tensor_for_model(model: nn.Module) -> torch.Tensor:
    parameter = next(model.parameters())
    return torch.zeros((), device=parameter.device)


def aggregate_regularization_losses(
    model: nn.Module,
    *,
    loss_weight: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Averages NSC capture, A-orthogonality, and capture-alignment losses."""

    layers = adapter_layers(model)
    if not layers:
        zero = _zero_tensor_for_model(model)
        return zero, zero, zero

    capture_layers = [
        layer
        for layer in layers
        if layer.use_capture_loss and layer.capture_enabled
    ]
    capture_terms = [
        layer.capture_loss(loss_weight=loss_weight) for layer in capture_layers
    ]
    alignment_layers = [
        layer
        for layer in layers
        if layer.use_capture_alignment_loss and layer.capture_enabled
    ]
    alignment_terms = [layer.alignment_loss() for layer in alignment_layers]
    orth_terms = [layer.orthogonality_loss() for layer in layers]
    capture_loss = torch.stack(capture_terms).mean() if capture_terms else torch.zeros((), device=layers[0].lora_A.device)
    orth_loss = torch.stack(orth_terms).mean() if orth_terms else torch.zeros((), device=layers[0].lora_A.device)
    alignment_loss = torch.stack(alignment_terms).mean() if alignment_terms else torch.zeros((), device=layers[0].lora_A.device)
    return capture_loss, orth_loss, alignment_loss


def clear_adapter_caches(model: nn.Module) -> None:
    """Releases graph-carrying per-step adapter tensors after backward."""

    for layer in adapter_layers(model):
        layer.clear_step_cache()


def count_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    """Returns total and trainable parameter counts."""

    total = 0
    trainable = 0
    for parameter in model.parameters():
        numel = parameter.numel()
        total += numel
        if parameter.requires_grad:
            trainable += numel
    return {"total_parameters": total, "trainable_parameters": trainable}


def resolve_metric_layers(
    layers: Sequence[BaseLoRALinear],
    selector: object,
) -> List[BaseLoRALinear]:
    """Selects a deterministic subset of layers for metric computation."""

    layers = list(layers)
    if not layers:
        return []
    if selector in (None, "all"):
        return layers
    if selector == "last_quarter":
        start = max(len(layers) * 3 // 4, 0)
        return layers[start:]
    if selector == "first_quarter":
        end = max(len(layers) // 4, 1)
        return layers[:end]
    if selector == "last_half":
        start = max(len(layers) // 2, 0)
        return layers[start:]
    if isinstance(selector, int):
        return layers[: max(selector, 0)]
    if isinstance(selector, str):
        return [layer for layer in layers if layer.layer_name.endswith(selector) or layer.layer_name == selector]
    if isinstance(selector, Iterable):
        selected_names = {str(item) for item in selector}
        return [
            layer
            for layer in layers
            if layer.layer_name in selected_names
            or layer.layer_name.split(".")[-1] in selected_names
        ]
    raise ValueError(f"Unsupported layer selector: {selector!r}")


def _layer_rho_basis(
    layer: BaseLoRALinear,
    *,
    max_k: int,
    svd_cache_dir: str | Path,
) -> tuple[torch.Tensor, Dict[str, object]]:
    """Loads or computes the left singular basis needed for rho_k metrics."""

    if max_k <= 0:
        empty = torch.empty(layer.out_features, 0, device=layer.base_layer.weight.device, dtype=torch.float32)
        return empty, {
            "cache_hit": True,
            "build_time_sec": 0.0,
            "rank_k_used": 0,
        }

    cached_basis = layer._metric_left_basis_cache.get(max_k)
    if cached_basis is not None:
        return cached_basis, {
            "cache_hit": True,
            "build_time_sec": 0.0,
            "rank_k_used": max_k,
        }

    basis_result = build_projection_basis(
        layer.base_layer.weight.detach(),
        rank_k=max_k,
        cache_dir=svd_cache_dir,
        cache_key=f"rho|{layer.layer_name}|{tuple(layer.base_layer.weight.shape)}|k={max_k}",
        left_source="svd",
        right_source="none",
        enforce_rank_mode="clip",
        random_seed=0,
        cache_svd=True,
    )
    left_basis = basis_result.basis.left_basis
    if left_basis is None:
        left_basis = torch.empty(layer.out_features, 0, dtype=torch.float32)
    layer._metric_left_basis_cache[max_k] = left_basis
    return left_basis, basis_result.metadata


def collect_mechanism_metrics(
    model: nn.Module,
    *,
    step: int,
    method: str,
    metrics_config: Dict[str, object],
    svd_cache_dir: str | Path,
) -> MechanismCollectionResult:
    """Collects per-layer mechanism metrics from injected adapters."""

    layers = adapter_layers(model)
    if not layers:
        return MechanismCollectionResult(
            records=[],
            capture_metric_time_sec=0.0,
            rho_metric_time_sec=0.0,
            svd_time_sec=0.0,
            svd_cache_hits=0,
            svd_cache_misses=0,
        )

    compute_omega = bool(metrics_config.get("compute_omega", True))
    compute_rho_k = bool(metrics_config.get("compute_rho_k", True))
    compute_orth_error = bool(metrics_config.get("compute_orth_error", True))
    compute_update_norms = bool(metrics_config.get("compute_update_norms", True))
    rho_k_values = [int(value) for value in metrics_config.get("rho_k_values", [int(layers[0].projection_rank_k or 0)])]
    rho_k_values = sorted({value for value in rho_k_values if value >= 0})

    capture_layers = resolve_metric_layers(layers, metrics_config.get("omega_capture_layers", "all"))
    rho_max_layers = metrics_config.get("rho_max_layers")
    rho_layers = layers if rho_max_layers in (None, 0) else layers[: int(rho_max_layers)]
    tracked_layers = list(dict.fromkeys(capture_layers + rho_layers))

    capture_start = time.perf_counter()
    layer_summaries: Dict[str, Dict[str, object]] = {}
    for layer in tracked_layers:
        record = {
            "step": step,
            "method": method,
            "layer_name": layer.layer_name,
            "module_name": layer.layer_name.split(".")[-1],
            "lora_alpha": float(layer.alpha),
            "lora_rank": int(layer.rank),
            "projection_rank_k": int(layer.projection_rank_k),
            "left_projection_source": layer.left_projection_source,
            "right_projection_source": layer.right_projection_source,
            "capture_basis_source": layer.capture_basis_source,
            "capture_rank_k": int(layer.capture_rank_k),
            "capture_basis_residualized": bool(layer.capture_basis_metadata.get("residualized_against_preserve", False)),
            "use_capture_alignment_loss": bool(layer.use_capture_alignment_loss),
            "lambda_cap": None,
            "lambda_align": None,
            "capture_weighting": layer.capture_weighting,
            "lambda_orth": None,
            "k": None,
            "rho_k": None,
            "delta_w_norm": None,
            "q_delta_w_norm": None,
            "lora_update_norm": None,
            "omega_full_mean": None,
            "omega_full_std": None,
            "omega_full_min": None,
            "omega_full_max": None,
            "omega_residual_mean": None,
            "omega_residual_std": None,
            "omega_residual_min": None,
            "omega_residual_max": None,
            "captured_energy_full_mean": None,
            "captured_energy_residual_mean": None,
            "total_energy_mean": None,
            "cap_loss": None,
            "actcov_alignment_loss": None,
            "actcov_alignment_score": None,
            "capture_alignment_loss": None,
            "capture_alignment_score": None,
            "capture_energy_task": None,
            "orth_error_A": None,
            "orth_error_A_norm": None,
            "orth_loss": None,
            "A_norm": None,
            "B_norm": None,
            "effective_update_norm": None,
            "raw_BA_norm": None,
            "projected_delta_w_norm": None,
            "projection_energy_removed_left": None,
            "projection_energy_removed_right": None,
        }

        if compute_omega and layer in capture_layers:
            state = layer.capture_metric_state()
            if state.omega_full is not None:
                full_stats = RunningScalarStats()
                full_stats.update(state.omega_full)
                record.update(full_stats.to_dict("omega_full"))
            if state.omega_residual is not None:
                residual_stats = RunningScalarStats()
                residual_stats.update(state.omega_residual)
                record.update(residual_stats.to_dict("omega_residual"))
            if state.captured_energy_full is not None:
                captured_full_stats = RunningScalarStats()
                captured_full_stats.update(state.captured_energy_full)
                record["captured_energy_full_mean"] = captured_full_stats.to_dict("captured_energy_full")["captured_energy_full_mean"]
            if state.captured_energy_residual is not None:
                captured_residual_stats = RunningScalarStats()
                captured_residual_stats.update(state.captured_energy_residual)
                record["captured_energy_residual_mean"] = captured_residual_stats.to_dict("captured_energy_residual")["captured_energy_residual_mean"]
            total_energy = state.total_energy_residual if state.total_energy_residual is not None else state.total_energy_full
            if total_energy is not None:
                total_stats = RunningScalarStats()
                total_stats.update(total_energy)
                record["total_energy_mean"] = total_stats.to_dict("total_energy")["total_energy_mean"]
            if state.capture_loss is not None:
                record["cap_loss"] = float(state.capture_loss.detach().cpu().item())
            if state.actcov_alignment_loss is not None:
                record["actcov_alignment_loss"] = float(state.actcov_alignment_loss.detach().cpu().item())
            if state.actcov_alignment_score is not None:
                record["actcov_alignment_score"] = float(state.actcov_alignment_score.detach().cpu().item())
            if state.capture_alignment_loss is not None:
                record["capture_alignment_loss"] = float(state.capture_alignment_loss.detach().cpu().item())
            if state.capture_alignment_score is not None:
                score = float(state.capture_alignment_score.detach().cpu().item())
                record["capture_alignment_score"] = score
                record["capture_energy_task"] = score

        if compute_orth_error:
            record["orth_error_A"] = float(layer.orthogonality_error().cpu().item())
            record["orth_error_A_norm"] = float(layer.orthogonality_error_normalized().cpu().item())
            record["orth_loss"] = float(orthogonality_loss_value(layer.lora_A).detach().cpu().item())

        if compute_update_norms:
            record.update(layer.parameter_norms())
            record.update(layer.projection_energy_metrics())
            record["delta_w_norm"] = record["effective_update_norm"]
            record["lora_update_norm"] = record["effective_update_norm"]

        layer_summaries[layer.layer_name] = record

    capture_metric_time_sec = float(time.perf_counter() - capture_start)

    rho_start = time.perf_counter()
    records: List[Dict[str, object]] = []
    svd_time_sec = 0.0
    svd_cache_hits = 0
    svd_cache_misses = 0
    max_requested_k = max(rho_k_values, default=0)
    for layer in tracked_layers:
        base_record = dict(layer_summaries[layer.layer_name])
        if not compute_rho_k or layer not in rho_layers:
            records.append(base_record)
            continue

        left_basis, metadata = _layer_rho_basis(
            layer,
            max_k=max_requested_k,
            svd_cache_dir=svd_cache_dir,
        )
        svd_time_sec += float(metadata.get("build_time_sec", 0.0))
        if bool(metadata.get("cache_hit", False)):
            svd_cache_hits += 1
        else:
            svd_cache_misses += 1

        available_k = min(int(metadata.get("rank_k_used", left_basis.shape[1])), left_basis.shape[1])
        candidate_k_values = [k for k in rho_k_values if k <= available_k] or [available_k]
        candidate_k_values = [k for k in candidate_k_values if k >= 0]
        if not candidate_k_values:
            records.append(base_record)
            continue

        for k in candidate_k_values:
            layer_record = dict(base_record)
            layer_record["k"] = int(k)
            if k == 0 or left_basis.numel() == 0:
                rho_stats = {
                    "rho_k": 0.0,
                    "delta_w_norm": float(torch.norm(layer.dense_delta_weight().detach()).cpu().item()),
                    "q_delta_w_norm": 0.0,
                }
            else:
                rho_tensor_stats = layer.rho_k_with_basis(left_basis[:, :k])
                rho_stats = {
                    "rho_k": float(rho_tensor_stats["rho_k_tensor"].detach().cpu().item()),
                    "delta_w_norm": float(rho_tensor_stats["delta_w_norm_tensor"].detach().cpu().item()),
                    "q_delta_w_norm": float(rho_tensor_stats["q_delta_w_norm_tensor"].detach().cpu().item()),
                }
            layer_record.update(rho_stats)
            records.append(layer_record)

    rho_metric_time_sec = float(time.perf_counter() - rho_start)
    return MechanismCollectionResult(
        records=records,
        capture_metric_time_sec=capture_metric_time_sec,
        rho_metric_time_sec=rho_metric_time_sec,
        svd_time_sec=svd_time_sec,
        svd_cache_hits=svd_cache_hits,
        svd_cache_misses=svd_cache_misses,
    )
