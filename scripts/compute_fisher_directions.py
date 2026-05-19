#!/usr/bin/env python
"""Computes approximate empirical-Fisher projection directions.

For a linear layer y = W z, the full Fisher over W is too large. This script
uses Kronecker-style input/output factors from a small calibration set:

  C_in  = E[ ||grad_y||^2 z z^T ]
  C_out = E[ ||z||^2 grad_y grad_y^T ]

The top eigenvectors of C_in replace SVD right singular directions. C_out is
also saved so left-projection experiments can be added later.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.eval_lm import move_batch_to_device
from src.methods.lora_base import iter_target_linear_modules, resolve_target_layer_enabled
from src.training.trainer import (
    build_dataloaders,
    build_model_and_tokenizer,
    resolve_model_slug,
    set_seed,
    slugify,
)
from src.utils.config import load_config_stack, save_yaml
from src.utils.gpu_select import apply_gpu_override, apply_selected_gpu, selection_to_json
from src.utils.logging import save_json
from src.utils.projection_directions import sanitize_module_name, top_eigenvectors


@dataclass
class FisherAccumulator:
    """Streaming empirical-Fisher input/output factors for one linear module."""

    module_name: str
    in_features: int
    out_features: int
    right_factor: torch.Tensor
    left_factor: torch.Tensor
    count: int = 0
    right_weight_sum: float = 0.0
    left_weight_sum: float = 0.0

    @classmethod
    def create(cls, module_name: str, in_features: int, out_features: int) -> "FisherAccumulator":
        return cls(
            module_name=module_name,
            in_features=int(in_features),
            out_features=int(out_features),
            right_factor=torch.zeros(int(in_features), int(in_features), dtype=torch.float32),
            left_factor=torch.zeros(int(out_features), int(out_features), dtype=torch.float32),
        )

    def update(self, inputs: torch.Tensor, grad_outputs: torch.Tensor, mask: Optional[torch.Tensor]) -> None:
        """Adds one batch of flattened token activations/gradients."""

        z = inputs.detach()
        g = grad_outputs.detach()
        if z.ndim == 3:
            z = z.reshape(-1, z.shape[-1])
        if g.ndim == 3:
            g = g.reshape(-1, g.shape[-1])
        if mask is not None:
            flat_mask = mask.reshape(-1).to(dtype=torch.bool)
            if flat_mask.numel() == z.shape[0]:
                z = z[flat_mask]
                g = g[flat_mask]
        if z.numel() == 0 or g.numel() == 0:
            return
        z = z.to(dtype=torch.float32, device="cpu")
        g = g.to(dtype=torch.float32, device="cpu")

        right_weights = torch.sum(g * g, dim=-1).clamp_min(0.0)
        left_weights = torch.sum(z * z, dim=-1).clamp_min(0.0)
        self.right_factor.add_(z.T @ (z * right_weights[:, None]))
        self.left_factor.add_(g.T @ (g * left_weights[:, None]))
        self.count += int(z.shape[0])
        self.right_weight_sum += float(right_weights.sum().item())
        self.left_weight_sum += float(left_weights.sum().item())

    def normalized_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns scale-normalized Fisher factors for eigendecomposition."""

        right_scale = max(float(self.right_weight_sum), 1e-12)
        left_scale = max(float(self.left_weight_sum), 1e-12)
        return self.right_factor / right_scale, self.left_factor / left_scale


class FisherHookCollector:
    """Registers hooks and updates empirical-Fisher factors during backward."""

    def __init__(self, targets: List[Tuple[str, torch.nn.Linear]]):
        self.accumulators: Dict[str, FisherAccumulator] = {
            name: FisherAccumulator.create(name, module.in_features, module.out_features)
            for name, module in targets
        }
        self._handles = []
        self._cached_inputs: Dict[str, torch.Tensor] = {}
        self._cached_masks: Dict[str, Optional[torch.Tensor]] = {}
        self._attention_mask: Optional[torch.Tensor] = None
        for name, module in targets:
            self._handles.append(module.register_forward_hook(self._make_forward_hook(name)))
            self._handles.append(module.register_full_backward_hook(self._make_backward_hook(name)))

    def set_attention_mask(self, attention_mask: Optional[torch.Tensor]) -> None:
        """Sets the current batch mask used by subsequent hooks."""

        self._attention_mask = None if attention_mask is None else attention_mask.detach().to(device="cpu")

    def close(self) -> None:
        """Removes registered hooks."""

        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def min_count(self) -> int:
        """Returns the minimum collected token count across targets."""

        if not self.accumulators:
            return 0
        return min(acc.count for acc in self.accumulators.values())

    def _make_forward_hook(self, name: str):
        def hook(_module, inputs, _output):
            if not inputs:
                return
            tensor = inputs[0]
            if not isinstance(tensor, torch.Tensor):
                return
            self._cached_inputs[name] = tensor.detach().to(dtype=torch.float32, device="cpu")
            self._cached_masks[name] = self._attention_mask

        return hook

    def _make_backward_hook(self, name: str):
        def hook(_module, _grad_input, grad_output):
            if not grad_output:
                return
            grad = grad_output[0]
            if not isinstance(grad, torch.Tensor):
                return
            inputs = self._cached_inputs.pop(name, None)
            mask = self._cached_masks.pop(name, None)
            if inputs is None:
                return
            self.accumulators[name].update(
                inputs,
                grad.detach().to(dtype=torch.float32, device="cpu"),
                mask,
            )

        return hook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--gpu", default=None)
    return parser.parse_args()


def direction_output_dir(config: Dict[str, Any], *, source: str) -> Path:
    """Resolves output directory for a direction source."""

    direction_cfg = config.get("projection_directions", {})
    configured = direction_cfg.get("output_dir") or direction_cfg.get(f"{source}_dir")
    if configured:
        return Path(str(configured))
    model_slug = resolve_model_slug(str(config["model"].get("name_or_path") or config["model"].get("synthetic_name", "model")))
    dataset_slug = slugify(Path(str(config["training"].get("train_file") or config["training"].get("dataset_name", "dataset"))).stem)
    return Path("outputs") / "directions" / source / f"{model_slug}_{dataset_slug}"


def iter_selected_modules(model: torch.nn.Module, config: Dict[str, Any]) -> List[Tuple[str, torch.nn.Linear]]:
    """Returns target linear modules selected by suffix and optional layer index."""

    direction_cfg = config.get("projection_directions", {})
    target_modules = direction_cfg.get("target_modules") or config["model"].get("target_modules", [])
    target_layers = direction_cfg.get("target_layers", config.get("lora", {}).get("target_layers", "all"))
    targets: List[Tuple[str, torch.nn.Linear]] = []
    for layer_name, module in iter_target_linear_modules(model, target_modules):
        if resolve_target_layer_enabled(layer_name, target_layers):
            targets.append((layer_name, module))
    return targets


def enable_input_grads(model: torch.nn.Module) -> None:
    """Ensures frozen model activations carry gradients for Fisher hooks."""

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return
    embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if embeddings is None:
        return

    def hook(_module, _inputs, output):
        if isinstance(output, torch.Tensor):
            output.requires_grad_(True)
        return output

    model._fisher_input_grad_hook = embeddings.register_forward_hook(hook)  # type: ignore[attr-defined]


def prepare_calibration_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Reduces dataset work for Fisher calibration without affecting training configs."""

    direction_cfg = config.get("projection_directions", {})
    training_cfg = config.setdefault("training", {})
    max_samples = int(direction_cfg.get("max_samples", 128))
    training_cfg["max_train_samples"] = max_samples
    training_cfg["max_eval_samples"] = 1
    if "batch_size" in direction_cfg:
        training_cfg["batch_size"] = int(direction_cfg["batch_size"])
    return config


def main() -> None:
    args = parse_args()
    config = load_config_stack(args.config, base_config_path=args.base_config, overrides=args.override)
    apply_gpu_override(config, args.gpu)
    selection = apply_selected_gpu(config)
    print(selection_to_json(selection), flush=True)

    seed = int(config["experiment"].get("seed", 0))
    set_seed(seed)
    config = prepare_calibration_config(config)
    direction_cfg = config.get("projection_directions", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, model_name = build_model_and_tokenizer(config, device=device)
    model.to(device)
    model.train()
    if hasattr(model, "config"):
        model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    enable_input_grads(model)

    train_loader, _ = build_dataloaders(
        config,
        tokenizer=tokenizer,
        vocab_size=int(config["model"].get("vocab_size", 128)),
    )
    targets = iter_selected_modules(model, config)
    if not targets:
        raise ValueError("No target modules selected for Fisher direction computation.")

    collector = FisherHookCollector(targets)
    max_batches = int(direction_cfg.get("max_batches", 32))
    max_tokens = int(direction_cfg.get("max_tokens", 4096))
    processed_batches = 0
    losses: List[float] = []
    try:
        for batch in train_loader:
            if processed_batches >= max_batches or collector.min_count() >= max_tokens:
                break
            model.zero_grad(set_to_none=True)
            collector.set_attention_mask(batch.get("attention_mask"))
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None:
                raise RuntimeError("Model output did not include a loss for Fisher calibration.")
            loss.backward()
            losses.append(float(loss.detach().to(dtype=torch.float32).cpu().item()))
            processed_batches += 1
    finally:
        collector.close()

    top_k_values = sorted({int(k) for k in direction_cfg.get("top_k_values", [config["lora"].get("projection_rank_k", 16)])})
    max_rank = max(top_k_values)
    eig_method = str(direction_cfg.get("eig_method", "randomized"))
    oversample = int(direction_cfg.get("oversample", 8))
    n_iter = int(direction_cfg.get("power_iterations", 2))
    output_dir = direction_output_dir(config, source="fisher")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(output_dir / "config_resolved.yaml", config)

    rows: List[Dict[str, Any]] = []
    for index, (layer_name, accumulator) in enumerate(collector.accumulators.items()):
        right_factor, left_factor = accumulator.normalized_factors()
        right_eigenvalues, right_basis = top_eigenvectors(
            right_factor,
            rank=max_rank,
            method=eig_method,
            oversample=oversample,
            n_iter=n_iter,
            seed=seed + index * 13,
        )
        left_eigenvalues, left_basis = top_eigenvectors(
            left_factor,
            rank=max_rank,
            method=eig_method,
            oversample=oversample,
            n_iter=n_iter,
            seed=seed + index * 13 + 7,
        )
        module_key = sanitize_module_name(layer_name)
        payload = {
            "source": "fisher",
            "module_name": layer_name,
            "module_key": module_key,
            "model_name_or_path": model_name,
            "rank": int(max_rank),
            "top_k_values": top_k_values,
            "count": int(accumulator.count),
            "right_weight_sum": float(accumulator.right_weight_sum),
            "left_weight_sum": float(accumulator.left_weight_sum),
            "right_basis": right_basis,
            "left_basis": left_basis,
            "right_eigenvalues": right_eigenvalues,
            "left_eigenvalues": left_eigenvalues,
            "matrix_trace": float(torch.trace(right_factor).item()),
            "eig_method": eig_method,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(payload, output_dir / f"{module_key}.pt")
        rows.append(
            {
                "module_name": layer_name,
                "module_key": module_key,
                "count": int(accumulator.count),
                "rank": int(max_rank),
                "right_weight_sum": float(accumulator.right_weight_sum),
                "left_weight_sum": float(accumulator.left_weight_sum),
                "top_right_eigenvalues": [float(v) for v in right_eigenvalues[: min(8, right_eigenvalues.numel())].tolist()],
            }
        )

    summary = {
        "source": "fisher",
        "output_dir": str(output_dir),
        "model_name_or_path": model_name,
        "num_modules": len(rows),
        "processed_batches": processed_batches,
        "loss_mean": sum(losses) / max(len(losses), 1),
        "top_k_values": top_k_values,
        "eig_method": eig_method,
        "targets": rows,
        "gpu_selection": json.loads(selection_to_json(selection)),
    }
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
