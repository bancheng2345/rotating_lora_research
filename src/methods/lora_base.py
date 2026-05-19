"""Standard LoRA layers and adapter injection helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn

from src.capture.directions import build_capture_basis
from src.capture.losses import capture_alignment_stats
from src.methods.capture_losses import orthogonality_loss, reduce_capture_loss
from src.methods.projections import ProjectionBasis, apply_left_projection, apply_right_projection, left_project_weight, right_project_weight
from src.utils.metrics import (
    compute_capture_projection_stats,
    compute_rho_k_from_basis,
    orthogonality_error_value,
    orthogonality_loss_value,
)
from src.utils.svd import build_projection_basis


@dataclass
class AdapterRegularizationState:
    """Per-layer regularization values cached from the last forward pass."""

    capture_loss: Optional[torch.Tensor]
    omega_full: Optional[torch.Tensor]
    omega_residual: Optional[torch.Tensor]
    captured_energy_full: Optional[torch.Tensor]
    captured_energy_residual: Optional[torch.Tensor]
    total_energy_full: Optional[torch.Tensor]
    total_energy_residual: Optional[torch.Tensor]
    actcov_alignment_loss: Optional[torch.Tensor] = None
    actcov_alignment_score: Optional[torch.Tensor] = None
    capture_alignment_loss: Optional[torch.Tensor] = None
    capture_alignment_score: Optional[torch.Tensor] = None


class BaseLoRALinear(nn.Module):
    """Wraps an nn.Linear layer with low-rank adaptation and optional projections."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        layer_name: str,
        rank: int,
        alpha: float,
        dropout: float,
        method_name: str,
        projection_basis: Optional[ProjectionBasis] = None,
        use_left_projection: bool = False,
        use_right_projection: bool = False,
        use_capture_loss: bool = False,
        use_capture_alignment_loss: bool = False,
        capture_space: str = "full",
        capture_weighting: str = "none",
        capture_basis: Optional[torch.Tensor] = None,
        capture_basis_source: str = "none",
        capture_basis_metadata: Optional[Dict[str, object]] = None,
        detach_capture_inputs: bool = True,
        capture_enabled: bool = True,
        projection_metadata: Optional[Dict[str, object]] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(base_layer)!r}")
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")

        self.base_layer = base_layer
        self.layer_name = layer_name
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.method_name = method_name
        self.use_left_projection = use_left_projection
        self.use_right_projection = use_right_projection
        self.use_capture_loss = use_capture_loss
        self.use_capture_alignment_loss = use_capture_alignment_loss
        self.capture_space = capture_space
        self.capture_weighting = capture_weighting
        self.capture_basis_source = capture_basis_source
        self.capture_basis_metadata = dict(capture_basis_metadata or {})
        self.detach_capture_inputs = detach_capture_inputs
        self.capture_enabled = capture_enabled
        self.eps = eps

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.dropout_layer = nn.Dropout(dropout)
        parameter_device = base_layer.weight.device
        parameter_dtype = (
            torch.float32
            if base_layer.weight.dtype in {torch.float16, torch.bfloat16, torch.float32}
            else base_layer.weight.dtype
        )
        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                self.in_features,
                device=parameter_device,
                dtype=parameter_dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.empty(
                self.out_features,
                rank,
                device=parameter_device,
                dtype=parameter_dtype,
            )
        )
        self.reset_lora_parameters()

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

        projection_basis = projection_basis or ProjectionBasis(None, None, 0)
        left_basis = projection_basis.left_basis
        right_basis = projection_basis.right_basis
        self.register_buffer(
            "left_basis",
            left_basis
            if left_basis is not None
            else torch.empty(
                self.out_features,
                0,
                device=parameter_device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "right_basis",
            right_basis
            if right_basis is not None
            else torch.empty(
                self.in_features,
                0,
                device=parameter_device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.projection_rank_k = projection_basis.rank_k
        self.left_projection_source = projection_basis.left_source
        self.right_projection_source = projection_basis.right_source
        self.projection_metadata = dict(projection_metadata or {})
        self.register_buffer(
            "capture_basis",
            capture_basis
            if capture_basis is not None
            else torch.empty(
                self.in_features,
                0,
                device=parameter_device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.capture_rank_k = int(self.capture_basis.shape[1])
        self._metric_left_basis_cache: Dict[int, torch.Tensor] = {}
        if (
            self.capture_space == "activation_cov"
            and self.right_basis.numel() == 0
            and self.capture_basis.numel() == 0
        ):
            raise ValueError(
                "capture_space=activation_cov requires right_projection_source=activation_cov "
                "or a non-empty capture_basis."
            )
        if self.use_capture_alignment_loss and self.capture_basis.numel() == 0:
            raise ValueError(
                "use_capture_alignment_loss=true requires lora.capture_basis_source "
                "to resolve a non-empty capture basis."
            )

        self._regularization_state = AdapterRegularizationState(
            capture_loss=None,
            omega_full=None,
            omega_residual=None,
            captured_energy_full=None,
            captured_energy_residual=None,
            total_energy_full=None,
            total_energy_residual=None,
            actcov_alignment_loss=None,
            actcov_alignment_score=None,
            capture_alignment_loss=None,
            capture_alignment_score=None,
        )
        self._capture_inputs: Optional[torch.Tensor] = None
        self._capture_residual_inputs: Optional[torch.Tensor] = None

    def reset_lora_parameters(self) -> None:
        """Initializes A with Kaiming and B with zeros, as in standard LoRA."""

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def _basis_or_none(self, basis: torch.Tensor) -> Optional[torch.Tensor]:
        if basis.numel() == 0:
            return None
        return basis

    def _capture_basis_or_none(self) -> Optional[torch.Tensor]:
        """Returns capture-only basis, falling back to legacy right_basis configs."""

        if self.capture_basis.numel() > 0:
            return self.capture_basis
        if self.capture_space == "activation_cov" and self.right_basis.numel() > 0:
            return self.right_basis
        return None

    def project_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transforms inputs before applying A."""

        if not self.use_right_projection:
            return inputs
        return apply_right_projection(inputs, self._basis_or_none(self.right_basis))

    def project_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        """Transforms outputs after applying B."""

        if not self.use_left_projection:
            return outputs
        return apply_left_projection(outputs, self._basis_or_none(self.left_basis))

    def compute_residual_input(self, inputs: torch.Tensor) -> torch.Tensor:
        """Computes P_R z for metrics/capture, even when right projection is disabled."""

        return apply_right_projection(inputs, self._basis_or_none(self.right_basis))

    def compute_capture_loss(
        self,
        inputs: torch.Tensor,
        residual_inputs: torch.Tensor,
        *,
        loss_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the configured capture loss for the current layer."""

        if self.capture_space == "activation_cov":
            return self.capture_alignment_loss()
        capture_inputs = residual_inputs if self.capture_space == "residual" else inputs
        if self.detach_capture_inputs:
            capture_inputs = capture_inputs.detach()
        omega = compute_capture_projection_stats(
            self.lora_A,
            capture_inputs,
            eps=self.eps,
        )["omega"]
        return reduce_capture_loss(
            omega,
            weighting=self.capture_weighting,
            weight=loss_weight,
        )

    def update_regularization_cache(self, inputs: torch.Tensor) -> None:
        """Caches omega metrics and unweighted capture loss from the latest forward pass."""

        self._capture_inputs = inputs.detach() if self.detach_capture_inputs else inputs
        self._capture_residual_inputs = self.compute_residual_input(self._capture_inputs)
        full_inputs = self._capture_inputs
        residual_inputs = self._capture_residual_inputs
        full_stats = compute_capture_projection_stats(self.lora_A, full_inputs, eps=self.eps)
        residual_stats = compute_capture_projection_stats(
            self.lora_A,
            residual_inputs,
            eps=self.eps,
        )
        omega_full = full_stats["omega"]
        omega_residual = residual_stats["omega"]

        capture_loss = None
        actcov_alignment_loss = None
        actcov_alignment_score = None
        capture_alignment_loss = None
        capture_alignment_score = None
        if self.capture_enabled:
            align_stats = self.capture_alignment_stats()
            if self._capture_basis_or_none() is not None:
                capture_alignment_loss = align_stats["alignment_loss"].detach()
                capture_alignment_score = align_stats["alignment_score"].detach()
                # Legacy fields are kept for old scripts that still aggregate
                # ActivationCov-specific names.
                actcov_alignment_loss = capture_alignment_loss
                actcov_alignment_score = capture_alignment_score
        if self.use_capture_loss and self.capture_enabled:
            if self.capture_space == "activation_cov":
                capture_loss = capture_alignment_loss
            else:
                capture_values = omega_residual if self.capture_space == "residual" else omega_full
                capture_loss = reduce_capture_loss(capture_values, weighting="none")

        self._regularization_state = AdapterRegularizationState(
            capture_loss=capture_loss,
            omega_full=omega_full.detach(),
            omega_residual=omega_residual.detach(),
            captured_energy_full=full_stats["captured_energy"].detach(),
            captured_energy_residual=residual_stats["captured_energy"].detach(),
            total_energy_full=full_stats["total_energy"].detach(),
            total_energy_residual=residual_stats["total_energy"].detach(),
            actcov_alignment_loss=actcov_alignment_loss,
            actcov_alignment_score=actcov_alignment_score,
            capture_alignment_loss=capture_alignment_loss,
            capture_alignment_score=capture_alignment_score,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies the frozen base layer plus low-rank adaptation."""

        base_output = self.base_layer(inputs)
        lora_inputs = self.dropout_layer(inputs).to(dtype=self.lora_A.dtype)
        projected_inputs = self.project_inputs(lora_inputs)
        a_output = F.linear(projected_inputs, self.lora_A)
        update = F.linear(a_output, self.lora_B)
        update = self.project_outputs(update)
        update = update.to(dtype=base_output.dtype)
        self.update_regularization_cache(inputs)
        return base_output + self.scaling * update

    def dense_delta_weight(self) -> torch.Tensor:
        """Materializes ΔW for logging/evaluation only."""

        delta = self.raw_delta_weight()
        if self.use_right_projection:
            delta = right_project_weight(delta, self._basis_or_none(self.right_basis))
        if self.use_left_projection:
            delta = left_project_weight(delta, self._basis_or_none(self.left_basis))
        return delta

    def raw_delta_weight(self) -> torch.Tensor:
        """Materializes the unprojected scaled BA update."""

        delta = torch.matmul(self.lora_B, self.lora_A)
        return self.scaling * delta

    def projection_energy_metrics(self) -> Dict[str, float]:
        """Returns norms for raw/projected updates and removed projection energy."""

        raw_delta = self.raw_delta_weight().detach().to(dtype=torch.float32)
        right_projected = right_project_weight(raw_delta, self._basis_or_none(self.right_basis))
        fully_projected = left_project_weight(right_projected, self._basis_or_none(self.left_basis))
        right_removed = raw_delta - right_projected
        left_removed = right_projected - fully_projected
        return {
            "raw_BA_norm": float(torch.norm(raw_delta).cpu().item()),
            "projected_delta_w_norm": float(torch.norm(fully_projected).cpu().item()),
            "effective_update_norm": float(torch.norm(fully_projected).cpu().item()),
            "projection_energy_removed_right": float(torch.norm(right_removed).cpu().item()),
            "projection_energy_removed_left": float(torch.norm(left_removed).cpu().item()),
        }

    def rho_k(self) -> Optional[torch.Tensor]:
        """Computes rho_k = ||Q_k ΔW||_F^2 / ||ΔW||_F^2 using U_k^T ΔW."""

        if self.left_basis.numel() == 0:
            return None
        return self.rho_k_with_basis(self.left_basis)["rho_k_tensor"]

    def rho_k_with_basis(self, left_basis: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes rho statistics for an arbitrary left singular basis."""

        delta = self.dense_delta_weight().detach()
        rho_stats = compute_rho_k_from_basis(delta, left_basis, eps=self.eps)
        return {
            "rho_k_tensor": delta.new_tensor(rho_stats["rho_k"]),
            "delta_w_norm_tensor": delta.new_tensor(rho_stats["delta_w_norm"]),
            "q_delta_w_norm_tensor": delta.new_tensor(rho_stats["q_delta_w_norm"]),
        }

    def orthogonality_loss(self) -> torch.Tensor:
        """Returns the orthogonality regularizer for A."""

        return orthogonality_loss(self.lora_A)

    def capture_alignment_stats(self) -> Dict[str, torch.Tensor]:
        """Returns rowspace(A) alignment stats against capture-only directions."""

        basis = self._capture_basis_or_none()
        if basis is None:
            zero = self.lora_A.new_tensor(0.0)
            return {"alignment_score": zero, "alignment_loss": zero}
        return capture_alignment_stats(
            self.lora_A,
            basis,
            eps=self.eps,
        )

    def actcov_alignment_stats(self) -> Dict[str, torch.Tensor]:
        """Backward-compatible alias for ActivationCov-guided configs."""

        return self.capture_alignment_stats()

    def capture_alignment_loss(self) -> torch.Tensor:
        """Loss that encourages rowspace(A) to span capture-only directions."""

        return self.capture_alignment_stats()["alignment_loss"]

    def actcov_alignment_loss(self) -> torch.Tensor:
        """Loss that encourages rowspace(A) to span ActivationCov high-energy directions."""

        return self.capture_alignment_loss()

    def capture_loss(self, *, loss_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes the per-layer capture loss using cached forward activations."""

        if not self.use_capture_loss or not self.capture_enabled:
            return self.lora_A.new_tensor(0.0)
        if self.capture_space == "activation_cov":
            return self.capture_alignment_loss()
        if self._capture_inputs is None or self._capture_residual_inputs is None:
            return self.lora_A.new_tensor(0.0)
        return self.compute_capture_loss(
            self._capture_inputs,
            self._capture_residual_inputs,
            loss_weight=loss_weight,
        )

    def alignment_loss(self) -> torch.Tensor:
        """Computes capture-basis alignment loss independent from NSC capture."""

        if not self.use_capture_alignment_loss or not self.capture_enabled:
            return self.lora_A.new_tensor(0.0)
        return self.capture_alignment_loss()

    def clear_step_cache(self) -> None:
        """Clears graph-carrying tensors from the previous forward pass."""

        self._capture_inputs = None
        self._capture_residual_inputs = None

    def orthogonality_error(self) -> torch.Tensor:
        """Returns detached orthogonality error for logging."""

        return orthogonality_error_value(self.lora_A).detach()

    def orthogonality_error_normalized(self) -> torch.Tensor:
        """Returns normalized Frobenius orthogonality error for logging."""

        return orthogonality_error_value(self.lora_A, normalize=True).detach()

    def lora_update_norm(self) -> torch.Tensor:
        """Returns the Frobenius norm of the current low-rank update."""

        return torch.norm(self.dense_delta_weight()).detach()

    def parameter_norms(self) -> Dict[str, float]:
        """Returns detached LoRA parameter norms."""

        return {
            "A_norm": float(torch.norm(self.lora_A.detach().to(dtype=torch.float32)).cpu().item()),
            "B_norm": float(torch.norm(self.lora_B.detach().to(dtype=torch.float32)).cpu().item()),
        }

    def capture_metric_state(self) -> AdapterRegularizationState:
        """Returns cached capture metrics from the latest forward pass."""

        return self._regularization_state

    def mechanism_metrics(self) -> Dict[str, Optional[float]]:
        """Returns detached per-layer mechanism metrics."""

        rho = self.rho_k()
        capture = self._regularization_state.capture_loss
        omega_full = self._regularization_state.omega_full
        omega_residual = self._regularization_state.omega_residual
        return {
            "omega_full": None if omega_full is None or omega_full.numel() == 0 else float(omega_full.mean().detach().cpu().item()),
            "omega_residual": None if omega_residual is None or omega_residual.numel() == 0 else float(omega_residual.mean().detach().cpu().item()),
            "rho_k": None if rho is None else float(rho.detach().cpu().item()),
            "orth_error_A": float(self.orthogonality_error().cpu().item()),
            "lora_update_norm": float(self.lora_update_norm().cpu().item()),
            "cap_loss": None if capture is None else float(capture.detach().cpu().item()),
            "actcov_alignment_loss": None if self._regularization_state.actcov_alignment_loss is None else float(self._regularization_state.actcov_alignment_loss.detach().cpu().item()),
            "actcov_alignment_score": None if self._regularization_state.actcov_alignment_score is None else float(self._regularization_state.actcov_alignment_score.detach().cpu().item()),
            "capture_alignment_loss": None if self._regularization_state.capture_alignment_loss is None else float(self._regularization_state.capture_alignment_loss.detach().cpu().item()),
            "capture_alignment_score": None if self._regularization_state.capture_alignment_score is None else float(self._regularization_state.capture_alignment_score.detach().cpu().item()),
            "orth_loss": float(orthogonality_loss_value(self.lora_A).detach().cpu().item()),
        }


class LoRALinear(BaseLoRALinear):
    """Standard LoRA layer."""

    def __init__(self, base_layer: nn.Linear, **kwargs):
        super().__init__(base_layer, method_name="lora", **kwargs)


def get_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    """Returns the parent module and final attribute name for a dotted module path."""

    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def iter_target_linear_modules(
    model: nn.Module,
    target_modules: Sequence[str],
) -> Iterator[Tuple[str, nn.Linear]]:
    """Yields named linear modules whose name suffix matches target_modules."""

    suffixes = (target_modules,) if isinstance(target_modules, str) else tuple(target_modules)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(suffixes):
            yield name, module


def extract_transformer_layer_index(layer_name: str) -> Optional[int]:
    """Extracts a transformer block index from common HF module paths."""

    for pattern in (r"\.layers\.(\d+)\.", r"\.h\.(\d+)\.", r"\.blocks\.(\d+)\."):
        match = re.search(pattern, layer_name)
        if match:
            return int(match.group(1))
    return None


def resolve_target_layer_enabled(layer_name: str, target_layers: object) -> bool:
    """Resolves optional layer-index filtering for LoRA injection."""

    if target_layers in (None, "all"):
        return True
    if isinstance(target_layers, Iterable) and not isinstance(target_layers, (str, bytes)):
        layer_index = extract_transformer_layer_index(layer_name)
        return layer_index in {int(value) for value in target_layers}
    raise ValueError(f"Unsupported lora.target_layers specification: {target_layers!r}")


def resolve_capture_enabled(layer_name: str, capture_layers: object) -> bool:
    """Resolves whether a layer participates in capture-loss computation."""

    if capture_layers in (None, "all"):
        return True
    if isinstance(capture_layers, str):
        return layer_name.endswith(capture_layers)
    if isinstance(capture_layers, Iterable):
        return any(layer_name.endswith(str(item)) or layer_name == str(item) for item in capture_layers)
    raise ValueError(f"Unsupported capture_layers specification: {capture_layers!r}")


def resolve_pattern_value(layer_name: str, pattern: object, default: object) -> object:
    """Resolves exact/suffix keyed per-layer config values."""

    if pattern in (None, {}):
        return default
    if not isinstance(pattern, dict):
        raise ValueError(f"Expected a dict pattern for {layer_name}, got {type(pattern)!r}.")
    for key, value in pattern.items():
        key_text = str(key)
        if layer_name == key_text or layer_name.endswith(key_text):
            return value
    return default


def resolve_lora_rank(layer_name: str, lora_config: Dict[str, object]) -> int:
    """Resolves optional per-layer LoRA rank."""

    rank = int(resolve_pattern_value(layer_name, lora_config.get("rank_pattern"), lora_config["rank"]))
    if rank <= 0:
        raise ValueError(f"Resolved LoRA rank must be > 0 for {layer_name}, got {rank}.")
    return rank


def resolve_lora_alpha(layer_name: str, lora_config: Dict[str, object], *, rank: int) -> float:
    """Resolves optional per-layer LoRA alpha.

    `alpha_strategy=rank_scaled` preserves the global alpha/rank scaling ratio
    when per-layer ranks vary.
    """

    alpha_value = resolve_pattern_value(layer_name, lora_config.get("alpha_pattern"), None)
    if alpha_value is not None:
        return float(alpha_value)
    strategy = str(lora_config.get("alpha_strategy", "fixed"))
    if strategy == "fixed":
        return float(lora_config["alpha"])
    if strategy == "rank_scaled":
        default_multiplier = float(lora_config["alpha"]) / float(lora_config["rank"])
        multiplier = float(lora_config.get("alpha_multiplier", default_multiplier))
        return float(rank * multiplier)
    raise ValueError(f"Unsupported lora.alpha_strategy: {strategy!r}")


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Freezes non-LoRA parameters and keeps LoRA matrices trainable."""

    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.endswith("lora_A") or name.endswith("lora_B")


def iter_lora_layers(model: nn.Module) -> Iterator[BaseLoRALinear]:
    """Iterates over injected LoRA-style layers."""

    for module in model.modules():
        if isinstance(module, BaseLoRALinear):
            yield module


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Collects only LoRA trainable weights for checkpointing."""

    state: Dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            state[name] = parameter.detach().cpu()
    return state


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Loads a trainable-only LoRA state dict."""

    named_parameters = dict(model.named_parameters())
    for name, tensor in state_dict.items():
        if name not in named_parameters:
            raise KeyError(f"Unknown LoRA parameter in checkpoint: {name}")
        named_parameters[name].data.copy_(tensor.to(named_parameters[name].device, dtype=named_parameters[name].dtype))


def inject_lora_layers(
    model: nn.Module,
    *,
    adapter_cls: Type[BaseLoRALinear],
    lora_config: Dict[str, object],
    target_modules: Sequence[str],
    svd_cache_dir: str,
    random_seed: int,
) -> List[Dict[str, object]]:
    """Replaces target linear layers with LoRA-family wrappers."""

    layer_summaries: List[Dict[str, object]] = []
    for index, (layer_name, linear_module) in enumerate(iter_target_linear_modules(model, target_modules)):
        if not resolve_target_layer_enabled(layer_name, lora_config.get("target_layers", "all")):
            continue
        adapter_rank = resolve_lora_rank(layer_name, lora_config)
        adapter_alpha = resolve_lora_alpha(layer_name, lora_config, rank=adapter_rank)
        basis_result = build_projection_basis(
            linear_module.weight.detach(),
            rank_k=int(lora_config.get("projection_rank_k", 0)),
            cache_dir=svd_cache_dir,
            cache_key=f"{layer_name}|{tuple(linear_module.weight.shape)}|k={lora_config.get('projection_rank_k', 0)}",
            layer_name=layer_name,
            left_source=str(lora_config.get("left_projection_source", "svd")),
            right_source=str(lora_config.get("right_projection_source", "svd")),
            activation_cov_dir=lora_config.get("activation_cov_dir"),
            projection_basis_dir=lora_config.get("projection_basis_dir"),
            enforce_rank_mode=str(lora_config.get("enforce_projection_rank", "clip")),
            random_seed=random_seed + index * 101,
            cache_svd=bool(lora_config.get("cache_svd", True)),
        )
        capture_rank_value = lora_config.get("capture_basis_rank_k")
        if capture_rank_value is None:
            capture_rank_value = lora_config.get("capture_rank_k", lora_config.get("projection_rank_k", 0))
        capture_basis_result = build_capture_basis(
            layer_name=layer_name,
            input_dim=linear_module.in_features,
            rank_k=int(capture_rank_value),
            source=str(lora_config.get("capture_basis_source", "none")),
            activation_cov_dir=lora_config.get("activation_cov_dir"),
            capture_basis_dir=lora_config.get("capture_basis_dir"),
            projection_basis_dir=lora_config.get("projection_basis_dir"),
            preserve_right_basis=basis_result.basis.right_basis,
            residualize_against_preserve=bool(lora_config.get("capture_basis_residualize_against_preserve", False)),
        )
        adapter = adapter_cls(
            linear_module,
            layer_name=layer_name,
            rank=adapter_rank,
            alpha=adapter_alpha,
            dropout=float(lora_config.get("dropout", 0.0)),
            projection_basis=basis_result.basis,
            use_left_projection=bool(lora_config.get("use_left_projection", False)),
            use_right_projection=bool(lora_config.get("use_right_projection", False)),
            use_capture_loss=bool(lora_config.get("use_capture_loss", False)),
            use_capture_alignment_loss=bool(lora_config.get("use_capture_alignment_loss", False)),
            capture_space=str(lora_config.get("capture_space", "full")),
            capture_weighting=str(lora_config.get("capture_weighting", "none")),
            capture_basis=capture_basis_result.basis,
            capture_basis_source=capture_basis_result.source,
            capture_basis_metadata=capture_basis_result.metadata,
            detach_capture_inputs=bool(lora_config.get("detach_capture_inputs", True)),
            capture_enabled=resolve_capture_enabled(layer_name, lora_config.get("capture_layers", "all")),
            projection_metadata=basis_result.metadata,
            eps=float(lora_config.get("eps", 1e-6)),
        )
        parent_module, child_name = get_parent_module(model, layer_name)
        setattr(parent_module, child_name, adapter)
        layer_summaries.append(
            {
                "layer_name": layer_name,
                "rank": adapter_rank,
                "alpha": adapter_alpha,
                "projection_rank_k": adapter.projection_rank_k,
                "left_projection_source": adapter.left_projection_source,
                "right_projection_source": adapter.right_projection_source,
                "capture_basis_source": adapter.capture_basis_source,
                "capture_rank_k": adapter.capture_rank_k,
                "capture_basis_residualized": capture_basis_result.metadata.get("residualized_against_preserve"),
                "capture_enabled": adapter.capture_enabled,
                "svd_cache_hit": basis_result.metadata.get("cache_hit"),
                "svd_time_sec": basis_result.metadata.get("build_time_sec"),
            }
        )
    mark_only_lora_trainable(model)
    return layer_summaries
