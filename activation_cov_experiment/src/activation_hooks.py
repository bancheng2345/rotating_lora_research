"""Activation hook utilities for collecting module input activations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from activation_cov_experiment.src.io_utils import ensure_dir


LAYER_PATTERNS = [
    re.compile(r"\.layers\.(\d+)\."),
    re.compile(r"\.h\.(\d+)\."),
    re.compile(r"\.blocks\.(\d+)\."),
]


def sanitize_module_name(name: str) -> str:
    """Converts a module path into a filesystem-safe identifier."""

    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


def extract_layer_index(module_name: str) -> Optional[int]:
    """Attempts to recover a transformer block index from a module path."""

    for pattern in LAYER_PATTERNS:
        match = pattern.search(module_name)
        if match:
            return int(match.group(1))
    return None


def module_io_dims(module: nn.Module) -> tuple[int, int]:
    """Infers input/output dimensions for linear-like modules."""

    if not hasattr(module, "weight"):
        raise ValueError(f"Module {module} has no weight attribute.")
    weight = module.weight
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError(f"Unsupported module weight shape for {module.__class__.__name__}.")
    if isinstance(module, nn.Linear):
        return int(weight.shape[1]), int(weight.shape[0])
    if hasattr(module, "nf"):  # GPT-2 Conv1D
        return int(weight.shape[0]), int(weight.shape[1])
    return int(weight.shape[1]), int(weight.shape[0])


@dataclass
class ResolvedTarget:
    """One resolved module target for activation or direction analysis."""

    module_name: str
    module_suffix: str
    module_key: str
    layer_index: Optional[int]
    input_dim: int
    output_dim: int


def available_module_names(model: nn.Module) -> List[str]:
    """Lists all named modules except the root module."""

    return [name for name, _ in model.named_modules() if name]


def resolve_targets(model: nn.Module, targets_cfg: Dict[str, object]) -> List[ResolvedTarget]:
    """Resolves configured layer/module selections into concrete module targets."""

    requested_modules = [str(name) for name in targets_cfg.get("modules", [])]
    if not requested_modules:
        raise ValueError("targets.modules must be non-empty.")

    candidates: List[ResolvedTarget] = []
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if not any(module_name.endswith(f".{requested}") or module_name == requested for requested in requested_modules):
            continue
        if not hasattr(module, "weight"):
            continue
        try:
            input_dim, output_dim = module_io_dims(module)
        except ValueError:
            continue
        candidates.append(
            ResolvedTarget(
                module_name=module_name,
                module_suffix=module_name.split(".")[-1],
                module_key=sanitize_module_name(module_name),
                layer_index=extract_layer_index(module_name),
                input_dim=input_dim,
                output_dim=output_dim,
            )
        )

    if not candidates:
        available = available_module_names(model)
        raise ValueError(
            "No matching target modules found. Requested modules: "
            f"{requested_modules}. Example available modules: {available[:40]}"
        )

    layer_selection = str(targets_cfg.get("layer_selection", "explicit"))
    selected_layers: Optional[set[int]]
    selected_layers = None
    if layer_selection == "explicit":
        selected_layers = {int(value) for value in targets_cfg.get("layers", [])}
    elif layer_selection == "last_n":
        last_n = int(targets_cfg.get("last_n", 1))
        layer_ids = sorted({target.layer_index for target in candidates if target.layer_index is not None})
        selected_layers = set(layer_ids[-last_n:])
    elif layer_selection == "all":
        selected_layers = None
    else:
        raise ValueError(f"Unsupported targets.layer_selection '{layer_selection}'.")

    if selected_layers is None:
        filtered = candidates
    else:
        filtered = [
            target
            for target in candidates
            if target.layer_index is not None and target.layer_index in selected_layers
        ]
    if not filtered:
        raise ValueError(
            f"No modules remained after applying layer selection '{layer_selection}' with "
            f"layers={targets_cfg.get('layers')}."
        )
    filtered.sort(key=lambda target: (target.layer_index if target.layer_index is not None else -1, target.module_name))
    return filtered


class ActivationChunkWriter:
    """Buffers token activations and writes them to chunked .pt files on CPU."""

    def __init__(self, target_dir: Path, *, chunk_tokens: int):
        self.target_dir = ensure_dir(target_dir)
        self.chunk_tokens = int(chunk_tokens)
        self.chunk_index = 0
        self.buffer: List[torch.Tensor] = []
        self.buffer_tokens = 0
        self.total_tokens = 0
        self.hidden_dim: Optional[int] = None
        self.saved_dtype: Optional[str] = None
        self.chunk_files: List[str] = []

    def append(self, values: torch.Tensor) -> None:
        """Appends a [tokens, hidden] tensor to the writer buffer."""

        if values.numel() == 0:
            return
        if values.ndim != 2:
            raise ValueError(f"Expected [tokens, hidden] activation tensor, got shape {tuple(values.shape)}.")
        self.hidden_dim = int(values.shape[-1])
        cpu_values = values.detach().to(device="cpu")
        if cpu_values.dtype in {torch.float16, torch.bfloat16}:
            cpu_values = cpu_values.to(dtype=torch.float16)
        else:
            cpu_values = cpu_values.to(dtype=torch.float32)
        self.saved_dtype = str(cpu_values.dtype).replace("torch.", "")
        self.buffer.append(cpu_values)
        self.buffer_tokens += int(cpu_values.shape[0])
        self.total_tokens += int(cpu_values.shape[0])
        if self.buffer_tokens >= self.chunk_tokens:
            self.flush()

    def flush(self) -> None:
        """Writes the current activation buffer to disk."""

        if not self.buffer:
            return
        chunk = torch.cat(self.buffer, dim=0)
        chunk_path = self.target_dir / f"chunk_{self.chunk_index:05d}.pt"
        torch.save(chunk, chunk_path)
        self.chunk_files.append(chunk_path.name)
        self.chunk_index += 1
        self.buffer = []
        self.buffer_tokens = 0

    def finalize(self) -> Dict[str, object]:
        """Flushes pending activations and returns metadata."""

        self.flush()
        return {
            "target_dir": str(self.target_dir),
            "chunk_files": self.chunk_files,
            "num_chunks": self.chunk_index,
            "num_tokens": self.total_tokens,
            "hidden_dim": self.hidden_dim,
            "saved_dtype": self.saved_dtype,
        }


class ActivationCollector:
    """Registers forward-pre hooks and writes module input activations to disk."""

    def __init__(
        self,
        model: nn.Module,
        targets: List[ResolvedTarget],
        *,
        output_dir: Path,
        chunk_tokens: int,
    ):
        self.model = model
        self.targets = targets
        self.output_dir = ensure_dir(output_dir)
        self.chunk_tokens = int(chunk_tokens)
        self.current_attention_mask: Optional[torch.Tensor] = None
        self.handles = []
        self.writers: Dict[str, ActivationChunkWriter] = {}
        named_modules = dict(model.named_modules())

        for target in targets:
            module = named_modules[target.module_name]
            writer = ActivationChunkWriter(self.output_dir / target.module_key, chunk_tokens=self.chunk_tokens)
            self.writers[target.module_name] = writer
            handle = module.register_forward_pre_hook(self._make_hook(target.module_name))
            self.handles.append(handle)

    def _make_hook(self, module_name: str):
        def hook(_module, inputs):
            if not inputs:
                return
            values = inputs[0]
            if not isinstance(values, torch.Tensor):
                return
            if values.ndim == 2:
                flat = values.reshape(-1, values.shape[-1])
            elif values.ndim == 3:
                if self.current_attention_mask is not None and self.current_attention_mask.shape[:2] == values.shape[:2]:
                    mask = self.current_attention_mask.to(device=values.device, dtype=torch.bool)
                    flat = values[mask]
                else:
                    flat = values.reshape(-1, values.shape[-1])
            else:
                flat = values.reshape(-1, values.shape[-1])
            self.writers[module_name].append(flat)

        return hook

    def set_attention_mask(self, attention_mask: Optional[torch.Tensor]) -> None:
        """Sets the current batch attention mask for token filtering."""

        self.current_attention_mask = attention_mask

    def finalize(self) -> List[Dict[str, object]]:
        """Flushes all writers and removes hooks, returning per-target metadata."""

        records: List[Dict[str, object]] = []
        for handle in self.handles:
            handle.remove()
        self.handles = []
        for target in self.targets:
            records.append(
                {
                    "module_name": target.module_name,
                    "module_suffix": target.module_suffix,
                    "module_key": target.module_key,
                    "layer_index": target.layer_index,
                    "input_dim": target.input_dim,
                    "output_dim": target.output_dim,
                    **self.writers[target.module_name].finalize(),
                }
            )
        return records
