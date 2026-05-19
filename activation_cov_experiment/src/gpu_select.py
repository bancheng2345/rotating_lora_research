"""Standalone GPU selection helpers for the isolated activation covariance experiment."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SelectedGPU:
    """Selected GPU decision metadata."""

    visible_devices: str
    mode: str
    reason: str
    detected_gpus: List[Dict[str, str]]


def query_gpus() -> List[Dict[str, str]]:
    """Returns detected GPU index/name pairs using nvidia-smi."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    pairs: List[Dict[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, name = [part.strip() for part in line.split(",", 1)]
        pairs.append({"index": index, "name": name})
    return pairs


def select_gpu(config: Dict[str, object]) -> SelectedGPU:
    """Selects a CUDA_VISIBLE_DEVICES value from runtime.gpu config."""

    gpu_cfg = config.get("runtime", {}).get("gpu", {})  # type: ignore[union-attr]
    mode = str(gpu_cfg.get("mode", "prefer_pcie"))
    requested = str(gpu_cfg.get("cuda_visible_devices", "auto"))
    fallback = str(gpu_cfg.get("fallback_cuda_visible_devices", "0"))
    detected = query_gpus()

    if requested not in {"", "auto"}:
        return SelectedGPU(requested, mode, "manual_override", detected)
    if mode == "disabled":
        return SelectedGPU("", mode, "gpu_selection_disabled", detected)
    if mode == "prefer_sxm4":
        for gpu in detected:
            if "SXM4" in gpu["name"]:
                return SelectedGPU(gpu["index"], mode, "matched_sxm4_name", detected)
    if mode == "prefer_pcie":
        for gpu in detected:
            if "PCIe" in gpu["name"]:
                return SelectedGPU(gpu["index"], mode, "matched_pcie_name", detected)
    if detected:
        return SelectedGPU(detected[0]["index"], mode, "selected_first_detected_gpu", detected)
    return SelectedGPU(fallback, mode, "fallback_cuda_visible_devices", [])


def apply_gpu_override(config: Dict[str, object], gpu_value: Optional[str]) -> Dict[str, object]:
    """Applies a CLI-level GPU override to runtime.gpu.cuda_visible_devices."""

    if gpu_value is None:
        return config
    runtime_cfg = config.setdefault("runtime", {})  # type: ignore[assignment]
    gpu_cfg = runtime_cfg.setdefault("gpu", {})  # type: ignore[assignment]
    gpu_cfg["cuda_visible_devices"] = gpu_value
    return config


def apply_selected_gpu(config: Dict[str, object]) -> SelectedGPU:
    """Sets CUDA_VISIBLE_DEVICES and returns the selection metadata."""

    selection = select_gpu(config)
    env_name = (
        config.get("runtime", {})  # type: ignore[union-attr]
        .get("gpu", {})
        .get("override_env", "CUDA_VISIBLE_DEVICES")
    )
    if selection.visible_devices == "":
        os.environ.pop(str(env_name), None)
    else:
        os.environ[str(env_name)] = selection.visible_devices
    return selection


def selection_to_json(selection: SelectedGPU) -> str:
    """Serializes selection metadata for CLI printing."""

    return json.dumps(
        {
            "visible_devices": selection.visible_devices,
            "mode": selection.mode,
            "reason": selection.reason,
            "detected_gpus": selection.detected_gpus,
        },
        ensure_ascii=False,
    )
