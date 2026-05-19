"""GPU selection helpers with SXM4 preference support."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SelectedGPU:
    """Selected GPU index and decision metadata."""

    visible_devices: str
    mode: str
    reason: str
    detected_gpus: List[Dict[str, str]]


def query_gpus() -> List[Dict[str, str]]:
    """Queries GPU index/name pairs via nvidia-smi."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    gpus: List[Dict[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, name = [part.strip() for part in line.split(",", 1)]
        gpus.append({"index": index, "name": name})
    return gpus


def select_gpu(config: Dict[str, object]) -> SelectedGPU:
    """Selects a CUDA_VISIBLE_DEVICES value from runtime.gpu config."""

    gpu_cfg = config.get("runtime", {}).get("gpu", {})  # type: ignore[union-attr]
    mode = str(gpu_cfg.get("mode", "prefer_sxm4"))
    requested = str(gpu_cfg.get("cuda_visible_devices", "auto"))
    fallback = str(gpu_cfg.get("fallback_cuda_visible_devices", "0"))
    detected = query_gpus()

    if requested not in {"", "auto"}:
        return SelectedGPU(
            visible_devices=requested,
            mode=mode,
            reason="manual_override",
            detected_gpus=detected,
        )
    if mode == "disabled":
        return SelectedGPU(
            visible_devices="",
            mode=mode,
            reason="gpu_selection_disabled",
            detected_gpus=detected,
        )
    if mode == "prefer_sxm4":
        for gpu in detected:
            if "SXM4" in gpu["name"]:
                return SelectedGPU(
                    visible_devices=gpu["index"],
                    mode=mode,
                    reason="matched_sxm4_name",
                    detected_gpus=detected,
                )
    if detected:
        return SelectedGPU(
            visible_devices=detected[0]["index"],
            mode=mode,
            reason="selected_first_detected_gpu",
            detected_gpus=detected,
        )
    return SelectedGPU(
        visible_devices=fallback,
        mode=mode,
        reason="fallback_cuda_visible_devices",
        detected_gpus=[],
    )


def apply_gpu_override(config: Dict[str, object], gpu_value: Optional[str]) -> Dict[str, object]:
    """Applies a CLI-level GPU override to runtime.gpu.cuda_visible_devices."""

    if gpu_value is None:
        return config
    runtime_cfg = config.setdefault("runtime", {})  # type: ignore[assignment]
    gpu_cfg = runtime_cfg.setdefault("gpu", {})  # type: ignore[assignment]
    gpu_cfg["cuda_visible_devices"] = gpu_value
    return config


def apply_selected_gpu(config: Dict[str, object]) -> SelectedGPU:
    """Sets CUDA_VISIBLE_DEVICES based on selection config and returns the decision."""

    selection = select_gpu(config)
    override_env = (
        config.get("runtime", {})  # type: ignore[union-attr]
        .get("gpu", {})
        .get("override_env", "CUDA_VISIBLE_DEVICES")
    )
    if selection.visible_devices == "":
        os.environ.pop(str(override_env), None)
    else:
        os.environ[str(override_env)] = selection.visible_devices
    return selection


def selection_to_json(selection: SelectedGPU) -> str:
    """Serializes selection metadata for CLI output."""

    return json.dumps(
        {
            "visible_devices": selection.visible_devices,
            "mode": selection.mode,
            "reason": selection.reason,
            "detected_gpus": selection.detected_gpus,
        },
        ensure_ascii=False,
    )
