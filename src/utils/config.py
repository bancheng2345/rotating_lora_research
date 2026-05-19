"""Configuration loading and override helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Loads a YAML file into a dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at {path}, got {type(data)!r}")
    return data


def deep_merge_dicts(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges mappings, with update values taking precedence."""

    merged = copy.deepcopy(base)
    for key, value in update.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_override_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def apply_overrides(config: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Applies dotlist overrides such as training.max_steps=10."""

    updated = copy.deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value.")
        key_path, raw_value = override.split("=", 1)
        value = _parse_override_value(raw_value)
        cursor = updated
        parts = key_path.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return updated


def load_config_stack(
    config_paths: Sequence[str | Path],
    *,
    base_config_path: Optional[str | Path] = None,
    overrides: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Loads base config plus an ordered stack of overlay configs."""

    config = {}
    if base_config_path is not None:
        config = load_yaml(base_config_path)
    for config_path in config_paths:
        overlay_config = load_yaml(config_path)
        config = deep_merge_dicts(config, overlay_config)
    return apply_overrides(config, overrides)


def load_config(
    config_path: str | Path,
    *,
    base_config_path: Optional[str | Path] = None,
    overrides: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Backward-compatible wrapper for loading a single overlay config."""

    return load_config_stack(
        [config_path],
        base_config_path=base_config_path,
        overrides=overrides,
    )


def save_yaml(path: str | Path, data: Dict[str, Any]) -> None:
    """Writes a YAML dictionary with stable key ordering."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)
