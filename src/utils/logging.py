"""Logging helpers for experiment outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Creates a directory if needed and returns it as a Path."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


class JsonlLogger:
    """Appends one JSON object per line to a file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: Dict[str, Any]) -> None:
        validate_finite_payload(payload, context=str(self.path))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Writes formatted JSON to disk."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    validate_finite_payload(payload, context=str(path))
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def save_yaml(path: str | Path, payload: Dict[str, Any]) -> None:
    """Writes YAML to disk."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def validate_finite_payload(payload: Dict[str, Any], *, context: str) -> None:
    """Raises a clear error if a flat logging payload contains NaN or Inf."""

    invalid = []
    for key, value in payload.items():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            invalid.append(f"{key}={value}")
    if invalid:
        joined = ", ".join(invalid)
        raise ValueError(f"Non-finite values in payload for {context}: {joined}")
