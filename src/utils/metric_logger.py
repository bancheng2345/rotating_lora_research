"""Structured JSONL loggers for train/eval/mechanism/efficiency metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.utils.logging import ensure_dir, validate_finite_payload


def timestamp_now() -> str:
    """Returns an ISO-8601 UTC timestamp string."""

    return datetime.now(timezone.utc).isoformat()


class JsonlMetricLogger:
    """Appends one JSON object per line and flushes immediately."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def log(self, payload: Dict[str, Any]) -> None:
        validate_finite_payload(payload, context=str(self.path))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()


@dataclass
class MetricLoggers:
    """Bundle of per-run metric loggers."""

    train: JsonlMetricLogger
    eval: JsonlMetricLogger
    forgetting: JsonlMetricLogger
    mechanism: JsonlMetricLogger
    efficiency: JsonlMetricLogger


def create_metric_loggers(run_dir: str | Path) -> MetricLoggers:
    """Creates the standard set of per-run metric loggers."""

    run_dir = ensure_dir(run_dir)
    return MetricLoggers(
        train=JsonlMetricLogger(run_dir / "train_metrics.jsonl"),
        eval=JsonlMetricLogger(run_dir / "eval_metrics.jsonl"),
        forgetting=JsonlMetricLogger(run_dir / "forgetting_metrics.jsonl"),
        mechanism=JsonlMetricLogger(run_dir / "mechanism_metrics.jsonl"),
        efficiency=JsonlMetricLogger(run_dir / "efficiency_metrics.jsonl"),
    )
