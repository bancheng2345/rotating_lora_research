"""Run-level metric aggregation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from src.utils.logging import save_json
from src.utils.metrics import summarize_metric_records


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Loads a JSONL file, returning an empty list when the file is absent."""

    target = Path(path)
    if not target.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _best_metric(records: Iterable[Dict[str, Any]], *, metric_mode: str) -> float | None:
    values = [record.get("metric_value") for record in records if record.get("metric_value") is not None]
    if not values:
        return None
    return float(min(values) if metric_mode == "min" else max(values))


def _final_metric(records: List[Dict[str, Any]]) -> float | None:
    if not records:
        return None
    value = records[-1].get("metric_value")
    return None if value is None else float(value)


def _best_forgetting_score(records: Iterable[Dict[str, Any]]) -> float | None:
    values = [record.get("current_score") for record in records if record.get("current_score") is not None]
    if not values:
        return None
    return float(min(values))


def _final_forgetting_score(records: List[Dict[str, Any]]) -> float | None:
    if not records:
        return None
    value = records[-1].get("current_score")
    return None if value is None else float(value)


def _metric_value_for_record(record: Dict[str, Any], *, metric_name: str) -> float | None:
    """Extracts a metric value from new or legacy eval records."""

    if record.get("metric_value") is not None and str(record.get("metric_name")) == metric_name:
        return float(record["metric_value"])
    if metric_name == "perplexity":
        value = record.get("perplexity") if record.get("perplexity") is not None else record.get("eval_metric")
        return None if value is None else float(value)
    if metric_name == "accuracy":
        value = record.get("eval_accuracy")
        return None if value is None else float(value)
    if metric_name in {"exact_match", "em"}:
        value = record.get("eval_exact_match")
        return None if value is None else float(value)
    return None


def _record_matches_metric(record: Dict[str, Any], *, metric_name: str) -> bool:
    """Checks whether an eval record corresponds to the configured primary metric."""

    if record.get("metric_name") is not None:
        return str(record.get("metric_name")) == metric_name
    return _metric_value_for_record(record, metric_name=metric_name) is not None


def build_final_summary(run_dir: str | Path) -> Dict[str, Any]:
    """Builds a final_summary payload from resolved config and jsonl logs."""

    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    eval_records = load_jsonl(run_dir / "eval_metrics.jsonl")
    forgetting_records = load_jsonl(run_dir / "forgetting_metrics.jsonl")
    mechanism_records = load_jsonl(run_dir / "mechanism_metrics.jsonl")
    efficiency_records = load_jsonl(run_dir / "efficiency_metrics.jsonl")
    math_generation_summary_path = run_dir / "math_generation_eval.json"
    commonsense_8task_summary_path = run_dir / "commonsense_8task_eval.json"

    primary_metric_name = str(config["training"].get("eval_metric_name", "perplexity"))
    in_domain_records = [record for record in eval_records if record.get("eval_type", "in_domain") == "in_domain"]
    primary_metric_records = [
        record for record in in_domain_records if _record_matches_metric(record, metric_name=primary_metric_name)
    ] or in_domain_records
    final_mechanism_step = max((record["step"] for record in mechanism_records), default=None)
    if final_mechanism_step is None:
        final_mechanism = []
    else:
        final_mechanism = [record for record in mechanism_records if record["step"] == final_mechanism_step]

    rho_target_k = int(config["lora"].get("projection_rank_k", 0))
    metric_mode = str(config["training"].get("eval_metric_mode", "min"))
    rho_records = [
        record
        for record in final_mechanism
        if record.get("rho_k") is not None
        and (record.get("k") == rho_target_k or record.get("k") is None)
    ]
    summary = {
        "method": config["lora"]["method"],
        "model": config["model"].get("name_or_path") or config["model"].get("synthetic_name"),
        "dataset": config["training"].get("train_file") or config["training"].get("dataset_name"),
        "train_domain": config.get("forgetting_eval", {}).get("train_domain") or config["training"].get("domain"),
        "lora_rank": int(config["lora"]["rank"]),
        "projection_rank_k": int(config["lora"].get("projection_rank_k", 0)),
        "lambda_cap": float(config["lora"].get("lambda_cap", 0.0)),
        "lambda_align": float(config["lora"].get("lambda_align", 0.0)),
        "lambda_orth": float(config["lora"].get("lambda_orth", 0.0)),
        "best_eval_metric": _best_metric(
            [
                {"metric_value": _metric_value_for_record(record, metric_name=primary_metric_name)}
                for record in primary_metric_records
                if _metric_value_for_record(record, metric_name=primary_metric_name) is not None
            ],
            metric_mode=metric_mode,
        ),
        "final_eval_metric": _final_metric(
            [
                {"metric_value": _metric_value_for_record(record, metric_name=primary_metric_name)}
                for record in primary_metric_records
                if _metric_value_for_record(record, metric_name=primary_metric_name) is not None
            ]
        ),
        "best_forgetting_score": _best_forgetting_score(forgetting_records),
        "final_forgetting_score": _final_forgetting_score(forgetting_records),
        "final_rho_k_mean": summarize_metric_records(rho_records, "rho_k"),
        "final_omega_full_mean": summarize_metric_records(final_mechanism, "omega_full_mean"),
        "final_omega_residual_mean": summarize_metric_records(final_mechanism, "omega_residual_mean"),
        "final_actcov_alignment_loss": summarize_metric_records(final_mechanism, "actcov_alignment_loss"),
        "final_actcov_alignment_score": summarize_metric_records(final_mechanism, "actcov_alignment_score"),
        "final_capture_alignment_loss": summarize_metric_records(final_mechanism, "capture_alignment_loss"),
        "final_capture_alignment_score": summarize_metric_records(final_mechanism, "capture_alignment_score"),
        "final_capture_energy_task": summarize_metric_records(final_mechanism, "capture_energy_task"),
        "final_orth_error_A": summarize_metric_records(final_mechanism, "orth_error_A"),
        "total_train_time_sec": efficiency_records[-1].get("total_train_time_sec") if efficiency_records else None,
        "peak_gpu_memory_gb": summarize_metric_records(efficiency_records, "peak_gpu_memory_gb"),
        "output_dir": str(run_dir),
    }
    if primary_metric_records:
        summary["eval_metric_name"] = primary_metric_name
        summary["eval_metric_value"] = _metric_value_for_record(
            primary_metric_records[-1],
            metric_name=primary_metric_name,
        )
    else:
        summary["eval_metric_name"] = None
        summary["eval_metric_value"] = None
    if math_generation_summary_path.exists():
        math_generation_summary = json.loads(math_generation_summary_path.read_text(encoding="utf-8"))
        summary["math_generation_exact_match"] = math_generation_summary.get("exact_match")
        summary["math_generation_exact_match_count"] = math_generation_summary.get("exact_match_count")
        summary["math_generation_num_samples"] = math_generation_summary.get("num_samples")
        summary["math_generation_checkpoint_step"] = math_generation_summary.get("checkpoint_step")
        summary["math_generation_checkpoint"] = math_generation_summary.get("checkpoint_path")
    if commonsense_8task_summary_path.exists():
        commonsense_summary = json.loads(commonsense_8task_summary_path.read_text(encoding="utf-8"))
        summary["commonsense_8task_macro_accuracy"] = commonsense_summary.get("macro_accuracy")
        summary["commonsense_8task_micro_accuracy"] = commonsense_summary.get("micro_accuracy")
        summary["commonsense_8task_num_samples"] = commonsense_summary.get("num_samples")
        summary["commonsense_8task_checkpoint_step"] = commonsense_summary.get("checkpoint_step")
        summary["commonsense_8task_checkpoint"] = commonsense_summary.get("checkpoint_path")
        summary["commonsense_8task_per_task"] = commonsense_summary.get("per_task")
    return summary


def save_final_summary(run_dir: str | Path) -> Dict[str, Any]:
    """Writes final_summary.json and returns the payload."""

    summary = build_final_summary(run_dir)
    save_json(Path(run_dir) / "final_summary.json", summary)
    return summary
