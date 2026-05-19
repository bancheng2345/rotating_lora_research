#!/usr/bin/env python
"""Summarizes MetaMathQA M_WA comparison runs into a compact table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def latest_complete_runs(
    root: Path,
    *,
    methods: Iterable[str],
    dataset_contains: str,
    notes_contains: Optional[str],
) -> Dict[str, Path]:
    selected: Dict[str, Path] = {}
    for method in methods:
        candidates = []
        for run_dir in root.glob(f"{method}_*"):
            summary_path = run_dir / "final_summary.json"
            if not summary_path.exists() or dataset_contains not in run_dir.name:
                continue
            try:
                summary = load_json(summary_path)
            except Exception:
                continue
            if summary.get("method") != method:
                continue
            if notes_contains:
                config_path = run_dir / "config_resolved.yaml"
                text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
                if notes_contains not in text and notes_contains not in str(summary.get("notes", "")):
                    continue
            candidates.append(run_dir)
        candidates.sort(key=lambda path: (path / "final_summary.json").stat().st_mtime, reverse=True)
        if candidates:
            selected[method] = candidates[0]
    return selected


def final_in_domain_eval(run_dir: Path) -> Dict[str, Any]:
    records = [
        record
        for record in load_jsonl(run_dir / "eval_metrics.jsonl")
        if record.get("eval_type", "in_domain") == "in_domain"
        and (record.get("eval_exact_match") is not None or record.get("perplexity") is not None)
    ]
    if not records:
        return {}
    records.sort(key=lambda record: int(record.get("step") or -1))
    return records[-1]


def mean_last_mechanism(run_dir: Path, key: str) -> Optional[float]:
    records = [record for record in load_jsonl(run_dir / "mechanism_metrics.jsonl") if record.get(key) is not None]
    if not records:
        return None
    max_step = max(int(record.get("step") or 0) for record in records)
    values = [float(record[key]) for record in records if int(record.get("step") or 0) == max_step]
    return mean(values) if values else None


def summarize_run(method: str, run_dir: Path) -> Dict[str, Any]:
    summary = load_json(run_dir / "final_summary.json")
    eval_record = final_in_domain_eval(run_dir)
    return {
        "method": method,
        "run_dir": str(run_dir),
        "steps": summary.get("max_steps"),
        "trainable_parameters": (summary.get("parameter_counts") or {}).get("trainable_parameters"),
        "ppl": eval_record.get("perplexity") or summary.get("final_eval_metric"),
        "eval_loss": eval_record.get("eval_loss") or eval_record.get("loss"),
        "exact_match": summary.get("math_generation_exact_match") or eval_record.get("eval_exact_match"),
        "exact_match_type": "generation_final_answer" if summary.get("math_generation_exact_match") is not None else eval_record.get("eval_exact_match_type"),
        "token_acc": eval_record.get("eval_token_accuracy"),
        "teacher_forced_sequence_exact_match": eval_record.get("eval_sequence_exact_match"),
        "generation_exact_match": summary.get("math_generation_exact_match"),
        "generation_exact_match_count": summary.get("math_generation_exact_match_count"),
        "generation_num_samples": summary.get("math_generation_num_samples"),
        "omega_full": mean_last_mechanism(run_dir, "omega_full_mean") or summary.get("final_omega_full_mean"),
        "omega_residual": mean_last_mechanism(run_dir, "omega_residual_mean") or summary.get("final_omega_residual_mean"),
        "capture_alignment_score": (
            mean_last_mechanism(run_dir, "capture_alignment_score")
            or summary.get("final_capture_alignment_score")
            or summary.get("final_actcov_alignment_score")
        ),
        "capture_alignment_loss": (
            mean_last_mechanism(run_dir, "capture_alignment_loss")
            or summary.get("final_capture_alignment_loss")
            or summary.get("final_actcov_alignment_loss")
        ),
        "rho_k": mean_last_mechanism(run_dir, "rho_k") or summary.get("final_rho_k_mean"),
        "orth_error_A": mean_last_mechanism(run_dir, "orth_error_A") or summary.get("final_orth_error_A"),
        "peak_gpu_memory_gb": summary.get("peak_gpu_memory_gb"),
        "total_train_time_sec": summary.get("total_train_time_sec"),
    }


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    columns = [
        "method",
        "steps",
        "exact_match",
        "exact_match_type",
        "ppl",
        "token_acc",
        "omega_residual",
        "capture_alignment_score",
        "rho_k",
        "peak_gpu_memory_gb",
        "total_train_time_sec",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column)
                if isinstance(value, float):
                    values.append(f"{value:.6g}")
                elif value is None:
                    values.append("")
                else:
                    values.append(str(value))
            handle.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/runs/meta_llama_llama_3_1_8b")
    parser.add_argument("--methods", default="lora,oplora,pc_lora,residual_mwa_pc_lora")
    parser.add_argument("--dataset-contains", default="metamathqa_train_100000")
    parser.add_argument("--notes-contains", default=None)
    parser.add_argument("--out", default="results/tables/metamathqa_mwa_compare.csv")
    args = parser.parse_args()

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    runs = latest_complete_runs(
        Path(args.root),
        methods=methods,
        dataset_contains=str(args.dataset_contains),
        notes_contains=args.notes_contains,
    )
    rows = [summarize_run(method, runs[method]) for method in methods if method in runs]
    if not rows:
        raise SystemExit("No matching completed runs found.")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, output_path.with_suffix(".md"))
    print(json.dumps({"rows": len(rows), "csv": str(output_path), "markdown": str(output_path.with_suffix(".md"))}, indent=2))


if __name__ == "__main__":
    main()
