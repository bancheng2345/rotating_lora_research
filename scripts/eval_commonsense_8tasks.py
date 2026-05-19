#!/usr/bin/env python
"""Evaluate an adapter run on the 8 commonsense reasoning benchmarks."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.commonsense_8tasks import TASKS_8, evaluate_commonsense_tasks, write_jsonl
from src.training.trainer import (
    apply_method_to_model,
    build_model_and_tokenizer,
    configure_runtime_cache_dirs,
    load_adapter_checkpoint,
)
from src.utils.gpu_select import apply_gpu_override, apply_selected_gpu, selection_to_json
from src.utils.logging import JsonlLogger, save_json
from src.utils.metric_logger import timestamp_now
from src.utils.run_summary import load_jsonl, save_final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True, help="Existing run directory to evaluate.")
    parser.add_argument("--gpu", default=None, help="Optional CUDA_VISIBLE_DEVICES override.")
    parser.add_argument("--data-root", default="data/commonsense_eval", help="Prepared 8-task JSONL root.")
    parser.add_argument("--split", default="validation", help="Prepared split name.")
    parser.add_argument("--task", action="append", choices=TASKS_8, help="Task to evaluate. Repeatable.")
    parser.add_argument("--checkpoint-mode", choices=["best", "latest"], default="best")
    parser.add_argument("--batch-size", type=int, default=8, help="Candidate scoring microbatch size.")
    parser.add_argument("--max-examples-per-task", type=int, default=None)
    parser.add_argument("--no-length-normalize", action="store_true", help="Use sum log-prob instead of mean log-prob.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    return parser.parse_args()


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    """Loads config_resolved.yaml from a run directory."""

    return yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))


def checkpoint_step_pairs(run_dir: Path) -> List[Tuple[int, Path]]:
    """Lists adapter checkpoints as sorted (step, path) pairs."""

    pairs: List[Tuple[int, Path]] = []
    for path in sorted((run_dir / "checkpoints").glob("step_*.pt")):
        try:
            step = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        pairs.append((step, path))
    return pairs


def select_checkpoint_path(run_dir: Path, config: Dict[str, Any], mode: str) -> Tuple[int, Path]:
    """Selects latest or best checkpoint using in-domain eval logs."""

    checkpoints = checkpoint_step_pairs(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {run_dir / 'checkpoints'}.")
    if mode == "latest":
        return checkpoints[-1]

    metric_name = str(config.get("training", {}).get("eval_metric_name", "perplexity"))
    metric_mode = str(config.get("training", {}).get("eval_metric_mode", "min"))
    eval_records = load_jsonl(run_dir / "eval_metrics.jsonl")
    in_domain = [
        record
        for record in eval_records
        if record.get("eval_type", "in_domain") == "in_domain"
        and str(record.get("metric_name")) == metric_name
        and record.get("metric_value") is not None
        and record.get("generation_checkpoint_step") is None
    ]
    if not in_domain:
        return checkpoints[-1]
    best_record = min(in_domain, key=lambda record: float(record["metric_value"]))
    if metric_mode == "max":
        best_record = max(in_domain, key=lambda record: float(record["metric_value"]))
    step = int(best_record["step"])
    checkpoint_path = run_dir / "checkpoints" / f"step_{step}.pt"
    return (step, checkpoint_path) if checkpoint_path.exists() else checkpoints[-1]


def append_eval_records(run_dir: Path, result: Dict[str, Any], *, checkpoint_step: int, data_root: str, split: str) -> None:
    """Appends per-task and aggregate evaluation records to eval_metrics.jsonl."""

    logger = JsonlLogger(run_dir / "eval_metrics.jsonl")
    common = {
        "step": checkpoint_step,
        "epoch": None,
        "split": split,
        "eval_type": "commonsense_8task",
        "train_domain": "commonsense",
        "metric_name": "accuracy",
        "metric_mode": "max",
        "dataset": data_root,
        "eval_dataset": data_root,
        "timestamp": timestamp_now(),
    }
    for task, row in result["per_task"].items():
        logger.log(
            {
                **common,
                "domain": "commonsense",
                "eval_domain": task,
                "task": task,
                "metric_value": row["accuracy"],
                "eval_metric_name": "accuracy",
                "eval_metric_value": row["accuracy"],
                "eval_accuracy": row["accuracy"],
                "eval_exact_match": None,
                "num_correct": row["num_correct"],
                "num_samples": row["num_samples"],
                "eval_num_samples": row["num_samples"],
            }
        )
    logger.log(
        {
            **common,
            "domain": "commonsense",
            "eval_domain": "commonsense_8task",
            "task": "commonsense_8task",
            "metric_value": result["macro_accuracy"],
            "eval_metric_name": "macro_accuracy",
            "eval_metric_value": result["macro_accuracy"],
            "eval_accuracy": result["macro_accuracy"],
            "micro_accuracy": result["micro_accuracy"],
            "num_correct": result["num_correct"],
            "num_samples": result["num_samples"],
            "eval_num_samples": result["num_samples"],
        }
    )


def update_summary(run_dir: Path, result: Dict[str, Any], checkpoint_step: int, checkpoint_path: Path) -> None:
    """Updates final_summary.json with commonsense 8-task metrics."""

    summary = save_final_summary(run_dir)
    summary.update(
        {
            "commonsense_8task_macro_accuracy": result["macro_accuracy"],
            "commonsense_8task_micro_accuracy": result["micro_accuracy"],
            "commonsense_8task_num_samples": result["num_samples"],
            "commonsense_8task_checkpoint_step": checkpoint_step,
            "commonsense_8task_checkpoint": str(checkpoint_path),
            "commonsense_8task_per_task": result["per_task"],
        }
    )
    save_json(run_dir / "final_summary.json", summary)


def evaluate_run(
    run_dir: Path,
    *,
    gpu: str | None,
    data_root: str,
    split: str,
    tasks: List[str],
    checkpoint_mode: str,
    batch_size: int,
    max_examples_per_task: int | None,
    length_normalize: bool,
    show_progress: bool,
) -> Dict[str, Any]:
    """Loads one adapter run and evaluates it on the requested tasks."""

    config = load_run_config(run_dir)
    apply_gpu_override(config, gpu)
    configure_runtime_cache_dirs(config)
    selection = apply_selected_gpu(config)
    print(selection_to_json(selection), flush=True)

    checkpoint_step, checkpoint_path = select_checkpoint_path(run_dir, config, checkpoint_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = build_model_and_tokenizer(config, device=device)
    apply_method_to_model(model, config, run_dir=run_dir)
    model.to(device)
    load_adapter_checkpoint(model, checkpoint_path)

    result = evaluate_commonsense_tasks(
        model,
        tokenizer,
        data_root=data_root,
        tasks=tasks,
        split=split,
        device=device,
        batch_size=batch_size,
        max_examples_per_task=max_examples_per_task,
        length_normalize=length_normalize,
        show_progress=show_progress,
    )
    result.update(
        {
            "run_dir": str(run_dir),
            "checkpoint_step": checkpoint_step,
            "checkpoint_path": str(checkpoint_path),
            "timestamp": timestamp_now(),
        }
    )

    save_json(run_dir / "commonsense_8task_eval.json", {k: v for k, v in result.items() if k != "predictions"})
    pred_dir = run_dir / "commonsense_8task_predictions"
    pred_dir.mkdir(exist_ok=True)
    for task, rows in result["predictions"].items():
        write_jsonl(pred_dir / f"{task}_{split}.jsonl", rows)

    append_eval_records(run_dir, result, checkpoint_step=checkpoint_step, data_root=data_root, split=split)
    update_summary(run_dir, result, checkpoint_step, checkpoint_path)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "run_dir": str(run_dir),
        "checkpoint_step": checkpoint_step,
        "macro_accuracy": result["macro_accuracy"],
        "micro_accuracy": result["micro_accuracy"],
        "per_task": result["per_task"],
    }


def main() -> None:
    args = parse_args()
    tasks = args.task or TASKS_8
    summaries = []
    for run_dir in [Path(path) for path in args.run_dir]:
        summaries.append(
            evaluate_run(
                run_dir,
                gpu=args.gpu,
                data_root=args.data_root,
                split=args.split,
                tasks=tasks,
                checkpoint_mode=args.checkpoint_mode,
                batch_size=int(args.batch_size),
                max_examples_per_task=args.max_examples_per_task,
                length_normalize=not bool(args.no_length_normalize),
                show_progress=not bool(args.no_progress),
            )
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

