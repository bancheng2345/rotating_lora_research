#!/usr/bin/env python
"""Generation-based math exact-match evaluation for existing run directories."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.eval_math import evaluate_math_generation, load_jsonl_records
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
    parser.add_argument("--run-dir", action="append", required=True, help="Existing run directory to reevaluate.")
    parser.add_argument("--gpu", default=None, help="Optional CUDA_VISIBLE_DEVICES override.")
    parser.add_argument(
        "--checkpoint-mode",
        choices=["best", "latest"],
        default="best",
        help="Which existing checkpoint to reevaluate.",
    )
    parser.add_argument("--eval-file", default=None, help="Optional override eval jsonl path.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap on eval examples.")
    parser.add_argument("--max-prompt-length", type=int, default=384, help="Prompt truncation length.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generated answer budget.")
    parser.add_argument("--batch-size", type=int, default=1, help="Generation microbatch size.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable generation progress bar/logging.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding. Defaults to greedy.",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    return parser.parse_args()


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    """Loads the resolved config of an existing run."""

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
    """Selects the target checkpoint from existing eval logs."""

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
    ]
    if not in_domain:
        return checkpoints[-1]

    best_record = min(in_domain, key=lambda record: float(record["metric_value"]))
    if metric_mode == "max":
        best_record = max(in_domain, key=lambda record: float(record["metric_value"]))
    step = int(best_record["step"])
    checkpoint_path = run_dir / "checkpoints" / f"step_{step}.pt"
    if checkpoint_path.exists():
        return step, checkpoint_path
    return checkpoints[-1]


def eval_dataset_path(config: Dict[str, Any], override: str | None) -> Path:
    """Resolves the evaluation dataset path."""

    if override:
        candidate = Path(override)
        return candidate if candidate.is_absolute() else ROOT / candidate
    candidate = config.get("training", {}).get("eval_file")
    if not candidate:
        raise ValueError("training.eval_file is required for math generation evaluation.")
    candidate_path = Path(str(candidate))
    return candidate_path if candidate_path.is_absolute() else ROOT / candidate_path


def prompt_template_from_config(config: Dict[str, Any]) -> str:
    """Returns the prompt template used for supervised fine-tuning."""

    return str(
        config.get("training", {}).get(
            "prompt_template",
            "Question:\n{question}\n\nAnswer:\n",
        )
    )


def prompt_response_columns_from_config(config: Dict[str, Any]) -> Tuple[str, str]:
    """Returns the prompt/response columns used by the run dataset."""

    training_cfg = config.get("training", {})
    return (
        str(training_cfg.get("prompt_column", "question")),
        str(training_cfg.get("response_column", "answer")),
    )


def append_eval_record(run_dir: Path, record: Dict[str, Any]) -> None:
    """Appends a post-hoc evaluation record to eval_metrics.jsonl."""

    JsonlLogger(run_dir / "eval_metrics.jsonl").log(record)


def update_final_summary_from_generation(run_dir: Path, generation_summary: Dict[str, Any]) -> None:
    """Persists generation summary and refreshes final_summary.json."""

    save_json(run_dir / "math_generation_eval.json", generation_summary)
    summary = save_final_summary(run_dir)
    summary.update(
        {
            "math_generation_exact_match": generation_summary.get("exact_match"),
            "math_generation_exact_match_count": generation_summary.get("exact_match_count"),
            "math_generation_num_samples": generation_summary.get("num_samples"),
            "math_generation_checkpoint_step": generation_summary.get("checkpoint_step"),
            "math_generation_checkpoint": generation_summary.get("checkpoint_path"),
            "math_generation_batch_size": generation_summary.get("batch_size"),
        }
    )
    save_json(run_dir / "final_summary.json", summary)


def evaluate_run(
    run_dir: Path,
    *,
    gpu: str | None,
    checkpoint_mode: str,
    eval_file_override: str | None,
    max_examples: int | None,
    max_prompt_length: int,
    max_new_tokens: int,
    batch_size: int,
    show_progress: bool,
    do_sample: bool,
    temperature: float,
) -> Dict[str, Any]:
    """Loads one run, reevaluates it with generation EM, and saves artifacts in-place."""

    config = load_run_config(run_dir)
    apply_gpu_override(config, gpu)
    configure_runtime_cache_dirs(config)
    selection = apply_selected_gpu(config)
    print(selection_to_json(selection), flush=True)

    checkpoint_step, checkpoint_path = select_checkpoint_path(run_dir, config, checkpoint_mode)
    dataset_path = eval_dataset_path(config, eval_file_override)
    examples = load_jsonl_records(dataset_path, max_examples=max_examples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = build_model_and_tokenizer(config, device=device)
    apply_method_to_model(model, config, run_dir=run_dir)
    model.to(device)
    load_adapter_checkpoint(model, checkpoint_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model.eval()

    result = evaluate_math_generation(
        model,
        tokenizer,
        examples,
        device=device,
        prompt_template=prompt_template_from_config(config),
        prompt_column=prompt_response_columns_from_config(config)[0],
        response_column=prompt_response_columns_from_config(config)[1],
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        show_progress=show_progress,
        progress_desc=f"{run_dir.name} generation",
        do_sample=do_sample,
        temperature=temperature,
    )
    result.update(
        {
            "run_dir": str(run_dir),
            "checkpoint_step": checkpoint_step,
            "checkpoint_path": str(checkpoint_path),
            "dataset": str(dataset_path),
            "metric_name": "exact_match",
            "metric_mode": "max",
            "exact_match_type": "generation_final_answer",
            "batch_size": int(batch_size),
            "timestamp": timestamp_now(),
        }
    )

    predictions_path = run_dir / f"math_generation_predictions_step_{checkpoint_step}.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in result["predictions"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    append_eval_record(
        run_dir,
        {
            "step": checkpoint_step,
            "epoch": None,
            "split": str(config.get("training", {}).get("eval_split", "validation")),
            "dataset": str(dataset_path),
            "domain": config.get("training", {}).get("eval_domain") or config.get("training", {}).get("domain"),
            "eval_dataset": str(dataset_path),
            "eval_domain": config.get("training", {}).get("eval_domain") or config.get("training", {}).get("domain"),
            "eval_type": "in_domain",
            "train_domain": config.get("forgetting_eval", {}).get("train_domain") or config.get("training", {}).get("domain"),
            "metric_name": "exact_match",
            "metric_mode": "max",
            "metric_value": result["exact_match"],
            "eval_metric_name": "exact_match",
            "eval_metric_value": result["exact_match"],
            "loss": None,
            "eval_loss": None,
            "perplexity": None,
            "eval_accuracy": result["exact_match"],
            "eval_token_accuracy": None,
            "eval_exact_match": result["exact_match"],
            "eval_exact_match_count": result.get("exact_match_count", result.get("num_correct")),
            "eval_exact_match_type": "generation_final_answer",
            "eval_final_answer_exact_match": result["exact_match"],
            "eval_sequence_exact_match": None,
            "eval_pass_at_1": None,
            "eval_metric": result["exact_match"],
            "eval_tokens": None,
            "eval_num_samples": result["num_samples"],
            "num_samples": result["num_samples"],
            "generation_checkpoint_step": checkpoint_step,
            "generation_checkpoint_path": str(checkpoint_path),
            "generation_max_new_tokens": int(max_new_tokens),
            "generation_max_prompt_length": int(max_prompt_length),
            "generation_batch_size": int(batch_size),
            "timestamp": timestamp_now(),
        },
    )
    update_final_summary_from_generation(run_dir, result)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "run_dir": str(run_dir),
        "checkpoint_step": checkpoint_step,
        "exact_match": result["exact_match"],
        "exact_match_count": result.get("exact_match_count", result.get("num_correct")),
        "num_samples": result["num_samples"],
        "predictions_path": str(predictions_path),
    }


def main() -> None:
    args = parse_args()
    run_dirs = [Path(path) for path in args.run_dir]
    summaries = []
    for run_dir in run_dirs:
        summaries.append(
            evaluate_run(
                run_dir,
                gpu=args.gpu,
                checkpoint_mode=args.checkpoint_mode,
                eval_file_override=args.eval_file,
                max_examples=args.max_examples,
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=args.max_new_tokens,
                batch_size=int(args.batch_size),
                show_progress=not bool(args.no_progress),
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
            )
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
