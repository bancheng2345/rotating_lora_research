"""Utilities for Commonsense170k-style 8-task evaluation.

The evaluator uses multiple-choice conditional log-likelihood.  For each
candidate answer, it scores only the answer continuation tokens conditioned on
the prompt, and predicts the candidate with the highest normalized log-prob.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F

from src.evaluation.eval_lm import move_batch_to_device

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None


TASKS_8 = ["boolq", "piqa", "siqa", "hellaswag", "winogrande", "arc_e", "arc_c", "obqa"]


@dataclass(frozen=True)
class Candidate:
    """One answer candidate for a multiple-choice example."""

    example_index: int
    choice_index: int
    prompt: str
    continuation: str


def read_jsonl(path: str | Path, *, max_examples: int | None = None) -> List[Dict[str, Any]]:
    """Reads JSONL examples from disk."""

    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_examples is not None and len(rows) >= int(max_examples):
                break
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Writes JSONL rows to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def task_file(root: str | Path, task: str, split: str = "validation") -> Path:
    """Returns the unified JSONL path for a task split."""

    return Path(root) / task / f"{split}.jsonl"


def _format_answer_continuation(text: str) -> str:
    """Formats candidate text as a continuation after an answer prompt."""

    text = str(text).strip()
    return f" {text}" if text else " "


def _tokenize_candidate(tokenizer, prompt: str, continuation: str) -> Dict[str, List[int]]:
    """Tokenizes one prompt+continuation pair and masks prompt tokens."""

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt + continuation, add_special_tokens=True)["input_ids"]
    prefix_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prefix_len + full_ids[prefix_len:]
    if all(label == -100 for label in labels) and len(full_ids) > 0:
        labels[-1] = full_ids[-1]
    return {"input_ids": full_ids, "labels": labels}


def _collate_candidate_batch(tokenizer, encoded: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    """Pads encoded candidate sequences into tensors."""

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    max_len = max(len(row["input_ids"]) for row in encoded)
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    labels: List[List[int]] = []
    for row in encoded:
        pad_len = max_len - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad_len)
        attention_mask.append([1] * len(row["input_ids"]) + [0] * pad_len)
        labels.append(row["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


@torch.no_grad()
def score_candidates(
    model: torch.nn.Module,
    tokenizer,
    candidates: Sequence[Candidate],
    *,
    device: torch.device,
    batch_size: int,
    length_normalize: bool = True,
    show_progress: bool = True,
    progress_desc: str = "Scoring candidates",
) -> List[float]:
    """Scores answer candidates by conditional log-likelihood."""

    model.eval()
    scores: List[float] = []
    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(total=len(candidates), desc=progress_desc, unit="choice", dynamic_ncols=True)
    try:
        for start in range(0, len(candidates), max(1, int(batch_size))):
            batch = candidates[start : start + max(1, int(batch_size))]
            encoded = [
                _tokenize_candidate(tokenizer, candidate.prompt, candidate.continuation)
                for candidate in batch
            ]
            tensors = move_batch_to_device(_collate_candidate_batch(tokenizer, encoded), device)
            labels = tensors.pop("labels")
            outputs = model(**tensors)
            logits = outputs.logits.float()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(shift_labels.shape)
            valid = shift_labels.ne(-100)
            token_counts = valid.sum(dim=-1).clamp_min(1)
            log_probs = -(losses * valid).sum(dim=-1)
            if length_normalize:
                log_probs = log_probs / token_counts
            scores.extend(float(value.detach().cpu().item()) for value in log_probs)
            if progress is not None:
                progress.update(len(batch))
            elif show_progress:
                done = min(start + len(batch), len(candidates))
                if done == len(candidates) or done % max(1, int(batch_size) * 20) == 0:
                    print(f"{progress_desc}: {done}/{len(candidates)} choices", flush=True)
    finally:
        if progress is not None:
            progress.close()
    return scores


@torch.no_grad()
def evaluate_commonsense_tasks(
    model: torch.nn.Module,
    tokenizer,
    *,
    data_root: str | Path,
    tasks: Sequence[str] = TASKS_8,
    split: str = "validation",
    device: torch.device,
    batch_size: int = 8,
    max_examples_per_task: int | None = None,
    length_normalize: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Evaluates all requested commonsense tasks and returns accuracy metrics."""

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    per_task: Dict[str, Dict[str, Any]] = {}
    predictions: Dict[str, List[Dict[str, Any]]] = {}
    total_correct = 0
    total_examples = 0

    for task in tasks:
        path = task_file(data_root, task, split)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing prepared task file: {path}. Run scripts/prepare_commonsense_eval.py first."
            )
        examples = read_jsonl(path, max_examples=max_examples_per_task)
        candidates: List[Candidate] = []
        for example_index, example in enumerate(examples):
            choices = example.get("choices") or []
            for choice_index, choice in enumerate(choices):
                text = choice.get("text") if isinstance(choice, dict) else str(choice)
                candidates.append(
                    Candidate(
                        example_index=example_index,
                        choice_index=choice_index,
                        prompt=str(example["prompt"]),
                        continuation=_format_answer_continuation(str(text)),
                    )
                )

        scores = score_candidates(
            model,
            tokenizer,
            candidates,
            device=device,
            batch_size=batch_size,
            length_normalize=length_normalize,
            show_progress=show_progress,
            progress_desc=f"{task} {split}",
        )

        grouped: List[List[float]] = [[] for _ in examples]
        for candidate, score in zip(candidates, scores):
            grouped[candidate.example_index].append(score)

        task_predictions: List[Dict[str, Any]] = []
        correct = 0
        for example, choice_scores in zip(examples, grouped):
            if not choice_scores:
                continue
            predicted_idx = max(range(len(choice_scores)), key=lambda idx: choice_scores[idx])
            answer_idx = int(example["answer_idx"])
            is_correct = predicted_idx == answer_idx
            correct += int(is_correct)
            choices = example.get("choices") or []
            task_predictions.append(
                {
                    "id": example.get("id"),
                    "task": task,
                    "answer_idx": answer_idx,
                    "predicted_idx": predicted_idx,
                    "answer_text": choices[answer_idx]["text"] if answer_idx < len(choices) else None,
                    "predicted_text": choices[predicted_idx]["text"] if predicted_idx < len(choices) else None,
                    "scores": choice_scores,
                    "correct": is_correct,
                }
            )

        num_examples = len(task_predictions)
        accuracy = correct / num_examples if num_examples else float("nan")
        per_task[task] = {
            "accuracy": accuracy,
            "num_correct": int(correct),
            "num_samples": int(num_examples),
        }
        predictions[task] = task_predictions
        total_correct += int(correct)
        total_examples += int(num_examples)

    macro_accuracy = sum(row["accuracy"] for row in per_task.values()) / len(per_task) if per_task else float("nan")
    micro_accuracy = total_correct / total_examples if total_examples else float("nan")
    return {
        "split": split,
        "tasks": list(tasks),
        "per_task": per_task,
        "macro_accuracy": macro_accuracy,
        "micro_accuracy": micro_accuracy,
        "num_correct": int(total_correct),
        "num_samples": int(total_examples),
        "length_normalize": bool(length_normalize),
        "predictions": predictions,
    }

