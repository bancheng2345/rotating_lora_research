#!/usr/bin/env python
"""Download and normalize the 8 commonsense reasoning evaluation datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.commonsense_8tasks import TASKS_8, write_jsonl


DATASET_SPECS = {
    "boolq": ("google/boolq", None),
    "piqa": ("piqa", None),
    "siqa": ("social_i_qa", None),
    "hellaswag": ("hellaswag", None),
    "winogrande": ("winogrande", "winogrande_xl"),
    "arc_e": ("ai2_arc", "ARC-Easy"),
    "arc_c": ("ai2_arc", "ARC-Challenge"),
    "obqa": ("openbookqa", "main"),
}


def _choice(label: str, text: Any) -> Dict[str, str]:
    return {"label": str(label), "text": str(text).strip()}


def _letter_index(answer: Any, labels: List[str]) -> int:
    answer_str = str(answer).strip()
    if answer_str in labels:
        return labels.index(answer_str)
    upper_labels = [label.upper() for label in labels]
    if answer_str.upper() in upper_labels:
        return upper_labels.index(answer_str.upper())
    raise ValueError(f"Could not map answer {answer!r} into labels {labels!r}")


def _normalize_hf_choices(choices_raw: Any) -> tuple[List[str], List[str]]:
    """Normalizes HF choices from either list-of-dicts or dict-of-lists."""

    if isinstance(choices_raw, dict):
        labels = [str(label) for label in choices_raw.get("label", [])]
        texts = [str(text) for text in choices_raw.get("text", [])]
        return labels, texts
    labels: List[str] = []
    texts: List[str] = []
    for idx, choice in enumerate(choices_raw or []):
        if isinstance(choice, dict):
            labels.append(str(choice.get("label", idx)))
            texts.append(str(choice.get("text", "")))
        else:
            labels.append(str(idx))
            texts.append(str(choice))
    return labels, texts


def convert_example(task: str, row: Dict[str, Any], index: int, split: str) -> Dict[str, Any] | None:
    """Converts one HF dataset row to the unified multiple-choice schema."""

    if task == "boolq":
        answer = bool(row["answer"])
        prompt = (
            "Passage:\n"
            f"{row['passage']}\n\n"
            f"Question: {row['question']}\n"
            "Choose the correct answer: no or yes.\n"
            "Answer:"
        )
        return {
            "id": row.get("id", f"{task}-{split}-{index}"),
            "task": task,
            "split": split,
            "prompt": prompt,
            "choices": [_choice("no", "no"), _choice("yes", "yes")],
            "answer_idx": 1 if answer else 0,
        }

    if task == "piqa":
        label = row.get("label")
        if label is None or int(label) < 0:
            return None
        prompt = (
            f"Goal: {row['goal']}\n"
            "Which solution is more plausible?\n"
            "Answer:"
        )
        return {
            "id": row.get("id", f"{task}-{split}-{index}"),
            "task": task,
            "split": split,
            "prompt": prompt,
            "choices": [_choice("A", row["sol1"]), _choice("B", row["sol2"])],
            "answer_idx": int(label),
        }

    if task == "siqa":
        label = str(row.get("label", "")).strip()
        if not label:
            return None
        prompt = (
            f"Context: {row['context']}\n"
            f"Question: {row['question']}\n"
            "Choose the best answer.\n"
            "Answer:"
        )
        return {
            "id": row.get("id", f"{task}-{split}-{index}"),
            "task": task,
            "split": split,
            "prompt": prompt,
            "choices": [_choice("1", row["answerA"]), _choice("2", row["answerB"]), _choice("3", row["answerC"])],
            "answer_idx": int(label) - 1,
        }

    if task == "hellaswag":
        label = row.get("label")
        if label is None or str(label).strip() == "":
            return None
        context = str(row.get("ctx") or f"{row.get('ctx_a', '')} {row.get('ctx_b', '')}").strip()
        prompt = (
            f"Context: {context}\n"
            "Choose the most plausible ending.\n"
            "Answer:"
        )
        endings = row["endings"]
        return {
            "id": row.get("ind", f"{task}-{split}-{index}"),
            "task": task,
            "split": split,
            "prompt": prompt,
            "choices": [_choice(chr(ord("A") + idx), ending) for idx, ending in enumerate(endings)],
            "answer_idx": int(label),
        }

    if task == "winogrande":
        answer = str(row.get("answer", "")).strip()
        if answer not in {"1", "2"}:
            return None
        prompt = (
            f"Sentence: {row['sentence']}\n"
            "Which option best fills the blank?\n"
            "Answer:"
        )
        return {
            "id": row.get("qID", f"{task}-{split}-{index}"),
            "task": task,
            "split": split,
            "prompt": prompt,
            "choices": [_choice("1", row["option1"]), _choice("2", row["option2"])],
            "answer_idx": int(answer) - 1,
        }

    if task in {"arc_e", "arc_c", "obqa"}:
        question = row.get("question") or row.get("question_stem") or row.get("query")
        if isinstance(question, dict):
            question_text = str(question.get("stem", ""))
            choices_raw = question.get("choices", [])
        else:
            question_text = str(question or "")
            choices_raw = row.get("choices", [])
        labels, texts = _normalize_hf_choices(choices_raw)
        answer_key = row.get("answerKey") or row.get("answer_key") or row.get("answer")
        if not question_text or not labels or answer_key is None:
            return None
        prompt = (
            f"Question: {question_text}\n"
            "Choose the correct answer.\n"
            "Answer:"
        )
        return {
            "id": row.get("id", f"{task}-{split}-{index}"),
            "task": task,
            "split": split,
            "prompt": prompt,
            "choices": [_choice(label, text) for label, text in zip(labels, texts)],
            "answer_idx": _letter_index(answer_key, labels),
        }

    raise KeyError(f"Unsupported task: {task}")


def load_hf_dataset(task: str, split: str):
    """Loads one task from HuggingFace datasets."""

    from datasets import load_dataset

    name, subset = DATASET_SPECS[task]
    if subset:
        return load_dataset(name, subset, split=split, trust_remote_code=True)
    return load_dataset(name, split=split, trust_remote_code=True)


def prepare_task(task: str, split: str, output_root: Path, max_examples: int | None) -> Dict[str, Any]:
    """Downloads and writes one normalized task split."""

    dataset = load_hf_dataset(task, split)
    rows: List[Dict[str, Any]] = []
    skipped = 0
    for index, row in enumerate(dataset):
        converted = convert_example(task, dict(row), index, split)
        if converted is None:
            skipped += 1
            continue
        rows.append(converted)
        if max_examples is not None and len(rows) >= int(max_examples):
            break

    output_path = output_root / task / f"{split}.jsonl"
    write_jsonl(output_path, rows)
    metadata = {
        "task": task,
        "split": split,
        "dataset_name": DATASET_SPECS[task][0],
        "dataset_config": DATASET_SPECS[task][1],
        "num_examples": len(rows),
        "num_skipped": skipped,
        "output_path": str(output_path),
    }
    (output_root / task / f"{split}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="data/commonsense_eval", help="Directory for normalized JSONL files.")
    parser.add_argument("--split", default="validation", help="HF split to download, usually validation.")
    parser.add_argument("--task", action="append", choices=TASKS_8, help="Task to prepare. Repeatable.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap per task for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = args.task or TASKS_8
    output_root = Path(args.output_root)
    summaries = [prepare_task(task, args.split, output_root, args.max_examples) for task in tasks]
    (output_root / f"{args.split}_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
