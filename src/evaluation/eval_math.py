"""Generation-based math evaluation helpers."""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from src.evaluation.eval_lm import move_batch_to_device

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None


_BOXED_PATTERN = re.compile(r"\\boxed\s*\{")
_FINAL_ANSWER_PATTERNS = [
    re.compile(r"(?is)(?:final answer|answer)\s*[:：]\s*(.+)$"),
    re.compile(r"(?is)(?:the\s+)?final answer is\s+(.+)$"),
    re.compile(r"(?is)(?:the answer is|thus,? the answer is|therefore,? the answer is)\s+(.+)$"),
]
_SIMPLE_FRACTION_PATTERN = re.compile(r"^(-?\d+)\s*/\s*(-?\d+)$")
_LATEX_FRACTION_PATTERN = re.compile(r"^\\frac\{([^{}]+)\}\{([^{}]+)\}$")


def load_jsonl_records(path: str | Path, *, max_examples: int | None = None) -> List[Dict[str, Any]]:
    """Loads JSONL rows from disk."""

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


def _extract_last_boxed(text: str) -> str | None:
    """Extracts the content of the last \\boxed{...} span when present."""

    matches = list(_BOXED_PATTERN.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    chars: List[str] = []
    for char in text[start:]:
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars)
            chars.append(char)
        else:
            chars.append(char)
    return None


def extract_final_answer(text: str) -> str:
    """Heuristically extracts the final answer span from a generation."""

    value = str(text or "").strip()
    if not value:
        return ""

    boxed = _extract_last_boxed(value)
    if boxed is not None:
        return boxed.strip()

    if "####" in value:
        return value.rsplit("####", maxsplit=1)[-1].strip()

    for pattern in _FINAL_ANSWER_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(1).strip()

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return lines[-1] if lines else value


def _strip_outer_wrappers(text: str) -> str:
    """Repeatedly strips trivial outer wrappers like $, (), and braces."""

    value = text.strip()
    changed = True
    while changed and value:
        changed = False
        for left, right in [("$", "$"), ("\\(", "\\)"), ("(", ")"), ("[", "]"), ("{", "}")]:
            if value.startswith(left) and value.endswith(right) and len(value) > len(left) + len(right):
                value = value[len(left) : len(value) - len(right)].strip()
                changed = True
    return value


def _normalize_decimal(text: str) -> str:
    """Normalizes plain integer/decimal strings."""

    candidate = text.strip()
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", candidate):
        return candidate
    try:
        normalized = format(Decimal(candidate).normalize(), "f")
    except InvalidOperation:
        return candidate
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _normalize_simple_fraction(text: str) -> str:
    """Normalizes simple a/b fractions by sign and gcd."""

    candidate = text.strip()
    match = _LATEX_FRACTION_PATTERN.fullmatch(candidate)
    if match:
        candidate = f"{match.group(1)}/{match.group(2)}"
    match = _SIMPLE_FRACTION_PATTERN.fullmatch(candidate)
    if not match:
        return candidate
    numerator = int(match.group(1))
    denominator = int(match.group(2))
    if denominator == 0:
        return candidate
    sign = -1 if numerator * denominator < 0 else 1
    numerator = abs(numerator)
    denominator = abs(denominator)
    gcd = math.gcd(numerator, denominator)
    numerator //= gcd
    denominator //= gcd
    normalized = f"{numerator}/{denominator}"
    return f"-{normalized}" if sign < 0 else normalized


def normalize_math_answer(text: str) -> str:
    """Normalizes a math answer string for exact-match comparison."""

    value = extract_final_answer(text)
    value = value.replace("\u2212", "-")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("$", " ")
    value = re.sub(r"\s+", " ", value).strip()
    value = value.rstrip(".。;；,，:")
    value = _strip_outer_wrappers(value)
    value = re.sub(r"\s*,\s*", ",", value)
    value = re.sub(r"\s*/\s*", "/", value)
    value = re.sub(r"\s+", "", value)
    value = _normalize_simple_fraction(value)
    value = _normalize_decimal(value)
    return value


def exact_match(prediction: str, target: str) -> bool:
    """Returns normalized exact-match status."""

    normalized_target = normalize_math_answer(target)
    normalized_prediction = normalize_math_answer(prediction)
    return bool(normalized_target) and normalized_prediction == normalized_target


def _batched(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    """Yields fixed-size batches from a list of examples."""

    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@torch.no_grad()
def evaluate_math_generation(
    model: torch.nn.Module,
    tokenizer,
    examples: Iterable[Dict[str, Any]],
    *,
    device: torch.device,
    prompt_template: str,
    prompt_column: str = "question",
    response_column: str = "answer",
    max_prompt_length: int,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    batch_size: int = 1,
    show_progress: bool = True,
    progress_desc: str = "Generating math answers",
) -> Dict[str, Any]:
    """Runs greedy/sampled generation and computes normalized exact match."""

    model.eval()
    examples_list = list(examples)
    predictions: List[Dict[str, Any]] = []
    correct = 0
    total = 0
    subject_hits: Dict[str, int] = {}
    subject_counts: Dict[str, int] = {}

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(temperature)

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    batches = list(_batched(examples_list, batch_size))
    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(total=len(examples_list), desc=progress_desc, unit="sample", dynamic_ncols=True)
    try:
        for batch_index, batch in enumerate(batches, start=1):
            prompts: List[str] = []
            questions: List[str] = []
            gold_answers: List[str] = []
            subjects: List[str] = []
            for example in batch:
                # Math datasets in this repo use either question/answer or query/response.
                question = str(
                    example.get(prompt_column)
                    or example.get("question")
                    or example.get("query")
                    or example.get("original_question")
                    or ""
                )
                gold_answer = str(
                    example.get(response_column)
                    or example.get("answer")
                    or example.get("response")
                    or ""
                )
                prompts.append(
                    prompt_template.format(
                        prompt=question,
                        question=question,
                        answer=gold_answer,
                        response=gold_answer,
                    )
                )
                questions.append(question)
                gold_answers.append(gold_answer)
                subjects.append(str(example.get("metadata", {}).get("subject") or example.get("type") or "unknown"))

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_prompt_length),
            )
            inputs = move_batch_to_device(inputs, device)
            input_length = int(inputs["input_ids"].shape[1])
            generated = model.generate(**inputs, **generation_kwargs)
            continuations = generated[:, input_length:]
            prediction_texts = tokenizer.batch_decode(continuations, skip_special_tokens=True)

            for example, question, gold_answer, subject, prediction_text in zip(
                batch, questions, gold_answers, subjects, prediction_texts
            ):
                prediction_norm = normalize_math_answer(prediction_text)
                gold_norm = normalize_math_answer(gold_answer)
                is_correct = bool(gold_norm) and prediction_norm == gold_norm
                correct += int(is_correct)
                total += 1

                subject_counts[subject] = subject_counts.get(subject, 0) + 1
                subject_hits[subject] = subject_hits.get(subject, 0) + int(is_correct)

                predictions.append(
                    {
                        "id": example.get("id"),
                        "dataset": example.get("dataset"),
                        "subject": subject,
                        "level": example.get("metadata", {}).get("level"),
                        "question": question,
                        "gold_answer": gold_answer,
                        "gold_answer_normalized": gold_norm,
                        "prediction_text": prediction_text,
                        "prediction_normalized": prediction_norm,
                        "exact_match": is_correct,
                    }
                )
            if progress is not None:
                progress.update(len(batch))
            elif show_progress and (batch_index == 1 or batch_index == len(batches) or batch_index % 10 == 0):
                processed = min(batch_index * max(1, int(batch_size)), len(examples_list))
                print(
                    f"{progress_desc}: {processed}/{len(examples_list)} samples",
                    flush=True,
                )
    finally:
        if progress is not None:
            progress.close()
        tokenizer.padding_side = original_padding_side

    exact_match_score = (correct / total) if total > 0 else 0.0
    subject_breakdown = {
        subject: {
            "num_samples": count,
            "exact_match": (subject_hits.get(subject, 0) / count) if count > 0 else 0.0,
        }
        for subject, count in sorted(subject_counts.items())
    }
    return {
        "exact_match": exact_match_score,
        "exact_match_count": int(correct),
        "num_samples": total,
        "num_correct": correct,
        "subject_breakdown": subject_breakdown,
        "predictions": predictions,
    }
