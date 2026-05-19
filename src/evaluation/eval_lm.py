"""Causal language-model evaluation helpers."""

from __future__ import annotations

import math
from typing import Dict

import torch


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Moves a tensor batch dictionary to a target device."""

    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_causal_lm(
    model: torch.nn.Module,
    dataloader,
    *,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates a causal LM and returns loss/perplexity metrics."""

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(**batch)
        labels = batch["labels"]
        valid_tokens = int(labels.ne(-100).sum().item())
        if valid_tokens == 0:
            continue
        total_loss += float(outputs.loss.detach().cpu().item()) * valid_tokens
        total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")
    perplexity = float(math.exp(min(avg_loss, 20.0)))
    return {
        "eval_loss": avg_loss,
        "perplexity": perplexity,
        "eval_metric": perplexity,
        "eval_tokens": total_tokens,
    }
