"""Generic classification evaluation placeholder."""

from __future__ import annotations

from typing import Dict

import torch

from src.evaluation.eval_lm import move_batch_to_device


@torch.no_grad()
def evaluate_classification(
    model: torch.nn.Module,
    dataloader,
    *,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates accuracy for batches containing a `labels` field."""

    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(**batch)
        logits = outputs.logits
        labels = batch["labels"]
        predictions = logits.argmax(dim=-1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
        if getattr(outputs, "loss", None) is not None:
            total_loss += float(outputs.loss.detach().cpu().item()) * labels.numel()

    average_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return {
        "eval_loss": average_loss,
        "accuracy": accuracy,
        "eval_metric": accuracy,
    }

