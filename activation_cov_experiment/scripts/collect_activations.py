#!/usr/bin/env python
"""Collects layer/module input activations for activation covariance analysis."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from activation_cov_experiment.src.activation_hooks import ActivationCollector, available_module_names, resolve_targets
from activation_cov_experiment.src.gpu_select import apply_gpu_override, apply_selected_gpu, selection_to_json
from activation_cov_experiment.src.io_utils import (
    activation_run_dir,
    build_model_and_tokenizer,
    ensure_dir,
    load_config_stack,
    load_text_samples,
    save_json,
    save_yaml,
    tokenize_text_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Overlay config YAML path. Repeat to stack multiple overlays.",
    )
    parser.add_argument(
        "--base-config",
        default="activation_cov_experiment/configs/activation_cov_base.yaml",
        help="Base config YAML path.",
    )
    parser.add_argument("--override", action="append", default=[], help="Dotlist override key=value.")
    parser.add_argument("--gpu", default=None, help="Explicit CUDA_VISIBLE_DEVICES override.")
    return parser.parse_args()


def batched(items: List[str], batch_size: int) -> List[List[str]]:
    """Splits a list into fixed-size batches."""

    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def main() -> None:
    args = parse_args()
    config = load_config_stack(args.config, base_config_path=args.base_config, overrides=args.override)
    apply_gpu_override(config, args.gpu)
    selection = apply_selected_gpu(config)
    print(selection_to_json(selection), flush=True)

    run_dir = ensure_dir(activation_run_dir(config))
    save_yaml(run_dir / "config_resolved.yaml", config)

    device_name = str(config.get("model", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    model, tokenizer, model_name = build_model_and_tokenizer(config, device=device)
    model.eval()

    texts = load_text_samples(config)
    if not texts:
        raise ValueError("No text samples available for activation collection.")

    try:
        targets = resolve_targets(model, config["targets"])
    except ValueError as exc:
        available = available_module_names(model)[:200]
        raise ValueError(f"{exc}\nAvailable module names (first 200): {available}") from exc

    collector = ActivationCollector(
        model,
        targets,
        output_dir=run_dir,
        chunk_tokens=int(config.get("collector", {}).get("chunk_tokens", 8192)),
    )

    max_seq_len = int(config["data"].get("max_seq_len", 128))
    batch_size = int(config["data"].get("batch_size", 4))
    model_vocab_size = int(getattr(model.config, "vocab_size", config["model"].get("vocab_size", 259)))
    total_batches = int(math.ceil(len(texts) / batch_size))

    with torch.no_grad():
        for batch_texts in batched(texts, batch_size):
            batch = tokenize_text_batch(
                batch_texts,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                model_vocab_size=model_vocab_size,
            )
            attention_mask = batch.get("attention_mask")
            collector.set_attention_mask(attention_mask)
            batch = {
                key: value.to(device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            _ = model(**batch)

    target_records = collector.finalize()
    metadata: Dict[str, Any] = {
        "run_name": str(config["output"]["run_name"]),
        "model_name_or_path": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "gpu_selection": json.loads(selection_to_json(selection)),
        "num_text_samples": len(texts),
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "total_batches": total_batches,
        "target_modules": [record["module_name"] for record in target_records],
        "targets": target_records,
    }
    save_json(run_dir / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
