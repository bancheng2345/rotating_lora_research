#!/usr/bin/env python
"""Training entrypoint for LoRA / OPLoRA / PC-LoRA experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config_stack
from src.utils.gpu_select import apply_gpu_override, apply_selected_gpu, selection_to_json


def apply_cache_env_from_config(config: dict) -> None:
    cache_cfg = config.get("runtime", {}).get("cache", {})
    mapping = {
        "HF_HOME": cache_cfg.get("hf_home"),
        "HF_DATASETS_CACHE": cache_cfg.get("hf_datasets_cache"),
    }
    for env_name, path in mapping.items():
        if not path:
            continue
        resolved = Path(str(path))
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        resolved.mkdir(parents=True, exist_ok=True)
        os.environ[env_name] = str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Overlay config YAML path. Repeat this flag to stack multiple presets.",
    )
    parser.add_argument("--base-config", default="configs/base.yaml", help="Base config YAML path.")
    parser.add_argument(
        "--gpu",
        default=None,
        help="Explicit CUDA_VISIBLE_DEVICES value, e.g. 0 or 1. Overrides config auto-selection.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotlist override, e.g. training.max_steps=5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_stack(
        config_paths=args.config,
        base_config_path=args.base_config,
        overrides=args.override,
    )
    apply_gpu_override(config, args.gpu)
    apply_cache_env_from_config(config)
    selection = apply_selected_gpu(config)
    print(selection_to_json(selection), flush=True)

    from src.training.trainer import ExperimentTrainer

    trainer = ExperimentTrainer(config)
    summary = trainer.train()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
