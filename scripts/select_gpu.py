#!/usr/bin/env python
"""CLI wrapper for GPU auto-selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config_stack
from src.utils.gpu_select import apply_gpu_override, select_gpu, selection_to_json


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
        "--format",
        default="shell",
        choices=["shell", "index", "json"],
        help="Output format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_stack(args.config, base_config_path=args.base_config)
    apply_gpu_override(config, args.gpu)
    selection = select_gpu(config)
    if args.format == "index":
        print(selection.visible_devices)
        return
    if args.format == "json":
        print(selection_to_json(selection))
        return
    env_name = config.get("runtime", {}).get("gpu", {}).get("override_env", "CUDA_VISIBLE_DEVICES")
    if selection.visible_devices == "":
        print(f"unset {env_name}")
    else:
        print(f"export {env_name}={selection.visible_devices}")


if __name__ == "__main__":
    main()
