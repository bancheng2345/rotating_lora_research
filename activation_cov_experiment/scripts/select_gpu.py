#!/usr/bin/env python
"""CLI wrapper for isolated activation-covariance GPU selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from activation_cov_experiment.src.gpu_select import apply_gpu_override, select_gpu, selection_to_json
from activation_cov_experiment.src.io_utils import load_config_stack


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
    parser.add_argument("--gpu", default=None, help="Explicit CUDA_VISIBLE_DEVICES override.")
    parser.add_argument("--format", default="shell", choices=["shell", "json", "index"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_stack(args.config, base_config_path=args.base_config)
    apply_gpu_override(config, args.gpu)
    selection = select_gpu(config)
    if args.format == "json":
        print(selection_to_json(selection))
        return
    if args.format == "index":
        print(selection.visible_devices)
        return
    env_name = config.get("runtime", {}).get("gpu", {}).get("override_env", "CUDA_VISIBLE_DEVICES")
    if selection.visible_devices == "":
        print(f"unset {env_name}")
    else:
        print(f"export {env_name}={selection.visible_devices}")


if __name__ == "__main__":
    main()
