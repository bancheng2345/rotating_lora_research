#!/usr/bin/env python
"""Computes weight-SVD directions for the same modules used in activation collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from activation_cov_experiment.src.io_utils import activation_run_dir, build_model_and_tokenizer, direction_run_dir, ensure_dir, load_activation_metadata, load_config_stack, save_json, save_yaml
from activation_cov_experiment.src.svd_utils import compute_weight_svd
from activation_cov_experiment.src.subspace_metrics import orthogonality_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument(
        "--base-config",
        default="activation_cov_experiment/configs/activation_cov_base.yaml",
    )
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_stack(args.config, base_config_path=args.base_config, overrides=args.override)
    activations_dir = activation_run_dir(config)
    metadata = load_activation_metadata(activations_dir / "metadata.json")
    output_dir = ensure_dir(direction_run_dir(config) / "weight_svd")
    save_yaml(output_dir / "config_resolved.yaml", config)

    device = torch.device("cpu")
    model, _, model_name = build_model_and_tokenizer(config, device=device)
    model.eval()
    modules = dict(model.named_modules())
    top_k_values = sorted({int(value) for value in config["svd"].get("top_k_values", [4, 8, 16, 32])})
    max_k = max(top_k_values)
    use_lowrank = bool(config["svd"].get("use_lowrank", False))

    summary_rows: List[Dict[str, Any]] = []
    for target in metadata["targets"]:
        module = modules[target["module_name"]]
        svd_payload = compute_weight_svd(module, max_k=max_k, use_lowrank=use_lowrank)
        input_vectors = svd_payload["input_vectors"]
        output_vectors = svd_payload["output_vectors"]
        singular_values = svd_payload["singular_values"]
        payload = {
            "module_name": target["module_name"],
            "module_key": target["module_key"],
            "layer_index": target["layer_index"],
            "model_name_or_path": model_name,
            "singular_values": singular_values,
            "input_vectors": input_vectors,
            "output_vectors": output_vectors,
            "top_k_values": top_k_values,
            "use_lowrank": use_lowrank,
            "orthogonality_error_input": orthogonality_error(input_vectors),
            "orthogonality_error_output": orthogonality_error(output_vectors),
            "matrix_shape": list(svd_payload["matrix"].shape),
        }
        torch.save(payload, output_dir / f"{target['module_key']}.pt")
        summary_rows.append(
            {
                "module_name": target["module_name"],
                "module_key": target["module_key"],
                "layer_index": target["layer_index"],
                "matrix_shape": list(svd_payload["matrix"].shape),
                "top_singular_values": [float(value) for value in singular_values[: min(8, len(singular_values))].tolist()],
                "orthogonality_error_input": orthogonality_error(input_vectors),
                "orthogonality_error_output": orthogonality_error(output_vectors),
            }
        )

    save_json(output_dir / "summary.json", {"targets": summary_rows, "top_k_values": top_k_values, "use_lowrank": use_lowrank})
    print(json.dumps({"output_dir": str(output_dir), "targets": summary_rows}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
