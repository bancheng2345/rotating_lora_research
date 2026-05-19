#!/usr/bin/env python
"""Computes CUR-style projection directions from pretrained weights.

For a linear weight W in R[out, in], the input-side right basis is estimated
from the row space of important rows selected by row norms. The output-side
left basis is estimated from important columns selected by column norms.
This is a lightweight CUR surrogate for SVD directions.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods.lora_base import iter_target_linear_modules, resolve_target_layer_enabled
from src.training.trainer import build_model_and_tokenizer, resolve_model_slug, set_seed, slugify
from src.utils.config import load_config_stack, save_yaml
from src.utils.gpu_select import apply_gpu_override, apply_selected_gpu, selection_to_json
from src.utils.logging import save_json
from src.utils.projection_directions import orthonormalize_basis, sanitize_module_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--gpu", default=None)
    return parser.parse_args()


def direction_output_dir(config: Dict[str, Any], *, source: str) -> Path:
    """Resolves output directory for a direction source."""

    direction_cfg = config.get("projection_directions", {})
    configured = direction_cfg.get("output_dir") or direction_cfg.get(f"{source}_dir")
    if configured:
        return Path(str(configured))
    model_slug = resolve_model_slug(str(config["model"].get("name_or_path") or config["model"].get("synthetic_name", "model")))
    dataset_slug = slugify(Path(str(config["training"].get("train_file") or config["training"].get("dataset_name", "dataset"))).stem)
    return Path("outputs") / "directions" / source / f"{model_slug}_{dataset_slug}"


def top_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Returns indices of the top-k scores in descending order."""

    k = min(max(int(k), 0), int(scores.numel()))
    if k == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.argsort(scores.detach().to(dtype=torch.float32), descending=True)[:k].contiguous()


def iter_selected_modules(model: torch.nn.Module, config: Dict[str, Any]) -> Iterable[Tuple[str, torch.nn.Linear]]:
    """Yields target linear modules selected by suffix and optional layer index."""

    direction_cfg = config.get("projection_directions", {})
    target_modules = direction_cfg.get("target_modules") or config["model"].get("target_modules", [])
    target_layers = direction_cfg.get("target_layers", config.get("lora", {}).get("target_layers", "all"))
    for layer_name, module in iter_target_linear_modules(model, target_modules):
        if resolve_target_layer_enabled(layer_name, target_layers):
            yield layer_name, module


def compute_cur_basis(weight: torch.Tensor, *, rank: int) -> Dict[str, Any]:
    """Builds CUR-style left/right bases from row/column norm selections."""

    weight_f32 = weight.detach().to(dtype=torch.float32, device="cpu")
    out_features, in_features = int(weight_f32.shape[0]), int(weight_f32.shape[1])
    rank = min(max(int(rank), 0), out_features, in_features)
    if rank == 0:
        return {
            "left_basis": torch.empty(out_features, 0, dtype=torch.float32),
            "right_basis": torch.empty(in_features, 0, dtype=torch.float32),
            "selected_rows": [],
            "selected_columns": [],
            "row_scores_top": [],
            "column_scores_top": [],
        }

    row_scores = torch.sum(weight_f32 * weight_f32, dim=1)
    column_scores = torch.sum(weight_f32 * weight_f32, dim=0)
    selected_rows = top_indices(row_scores, rank)
    selected_columns = top_indices(column_scores, rank)

    # CUR: selected rows span the input-side row space; selected columns span
    # the output-side column space. No dense projector is materialized.
    right_basis = orthonormalize_basis(weight_f32[selected_rows, :].T, max_rank=rank)
    left_basis = orthonormalize_basis(weight_f32[:, selected_columns], max_rank=rank)
    return {
        "left_basis": left_basis,
        "right_basis": right_basis,
        "selected_rows": selected_rows.cpu().tolist(),
        "selected_columns": selected_columns.cpu().tolist(),
        "row_scores_top": row_scores[selected_rows].cpu().tolist(),
        "column_scores_top": column_scores[selected_columns].cpu().tolist(),
    }


def main() -> None:
    args = parse_args()
    config = load_config_stack(args.config, base_config_path=args.base_config, overrides=args.override)
    apply_gpu_override(config, args.gpu)
    selection = apply_selected_gpu(config)
    print(selection_to_json(selection), flush=True)

    seed = int(config["experiment"].get("seed", 0))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, model_name = build_model_and_tokenizer(config, device=device)
    model.to(device)
    model.eval()

    direction_cfg = config.get("projection_directions", {})
    top_k_values = sorted({int(k) for k in direction_cfg.get("top_k_values", [config["lora"].get("projection_rank_k", 16)])})
    max_rank = max(top_k_values)
    output_dir = direction_output_dir(config, source="cur")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(output_dir / "config_resolved.yaml", config)

    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for layer_name, module in iter_selected_modules(model, config):
            payload = compute_cur_basis(module.weight, rank=max_rank)
            module_key = sanitize_module_name(layer_name)
            save_payload = {
                "source": "cur",
                "module_name": layer_name,
                "module_key": module_key,
                "model_name_or_path": model_name,
                "shape": list(module.weight.shape),
                "rank": int(max_rank),
                "top_k_values": top_k_values,
                "left_basis": payload["left_basis"],
                "right_basis": payload["right_basis"],
                "selected_rows": payload["selected_rows"],
                "selected_columns": payload["selected_columns"],
                "row_scores_top": payload["row_scores_top"],
                "column_scores_top": payload["column_scores_top"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            torch.save(save_payload, output_dir / f"{module_key}.pt")
            rows.append(
                {
                    "module_name": layer_name,
                    "module_key": module_key,
                    "shape": list(module.weight.shape),
                    "rank": int(max_rank),
                    "selected_rows_head": payload["selected_rows"][:8],
                    "selected_columns_head": payload["selected_columns"][:8],
                }
            )

    summary = {
        "source": "cur",
        "output_dir": str(output_dir),
        "model_name_or_path": model_name,
        "num_modules": len(rows),
        "top_k_values": top_k_values,
        "targets": rows,
        "gpu_selection": json.loads(selection_to_json(selection)),
    }
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
