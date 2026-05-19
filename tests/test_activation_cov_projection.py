"""ActivationCov projection integration tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods.lora_base import (
    extract_transformer_layer_index,
    resolve_lora_alpha,
    resolve_lora_rank,
    resolve_target_layer_enabled,
)
from src.methods.projections import apply_right_projection
from src.utils.metrics import rowspace_alignment_stats
from src.utils.svd import build_projection_basis


class ActivationCovProjectionTest(unittest.TestCase):
    def test_loads_activation_cov_right_basis(self) -> None:
        layer_name = "model.layers.8.self_attn.q_proj"
        weight = torch.randn(10, 12)
        q, _ = torch.linalg.qr(torch.randn(12, 6), mode="reduced")
        with tempfile.TemporaryDirectory() as tmpdir:
            direction_dir = Path(tmpdir)
            torch.save({"eigenvectors": q, "layer_index": 8}, direction_dir / "model_layers_8_self_attn_q_proj.pt")
            result = build_projection_basis(
                weight,
                rank_k=4,
                cache_dir=tmpdir,
                cache_key="actcov-right",
                layer_name=layer_name,
                left_source="none",
                right_source="activation_cov",
                activation_cov_dir=direction_dir,
            )

        right_basis = result.basis.right_basis
        self.assertIsNotNone(right_basis)
        self.assertEqual(tuple(right_basis.shape), (12, 4))
        gram = right_basis.transpose(0, 1) @ right_basis
        self.assertLess(float((gram - torch.eye(4)).abs().max().item()), 1e-5)
        self.assertEqual(result.metadata["right_activation_cov"]["activation_cov_rank_used"], 4)

    def test_activation_cov_projection_removes_basis_overlap(self) -> None:
        layer_name = "model.layers.0.self_attn.v_proj"
        weight = torch.randn(9, 7)
        q, _ = torch.linalg.qr(torch.randn(7, 4), mode="reduced")
        inputs = torch.randn(5, 7)
        with tempfile.TemporaryDirectory() as tmpdir:
            direction_dir = Path(tmpdir)
            torch.save({"eigenvectors": q}, direction_dir / "model_layers_0_self_attn_v_proj.pt")
            basis = build_projection_basis(
                weight,
                rank_k=3,
                cache_dir=tmpdir,
                cache_key="actcov-proj",
                layer_name=layer_name,
                left_source="none",
                right_source="activation_cov",
                activation_cov_dir=direction_dir,
            ).basis
        residual = apply_right_projection(inputs, basis.right_basis)
        overlap = residual @ basis.right_basis
        self.assertLess(float(overlap.abs().max().item()), 1e-5)

    def test_target_layer_filter(self) -> None:
        self.assertEqual(extract_transformer_layer_index("model.layers.24.self_attn.q_proj"), 24)
        self.assertTrue(resolve_target_layer_enabled("model.layers.24.self_attn.q_proj", [0, 24]))
        self.assertFalse(resolve_target_layer_enabled("model.layers.23.self_attn.q_proj", [0, 24]))

    def test_rank_pattern_and_rank_scaled_alpha(self) -> None:
        config = {
            "rank": 8,
            "alpha": 16,
            "alpha_strategy": "rank_scaled",
            "rank_pattern": {
                "model.layers.0.self_attn.q_proj": 4,
                "v_proj": 14,
            },
        }
        self.assertEqual(resolve_lora_rank("model.layers.0.self_attn.q_proj", config), 4)
        self.assertEqual(resolve_lora_rank("model.layers.8.self_attn.v_proj", config), 14)
        self.assertEqual(resolve_lora_rank("model.layers.8.self_attn.o_proj", config), 8)
        self.assertEqual(resolve_lora_alpha("model.layers.0.self_attn.q_proj", config, rank=4), 8.0)
        self.assertEqual(resolve_lora_alpha("model.layers.8.self_attn.v_proj", config, rank=14), 28.0)

    def test_rowspace_alignment_prefers_matching_basis(self) -> None:
        target = torch.eye(6, 3)
        aligned_a = target.transpose(0, 1).contiguous()
        random_a = torch.randn(3, 6)
        aligned = rowspace_alignment_stats(aligned_a, target)
        random = rowspace_alignment_stats(random_a, target)
        self.assertGreater(float(aligned["alignment_score"].item()), 0.999)
        self.assertLess(float(aligned["alignment_loss"].item()), 1e-5)
        self.assertLess(float(random["alignment_score"].item()), 1.0)


if __name__ == "__main__":
    unittest.main()
