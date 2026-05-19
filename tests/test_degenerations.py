"""Degeneration tests for LoRA-family layers."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods.lora_base import LoRALinear
from src.methods.oplora import OPLoRALinear
from src.methods.pc_lora import PCLoRALinear
from src.utils.svd import build_projection_basis


class DegenerationTest(unittest.TestCase):
    def _sync_parameters(self, left, right) -> None:
        right.lora_A.data.copy_(left.lora_A.data)
        right.lora_B.data.copy_(left.lora_B.data)

    def test_oplora_reduces_to_lora_when_k_zero(self) -> None:
        base = nn.Linear(10, 8, bias=False)
        inputs = torch.randn(4, 10)
        with tempfile.TemporaryDirectory() as tmpdir:
            basis = build_projection_basis(
                base.weight,
                rank_k=0,
                cache_dir=tmpdir,
                cache_key="degeneration_k_zero",
            ).basis
            lora = LoRALinear(
                copy.deepcopy(base),
                layer_name="layer",
                rank=4,
                alpha=8,
                dropout=0.0,
                projection_basis=basis,
            )
            oplora = OPLoRALinear(
                copy.deepcopy(base),
                layer_name="layer",
                rank=4,
                alpha=8,
                dropout=0.0,
                projection_basis=basis,
                use_left_projection=True,
                use_right_projection=True,
            )
        self._sync_parameters(lora, oplora)
        self.assertTrue(torch.allclose(lora(inputs), oplora(inputs), atol=1e-6, rtol=1e-6))

    def test_oplora_reduces_to_lora_when_projections_disabled(self) -> None:
        base = nn.Linear(10, 8, bias=False)
        inputs = torch.randn(4, 10)
        with tempfile.TemporaryDirectory() as tmpdir:
            basis = build_projection_basis(
                base.weight,
                rank_k=3,
                cache_dir=tmpdir,
                cache_key="degeneration_disabled",
            ).basis
            lora = LoRALinear(
                copy.deepcopy(base),
                layer_name="layer",
                rank=4,
                alpha=8,
                dropout=0.0,
                projection_basis=basis,
            )
            oplora = OPLoRALinear(
                copy.deepcopy(base),
                layer_name="layer",
                rank=4,
                alpha=8,
                dropout=0.0,
                projection_basis=basis,
                use_left_projection=False,
                use_right_projection=False,
            )
        self._sync_parameters(lora, oplora)
        self.assertTrue(torch.allclose(lora(inputs), oplora(inputs), atol=1e-6, rtol=1e-6))

    def test_pc_lora_forward_matches_oplora_when_lambda_cap_zero(self) -> None:
        base = nn.Linear(10, 8, bias=False)
        inputs = torch.randn(4, 10)
        with tempfile.TemporaryDirectory() as tmpdir:
            basis = build_projection_basis(
                base.weight,
                rank_k=3,
                cache_dir=tmpdir,
                cache_key="degeneration_pc",
            ).basis
            oplora = OPLoRALinear(
                copy.deepcopy(base),
                layer_name="layer",
                rank=4,
                alpha=8,
                dropout=0.0,
                projection_basis=basis,
                use_left_projection=True,
                use_right_projection=True,
            )
            pc_lora = PCLoRALinear(
                copy.deepcopy(base),
                layer_name="layer",
                rank=4,
                alpha=8,
                dropout=0.0,
                projection_basis=basis,
                use_left_projection=True,
                use_right_projection=True,
                use_capture_loss=True,
                capture_space="residual",
            )
        self._sync_parameters(oplora, pc_lora)
        self.assertTrue(torch.allclose(oplora(inputs), pc_lora(inputs), atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()

