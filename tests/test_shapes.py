"""Shape tests for LoRA-family linear layers."""

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


class LoRALayerShapeTest(unittest.TestCase):
    def test_lora_parameter_dtype_uses_fp32_for_low_precision_base_layer(self) -> None:
        base = nn.Linear(16, 12, bias=False, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as tmpdir:
            basis = build_projection_basis(
                base.weight,
                rank_k=4,
                cache_dir=tmpdir,
                cache_key="dtype_test",
            ).basis
            layer = LoRALinear(
                copy.deepcopy(base),
                layer_name="layer",
                rank=4,
                alpha=8,
                dropout=0.0,
                projection_basis=basis,
                use_left_projection=False,
                use_right_projection=False,
            )

        self.assertEqual(layer.lora_A.dtype, torch.float32)
        self.assertEqual(layer.lora_B.dtype, torch.float32)

    def test_forward_shapes_match(self) -> None:
        base = nn.Linear(16, 12, bias=False)
        inputs = torch.randn(2, 3, 16)
        with tempfile.TemporaryDirectory() as tmpdir:
            basis = build_projection_basis(
                base.weight,
                rank_k=4,
                cache_dir=tmpdir,
                cache_key="shape_test",
            ).basis

            lora = LoRALinear(
                copy.deepcopy(base),
                layer_name="layer",
                rank=4,
                alpha=8,
                dropout=0.0,
                projection_basis=basis,
                use_left_projection=False,
                use_right_projection=False,
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

        self.assertEqual(lora(inputs).shape, torch.Size([2, 3, 12]))
        self.assertEqual(oplora(inputs).shape, torch.Size([2, 3, 12]))
        self.assertEqual(pc_lora(inputs).shape, torch.Size([2, 3, 12]))


if __name__ == "__main__":
    unittest.main()
