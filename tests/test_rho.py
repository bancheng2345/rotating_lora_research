"""rho_k behavior tests."""

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
from src.utils.svd import build_projection_basis


class RhoMetricTest(unittest.TestCase):
    def test_oplora_has_lower_rho_than_lora(self) -> None:
        base = nn.Linear(16, 12, bias=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            basis = build_projection_basis(
                base.weight,
                rank_k=4,
                cache_dir=tmpdir,
                cache_key="rho_test",
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

        torch.manual_seed(7)
        a_matrix = torch.randn_like(lora.lora_A)
        b_matrix = torch.randn_like(lora.lora_B)
        lora.lora_A.data.copy_(a_matrix)
        lora.lora_B.data.copy_(b_matrix)
        oplora.lora_A.data.copy_(a_matrix)
        oplora.lora_B.data.copy_(b_matrix)

        rho_lora = float(lora.rho_k().item())
        rho_oplora = float(oplora.rho_k().item())
        self.assertLess(rho_oplora, rho_lora)
        self.assertLess(rho_oplora, 1e-5)


if __name__ == "__main__":
    unittest.main()

