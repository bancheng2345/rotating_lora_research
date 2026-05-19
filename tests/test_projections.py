"""Projection-orthogonality tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods.projections import apply_left_projection, apply_right_projection
from src.utils.svd import build_projection_basis


class ProjectionTest(unittest.TestCase):
    def test_right_projection_is_orthogonal_to_v(self) -> None:
        weight = nn.Linear(12, 10, bias=False).weight.detach().to(torch.float64)
        inputs = torch.randn(5, 12, dtype=torch.float64)
        with tempfile.TemporaryDirectory() as tmpdir:
            basis = build_projection_basis(
                weight,
                rank_k=4,
                cache_dir=tmpdir,
                cache_key="proj_right",
            ).basis
        residual = apply_right_projection(inputs, basis.right_basis.to(dtype=torch.float64))
        overlap = torch.matmul(residual, basis.right_basis.to(dtype=torch.float64))
        self.assertLess(float(overlap.abs().max().item()), 1e-8)

    def test_left_projection_is_orthogonal_to_u(self) -> None:
        weight = nn.Linear(12, 10, bias=False).weight.detach().to(torch.float64)
        outputs = torch.randn(5, 10, dtype=torch.float64)
        with tempfile.TemporaryDirectory() as tmpdir:
            basis = build_projection_basis(
                weight,
                rank_k=4,
                cache_dir=tmpdir,
                cache_key="proj_left",
            ).basis
        residual = apply_left_projection(outputs, basis.left_basis.to(dtype=torch.float64))
        overlap = torch.matmul(residual, basis.left_basis.to(dtype=torch.float64))
        self.assertLess(float(overlap.abs().max().item()), 1e-8)


if __name__ == "__main__":
    unittest.main()

