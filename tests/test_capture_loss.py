"""Tests for omega and orthogonality regularizers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods.capture_losses import compute_omega, orthogonality_loss


class CaptureLossTest(unittest.TestCase):
    def test_omega_is_bounded(self) -> None:
        q, _ = torch.linalg.qr(torch.randn(8, 8))
        a_matrix = q[:4, :]
        activations = torch.randn(6, 8)
        omega = compute_omega(a_matrix, activations)
        self.assertTrue(bool(torch.all(omega >= 0.0)))
        self.assertTrue(bool(torch.all(omega <= 1.0)))

    def test_orthogonality_loss_is_finite(self) -> None:
        q, _ = torch.linalg.qr(torch.randn(8, 8))
        a_matrix = q[:4, :]
        loss = orthogonality_loss(a_matrix)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertLessEqual(float(loss.item()), 1e-4)


if __name__ == "__main__":
    unittest.main()

