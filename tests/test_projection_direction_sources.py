from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.compute_cur_directions import compute_cur_basis
from src.utils.projection_directions import randomized_top_eigenvectors
from src.utils.svd import build_projection_basis


class ProjectionDirectionSourceTests(unittest.TestCase):
    def test_randomized_top_eigenvectors_returns_orthonormal_basis(self) -> None:
        matrix = torch.diag(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]))
        values, vectors = randomized_top_eigenvectors(matrix, rank=2, oversample=2, n_iter=2, seed=0)
        self.assertEqual(tuple(values.shape), (2,))
        self.assertEqual(tuple(vectors.shape), (5, 2))
        gram = vectors.T @ vectors
        self.assertTrue(torch.allclose(gram, torch.eye(2), atol=1e-5))

    def test_cur_basis_shapes_and_orthogonality(self) -> None:
        weight = torch.randn(6, 5)
        payload = compute_cur_basis(weight, rank=3)
        self.assertEqual(tuple(payload["left_basis"].shape), (6, 3))
        self.assertEqual(tuple(payload["right_basis"].shape), (5, 3))
        self.assertTrue(torch.allclose(payload["right_basis"].T @ payload["right_basis"], torch.eye(3), atol=1e-5))

    def test_build_projection_basis_loads_fisher_right_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            basis = torch.linalg.qr(torch.randn(4, 3), mode="reduced")[0]
            torch.save({"right_basis": basis}, root / "model_layers_0_self_attn_q_proj.pt")
            result = build_projection_basis(
                torch.randn(4, 4),
                rank_k=2,
                cache_dir=root / "cache",
                cache_key="fisher-test",
                layer_name="model.layers.0.self_attn.q_proj",
                left_source="none",
                right_source="fisher",
                projection_basis_dir=root,
                cache_svd=False,
            )
            self.assertEqual(result.basis.right_source, "fisher")
            self.assertEqual(tuple(result.basis.right_basis.shape), (4, 2))
            self.assertIn("right_fisher", result.metadata)

    def test_build_projection_basis_loads_cur_right_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            basis = torch.linalg.qr(torch.randn(5, 4), mode="reduced")[0]
            torch.save({"right_basis": basis}, root / "model_layers_1_self_attn_v_proj.pt")
            result = build_projection_basis(
                torch.randn(6, 5),
                rank_k=3,
                cache_dir=root / "cache",
                cache_key="cur-test",
                layer_name="model.layers.1.self_attn.v_proj",
                left_source="none",
                right_source="cur",
                projection_basis_dir=root,
                cache_svd=False,
            )
            self.assertEqual(result.basis.right_source, "cur")
            self.assertEqual(tuple(result.basis.right_basis.shape), (5, 3))
            self.assertIn("right_cur", result.metadata)


if __name__ == "__main__":
    unittest.main()
