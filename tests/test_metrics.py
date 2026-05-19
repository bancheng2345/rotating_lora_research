"""Metric-system tests for omega, rho, logging, and final summaries."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods.lora_base import LoRALinear
from src.methods.oplora import OPLoRALinear
from src.utils.metric_logger import JsonlMetricLogger
from src.utils.metrics import (
    compute_omega_values,
    orthogonality_error_value,
)
from src.utils.run_summary import save_final_summary
from src.utils.svd import build_projection_basis


class MetricSystemTest(unittest.TestCase):
    def test_omega_is_bounded(self) -> None:
        q, _ = torch.linalg.qr(torch.randn(8, 8))
        a_matrix = q[:4, :]
        activations = torch.randn(6, 8)
        omega = compute_omega_values(a_matrix, activations)
        self.assertTrue(bool(torch.all(omega >= 0.0)))
        self.assertTrue(bool(torch.all(omega <= 1.0)))

    def test_omega_is_small_inside_row_space(self) -> None:
        q, _ = torch.linalg.qr(torch.randn(8, 8))
        a_matrix = q[:4, :]
        coefficients = torch.randn(5, 4)
        activations = torch.matmul(coefficients, a_matrix)
        omega = compute_omega_values(a_matrix, activations)
        self.assertLess(float(omega.max().item()), 5e-3)

    def test_omega_is_large_in_null_space(self) -> None:
        q, _ = torch.linalg.qr(torch.randn(8, 8))
        a_matrix = q[:4, :]
        null_basis = q[4:, :]
        coefficients = torch.randn(5, 4)
        activations = torch.matmul(coefficients, null_basis)
        omega = compute_omega_values(a_matrix, activations)
        self.assertGreater(float(omega.min().item()), 0.999)

    def test_orth_error_is_small_for_orthogonal_rows(self) -> None:
        q, _ = torch.linalg.qr(torch.randn(8, 8))
        a_matrix = q[:4, :]
        error = orthogonality_error_value(a_matrix)
        self.assertLess(float(error.item()), 1e-4)

    def test_rho_k_is_lower_for_projected_update(self) -> None:
        base = torch.nn.Linear(16, 12, bias=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            basis = build_projection_basis(
                base.weight,
                rank_k=4,
                cache_dir=tmpdir,
                cache_key="metric_rho_test",
            ).basis
            lora = LoRALinear(
                torch.nn.Linear(16, 12, bias=False),
                layer_name="layer",
                rank=4,
                alpha=8,
                dropout=0.0,
                projection_basis=basis,
                use_left_projection=False,
                use_right_projection=False,
            )
            oplora = OPLoRALinear(
                torch.nn.Linear(16, 12, bias=False),
                layer_name="layer",
                rank=4,
                alpha=8,
                dropout=0.0,
                projection_basis=basis,
                use_left_projection=True,
                use_right_projection=True,
            )
        torch.manual_seed(3)
        a_matrix = torch.randn_like(lora.lora_A)
        b_matrix = torch.randn_like(lora.lora_B)
        lora.lora_A.data.copy_(a_matrix)
        lora.lora_B.data.copy_(b_matrix)
        oplora.lora_A.data.copy_(a_matrix)
        oplora.lora_B.data.copy_(b_matrix)
        rho_lora = float(lora.rho_k().item())
        rho_oplora = float(oplora.rho_k().item())
        self.assertLess(rho_oplora, rho_lora)

    def test_jsonl_logger_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = JsonlMetricLogger(Path(tmpdir) / "metrics.jsonl")
            logger.log({"step": 1, "value": 0.5})
            payload = json.loads((Path(tmpdir) / "metrics.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(payload["step"], 1)
            self.assertEqual(payload["value"], 0.5)

    def test_final_summary_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            config = {
                "model": {"name_or_path": "toy-model"},
                "training": {"train_file": "data/train.jsonl", "domain": "math"},
                "lora": {
                    "method": "pc_lora",
                    "rank": 8,
                    "projection_rank_k": 16,
                    "lambda_cap": 0.05,
                    "lambda_orth": 0.01,
                },
                "forgetting_eval": {"train_domain": "math"},
            }
            (run_dir / "config_resolved.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            (run_dir / "eval_metrics.jsonl").write_text(
                json.dumps({
                    "step": 10,
                    "eval_type": "in_domain",
                    "metric_name": "perplexity",
                    "metric_value": 5.0,
                }) + "\n",
                encoding="utf-8",
            )
            (run_dir / "forgetting_metrics.jsonl").write_text(
                json.dumps({
                    "step": 10,
                    "current_score": 6.0,
                }) + "\n",
                encoding="utf-8",
            )
            (run_dir / "mechanism_metrics.jsonl").write_text(
                json.dumps({
                    "step": 10,
                    "k": 16,
                    "rho_k": 0.01,
                    "omega_full_mean": 0.9,
                    "omega_residual_mean": 0.8,
                    "orth_error_A": 0.2,
                }) + "\n",
                encoding="utf-8",
            )
            (run_dir / "efficiency_metrics.jsonl").write_text(
                json.dumps({
                    "step": 10,
                    "total_train_time_sec": 12.5,
                    "peak_gpu_memory_gb": 3.0,
                }) + "\n",
                encoding="utf-8",
            )
            summary = save_final_summary(run_dir)
            self.assertEqual(summary["method"], "pc_lora")
            self.assertEqual(summary["best_eval_metric"], 5.0)
            self.assertEqual(summary["final_rho_k_mean"], 0.01)
            self.assertTrue((run_dir / "final_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
