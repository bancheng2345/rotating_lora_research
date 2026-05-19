"""Tests for generation-based math evaluation helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.eval_math import exact_match, extract_final_answer, normalize_math_answer
from src.utils.run_summary import build_final_summary


class MathEvalTests(unittest.TestCase):
    def test_extract_boxed_answer(self) -> None:
        prediction = "We compute the roots and obtain \\boxed{\\frac{11}{2}}."
        self.assertEqual(extract_final_answer(prediction), "\\frac{11}{2}")

    def test_normalize_fraction_variants(self) -> None:
        self.assertEqual(normalize_math_answer("\\boxed{\\frac{10}{20}}"), "1/2")
        self.assertEqual(normalize_math_answer("  0.5000 "), "0.5")

    def test_exact_match_uses_normalization(self) -> None:
        self.assertTrue(exact_match("The answer is \\boxed{12}.", "12"))
        self.assertTrue(exact_match("\\frac{10}{20}", "1/2"))
        self.assertFalse(exact_match("13", "12"))

    def test_run_summary_filters_primary_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "config_resolved.yaml").write_text(
                "\n".join(
                    [
                        "model:",
                        "  name_or_path: tiny",
                        "training:",
                        "  eval_metric_name: perplexity",
                        "  eval_metric_mode: min",
                        "lora:",
                        "  method: lora",
                        "  rank: 8",
                        "  projection_rank_k: 16",
                        "  lambda_cap: 0.05",
                        "  lambda_orth: 0.01",
                    ]
                ),
                encoding="utf-8",
            )
            eval_records = [
                {
                    "step": 100,
                    "eval_type": "in_domain",
                    "metric_name": "perplexity",
                    "metric_value": 6.0,
                },
                {
                    "step": 100,
                    "eval_type": "in_domain",
                    "metric_name": "exact_match_generation",
                    "metric_value": 0.4,
                },
            ]
            with (run_dir / "eval_metrics.jsonl").open("w", encoding="utf-8") as handle:
                for record in eval_records:
                    handle.write(json.dumps(record) + "\n")
            (run_dir / "forgetting_metrics.jsonl").write_text("", encoding="utf-8")
            (run_dir / "mechanism_metrics.jsonl").write_text("", encoding="utf-8")
            (run_dir / "efficiency_metrics.jsonl").write_text("", encoding="utf-8")
            (run_dir / "math_generation_eval.json").write_text(
                json.dumps(
                    {
                        "exact_match": 0.4,
                        "num_samples": 50,
                        "checkpoint_step": 100,
                        "checkpoint_path": "checkpoints/step_100.pt",
                    }
                ),
                encoding="utf-8",
            )

            summary = build_final_summary(run_dir)
            self.assertEqual(summary["best_eval_metric"], 6.0)
            self.assertEqual(summary["math_generation_exact_match"], 0.4)


if __name__ == "__main__":
    unittest.main()
