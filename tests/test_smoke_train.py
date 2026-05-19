"""End-to-end smoke train test."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.trainer import ExperimentTrainer


class SmokeTrainTest(unittest.TestCase):
    def test_two_step_training_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "runtime": {
                    "gpu": {
                        "mode": "disabled",
                        "cuda_visible_devices": "auto",
                        "fallback_cuda_visible_devices": "0",
                        "override_env": "CUDA_VISIBLE_DEVICES",
                    }
                },
                "model": {
                    "init_from_scratch": True,
                    "synthetic_name": "tiny_llama_test",
                    "architecture": "llama",
                    "vocab_size": 64,
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "max_position_embeddings": 64,
                    "pad_token_id": 0,
                    "bos_token_id": 1,
                    "eos_token_id": 2,
                    "attention_dropout": 0.0,
                    "target_modules": ["q_proj", "v_proj"],
                    "torch_dtype": "float32",
                    "attn_impl": "eager",
                    "local_files_only": True,
                },
                "lora": {
                    "method": "pc_lora",
                    "rank": 4,
                    "alpha": 8,
                    "dropout": 0.0,
                    "projection_rank_k": 4,
                    "left_projection_source": "svd",
                    "right_projection_source": "svd",
                    "use_left_projection": True,
                    "use_right_projection": True,
                    "use_capture_loss": True,
                    "capture_space": "residual",
                    "capture_weighting": "none",
                    "capture_layers": "all",
                    "lambda_cap": 0.01,
                    "lambda_orth": 0.01,
                    "use_output_capture": False,
                    "detach_capture_inputs": True,
                    "cache_svd": True,
                    "svd_cache_dir": str(Path(tmpdir) / "svd_cache"),
                    "enforce_projection_rank": "clip",
                    "eps": 1.0e-6,
                },
                "training": {
                    "dataset_name": "synthetic_lm",
                    "seq_length": 16,
                    "synthetic_train_samples": 8,
                    "synthetic_eval_samples": 4,
                    "max_steps": 2,
                    "batch_size": 2,
                    "eval_batch_size": 2,
                    "grad_accum_steps": 1,
                    "learning_rate": 1.0e-3,
                    "warmup_ratio": 0.0,
                    "weight_decay": 0.0,
                    "logging_steps": 1,
                    "eval_steps": 1,
                    "save_steps": 2,
                    "gradient_checkpointing": False,
                },
                "experiment": {
                    "output_dir": str(Path(tmpdir) / "runs/{method}_{timestamp}"),
                    "seed": 0,
                    "notes": "unit test smoke run",
                },
            }

            trainer = ExperimentTrainer(config)
            summary = trainer.train()
            run_dir = Path(summary["run_dir"])

            self.assertTrue((run_dir / "config_resolved.yaml").exists())
            self.assertTrue((run_dir / "train_metrics.jsonl").exists())
            self.assertTrue((run_dir / "eval_metrics.jsonl").exists())
            self.assertTrue((run_dir / "mechanism_metrics.jsonl").exists())
            self.assertTrue((run_dir / "final_summary.json").exists())
            self.assertTrue((run_dir / "checkpoints" / "step_2.pt").exists())


if __name__ == "__main__":
    unittest.main()

