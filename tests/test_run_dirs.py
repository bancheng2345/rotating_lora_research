"""Run-directory resolution tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.trainer import resolve_model_slug, resolve_run_dir


class RunDirectoryResolutionTest(unittest.TestCase):
    def test_snapshot_path_resolves_to_readable_model_slug(self) -> None:
        model_path = (
            "/data/zck/code/models/models--meta-llama--Llama-3.2-3B/"
            "snapshots/13afe5124825b4f3751f836b40dafda64c1ed062"
        )
        self.assertEqual(resolve_model_slug(model_path), "meta_llama_llama_3_2_3b")

    def test_run_dir_groups_outputs_under_model_folder(self) -> None:
        config = {
            "model": {
                "name_or_path": (
                    "/data/zck/code/models/models--meta-llama--Llama-3.2-1B/"
                    "snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
                )
            },
            "training": {"train_file": "data/splits/math500_train_450.jsonl"},
            "lora": {"method": "pc_lora", "rank": 8, "projection_rank_k": 16},
            "experiment": {
                "output_dir": (
                    "outputs/runs/{model_dir}/"
                    "{method}_{model}_{dataset}_r{rank}_k{k}_seed{seed}_{timestamp}"
                ),
                "seed": 0,
            },
        }

        run_dir = resolve_run_dir(config)

        self.assertEqual(run_dir.parent.name, "meta_llama_llama_3_2_1b")
        self.assertTrue(run_dir.name.startswith("pc_lora_meta_llama_llama_3_2_1b_math500_train_450"))


if __name__ == "__main__":
    unittest.main()
