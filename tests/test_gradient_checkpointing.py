"""Gradient-checkpointing compatibility tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.trainer import enable_gradient_checkpointing_for_adapters


class DummyModelWithAPI(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gc_enabled = False
        self.input_grads_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.gc_enabled = True

    def enable_input_require_grads(self) -> None:
        self.input_grads_enabled = True


class DummyModelWithEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.gc_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.gc_enabled = True

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding


class GradientCheckpointingCompatibilityTest(unittest.TestCase):
    def test_prefers_enable_input_require_grads_api(self) -> None:
        model = DummyModelWithAPI()

        enable_gradient_checkpointing_for_adapters(model)

        self.assertTrue(model.gc_enabled)
        self.assertTrue(model.input_grads_enabled)

    def test_falls_back_to_embedding_hook(self) -> None:
        model = DummyModelWithEmbeddings()

        enable_gradient_checkpointing_for_adapters(model)
        outputs = model.get_input_embeddings()(torch.tensor([[1, 2, 3]]))

        self.assertTrue(model.gc_enabled)
        self.assertTrue(hasattr(model, "_pc_lora_input_require_grads_hook"))
        self.assertTrue(outputs.requires_grad)


if __name__ == "__main__":
    unittest.main()
