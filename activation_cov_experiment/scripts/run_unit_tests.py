#!/usr/bin/env python
"""Minimal unit tests for the isolated activation covariance experiment."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from activation_cov_experiment.src.covariance import StreamingCovariance
from activation_cov_experiment.src.subspace_metrics import capture_energy, orthogonality_error, principal_angle_stats, random_orthonormal


def assert_close(value: float, expected: float, *, atol: float, message: str) -> None:
    if abs(value - expected) > atol:
        raise AssertionError(f"{message}: got {value}, expected {expected} ± {atol}")


def test_random_orthonormal() -> None:
    q = random_orthonormal(32, 8)
    if orthogonality_error(q) > 1e-5:
        raise AssertionError("Random orthonormal basis failed Q^T Q ≈ I test.")


def test_capture_energy_bounds() -> None:
    q = random_orthonormal(16, 4)
    z = torch.randn(20, 16)
    energy = capture_energy(z, q)
    if not bool(((energy >= 0.0) & (energy <= 1.0)).all().item()):
        raise AssertionError("Capture energy must lie in [0, 1].")


def test_capture_energy_aligned() -> None:
    q = torch.eye(8, 1)
    z = torch.zeros(10, 8)
    z[:, 0] = 1.0
    energy = capture_energy(z, q)
    assert_close(float(energy.mean().item()), 1.0, atol=1e-6, message="Aligned capture energy")


def test_capture_energy_orthogonal() -> None:
    q = torch.eye(8, 1)
    z = torch.zeros(10, 8)
    z[:, 1] = 1.0
    energy = capture_energy(z, q)
    assert_close(float(energy.mean().item()), 0.0, atol=1e-6, message="Orthogonal capture energy")


def test_principal_angles() -> None:
    q = random_orthonormal(16, 4)
    stats = principal_angle_stats(q, q)
    if stats["mean_angle_deg"] > 5e-2:
        raise AssertionError(f"Principal angles for identical subspaces should be ~0, got {stats}")


def test_streaming_covariance() -> None:
    values = torch.randn(64, 12)
    streamer = StreamingCovariance(12)
    streamer.update(values[:20])
    streamer.update(values[20:40])
    streamer.update(values[40:])
    state = streamer.finalize(centered=False)
    direct = (values.T @ values) / values.shape[0]
    max_diff = float((state.matrix - direct).abs().max().item())
    if max_diff > 1e-5:
        raise AssertionError(f"Streaming second moment mismatch: max diff {max_diff}")


def main() -> None:
    tests = [
        test_random_orthonormal,
        test_capture_energy_bounds,
        test_capture_energy_aligned,
        test_capture_energy_orthogonal,
        test_principal_angles,
        test_streaming_covariance,
    ]
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")
    print(f"{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
