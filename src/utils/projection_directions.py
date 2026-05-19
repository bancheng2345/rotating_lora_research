"""Utilities for non-SVD projection directions.

These helpers support projection bases that are precomputed outside training,
such as empirical Fisher input factors or CUR row/column subspaces.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import torch


def sanitize_module_name(name: str) -> str:
    """Converts a module path into a stable filesystem key."""

    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


def orthonormalize_basis(matrix: torch.Tensor, *, max_rank: int | None = None) -> torch.Tensor:
    """Returns an orthonormal basis spanning the columns of `matrix`."""

    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D basis candidate, got shape {tuple(matrix.shape)}.")
    if matrix.numel() == 0:
        rank = 0 if max_rank is None else int(max_rank)
        return torch.empty(matrix.shape[0], 0 if rank <= 0 else min(rank, matrix.shape[1]), dtype=torch.float32)
    basis = matrix.detach().to(dtype=torch.float32, device="cpu")
    if max_rank is not None:
        basis = basis[:, : int(max_rank)]
    q, _ = torch.linalg.qr(basis, mode="reduced")
    if max_rank is not None:
        q = q[:, : int(max_rank)]
    return q.contiguous()


def randomized_top_eigenvectors(
    matrix: torch.Tensor,
    *,
    rank: int,
    oversample: int = 8,
    n_iter: int = 2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximates top eigenvectors of a symmetric PSD matrix.

    This avoids a full O(d^3) eigendecomposition for 4k-dim LLaMA layers. The
    returned eigenvalues are Ritz values from the projected matrix.
    """

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {tuple(matrix.shape)}.")
    dim = int(matrix.shape[0])
    rank = min(max(int(rank), 0), dim)
    if rank == 0:
        return torch.empty(0, dtype=torch.float32), torch.empty(dim, 0, dtype=torch.float32)

    mat = 0.5 * (matrix.detach().to(dtype=torch.float32, device="cpu") + matrix.detach().to(dtype=torch.float32, device="cpu").T)
    sketch_rank = min(dim, rank + max(int(oversample), 0))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    q = torch.randn(dim, sketch_rank, dtype=torch.float32, generator=generator)
    q, _ = torch.linalg.qr(mat @ q, mode="reduced")
    for _ in range(max(int(n_iter), 0)):
        q, _ = torch.linalg.qr(mat @ q, mode="reduced")
    small = q.T @ mat @ q
    evals, evecs = torch.linalg.eigh(0.5 * (small + small.T))
    order = torch.argsort(evals, descending=True)
    evals = evals[order][:rank].clamp_min(0.0).contiguous()
    vectors = (q @ evecs[:, order[:rank]]).contiguous()
    vectors = orthonormalize_basis(vectors, max_rank=rank)
    return evals, vectors


def top_eigenvectors(
    matrix: torch.Tensor,
    *,
    rank: int,
    method: str = "randomized",
    oversample: int = 8,
    n_iter: int = 2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes or approximates top eigenvectors for a symmetric matrix."""

    if method == "eigh":
        mat = 0.5 * (matrix.detach().to(dtype=torch.float32, device="cpu") + matrix.detach().to(dtype=torch.float32, device="cpu").T)
        evals, evecs = torch.linalg.eigh(mat)
        order = torch.argsort(evals, descending=True)[: int(rank)]
        return evals[order].clamp_min(0.0).contiguous(), evecs[:, order].contiguous()
    if method == "randomized":
        return randomized_top_eigenvectors(
            matrix,
            rank=rank,
            oversample=oversample,
            n_iter=n_iter,
            seed=seed,
        )
    raise ValueError(f"Unsupported eigen method: {method!r}")


def projection_direction_file(root: str | Path, source: str, layer_name: str) -> Path:
    """Resolves a precomputed projection direction file."""

    root_path = Path(root)
    module_key = sanitize_module_name(layer_name)
    candidates = [
        root_path / f"{module_key}.pt",
        root_path / source / f"{module_key}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing {source} projection basis for {layer_name}. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def load_projection_basis_file(
    *,
    root: str | Path,
    source: str,
    side: str,
    layer_name: str,
    dim: int,
    rank_k: int,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Loads a left/right basis from a generic projection direction file."""

    if side not in {"left", "right"}:
        raise ValueError(f"side must be left or right, got {side!r}.")
    path = projection_direction_file(root, source, layer_name)
    payload = torch.load(path, map_location="cpu")
    key = f"{side}_basis"
    alias_keys = [key]
    if side == "right":
        # M_WA direction files from activation_cov_experiment store input-side
        # bases as input_vectors. These are the directions used by P_R z.
        alias_keys.extend(["input_vectors", "basis"])
    else:
        alias_keys.extend(["output_vectors", "basis"])
    basis_key = next((candidate for candidate in alias_keys if candidate in payload), None)
    if basis_key is None:
        raise KeyError(f"{path} has none of the expected basis keys: {alias_keys}.")
    basis = payload[basis_key]
    basis = basis.detach().to(dtype=torch.float32, device="cpu")
    if basis.ndim != 2:
        raise ValueError(f"Expected 2D {key} in {path}, got shape {tuple(basis.shape)}.")
    if int(basis.shape[0]) != int(dim):
        raise ValueError(
            f"{source} {side} basis dim mismatch for {layer_name}: "
            f"basis dim={basis.shape[0]}, expected dim={dim}."
        )
    used_rank = min(int(rank_k), int(basis.shape[1]))
    result = orthonormalize_basis(basis[:, :used_rank], max_rank=used_rank)
    metadata = {
        "source": source,
        "side": side,
        "path": str(path),
        "basis_key": basis_key,
        "rank_available": int(basis.shape[1]),
        "rank_used": int(used_rank),
    }
    for optional_key in (
        "selected_rows",
        "selected_columns",
        "count",
        "weight_sum",
        "matrix_trace",
        "joint_eigenvalues",
        "covariance_rank_used",
        "small_mwa_trace",
        "orthogonality_error_input",
    ):
        if optional_key in payload:
            value = payload[optional_key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().tolist()
            metadata[optional_key] = value
    return result, metadata
