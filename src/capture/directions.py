"""Capture direction loading independent from preserve projections.

Preserve bases are used to mask LoRA updates. Capture bases are only used as
guidance targets for LoRA A row-space alignment, so task high-frequency
directions are not accidentally removed from the update path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.utils.projection_directions import (
    load_projection_basis_file,
    orthonormalize_basis,
    sanitize_module_name,
)


@dataclass
class CaptureBasisBuildResult:
    """A task/data capture basis plus provenance metadata."""

    basis: Optional[torch.Tensor]
    source: str
    rank_k: int
    metadata: Dict[str, Any]


def _activation_cov_file(root: str | Path, *, layer_name: str) -> Path:
    """Resolves an ActivationCov eigenvector file for one target module."""

    root_path = Path(root)
    module_key = sanitize_module_name(layer_name)
    candidates = [
        root_path / f"{module_key}.pt",
        root_path / "activation_cov" / f"{module_key}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing activation covariance capture basis for {layer_name}. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_activation_cov_basis(
    *,
    root: str | Path,
    layer_name: str,
    dim: int,
    rank_k: int,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Loads top-k task ActivationCov eigenvectors as an input-side basis."""

    path = _activation_cov_file(root, layer_name=layer_name)
    payload = torch.load(path, map_location="cpu")
    if "eigenvectors" not in payload:
        raise KeyError(f"Activation covariance file {path} has no 'eigenvectors' tensor.")
    basis = payload["eigenvectors"].detach().to(dtype=torch.float32, device="cpu")
    if basis.ndim != 2:
        raise ValueError(f"Expected 2D eigenvectors in {path}, got shape {tuple(basis.shape)}.")
    if int(basis.shape[0]) != int(dim):
        raise ValueError(
            f"Capture basis dim mismatch for {layer_name}: basis dim={basis.shape[0]}, expected dim={dim}."
        )
    used_rank = min(int(rank_k), int(basis.shape[1]))
    basis = orthonormalize_basis(basis[:, :used_rank], max_rank=used_rank)
    return basis, {
        "source": "activation_cov",
        "path": str(path),
        "rank_available": int(payload["eigenvectors"].shape[1]),
        "rank_used": int(used_rank),
    }


def residualize_capture_basis(
    capture_basis: torch.Tensor,
    preserve_basis: Optional[torch.Tensor],
    *,
    rank_k: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Projects capture directions into the preserve residual input space.

    For right-side preserve basis Q_p, this computes
    (I - Q_p Q_p^T) Q_capture and re-orthonormalizes the result. This keeps
    task/data guidance in the same residual space used by PC-LoRA capture.
    """

    if capture_basis.numel() == 0:
        return capture_basis
    basis = capture_basis.detach().to(dtype=torch.float32, device="cpu")
    if preserve_basis is not None and preserve_basis.numel() > 0:
        preserve = preserve_basis.detach().to(dtype=torch.float32, device="cpu")
        preserve = orthonormalize_basis(preserve, max_rank=preserve.shape[1])
        basis = basis - preserve @ (preserve.transpose(0, 1) @ basis)
    # QR can invent arbitrary columns for near-zero residual directions. SVD
    # keeps only directions with actual residual energy.
    u, s, _ = torch.linalg.svd(basis, full_matrices=False)
    keep = s > float(eps)
    if not bool(keep.any()):
        return torch.empty(basis.shape[0], 0, dtype=torch.float32)
    used_rank = min(int(rank_k), int(keep.sum().item()))
    return u[:, keep][:, :used_rank].contiguous()


def build_capture_basis(
    *,
    layer_name: str,
    input_dim: int,
    rank_k: int,
    source: str,
    activation_cov_dir: Optional[str | Path] = None,
    capture_basis_dir: Optional[str | Path] = None,
    projection_basis_dir: Optional[str | Path] = None,
    preserve_right_basis: Optional[torch.Tensor] = None,
    residualize_against_preserve: bool = False,
) -> CaptureBasisBuildResult:
    """Builds a capture-only input basis for LoRA A row-space guidance."""

    source = str(source or "none")
    if source == "none" or rank_k <= 0:
        return CaptureBasisBuildResult(
            basis=None,
            source=source,
            rank_k=0,
            metadata={"source": source, "rank_used": 0},
        )

    metadata: Dict[str, Any]
    if source == "activation_cov":
        if activation_cov_dir is None and capture_basis_dir is None:
            raise ValueError("capture_basis_source=activation_cov requires activation_cov_dir or capture_basis_dir.")
        basis, metadata = _load_activation_cov_basis(
            root=capture_basis_dir or activation_cov_dir,
            layer_name=layer_name,
            dim=input_dim,
            rank_k=rank_k,
        )
    elif source in {"mwa", "fisher", "cur"}:
        root = capture_basis_dir or projection_basis_dir
        if root is None:
            raise ValueError(f"capture_basis_source={source} requires capture_basis_dir or projection_basis_dir.")
        basis, metadata = load_projection_basis_file(
            root=root,
            source=source,
            side="right",
            layer_name=layer_name,
            dim=input_dim,
            rank_k=rank_k,
        )
    elif source == "preserve_right":
        if preserve_right_basis is None or preserve_right_basis.numel() == 0:
            raise ValueError("capture_basis_source=preserve_right requires a non-empty right preserve basis.")
        used_rank = min(int(rank_k), int(preserve_right_basis.shape[1]))
        basis = orthonormalize_basis(preserve_right_basis[:, :used_rank], max_rank=used_rank)
        metadata = {"source": source, "rank_used": used_rank}
    else:
        raise ValueError(f"Unsupported capture_basis_source: {source}")

    pre_residual_rank = int(basis.shape[1])
    if residualize_against_preserve:
        basis = residualize_capture_basis(
            basis,
            preserve_right_basis,
            rank_k=min(rank_k, pre_residual_rank),
        )
        metadata["residualized_against_preserve"] = True
        metadata["pre_residual_rank"] = pre_residual_rank
        metadata["rank_used"] = int(basis.shape[1])
    else:
        metadata["residualized_against_preserve"] = False

    return CaptureBasisBuildResult(
        basis=basis,
        source=source,
        rank_k=int(basis.shape[1]),
        metadata=metadata,
    )
