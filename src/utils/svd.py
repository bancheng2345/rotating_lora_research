"""SVD utilities and caching for projection bases."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.methods.projections import ProjectionBasis, random_orthonormal_basis, validate_projection_rank
from src.utils.projection_directions import (
    load_projection_basis_file,
    sanitize_module_name,
)


@dataclass
class ProjectionBuildResult:
    """Projection bases plus metadata for logging/debugging."""

    basis: ProjectionBasis
    metadata: Dict[str, Any]


def _cache_file(cache_dir: str | Path, cache_key: str) -> Path:
    safe_key = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"{safe_key}.pt"


def _metadata_file(cache_dir: str | Path, cache_key: str) -> Path:
    safe_key = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"{safe_key}.json"


def _compute_topk_svd(weight: torch.Tensor, rank_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    svd_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    weight_f32 = weight.detach().to(dtype=svd_dtype, device="cpu")
    if rank_k == 0:
        return (
            torch.empty(weight.shape[0], 0, dtype=svd_dtype),
            torch.empty(weight.shape[1], 0, dtype=svd_dtype),
        )
    u, _, vh = torch.linalg.svd(weight_f32, full_matrices=False)
    return u[:, :rank_k].contiguous(), vh[:rank_k, :].transpose(0, 1).contiguous()


def _random_basis_for_side(
    dim: int,
    rank_k: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return random_orthonormal_basis(
        dim,
        rank_k,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=generator,
    )


def _activation_cov_file(
    activation_cov_dir: str | Path,
    *,
    layer_name: str,
) -> Path:
    """Resolves the precomputed activation covariance direction file."""

    root = Path(activation_cov_dir)
    module_key = sanitize_module_name(layer_name)
    candidates = [
        root / f"{module_key}.pt",
        root / "activation_cov" / f"{module_key}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing activation covariance basis for "
        f"{layer_name}. Tried: {', '.join(str(path) for path in candidates)}"
    )


def _load_activation_cov_basis(
    *,
    activation_cov_dir: str | Path,
    layer_name: str,
    dim: int,
    rank_k: int,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Loads top-k activation covariance eigenvectors for one module input side."""

    path = _activation_cov_file(activation_cov_dir, layer_name=layer_name)
    payload = torch.load(path, map_location="cpu")
    if "eigenvectors" not in payload:
        raise KeyError(f"Activation covariance file {path} has no 'eigenvectors' tensor.")
    eigenvectors = payload["eigenvectors"].detach().to(dtype=torch.float32, device="cpu")
    if eigenvectors.ndim != 2:
        raise ValueError(f"Expected 2D eigenvectors in {path}, got shape {tuple(eigenvectors.shape)}.")
    if int(eigenvectors.shape[0]) != int(dim):
        raise ValueError(
            f"Activation covariance basis dim mismatch for {layer_name}: "
            f"basis dim={eigenvectors.shape[0]}, expected dim={dim}."
        )
    used_rank = min(int(rank_k), int(eigenvectors.shape[1]))
    if used_rank <= 0:
        return torch.empty(dim, 0, dtype=torch.float32), {
            "activation_cov_path": str(path),
            "activation_cov_rank_available": int(eigenvectors.shape[1]),
            "activation_cov_rank_used": 0,
        }
    basis = eigenvectors[:, :used_rank].contiguous()
    q, _ = torch.linalg.qr(basis, mode="reduced")
    return q[:, :used_rank].contiguous(), {
        "activation_cov_path": str(path),
        "activation_cov_rank_available": int(eigenvectors.shape[1]),
        "activation_cov_rank_used": used_rank,
    }


def build_projection_basis(
    weight: torch.Tensor,
    *,
    rank_k: int,
    cache_dir: str | Path,
    cache_key: str,
    layer_name: Optional[str] = None,
    left_source: str = "svd",
    right_source: str = "svd",
    activation_cov_dir: Optional[str | Path] = None,
    projection_basis_dir: Optional[str | Path] = None,
    enforce_rank_mode: str = "clip",
    random_seed: int = 0,
    cache_svd: bool = True,
) -> ProjectionBuildResult:
    """Builds left/right projection bases, caching SVD results when requested."""

    start_time = time.perf_counter()
    actual_rank = validate_projection_rank(
        rank_k=rank_k,
        out_features=weight.shape[0],
        in_features=weight.shape[1],
        mode=enforce_rank_mode,
    )
    cache_path = _cache_file(cache_dir, cache_key)
    meta_path = _metadata_file(cache_dir, cache_key)
    cache_hit = False
    generic_sources = {"fisher", "cur", "mwa"}

    if (
        cache_svd
        and actual_rank > 0
        and left_source == "svd"
        and right_source == "svd"
        and cache_path.exists()
    ):
        payload = torch.load(cache_path, map_location="cpu")
        left_basis = payload["left_basis"]
        right_basis = payload["right_basis"]
        cache_hit = True
    else:
        svd_left = None
        svd_right = None
        if actual_rank > 0 and (left_source == "svd" or right_source == "svd"):
            svd_left, svd_right = _compute_topk_svd(weight, actual_rank)
        if left_source == "svd":
            left_basis = svd_left
        elif left_source == "random_orthogonal":
            left_basis = _random_basis_for_side(weight.shape[0], actual_rank, random_seed + 17)
        elif left_source == "activation_cov":
            if activation_cov_dir is None or layer_name is None:
                raise ValueError("left_source=activation_cov requires activation_cov_dir and layer_name.")
            left_basis, left_actcov_metadata = _load_activation_cov_basis(
                activation_cov_dir=activation_cov_dir,
                layer_name=layer_name,
                dim=weight.shape[0],
                rank_k=actual_rank,
            )
        elif left_source in generic_sources:
            if projection_basis_dir is None or layer_name is None:
                raise ValueError(f"left_source={left_source} requires projection_basis_dir and layer_name.")
            left_basis, left_generic_metadata = load_projection_basis_file(
                root=projection_basis_dir,
                source=left_source,
                side="left",
                layer_name=layer_name,
                dim=weight.shape[0],
                rank_k=actual_rank,
            )
        elif left_source == "none":
            left_basis = None
        else:
            raise ValueError(f"Unsupported left projection source: {left_source}")

        if right_source == "svd":
            right_basis = svd_right
        elif right_source == "random_orthogonal":
            right_basis = _random_basis_for_side(weight.shape[1], actual_rank, random_seed + 29)
        elif right_source == "activation_cov":
            if activation_cov_dir is None or layer_name is None:
                raise ValueError("right_source=activation_cov requires activation_cov_dir and layer_name.")
            right_basis, right_actcov_metadata = _load_activation_cov_basis(
                activation_cov_dir=activation_cov_dir,
                layer_name=layer_name,
                dim=weight.shape[1],
                rank_k=actual_rank,
            )
        elif right_source in generic_sources:
            if projection_basis_dir is None or layer_name is None:
                raise ValueError(f"right_source={right_source} requires projection_basis_dir and layer_name.")
            right_basis, right_generic_metadata = load_projection_basis_file(
                root=projection_basis_dir,
                source=right_source,
                side="right",
                layer_name=layer_name,
                dim=weight.shape[1],
                rank_k=actual_rank,
            )
        elif right_source == "none":
            right_basis = None
        else:
            raise ValueError(f"Unsupported right projection source: {right_source}")

        if (
            cache_svd
            and actual_rank > 0
            and left_source == "svd"
            and right_source == "svd"
        ):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "left_basis": left_basis,
                    "right_basis": right_basis,
                },
                cache_path,
            )

    metadata = {
        "cache_key": cache_key,
        "cache_path": str(cache_path),
        "cache_hit": cache_hit,
        "build_time_sec": float(time.perf_counter() - start_time),
        "rank_k_requested": rank_k,
        "rank_k_used": actual_rank,
        "left_source": left_source,
        "right_source": right_source,
        "shape": list(weight.shape),
    }
    if "left_actcov_metadata" in locals():
        metadata["left_activation_cov"] = left_actcov_metadata
    if "right_actcov_metadata" in locals():
        metadata["right_activation_cov"] = right_actcov_metadata
    if "left_generic_metadata" in locals():
        metadata[f"left_{left_source}"] = left_generic_metadata
    if "right_generic_metadata" in locals():
        metadata[f"right_{right_source}"] = right_generic_metadata
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return ProjectionBuildResult(
        basis=ProjectionBasis(
            left_basis=left_basis,
            right_basis=right_basis,
            rank_k=actual_rank,
            left_source=left_source,
            right_source=right_source,
        ),
        metadata=metadata,
    )
