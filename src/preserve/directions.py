"""Preserve direction helpers.

This module intentionally wraps the existing projection builder instead of
moving it, keeping the old training path stable while making the Preserve /
Capture split explicit for new experiments.
"""

from src.utils.svd import build_projection_basis

__all__ = ["build_projection_basis"]
