"""Helpers for resolving project paths in scripts."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np


def project_root() -> Path:
    """Return repository root (two levels above this file)."""
    return Path(__file__).resolve().parents[2]


def add_root_to_path() -> Path:
    """Insert repo root into sys.path and return it."""
    root = project_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def infer_dof_from_npz(npz_path: Path, key: str) -> int:
    """Infer DoF from an .npz file array shape."""
    data = np.load(npz_path)
    arr = data[key]
    return int(arr.shape[-1])


def infer_steps_from_npz(npz_path: Path, key: str) -> int:
    """Infer time steps from an .npz file array shape."""
    data = np.load(npz_path)
    arr = data[key]
    # torque: (T, N, dof) -> steps=T
    # paths: (N, T, dof) -> steps=T
    if arr.ndim == 3 and key == "torque":
        return int(arr.shape[0])
    if arr.ndim >= 2:
        return int(arr.shape[-2])
    raise ValueError(f"Cannot infer steps from {npz_path}")
