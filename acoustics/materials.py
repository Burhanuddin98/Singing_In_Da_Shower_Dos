# acoustics/materials.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

# Octave centers we target: align with config.OCTAVE_CENTERS
DEFAULT_BANDS = [125, 250, 500, 1000, 2000, 4000]

@dataclass
class MaterialBands:
    name: str
    bands: List[int]
    alpha: np.ndarray      # shape (B,) absorption
    scatter: np.ndarray    # shape (B,) scattering [0..1], diffuse share
    tau: np.ndarray        # shape (B,) transmission

# A tiny “starter” library. Numbers are illustrative but reasonable.
# Replace/extend freely later or load from YAML.
def builtin_library(bands: List[int] = DEFAULT_BANDS) -> Dict[str, MaterialBands]:
    B = len(bands)

    def arr(vals):
        vals = list(vals)
        if len(vals) != B:
            raise ValueError(f"Material requires {B} band values.")
        return np.array(vals, dtype=np.float32)

    lib = {
        "Concrete": MaterialBands(
            "Concrete", bands,
            alpha=arr([0.01, 0.01, 0.015, 0.02, 0.02, 0.02]),
            scatter=arr([0.05, 0.05, 0.06, 0.08, 0.10, 0.10]),
            tau=arr([0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
        ),
        "Painted Brick": MaterialBands(
            "Painted Brick", bands,
            alpha=arr([0.02, 0.02, 0.03, 0.04, 0.05, 0.07]),
            scatter=arr([0.10, 0.10, 0.12, 0.14, 0.16, 0.18]),
            tau=arr([0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
        ),
        "Carpet on pad": MaterialBands(
            "Carpet on pad", bands,
            alpha=arr([0.02, 0.06, 0.14, 0.37, 0.60, 0.65]),
            scatter=arr([0.20, 0.20, 0.25, 0.30, 0.30, 0.30]),
            tau=arr([0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
        ),
        "Gypsum 12mm": MaterialBands(
            "Gypsum 12mm", bands,
            alpha=arr([0.29, 0.10, 0.05, 0.04, 0.07, 0.09]),
            scatter=arr([0.10, 0.10, 0.10, 0.12, 0.12, 0.15]),
            tau=arr([0.02, 0.02, 0.01, 0.01, 0.01, 0.01]),
        ),
        "Curtain heavy": MaterialBands(
            "Curtain heavy", bands,
            alpha=arr([0.05, 0.10, 0.15, 0.40, 0.70, 0.70]),
            scatter=arr([0.20, 0.25, 0.30, 0.35, 0.35, 0.35]),
            tau=arr([0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
        ),
        "Glass 6mm": MaterialBands(
            "Glass 6mm", bands,
            alpha=arr([0.18, 0.06, 0.04, 0.03, 0.02, 0.02]),
            scatter=arr([0.05, 0.05, 0.06, 0.06, 0.08, 0.10]),
            tau=arr([0.35, 0.40, 0.45, 0.45, 0.40, 0.35]),
        ),
    }
    return lib


def to_broadband(alpha_b: np.ndarray, tau_b: np.ndarray, method: str = "mean"):
    """Collapse per-band to a single broadband for legacy paths/preview UI."""
    if method == "mean":
        return float(np.clip(alpha_b.mean(), 0.0, 0.99)), float(np.clip(tau_b.mean(), 0.0, 0.99))
    elif method == "energy":
        # Energy-mean (weights high bands slightly more if you later weight by 1/3-oct width)
        return float(np.clip(np.mean(alpha_b), 0.0, 0.99)), float(np.clip(np.mean(tau_b), 0.0, 0.99))
    else:
        raise ValueError("Unknown broadband collapse method")
