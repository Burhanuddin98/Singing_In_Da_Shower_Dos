# acoustics/materials.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from .bands import resample_bands

@dataclass
class Material:
    name: str
    freqs: np.ndarray            # native band centers (Hz)
    alpha: np.ndarray            # (B,)
    tau:   np.ndarray            # (B,)
    scatter: np.ndarray          # (B,)

    def to_bands(self, dst_freqs: np.ndarray) -> "Material":
        return Material(
            name=self.name,
            freqs=np.asarray(dst_freqs, float),
            alpha=resample_bands(self.freqs, self.alpha, dst_freqs),
            tau=resample_bands(self.freqs, self.tau, dst_freqs),
            scatter=resample_bands(self.freqs, self.scatter, dst_freqs),
        )

def _mat(name, f, a, t, s):
    f = np.asarray(f, float)
    a = np.asarray(a, float)
    t = np.asarray(t, float)
    s = np.asarray(s, float)
    # Safety clamps
    a = np.clip(a, 0.0, 0.99)
    t = np.clip(t, 0.0, 0.99 - a)
    s = np.clip(s, 0.0, 1.0)
    return Material(name, f, a, t, s)

def builtin_library(native_mode: str = "octave") -> Dict[str, Material]:
    """
    Minimal starter library (expand as needed).
    The tables below are illustrative—replace with your measured/vendor data.
    """
    # native table centers (octaves)
    F = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)

    lib = {
        "Concrete": _mat("Concrete", F,
            a=np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.03]),
            t=np.zeros_like(F),
            s=np.array([0.03, 0.03, 0.03, 0.03, 0.02, 0.02])),
        "Brick": _mat("Brick", F,
            a=np.array([0.03, 0.03, 0.04, 0.04, 0.05, 0.05]),
            t=np.zeros_like(F),
            s=np.array([0.05, 0.05, 0.06, 0.06, 0.06, 0.06])),
        "Gypsum board": _mat("Gypsum board", F,
            a=np.array([0.29, 0.10, 0.05, 0.04, 0.07, 0.09]),
            t=np.zeros_like(F),
            s=np.array([0.20, 0.20, 0.25, 0.25, 0.30, 0.35])),
        "Carpet (on pad)": _mat("Carpet (on pad)", F,
            a=np.array([0.08, 0.24, 0.57, 0.69, 0.71, 0.73]),
            t=np.zeros_like(F),
            s=np.array([0.40, 0.60, 0.70, 0.75, 0.75, 0.75])),
        "Glass": _mat("Glass", F,
            a=np.array([0.18, 0.06, 0.04, 0.03, 0.02, 0.02]),
            t=np.array([0.00, 0.02, 0.03, 0.05, 0.05, 0.05]),
            s=np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])),
        "Acoustic tile": _mat("Acoustic tile", F,
            a=np.array([0.32, 0.51, 0.64, 0.70, 0.73, 0.75]),
            t=np.zeros_like(F),
            s=np.array([0.40, 0.55, 0.60, 0.65, 0.65, 0.65])),
    }
    return lib

def to_broadband(alpha_b: np.ndarray, tau_b: np.ndarray, method: str = "mean"):
    a = np.asarray(alpha_b, float); t = np.asarray(tau_b, float)
    if method == "mean":
        return float(np.mean(a)), float(np.mean(t))
    # energy-weight mean (weight highs a bit)
    w = 1.0 + np.linspace(0, 1, a.size)
    w /= w.sum()
    return float((a * w).sum()), float((t * w).sum())

# Simple presets mapping element role -> material name
PRESETS: Dict[str, List[str]] = {
    "Small office": ["Carpet (on pad)", "Gypsum board", "Glass", "Concrete"],
    "Lecture hall": ["Concrete", "Acoustic tile", "Gypsum board", "Carpet (on pad)"],
    "Stairwell":    ["Concrete", "Brick", "Glass", "Concrete"],
}
