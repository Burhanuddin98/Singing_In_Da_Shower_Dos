# acoustics/materials.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pathlib, csv, json

from .bands import standard_centers, resample_bands

@dataclass
class MaterialBands:
    name: str
    centers: List[float]          # Hz
    alpha: np.ndarray             # (B,)  0..1
    tau:   np.ndarray             # (B,)  0..1
    scatter: Optional[np.ndarray] = None  # (B,) 0..1

    def clamp01(self) -> "MaterialBands":
        a = np.clip(self.alpha, 0.0, 0.99)
        t = np.clip(self.tau,   0.0, 0.99 - a)
        s = None if self.scatter is None else np.clip(self.scatter, 0.0, 1.0)
        return MaterialBands(self.name, list(self.centers),
                             a.astype(np.float32), t.astype(np.float32),
                             None if s is None else s.astype(np.float32))

def to_broadband(alpha_b: np.ndarray, tau_b: np.ndarray, method: str = "mean") -> tuple[float, float]:
    a = np.asarray(alpha_b, float); t = np.asarray(tau_b, float)
    if method == "mean":
        return float(a.mean()), float(t.mean())
    if method == "max":
        return float(a.max()), float(t.max())
    return float(a.mean()), float(t.mean())

def _mk(name: str, centers: List[float], alpha_vals: List[float],
        tau_vals: Optional[List[float]] = None,
        scatter_vals: Optional[List[float]] = None) -> MaterialBands:
    a = np.asarray(alpha_vals, float)
    t = np.asarray(tau_vals if tau_vals is not None else np.zeros_like(a), float)
    s = None if scatter_vals is None else np.asarray(scatter_vals, float)
    return MaterialBands(name, list(centers), a, t, s).clamp01()

def builtin_library(centers: List[float]) -> Dict[str, MaterialBands]:
    """Small seed library; values are illustrative (add more below or load CSV/JSON)."""
    tgt = np.asarray(centers, float)
    base: Dict[str, MaterialBands] = {}

    # --- Hard surfaces ---
    base["Concrete"] = _mk("Concrete",
        [125, 250, 500, 1000, 2000, 4000],
        [0.01, 0.01, 0.015, 0.02, 0.02, 0.02], scatter_vals=[0.02]*6)
    base["Brick (painted)"] = _mk("Brick (painted)",
        [125, 250, 500, 1000, 2000, 4000],
        [0.01, 0.01, 0.02, 0.02, 0.03, 0.04], scatter_vals=[0.05]*6)
    base["Plaster"] = _mk("Plaster",
        [125, 250, 500, 1000, 2000, 4000],
        [0.01, 0.015, 0.02, 0.02, 0.03, 0.04], scatter_vals=[0.04]*6)

    # --- Floors ---
    base["Wood floor"] = _mk("Wood floor",
        [125, 250, 500, 1000, 2000, 4000],
        [0.15, 0.11, 0.10, 0.07, 0.06, 0.07], scatter_vals=[0.10]*6)
    base["Carpet (heavy, on pad)"] = _mk("Carpet (heavy, on pad)",
        [125, 250, 500, 1000, 2000, 4000],
        [0.08, 0.24, 0.57, 0.69, 0.71, 0.73], scatter_vals=[0.20]*6)

    # --- Ceilings / tiles ---
    base["Acoustic tile (mineral fiber)"] = _mk("Acoustic tile (mineral fiber)",
        [125, 250, 500, 1000, 2000, 4000],
        [0.40, 0.60, 0.70, 0.75, 0.80, 0.85], scatter_vals=[0.10]*6)

    # --- Glazing / panels ---
    base["Glass (single pane)"] = _mk("Glass (single pane)",
        [125, 250, 500, 1000, 2000, 4000],
        [0.35, 0.25, 0.18, 0.12, 0.07, 0.04], tau_vals=[0.25,0.30,0.35,0.40,0.45,0.50], scatter_vals=[0.02]*6)

    # --- Diffuse contents ---
    base["Bookshelves (filled)"] = _mk("Bookshelves (filled)",
        [125, 250, 500, 1000, 2000, 4000],
        [0.20, 0.25, 0.30, 0.35, 0.40, 0.45], scatter_vals=[0.60]*6)

    out: Dict[str, MaterialBands] = {}
    for name, m in base.items():
        if list(m.centers) == list(tgt):
            out[name] = m.clamp01()
        else:
            ai = resample_bands(np.asarray(m.centers,float), m.alpha, tgt)
            ti = resample_bands(np.asarray(m.centers,float), m.tau,   tgt)
            si = None if m.scatter is None else resample_bands(np.asarray(m.centers,float), m.scatter, tgt)
            out[name] = MaterialBands(name, list(tgt), ai, ti, si).clamp01()
    return out

# -------- External libraries (CSV / JSON) --------

def load_csv_library(path: str | pathlib.Path, centers: List[float]) -> Dict[str, MaterialBands]:
    """
    CSV schema:
      name,freq,alpha,tau,scatter
    where freq is semicolon-separated list, same length as alpha/tau/scatter.
    """
    p = pathlib.Path(path)
    if not p.exists(): return {}
    tgt = np.asarray(centers, float)
    out: Dict[str, MaterialBands] = {}
    with p.open("r", newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            name = row["name"].strip()
            freqs = np.array([float(x) for x in row["freq"].split(";")], float)
            alpha = np.array([float(x) for x in row["alpha"].split(";")], float)
            tau   = np.array([float(x) for x in row.get("tau","").split(";")], float) if row.get("tau") else np.zeros_like(alpha)
            scatter = None
            if row.get("scatter"):
                scatter = np.array([float(x) for x in row["scatter"].split(";")], float)
            ai = resample_bands(freqs, alpha, tgt)
            ti = resample_bands(freqs, tau,   tgt)
            si = None if scatter is None else resample_bands(freqs, scatter, tgt)
            out[name] = MaterialBands(name, list(tgt), ai, ti, si).clamp01()
    return out

def load_json_library(path: str | pathlib.Path, centers: List[float]) -> Dict[str, MaterialBands]:
    """
    JSON schema list/dict:
      {"name":"Concrete","freq":[...],"alpha":[...],"tau":[...],"scatter":[...]}
    """
    p = pathlib.Path(path)
    if not p.exists(): return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict): data = list(data.values())
    tgt = np.asarray(centers, float)
    out: Dict[str, MaterialBands] = {}
    for row in data:
        name = str(row["name"])
        freqs = np.asarray(row["freq"], float)
        alpha = np.asarray(row["alpha"], float)
        tau   = np.asarray(row.get("tau", np.zeros_like(alpha)), float)
        scatter = None if "scatter" not in row or row["scatter"] is None else np.asarray(row["scatter"], float)
        ai = resample_bands(freqs, alpha, tgt)
        ti = resample_bands(freqs, tau,   tgt)
        si = None if scatter is None else resample_bands(freqs, scatter, tgt)
        out[name] = MaterialBands(name, list(tgt), ai, ti, si).clamp01()
    return out

def merge_libraries(*libs: Dict[str, MaterialBands]) -> Dict[str, MaterialBands]:
    merged: Dict[str, MaterialBands] = {}
    for lib in libs:
        merged.update(lib)
    return merged
