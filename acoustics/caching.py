# acoustics/caching.py
from __future__ import annotations
from typing import List, Tuple, Optional
import hashlib
import numpy as np
import streamlit as st

try:
    import trimesh
except Exception:
    trimesh = None  # app guards this already

from .config import SimConfig, MaterialAuto, OCTAVE_CENTERS
from .geometry import Intersector, face_connected_components
from .tracing import Scene, path_trace, trace_preview_paths

# ------------ Helpers / hashing ------------

def mesh_hash_from_arrays(V: np.ndarray, F: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(V, dtype=np.float32).tobytes())
    h.update(np.asarray(F, dtype=np.int32).tobytes())
    return h.hexdigest()

def _build_trimesh(V: np.ndarray, F: np.ndarray) -> "trimesh.Trimesh":
    if trimesh is None:
        raise RuntimeError("trimesh is not available")
    return trimesh.Trimesh(vertices=np.asarray(V), faces=np.asarray(F), process=False)

# ------------ Caching primitives ------------

@st.cache_data(show_spinner=False)
def build_components_cached(V: np.ndarray, F: np.ndarray) -> List[np.ndarray]:
    mesh = _build_trimesh(V, F)
    return face_connected_components(mesh)

@st.cache_resource(show_spinner=False)
def build_intersector_cached(mesh_key: str, V: np.ndarray, F: np.ndarray) -> Intersector:
    mesh = _build_trimesh(V, F)
    return Intersector.build(mesh)

# Auto-material seeding (copied logic from original)
def _auto_alpha_per_face(mesh: "trimesh.Trimesh", mats: MaterialAuto) -> np.ndarray:
    F = mesh.faces.shape[0]
    alpha = np.full(F, float(mats.alpha_default), dtype=np.float32)
    centroids_z = mesh.triangles_center[:, 2]
    nz = np.abs(mesh.face_normals[:, 2])

    zl, zh = np.percentile(centroids_z, [mats.percentile_margin, 100 - mats.percentile_margin])
    alpha[centroids_z <= zl] = float(mats.alpha_floor)
    alpha[centroids_z >= zh] = float(mats.alpha_ceiling)
    alpha[nz <= float(mats.nz_thresh)] = float(mats.alpha_walls)
    return alpha

@st.cache_data(show_spinner=False)
def auto_alpha_cached(V: np.ndarray, F: np.ndarray, params: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    mesh = _build_trimesh(V, F)
    mats = MaterialAuto(*params)
    return _auto_alpha_per_face(mesh, mats)

def _expand_broadband_to_bands(alpha_face: np.ndarray, tau_face: np.ndarray, bands: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    nb = len(bands)
    F = len(alpha_face)
    A = np.tile(np.asarray(alpha_face, dtype=np.float32)[:, None], (1, nb))
    T = np.tile(np.asarray(tau_face, dtype=np.float32)[:, None],   (1, nb))
    return A, T

# ------------ Main cached tracer ------------

@st.cache_data(show_spinner=True)
def trace_cached(
    cfg_key: tuple,
    V: np.ndarray, F: np.ndarray,
    S: np.ndarray, R: np.ndarray,
    alpha_face: np.ndarray, tau_face: np.ndarray,
    *,
    band_mode: str = "broadband",
    alpha_face_b_override: Optional[np.ndarray] = None,
    tau_face_b_override: Optional[np.ndarray]   = None,
    bands_override: Optional[List[float]]       = None,
    scatter_face_b_override: Optional[np.ndarray] = None,
):
    """
    Cached high-level wrapper that:
      - builds the intersector
      - prepares per-band material arrays
      - runs the tracer
      - produces a small set of preview polylines

    Returns (h, arrivals, polylines)
      h: (N,) for broadband or (B, N) for banded
    """
    # Rebuild SimConfig from cfg_key (field order must match SimConfig definition)
    cfg = SimConfig(*cfg_key)

    # Mesh / intersector
    inter = build_intersector_cached(mesh_hash_from_arrays(V, F), V, F)
    mesh = _build_trimesh(V, F)

    # Decide bands
    if band_mode == "octave":
        bands = list(bands_override) if (bands_override is not None) else list(OCTAVE_CENTERS)
        if alpha_face_b_override is not None and tau_face_b_override is not None:
            A_b = np.asarray(alpha_face_b_override, dtype=np.float32)
            T_b = np.asarray(tau_face_b_override, dtype=np.float32)
        else:
            A_b, T_b = _expand_broadband_to_bands(alpha_face, tau_face, bands)
    else:
        # broadband → just one "band" to keep the math unified
        bands = [1000.0]
        A_b, T_b = _expand_broadband_to_bands(alpha_face, tau_face, bands)

    # Scene
    scene = Scene(
        mesh=mesh,
        S=np.asarray(S, dtype=float),
        R=np.asarray(R, dtype=float),
        inter=inter,
        alpha_face_b=A_b,
        tau_face_b=T_b,
        bands=bands,
        scatter_face_b=np.asarray(scatter_face_b_override, dtype=np.float32) if scatter_face_b_override is not None else None,
    )

    # Trace
    h, arrivals = path_trace(scene, cfg)

    # Quick polyline preview (bounded count)
    polylines, _ = trace_preview_paths(scene, int(cfg.max_bounces), min(300, 3 * 50), int(cfg.rng_seed) + 1234, float(cfg.scattering_deg))

    return h, arrivals, polylines
