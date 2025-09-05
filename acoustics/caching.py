# acoustics/caching.py
from __future__ import annotations

from typing import Optional, Tuple, List
import hashlib

import numpy as np
import streamlit as st
import trimesh

from .config import SimConfig, MaterialAuto
from .geometry import Intersector, face_connected_components
from .tracing import Scene, path_trace, trace_preview_paths
from .bands import standard_centers


# ------------ Helpers / hashing ------------

def _build_trimesh(V: np.ndarray, F: np.ndarray) -> "trimesh.Trimesh":
    if trimesh is None:
        raise RuntimeError("trimesh is not available")
    return trimesh.Trimesh(vertices=np.asarray(V), faces=np.asarray(F), process=False)


def mesh_hash_from_arrays(V: np.ndarray, F: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(V, dtype=np.float32).tobytes())
    h.update(np.asarray(F, dtype=np.int32).tobytes())
    return h.hexdigest()


# ------------ Caching primitives ------------

@st.cache_data(show_spinner=False)
def build_components_cached(V: np.ndarray, F: np.ndarray) -> List[np.ndarray]:
    mesh = _build_trimesh(V, F)
    return face_connected_components(mesh)


@st.cache_resource(show_spinner=False)
def build_intersector_cached(mesh_key: str, V: np.ndarray, F: np.ndarray) -> Intersector:
    mesh = _build_trimesh(V, F)
    return Intersector.build(mesh)


# --- Auto-material seeding (broadband α only; τ=0) ---

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
    scatter_face_b_override: Optional[np.ndarray] = None,
    bands_override: Optional[List[float]]       = None,
):
    """
    Returns (h, arrivals, polylines)
      - h: (N,) for broadband, or (B, N) for banded
      - arrivals: list of (t, amp or amp_vec, bounce)
      - polylines: list of polyline arrays for preview
    """
    cfg = SimConfig(*cfg_key)

    # intersector & mesh
    inter = build_intersector_cached(mesh_hash_from_arrays(V, F), V, F)
    mesh = _build_trimesh(V, F)

    # Choose band centers
    bands = list(bands_override) if bands_override is not None else list(standard_centers(band_mode))
    nb = max(1, len(bands))

    # Build per-face per-band α/τ
    if alpha_face_b_override is not None and tau_face_b_override is not None:
        A_b = np.asarray(alpha_face_b_override, dtype=np.float32)
        T_b = np.asarray(tau_face_b_override, dtype=np.float32)
    else:
        # broadcast broadband → bands
        A_b = np.tile(np.asarray(alpha_face, np.float32)[:, None], (1, nb))
        T_b = np.tile(np.asarray(tau_face,  np.float32)[:, None], (1, nb))

    # ---- Scatter (banded) ----
    if scatter_face_b_override is not None:
        S_b = np.asarray(scatter_face_b_override, dtype=np.float32)
        if S_b.shape != A_b.shape:
            S_b = np.broadcast_to(S_b, A_b.shape).astype(np.float32)
    else:
        S_b = np.zeros_like(A_b, dtype=np.float32)

    scene = Scene(
        mesh=mesh,
        S=np.asarray(S, float),
        R=np.asarray(R, float),
        inter=inter,
        alpha_face_b=A_b,
        tau_face_b=T_b,
        scatter_face_b=S_b,
        bands=bands,
    )

    h, arrivals = path_trace(scene, cfg)

    # A small set of preview paths (independent RNG)
    try:
        polylines, _ = trace_preview_paths(
            scene,
            int(getattr(cfg, "max_bounces", 8)),
            min(300, 150),
            int(getattr(cfg, "rng_seed", 0)) + 1234,
            float(getattr(cfg, "scattering_deg", 0.0)),
        )
    except Exception:
        polylines = []

    return h, arrivals, polylines
