# acoustics/caching.py
from __future__ import annotations
import hashlib
import numpy as np
import streamlit as st

from .config import SimConfig, MaterialAuto, OCTAVE_CENTERS
from .geometry import Intersector, face_connected_components, build_trimesh_from_arrays
from .tracing import Scene, path_trace, trace_preview_paths


# -----------------------------
# Hashing
# -----------------------------

def mesh_hash_from_arrays(V: np.ndarray, F: np.ndarray) -> str:
    """Stable hash for (V, F) arrays to key intersectors/caches."""
    return hashlib.sha1(
        V.astype(np.float32).tobytes() + F.astype(np.int32).tobytes()
    ).hexdigest()


# -----------------------------
# Connected components (per-element grouping)
# -----------------------------

@st.cache_data(show_spinner=False)
def build_components_cached(V: np.ndarray, F: np.ndarray):
    mesh = build_trimesh_from_arrays(V, F)
    return face_connected_components(mesh)


# -----------------------------
# Intersector cache (Embree / Triangle)
# -----------------------------

@st.cache_resource(show_spinner=False)
def build_intersector_cached(mesh_key: str, V: np.ndarray, F: np.ndarray):
    mesh = build_trimesh_from_arrays(V, F)
    return Intersector.build(mesh)


# -----------------------------
# Auto-material seeding (per-face α)
# -----------------------------

def _auto_alpha_per_face(mesh, mats: MaterialAuto) -> np.ndarray:
    """Heuristic: floors/ceilings/walls by z-centroid and |n_z|."""
    F = mesh.faces.shape[0]
    alpha = np.full(F, mats.alpha_default, dtype=np.float32)

    # z-centroid bands
    centroids_z = mesh.triangles_center[:, 2]
    nz = np.abs(mesh.face_normals[:, 2])

    zl, zh = np.percentile(
        centroids_z, [mats.percentile_margin, 100.0 - mats.percentile_margin]
    )

    alpha[centroids_z <= zl] = mats.alpha_floor
    alpha[centroids_z >= zh] = mats.alpha_ceiling
    alpha[nz <= mats.nz_thresh] = mats.alpha_walls
    return alpha


@st.cache_data(show_spinner=False)
def auto_alpha_cached(V: np.ndarray, F: np.ndarray, params: tuple):
    """Cache-friendly wrapper around _auto_alpha_per_face."""
    mesh = build_trimesh_from_arrays(V, F)
    mats = MaterialAuto(*params)
    return _auto_alpha_per_face(mesh, mats)


# -----------------------------
# Band expansion helpers
# -----------------------------

def expand_broadband_to_bands(alpha_face: np.ndarray,
                              tau_face: np.ndarray,
                              bands: list[int]):
    """Broadcast 1D per-face α/τ to (F x B)."""
    nb = len(bands)
    Fcount = alpha_face.shape[0]
    A = np.tile(alpha_face.reshape(Fcount, 1), (1, nb))
    T = np.tile(tau_face.reshape(Fcount, 1), (1, nb))
    return A.astype(np.float32), T.astype(np.float32)


# -----------------------------
# Main cached tracing entry point
# -----------------------------

@st.cache_data(show_spinner=True)
def trace_cached(cfg_key: tuple,
                 V: np.ndarray, F: np.ndarray,
                 S: np.ndarray, R: np.ndarray,
                 alpha_face: np.ndarray, tau_face: np.ndarray,
                 band_mode: str,
                 alpha_face_b_override: np.ndarray | None = None,
                 tau_face_b_override: np.ndarray | None = None,
                 bands_override: list[int] | None = None):
    """
    Build scene, run tracer, and return:
      - h (1D or [B,N] if banded)
      - arrivals (debug)
      - polylines (for preview)
    Cache invalidates on any change to inputs.
    """
    mesh = build_trimesh_from_arrays(V, F)
    inter = build_intersector_cached(mesh_hash_from_arrays(V, F), V, F)

    # Rebuild cfg from cfg_key (ordered as dataclass fields in app.py)
    cfg = SimConfig(*cfg_key)

    # Bands
    if band_mode == "octave":
        if (alpha_face_b_override is not None) and (tau_face_b_override is not None):
            bands = bands_override if bands_override is not None else OCTAVE_CENTERS
            alpha_b = alpha_face_b_override.astype(np.float32)
            tau_b   = tau_face_b_override.astype(np.float32)
        else:
            bands = OCTAVE_CENTERS
            alpha_b, tau_b = expand_broadband_to_bands(alpha_face, tau_face, bands)
    else:
        bands = [1000]
        alpha_b = alpha_face.reshape(-1, 1).astype(np.float32)
        tau_b   = tau_face.reshape(-1, 1).astype(np.float32)

    # Scene
    scene = Scene(
        mesh=mesh, S=S, R=R, inter=inter,
        alpha_face_b=alpha_b, tau_face_b=tau_b, bands=bands
    )

    # Trace
    h, arrivals = path_trace(scene, cfg)

    # Preview paths (fixed budget)
    polylines, _ = trace_preview_paths(
        scene, cfg.max_bounces, 300, cfg.rng_seed + 1234, cfg.scattering_deg
    )

    return h, arrivals, polylines
