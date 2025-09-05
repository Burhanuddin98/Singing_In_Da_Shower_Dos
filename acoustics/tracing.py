# acoustics/tracing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import time, math
import numpy as np

from .config import SimConfig
from .physics import (
    unit, reflect, sample_unit_sphere, jitter_direction, cosine_hemisphere_sample,
    pick_spread_fn,              # returns pressure_spread (1/r) or intensity_spread (1/r^2)
    air_lin_vec,                 # 10^(-dB/20) vectorized over bands
    air_db_per_m_iso9613,        # per-band dB/m from ISO 9613 (T, RH, P)
    bin_arrivals_to_ir,
    bin_arrivals_to_ir_banded,
)

# Geometry intersector is provided by acoustics.geometry
# We only need its interface (first_hit, visible, eps)

@dataclass
class Scene:
    mesh: any
    S: np.ndarray
    R: np.ndarray
    inter: "Intersector"
    alpha_face_b: np.ndarray      # (F x B)
    tau_face_b: np.ndarray        # (F x B)
    scatter_face_b: np.ndarray    # (F x B) in [0,1]
    bands: List[float]


def _angle_adjust_alpha(a_vec: np.ndarray, d_in: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Optional: increase absorption at grazing incidence (very simple model)."""
    di = unit(d_in); nn = unit(n)
    cos_inc = -float(np.dot(di, nn))
    cos_inc = max(1e-3, cos_inc)
    # α_eff = 1 - (1-α)^(1/|cosθ|)
    return 1.0 - np.power(np.maximum(1.0 - a_vec, 0.0), 1.0 / cos_inc)


def _pick_outgoing(cfg: SimConfig, d_in: np.ndarray, n: np.ndarray, s_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Choose outgoing direction from a specular/Lambert mixture, then add RMS jitter."""
    if str(getattr(cfg, "brdf_model", "specular+jitter")) == "spec+lambert" and rng.random() < s_prob:
        d_new = cosine_hemisphere_sample(n, rng)  # cosine-weighted lobe around +n
    else:
        d_new = reflect(d_in, n)                  # perfect specular
    sigma = math.radians(float(getattr(cfg, "scattering_deg", 0.0)))
    return unit(jitter_direction(d_new, sigma, rng))


def _air_db_per_m_vec(scene: Scene, cfg: SimConfig) -> np.ndarray:
    """Per-band dB/m array according to selected air model."""
    bands = np.asarray(scene.bands, float)
    air_model = str(getattr(cfg, "air_model", "flat"))
    if air_model == "iso9613":
        return air_db_per_m_iso9613(
            bands,
            float(getattr(cfg, "air_temp_c", 20.0)),
            float(getattr(cfg, "air_rh_pct", 50.0)),
            float(getattr(cfg, "air_pressure_kpa", 101.325)),
        ).astype(np.float32)
    # flat model
    return np.full(bands.shape[0], float(getattr(cfg, "air_db_per_m", 0.0)), dtype=np.float32)


def _do_nee(cfg: SimConfig, scene: Scene, pos: np.ndarray, dvec: np.ndarray, traveled: float,
            amp_after: np.ndarray, ray_weight: float, spread_fn, arrivals: list, b: int, rng: np.random.Generator):
    """
    Next-event estimation toward receiver R. Adds a (time, amp_vec, bounce) arrival if visible.
    Uses simple scheduling:
      - Always for first cfg.nee_bounces bounces
      - Then with probability cfg.nee_prob if cfg.nee_all_bounces
    """
    nee_bounces = int(getattr(cfg, "nee_bounces", 0))
    nee_all_b = bool(getattr(cfg, "nee_all_bounces", False))
    nee_prob = float(getattr(cfg, "nee_prob", 0.0))

    do = False
    if b < nee_bounces:
        do = True
    elif nee_all_b and (rng.random() < nee_prob):
        do = True
    if not do:
        return

    toR = scene.R - pos
    dist = float(np.linalg.norm(toR))
    if dist <= 1e-6:
        return
    dir_to_R = toR / dist

    # avoid extreme backfacing NEE spikes
    if float(np.dot(dir_to_R, dvec)) < -0.25:
        return

    if not scene.inter.visible(pos, scene.R):
        return

    # Spreading + air (banded)
    spread = spread_fn(dist)
    db_per_m_vec = _air_db_per_m_vec(scene, cfg)
    air_vec = air_lin_vec(db_per_m_vec, dist)

    w_vec = amp_after * spread * air_vec
    if bool(getattr(cfg, "phys_normalization", False)):
        w_vec = w_vec * ray_weight

    # MIS-lite guard (optional)
    if bool(getattr(cfg, "nee_mis", False)) and int(getattr(cfg, "rays", 0)) > 0:
        w_vec = w_vec / float(getattr(cfg, "rays"))

    t = (traveled + dist) / float(getattr(cfg, "c", 343.0))
    arrivals.append((t, w_vec.copy(), b + 1))


def path_trace(scene: Scene, cfg: SimConfig):
    """
    Monte Carlo acoustic path tracer.
    Returns:
      - h: IR; shape (N,) for broadband or (B, N) for banded
      - arrivals: list of (t, amplitude or amplitude_vec, bounce_count)
    """
    rng = np.random.default_rng(int(getattr(cfg, "rng_seed", 0)))
    start_time = time.time()

    nb = int(len(scene.bands))
    assert nb >= 1, "Scene must have at least one band."
    spread_fn = pick_spread_fn(str(getattr(cfg, "scale_convention", "pressure")))

    # Uniform sphere sampling PDF and MC weight
    pdf_dir = 1.0 / (4.0 * math.pi)
    ray_count = int(max(1, int(getattr(cfg, "rays", 1024))))
    ray_weight = 1.0 / (ray_count * pdf_dir)

    arrivals: List[Tuple[float, np.ndarray, int]] = []

    # Deterministic direct path (if requested)
    if bool(getattr(cfg, "include_direct", True)) and str(getattr(cfg, "direct_mode", "deterministic")) == "deterministic":
        if scene.inter.visible(scene.S, scene.R):
            d = float(np.linalg.norm(scene.R - scene.S))
            spread = spread_fn(d)
            dbpm = _air_db_per_m_vec(scene, cfg)
            air_vec = air_lin_vec(dbpm, d)
            w0 = spread * air_vec
            if bool(getattr(cfg, "phys_normalization", False)):
                w0 = w0 * ray_weight
            arrivals.append((d / float(getattr(cfg, "c", 343.0)), w0.copy(), 0))

    # Ray directions
    dirs = sample_unit_sphere(ray_count, rng)
    eps = float(getattr(scene.inter, "eps", 1e-4))
    max_b = int(getattr(cfg, "max_bounces", 8))
    time_budget = float(getattr(cfg, "time_budget", 8.0))
    rr_enabled = bool(getattr(cfg, "russian_roulette", True))
    min_amp = float(getattr(cfg, "min_amp", 1e-6))
    angle_abs = bool(getattr(cfg, "angle_absorption", False))
    brdf_model = str(getattr(cfg, "brdf_model", "specular+jitter"))

    for i in range(ray_count):
        if (time.time() - start_time) > time_budget:
            break

        pos = scene.S.copy()
        dvec = dirs[i].copy()
        traveled = 0.0
        amp_reflect = np.ones(nb, dtype=np.float32)

        for b in range(max_b):
            # First hit
            hit, fidx, dist = scene.inter.first_hit(pos + dvec * eps, dvec)
            if hit is None or fidx is None or dist is None:
                # If direct is sampled, try a NEE-like add when ray escapes
                if bool(getattr(cfg, "include_direct", True)) and str(getattr(cfg, "direct_mode", "deterministic")) == "sampled":
                    _do_nee(cfg, scene, pos, dvec, traveled, amp_reflect, ray_weight, spread_fn, arrivals, b=-1, rng=rng)
                break

            fidx = int(fidx)
            dist = float(dist)
            traveled += dist

            n = scene.mesh.face_normals[fidx]
            a_vec = scene.alpha_face_b[fidx].astype(np.float32)
            t_vec = scene.tau_face_b[fidx].astype(np.float32)

            a_eff = _angle_adjust_alpha(a_vec, dvec, n) if angle_abs else a_vec

            # Energy reflectance -> amplitude reflectance
            r2 = np.clip(1.0 - a_eff - t_vec, 0.0, 1.0)
            Rf_vec = np.sqrt(r2, dtype=np.float32)
            amp_after = (amp_reflect * Rf_vec).astype(np.float32)

            # Early prune on vector amplitude
            if float(np.max(amp_after)) < min_amp:
                break

            # NEE contribution to receiver
            _do_nee(cfg, scene, hit, dvec, traveled, amp_after, ray_weight, spread_fn, arrivals, b=b, rng=rng)

            # Russian roulette after a couple of bounces (band-agnostic heuristic)
            if rr_enabled and b >= 2:
                p_survive = float(np.clip(np.max(amp_after) / max(min_amp, 1e-9), 0.05, 0.95))
                if rng.random() > p_survive:
                    break
                amp_after = amp_after / p_survive

            # Scatter probability from library bands (mean) or fallback to cfg.scatter_ratio
            if scene.scatter_face_b is not None and scene.scatter_face_b.size:
                s_lib = float(np.clip(scene.scatter_face_b[fidx].mean(), 0.0, 1.0))
            else:
                s_lib = 0.0
            s_user = float(np.clip(float(getattr(cfg, "scatter_ratio", 0.0)), 0.0, 1.0))
            if brdf_model == "spec+lambert":
                s_prob = float(np.clip(max(s_lib, s_user), 0.0, 1.0))
            else:
                s_prob = 0.0  # specular+jitter model ignores lambert share

            # Choose outgoing direction and continue
            dvec = _pick_outgoing(cfg, dvec, n, s_prob, rng)
            pos = hit + dvec * eps
            amp_reflect = amp_after

    # --- Bin to IR ---
    fs = int(getattr(cfg, "fs", 48000))
    dur = float(getattr(cfg, "duration_s", 3.0))
    if nb == 1:
        # Convert vector arrivals into scalar by taking first element
        arr_scalar: List[Tuple[float, float, int]] = []
        for (t, avec, b) in arrivals:
            a = float(np.asarray(avec).reshape(-1)[0])
            arr_scalar.append((t, a, b))
        h = bin_arrivals_to_ir(arr_scalar, fs, dur)
        return h, arrivals

    H = bin_arrivals_to_ir_banded(arrivals, fs, dur, nb)
    return H, arrivals


def trace_preview_paths(scene: Scene, max_bounces: int, n_paths: int, seed: int, jitter_deg: float):
    """Generate a small subset of polyline paths for visualization."""
    rng = np.random.default_rng(seed)
    sigma = math.radians(float(jitter_deg))
    dirs = sample_unit_sphere(int(n_paths), rng)

    polylines: List[np.ndarray] = []
    to_rec: List[np.ndarray] = []

    eps = float(getattr(scene.inter, "eps", 1e-4))
    for i in range(int(n_paths)):
        pos = scene.S.copy()
        dvec = dirs[i].copy()
        pts = [pos.copy()]

        for b in range(int(max_bounces)):
            hit, fidx, dist = scene.inter.first_hit(pos + dvec * eps, dvec)
            if hit is None:
                break
            pts.append(hit.copy())
            if scene.inter.visible(hit, scene.R):
                to_rec.append(np.vstack([hit.copy(), scene.R.copy()]))

            n = scene.mesh.face_normals[int(fidx)]
            dvec = jitter_direction(reflect(dvec, n), sigma, rng)
            pos = hit + dvec * eps

        if len(pts) > 1:
            polylines.append(np.vstack(pts))

    return polylines, to_rec
