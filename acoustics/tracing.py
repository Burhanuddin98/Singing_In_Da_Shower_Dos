# acoustics/tracing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time, math
import numpy as np

from .config import SimConfig
from .physics import (
    unit, reflect, sample_unit_sphere, jitter_direction, cosine_hemisphere_sample,
    pressure_spread, intensity_spread, pick_spread_fn,
    air_lin_vec, air_db_per_m_iso9613,
    bin_arrivals_to_ir, bin_arrivals_to_ir_banded,
)

# Geometry intersector is provided by acoustics.geometry
# We only need its interface (first_hit, visible)
@dataclass
class Scene:
    mesh: any
    S: np.ndarray
    R: np.ndarray
    inter: any  # Intersector
    # Per-face, per-band (F x B). For broadband tracing, we pass B=1 arrays.
    alpha_face_b: np.ndarray
    tau_face_b: np.ndarray
    bands: List[float]
    # Optional per-face, per-band scattering [0..1]
    scatter_face_b: Optional[np.ndarray] = None


def _angle_adjust_alpha(a_vec: np.ndarray, d_in: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Increase absorption at grazing incidence (optional model)."""
    cos_inc = -float(np.dot(unit(d_in), unit(n)))
    cos_inc = max(1e-3, cos_inc)
    # α_eff = 1 - (1-α)^(1/|cosθ|)
    return 1.0 - np.power((1.0 - a_vec), 1.0 / cos_inc)


def _pick_outgoing(cfg: SimConfig, d_in: np.ndarray, n: np.ndarray, s_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Specular + Lambert mixture."""
    if cfg.brdf_model == "spec+lambert" and rng.random() < s_prob:
        d_new = cosine_hemisphere_sample(n, rng)
    else:
        d_new = reflect(d_in, n)
    # small RMS jitter to avoid coherent caustics / aliasing
    sigma = math.radians(float(cfg.scattering_deg))
    return jitter_direction(d_new, sigma, rng)


def _do_nee(cfg: SimConfig, scene: Scene, pos: np.ndarray, dvec: np.ndarray, traveled: float,
            amp_after: np.ndarray, ray_weight: float, spread_fn, arrivals: list, b: int):
    """
    Next-event estimation toward receiver R.
    We approximate receiver as a small disk (radius cfg.receiver_radius_m).
    """
    # Bounce scheduling / probability
    do = False
    if b < int(cfg.nee_bounces):
        do = True
    elif cfg.nee_all_bounces and (np.random.random() < float(cfg.nee_prob)):
        do = True
    if not do:
        return

    # Vector to receiver
    toR = scene.R - pos
    dist = float(np.linalg.norm(toR))
    if dist <= 1e-6:
        return
    dir_to_R = toR / dist

    # Must roughly align with current direction to avoid backfacing NEE spikes
    if float(np.dot(dir_to_R, dvec)) < -0.25:  # heuristic, same as original spirit
        return

    # Visibility check
    if not scene.inter.visible(pos, scene.R):
        return

    # Spreading + air (banded)
    spread = spread_fn(dist)
    if cfg.air_model == "iso9613":
        db_per_m_vec = air_db_per_m_iso9613(np.asarray(scene.bands, float), cfg.air_temp_c, cfg.air_rh_pct, cfg.air_pressure_kpa)
    else:
        db_per_m_vec = np.full(len(scene.bands), float(cfg.air_db_per_m), dtype=np.float32)
    air_vec = air_lin_vec(db_per_m_vec, dist)

    w_vec = amp_after * spread * air_vec
    if cfg.phys_normalization:
        w_vec = w_vec * ray_weight

    # Optional MIS-lite: divide NEE leg by number of rays (prevents over-bright early spikes)
    if cfg.nee_mis and cfg.rays > 0:
        w_vec = w_vec / float(cfg.rays)

    t = (traveled + dist) / float(cfg.c)
    arrivals.append((t, w_vec.copy(), b + 1))


def path_trace(scene: Scene, cfg: SimConfig):
    """
    Monte Carlo acoustic path tracer.
    Returns:
      - h: IR; shape (N,) for broadband or (B, N) for banded
      - arrivals: list of (t, amplitude or amplitude_vec, bounce_count)
    """
    rng = np.random.default_rng(cfg.rng_seed)
    start_time = time.time()

    nb = len(scene.bands)
    spread_fn = pick_spread_fn(cfg.scale_convention)

    # Uniform sphere sampling PDF
    pdf_dir = 1.0 / (4.0 * math.pi)
    ray_weight = 1.0 / (max(1, int(cfg.rays)) * pdf_dir)

    arrivals: List[Tuple[float, np.ndarray, int]] = []

    # Direct (deterministic vs sampled)
    if cfg.include_direct and cfg.direct_mode == "deterministic":
        if scene.inter.visible(scene.S, scene.R):
            d = float(np.linalg.norm(scene.R - scene.S))
            spread = spread_fn(d)
            if cfg.air_model == "iso9613":
                dbpm = air_db_per_m_iso9613(np.asarray(scene.bands, float), cfg.air_temp_c, cfg.air_rh_pct, cfg.air_pressure_kpa)
            else:
                dbpm = np.full(nb, float(cfg.air_db_per_m), dtype=np.float32)
            air_vec = air_lin_vec(dbpm, d)
            w0 = spread * air_vec
            if cfg.phys_normalization:
                w0 = w0 * ray_weight
            arrivals.append((d / float(cfg.c), w0.copy(), 0))

    # Rays
    dirs = sample_unit_sphere(int(cfg.rays), rng)
    for i in range(int(cfg.rays)):
        if (time.time() - start_time) > float(cfg.time_budget):
            break

        pos = scene.S.copy()
        dvec = dirs[i].copy()
        traveled = 0.0

        amp_reflect = np.ones(nb, dtype=np.float32)

        for b in range(int(cfg.max_bounces)):
            # First hit
            hit, fidx, dist = scene.inter.first_hit(pos + dvec * scene.inter.eps, dvec)
            if hit is None or fidx is None or dist is None:
                # If direct was "sampled", add the current ray's direct when it escapes
                if cfg.include_direct and cfg.direct_mode == "sampled":
                    # approximate escape -> receiver visibility check
                    _do_nee(cfg, scene, pos, dvec, traveled, amp_reflect, ray_weight, spread_fn, arrivals, b=-1)
                break

            traveled += float(dist)
            n = scene.mesh.face_normals[int(fidx)]
            a_vec = scene.alpha_face_b[int(fidx)].astype(np.float32)
            t_vec = scene.tau_face_b[int(fidx)].astype(np.float32)

            # Angle-dependent absorption (optional)
            if cfg.angle_absorption:
                a_eff = _angle_adjust_alpha(a_vec, dvec, n)
            else:
                a_eff = a_vec

            # Amplitude reflectance
            r2 = np.maximum(0.0, 1.0 - a_eff - t_vec)  # energy reflectance
            Rf_vec = np.sqrt(r2).astype(np.float32)
            amp_after = amp_reflect * Rf_vec

            # Early prune
            if float(np.max(amp_after)) < float(cfg.min_amp):
                break

            # Next-event estimation (receiver connection)
            _do_nee(cfg, scene, hit, dvec, traveled, amp_after, ray_weight, spread_fn, arrivals, b)

            # Russian roulette on total remaining amplitude (band-agnostic heuristic)
            if cfg.russian_roulette and b >= 2:
                p_survive = float(np.clip(np.max(amp_after), 0.05, 0.99))
                if rng.random() > p_survive:
                    break
                amp_after = amp_after / p_survive

            # Choose outgoing direction: mixture specular + Lambert
            if scene.scatter_face_b is not None:
                # mean scatter across bands as probability
                s_prob = float(np.clip(float(scene.scatter_face_b[int(fidx)].mean()), 0.0, 1.0))
            else:
                s_prob = float(np.clip(cfg.scatter_ratio, 0.0, 1.0))

            dvec = _pick_outgoing(cfg, dvec, n, s_prob, rng)
            pos = hit + dvec * scene.inter.eps
            amp_reflect = amp_after

    # Bin to IR
    if nb == 1:
        # unpack scalar per-ray arrivals
        arr_scalar = []
        for (t, avec, b) in arrivals:
            a = float(np.asarray(avec).reshape(-1)[0])
            arr_scalar.append((t, a, b))
        h = bin_arrivals_to_ir(arr_scalar, int(cfg.fs), float(cfg.duration_s))
        return h, arrivals

    H = bin_arrivals_to_ir_banded(arrivals, int(cfg.fs), float(cfg.duration_s), nb)
    return H, arrivals


def trace_preview_paths(scene: Scene, max_bounces: int, n_paths: int, seed: int, jitter_deg: float):
    """Generate a small subset of polyline paths for visualization."""
    rng = np.random.default_rng(seed)
    sigma = math.radians(float(jitter_deg))
    dirs = sample_unit_sphere(int(n_paths), rng)

    polylines = []
    to_rec = []

    for i in range(int(n_paths)):
        pos = scene.S.copy()
        dvec = dirs[i].copy()
        pts = [pos.copy()]

        for b in range(int(max_bounces)):
            hit, fidx, dist = scene.inter.first_hit(pos + dvec * scene.inter.eps, dvec)
            if hit is None:
                break
            pts.append(hit.copy())
            if scene.inter.visible(hit, scene.R):
                to_rec.append(np.vstack([hit.copy(), scene.R.copy()]))

            n = scene.mesh.face_normals[int(fidx)]
            dvec = jitter_direction(reflect(dvec, n), sigma, rng)
            pos = hit + dvec * scene.inter.eps

        if len(pts) > 1:
            polylines.append(np.vstack(pts))

    return polylines, to_rec
