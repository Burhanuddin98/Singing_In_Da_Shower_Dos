# acoustics/tracing.py
from __future__ import annotations
import math, time
import numpy as np
from dataclasses import dataclass

from .config import SimConfig, AIR_DB_PER_M_OCT
from .geometry import Intersector
from .physics import (
    unit, reflect, sample_unit_sphere, jitter_direction, cosine_hemisphere_sample,
    pick_spread_fn, air_lin, air_lin_vec,
    bin_arrivals_to_ir, bin_arrivals_to_ir_banded,
)

@dataclass
class Scene:
    mesh: any
    S: np.ndarray
    R: np.ndarray
    inter: Intersector
    # Banded per-face coefficients (F x B); if band_mode="broadband", B==1
    alpha_face_b: np.ndarray
    tau_face_b: np.ndarray
    bands: list  # length B

def _add_synth_tail_inplace(h_1d: np.ndarray, cfg: SimConfig, seed_offset: int = 0):
    """Optional gentle late tail to avoid silent ends when sparse."""
    if h_1d.size == 0:
        return
    # Compute simple EDC to judge sparsity
    e = h_1d.astype(np.float64) ** 2
    edc = np.cumsum(e[::-1])[::-1]
    if edc.size == 0 or edc[0] <= 0:
        return
    edc /= edc[0]
    db = 10 * np.log10(np.maximum(edc, 1e-20))
    t = np.arange(len(h_1d)) / cfg.fs
    if (db <= -40).any() and t[np.argmax(db <= -40)] < min(0.2, 0.5 * cfg.duration_s):
        tau = cfg.synth_rt60_s / 6.91
        rng = np.random.default_rng(cfg.rng_seed + 123 + seed_offset)
        tail = rng.normal(size=h_1d.size).astype(np.float32)
        tail *= np.exp(-t / tau).astype(np.float32)
        splice = int(0.15 * cfg.duration_s * cfg.fs)
        level = 10 ** (cfg.synth_tail_level_db / 20.0)
        denom = np.max(np.abs(tail[:splice])) + 1e-12
        h_1d += tail * (level / denom)

def path_trace(scene: Scene, cfg: SimConfig):
    """Trace rays and return (IR, arrivals). IR is 1D or [B,N] depending on bands."""
    rng = np.random.default_rng(cfg.rng_seed)
    sigma = math.radians(cfg.scattering_deg)

    B = len(scene.bands)
    spread_fn = pick_spread_fn(cfg.scale_convention)

    # Uniform sphere sampling: pdf = 1/(4π)
    pdf_dir = 1.0 / (4.0 * math.pi)
    ray_weight = (1.0 / (cfg.rays * pdf_dir)) if cfg.phys_normalization else 1.0

    # Receiver disk solid-angle -> PDF over sphere (MIS-lite)
    def receiver_pdf_solid_angle(d: float, r_disk: float) -> float:
        if r_disk <= 0 or d <= 0:
            return 0.0
        omega = math.pi * (r_disk ** 2) / (d * d)
        return max(0.0, min(omega / (4.0 * math.pi), 1.0))

    def mis_weight(pdf_a: float, pdf_b: float) -> float:
        denom = pdf_a + pdf_b
        return (pdf_a / denom) if denom > 0 else 0.5

    arrivals = []  # list of (time_s, amp_scalar or amp_vec[B], bounces)
    start_time = time.time()

    # ---- Direct contribution ----
    if cfg.include_direct and scene.inter.visible(scene.S, scene.R):
        d = float(np.linalg.norm(scene.R - scene.S))
        if B == 1:
            w0 = spread_fn(d) * air_lin(cfg.air_db_per_m, d)
            if cfg.direct_mode == "sampled" and cfg.phys_normalization:
                w0 *= ray_weight
            arrivals.append((d / cfg.c, float(w0), 0))
        else:
            db_vec = np.array([AIR_DB_PER_M_OCT.get(f, cfg.air_db_per_m) for f in scene.bands], dtype=np.float32)
            wv = spread_fn(d) * air_lin_vec(db_vec, d)
            if cfg.direct_mode == "sampled" and cfg.phys_normalization:
                wv *= ray_weight
            arrivals.append((d / cfg.c, wv.astype(np.float32), 0))

    # ---- Primary rays ----
    dirs = sample_unit_sphere(cfg.rays, rng)

    for i in range(cfg.rays):
        if time.time() - start_time > cfg.time_budget:
            break

        # State queue: (pos, dvec, amp_vec[B], traveled, bounces, spawned_T: bool)
        amp0 = np.ones(B, dtype=np.float32)
        queue = [(scene.S.copy(), dirs[i], amp0, 0.0, 0, False)]

        while queue:
            pos, dvec, amp_reflect, traveled, b, spawned_T = queue.pop()

            if b >= cfg.max_bounces:
                continue

            hit, fidx, dist = scene.inter.first_hit(pos + dvec * scene.inter.eps, dvec)
            if hit is None or fidx is None or dist is None:
                continue

            traveled2 = traveled + float(dist)

            a_vec = scene.alpha_face_b[fidx].astype(np.float32)  # (B,)
            t_vec = scene.tau_face_b[fidx].astype(np.float32)    # (B,)
            # amplitude reflectance per band
            Rf_vec = np.sqrt(np.maximum(0.0, 1.0 - a_vec - t_vec)).astype(np.float32)
            amp_after = (amp_reflect * Rf_vec).astype(np.float32)

            if float(np.max(amp_after)) < cfg.min_amp:
                continue

            # ---- Next-Event Estimation to receiver (MIS-lite) ----
            attempt_nee = False
            inv_p = 1.0

            if b < cfg.nee_bounces:
                attempt_nee = True
            else:
                if cfg.nee_all_bounces and cfg.nee_prob > 0.0:
                    if rng.random() < cfg.nee_prob:
                        attempt_nee = True
                        inv_p = 1.0 / max(cfg.nee_prob, 1e-9)

            if attempt_nee:
                toR = scene.R - hit
                dist2 = float(np.dot(toR, toR))
                if dist2 <= (cfg.nee_max_dist_m ** 2):
                    toR_hat = unit(toR)
                    if float(np.dot(toR_hat, dvec)) >= cfg.nee_min_dot and scene.inter.visible(hit, scene.R):
                        leg = float(math.sqrt(dist2))
                        t = (traveled2 + leg) / cfg.c

                        # PDFs for MIS-lite (conservative)
                        pdf_pt = pdf_dir
                        pdf_nee = receiver_pdf_solid_angle(leg, cfg.receiver_radius_m)
                        w_nee = mis_weight(pdf_nee, pdf_pt) if cfg.nee_mis else 1.0

                        if B == 1:
                            base = float(amp_after[0]) * spread_fn(traveled2 + leg) * air_lin(cfg.air_db_per_m, traveled2 + leg)
                            contrib = base * w_nee * inv_p
                            if cfg.phys_normalization:
                                contrib *= ray_weight
                            arrivals.append((t, contrib, b + 1))
                        else:
                            db_vec = np.array([AIR_DB_PER_M_OCT.get(f, cfg.air_db_per_m) for f in scene.bands], dtype=np.float32)
                            air_vec = air_lin_vec(db_vec, traveled2 + leg)
                            wv = amp_after * spread_fn(traveled2 + leg) * air_vec
                            if cfg.nee_mis:
                                wv *= w_nee
                            wv *= inv_p
                            if cfg.phys_normalization:
                                wv *= ray_weight
                            arrivals.append((t, wv.astype(np.float32), b + 1))


            # ---- Russian roulette (after a few bounces) ----
            if cfg.russian_roulette and b >= 3:
                m = float(np.max(amp_after))
                p = min(1.0, max(0.05, m / 0.1))  # tune threshold
                if rng.random() > p:
                    continue
                amp_after /= max(p, 1e-6)

            # ---- Choose outgoing direction (BRDF) ----
            n = scene.mesh.face_normals[fidx]
            # --- Energy-conserving throughput for diffuse branch ---
            # r2 is the energy reflectance we already computed: r2 = 1 - a_vec - t_vec
            # If we chose diffuse this bounce, multiply throughput by r2 instead of sqrt(r2).
            # Because for a Lambertian BRDF sampled with cosine hemisphere, BRDF*cos/pdf = r2 (per band).
            if cfg.brdf_model == "spec+lambert" and cfg.scatter_ratio > 0.0:
                chose_diffuse = False
                # re-draw decision (we just made it above). Capture the outcome.
                if rng.random() < cfg.scatter_ratio:
                    chose_diffuse = True
                    d_new = cosine_hemisphere_sample(n, rng)
                else:
                    d_new = reflect(dvec, n)

                if chose_diffuse:
                    # Use r2 (not sqrt) for energy conservation
                    amp_after = (amp_reflect * np.maximum(0.0, 1.0 - a_vec - t_vec)).astype(np.float32)
                else:
                    # Specular stays with amplitude reflectance sqrt(r2)
                    amp_after = (amp_reflect * np.sqrt(np.maximum(0.0, 1.0 - a_vec - t_vec))).astype(np.float32)
            else:
                d_new = reflect(dvec, n)
                amp_after = (amp_reflect * np.sqrt(np.maximum(0.0, 1.0 - a_vec - t_vec))).astype(np.float32)

            d_new = jitter_direction(d_new, sigma, rng)

            # ---- Reflect path continues ----
            pos_next = hit + d_new * scene.inter.eps
            queue.append((pos_next, d_new, amp_after, traveled2, b + 1, spawned_T))

            # ---- Optional single transmission spawn (first hit only) ----
            if cfg.transmission_paths and (not spawned_T) and (b == 0):
                T_amp = (amp_reflect * np.sqrt(np.maximum(0.0, t_vec))).astype(np.float32)
                if float(np.max(T_amp)) >= cfg.min_amp:
                    pos_T = hit + dvec * scene.inter.eps  # straight through
                    queue.append((pos_T, dvec, T_amp, traveled2, b + 1, True))

    # ---- Bin arrivals into IR ----
    if B == 1:
        h = bin_arrivals_to_ir(arrivals, cfg.fs, cfg.duration_s)
        if cfg.synth_tail_if_sparse:
            _add_synth_tail_inplace(h, cfg)
        return h, arrivals
    else:
        H = bin_arrivals_to_ir_banded(arrivals, cfg.fs, cfg.duration_s, B)
        if cfg.synth_tail_if_sparse:
            for bi in range(B):
                _add_synth_tail_inplace(H[bi], cfg, seed_offset=bi)
        return H, arrivals

def trace_preview_paths(scene: Scene, max_bounces: int, n_paths: int, seed: int, jitter_deg: float):
    """Visual-only preview: specular+jitter paths (unchanged behavior)."""
    rng = np.random.default_rng(seed)
    sigma = math.radians(jitter_deg)
    dirs = sample_unit_sphere(n_paths, rng)
    polylines = []
    to_rec = []
    for i in range(n_paths):
        pos = scene.S.copy()
        dvec = dirs[i]
        pts = [pos.copy()]
        for _ in range(max_bounces):
            hit, fidx, dist = scene.inter.first_hit(pos + dvec * scene.inter.eps, dvec)
            if hit is None:
                break
            pts.append(hit.copy())
            if scene.inter.visible(hit, scene.R):
                to_rec.append(np.vstack([hit.copy(), scene.R.copy()]))
            n = scene.mesh.face_normals[fidx]
            dvec = jitter_direction(reflect(dvec, n), sigma, rng)
            pos = hit + dvec * scene.inter.eps
        if len(pts) > 1:
            polylines.append(np.vstack(pts))
    return polylines, to_rec
