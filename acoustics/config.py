# acoustics/config.py
from __future__ import annotations
from dataclasses import dataclass

# Always sample NEE for the first N bounces (kept behavior)
nee_bounces: int = 4

# NEW: enable NEE attempts beyond nee_bounces
nee_all_bounces: bool = True         # try NEE at every bounce (after N, probabilistically)
nee_prob: float = 0.30               # probability to attempt NEE per bounce after the first N


# -----------------------------
# Band definitions / lookups
# -----------------------------

# Octave-band centers (Hz). Extend if you need more.
OCTAVE_CENTERS = [125, 250, 500, 1000, 2000, 4000]

# Simple per-band air attenuation [dB/m].
# (Placeholder; later you can compute ISO 9613-1 from temperature/RH.)
AIR_DB_PER_M_OCT = {
    125: 0.01,
    250: 0.01,
    500: 0.02,
    1000: 0.04,
    2000: 0.10,
    4000: 0.30,
}

# -----------------------------
# Existing auto-material seed
# -----------------------------

@dataclass
class MaterialAuto:
    alpha_floor: float = 0.15
    alpha_ceiling: float = 0.20
    alpha_walls: float = 0.08
    alpha_default: float = 0.05
    nz_thresh: float = 0.3
    percentile_margin: float = 5.0

# -----------------------------
# Simulation configuration
# -----------------------------

@dataclass
class SimConfig:
    # ---- Base (kept from your original) ----
    c: float = 343.0
    fs: int = 48000
    duration_s: float = 3.0
    rays: int = 8000
    max_bounces: int = 8
    alpha_default: float = 0.05
    scattering_deg: float = 7.0
    air_db_per_m: float = 0.00
    rng_seed: int = 0
    time_budget: float = 8.0
    nee_bounces: int = 4
    nee_max_dist_m: float = 60.0
    nee_min_dot: float = -0.25
    min_amp: float = 1e-6
    include_direct: bool = True
    synth_tail_if_sparse: bool = False
    synth_rt60_s: float = 1.5
    synth_tail_level_db: float = -24.0

    # ---- Advanced toggles (new; default preserves current behavior) ----
    # Monte-Carlo estimator normalization (ray-count invariant)
    phys_normalization: bool = False

    # Banding mode: "broadband" (old behavior) or "octave" (new, per-band)
    band_mode: str = "broadband"  # "broadband" | "octave"

    # BRDF: legacy specular + Gaussian jitter, or specular + Lambert blend
    brdf_model: str = "specular+jitter"  # "specular+jitter" | "spec+lambert"
    scatter_ratio: float = 0.0           # (0..1) diffuse share when spec+lambert

    # Transmission as actual paths (single pass-through spawn at first hit)
    transmission_paths: bool = False

    # NEE multiple-importance sampling (lite) toggle (safe no-op if unsupported)
    nee_mis: bool = False

    # Russian roulette termination for deep, low-energy paths
    russian_roulette: bool = True

    # Spreading law convention for amplitudes
    scale_convention: str = "pressure"  # "pressure" (∝1/r) | "intensity" (∝1/r^2)

    # ---- New controls used by MIS-lite / direct handling ----
    # Receiver “disk” radius for NEE solid-angle approximation (meters)
    receiver_radius_m: float = 0.10

    # How to add the direct contribution
    direct_mode: str = "deterministic"  # "deterministic" | "sampled"

        # Always sample NEE for the first N bounces (existing)
    nee_bounces: int = 4

    # NEW — probabilistic NEE for all later bounces:
    nee_all_bounces: bool = True     # try NEE beyond first N bounces
    nee_prob: float = 0.30           # probability per-bounce after N

    # (Make sure these also exist if you use them elsewhere)
    receiver_radius_m: float = 0.10
    direct_mode: str = "deterministic"
