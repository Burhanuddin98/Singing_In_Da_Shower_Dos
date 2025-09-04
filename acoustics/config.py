# acoustics/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List

# Default octave centers (extend if you like)
OCTAVE_CENTERS: List[int] = [125, 250, 500, 1000, 2000, 4000]

# A simple fallback (rarely used now that ISO is available)
AIR_DB_PER_M_OCT = {
    125: 0.01,
    250: 0.01,
    500: 0.02,
    1000: 0.04,
    2000: 0.10,
    4000: 0.30,
}


@dataclass
class SimConfig:
    # Core acoustics
    c: float = 343.0
    fs: int = 48000
    duration_s: float = 3.0

    # Path tracing
    rays: int = 8000
    max_bounces: int = 8
    rng_seed: int = 0
    time_budget: float = 8.0
    min_amp: float = 1e-6
    include_direct: bool = True

    # Materials & scattering
    alpha_default: float = 0.05
    scattering_deg: float = 7.0  # jitter RMS (deg)

    # Air absorption (legacy flat slider)
    air_db_per_m: float = 0.00

    # Optional synthetic tail if IR is sparse
    synth_tail_if_sparse: bool = False
    synth_rt60_s: float = 1.5
    synth_tail_level_db: float = -24.0

    # === Advanced (ODEON-ish) controls ===
    phys_normalization: bool = False      # unbiased MC estimator
    band_mode: str = "broadband"   # "broadband" | "octave" | "third" | "twelfth"
    brdf_model: str = "specular+jitter"   # "specular+jitter" | "spec+lambert"
    scatter_ratio: float = 0.0            # used if no per-face scatter map provided
    transmission_paths: bool = False      # (stub: on hit, could spawn a transmit ray)
    nee_mis: bool = False                 # (lite) divide NEE by ray count
    russian_roulette: bool = True
    scale_convention: str = "pressure"    # "pressure" | "intensity"
    receiver_radius_m: float = 0.10       # NEE disk radius (approx solid angle)
    direct_mode: str = "deterministic"    # "deterministic" | "sampled"

    # NEE scheduling
    nee_bounces: int = 4                  # always sample NEE up to this bounce count
    nee_all_bounces: bool = True          # allow probabilistic NEE beyond nee_bounces
    nee_prob: float = 0.30                # probability per bounce for NEE when beyond nee_bounces

    # Angle-dependent absorption
    angle_absorption: bool = False

    # ISO-9613-1 air absorption (per band)
    air_model: str = "flat"               # "flat" | "iso9613"
    air_temp_c: float = 20.0
    air_rh_pct: float = 50.0
    air_pressure_kpa: float = 101.325


@dataclass
class MaterialAuto:
    alpha_floor: float = 0.15
    alpha_ceiling: float = 0.20
    alpha_walls: float = 0.08
    alpha_default: float = 0.05
    nz_thresh: float = 0.3
    percentile_margin: float = 5.0
