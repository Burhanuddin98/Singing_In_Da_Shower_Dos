# acoustics/physics.py
from __future__ import annotations
import math
import numpy as np

# =========================
# Vector math / sampling
# =========================

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n

def reflect(dir_in: np.ndarray, n: np.ndarray) -> np.ndarray:
    # Reflect direction about normal (assumes both approx. unit)
    if float(np.dot(dir_in, n)) > 0:
        n = -n
    return unit(dir_in - 2.0 * float(np.dot(dir_in, n)) * n)

def sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return unit(v)

def jitter_direction(d: np.ndarray, sigma_rad: float, rng: np.random.Generator) -> np.ndarray:
    if sigma_rad <= 0:
        return d
    return unit(d + sigma_rad * rng.normal(size=3))

def cosine_hemisphere_sample(n: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cosine-weighted hemisphere sampling around normal n."""
    u1 = rng.random(); u2 = rng.random()
    r = math.sqrt(u1); theta = 2 * math.pi * u2
    x = r * math.cos(theta); y = r * math.sin(theta); z = math.sqrt(max(0.0, 1.0 - u1))
    n = unit(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t = unit(np.cross(a, n))
    b = unit(np.cross(n, t))
    return unit(x * t + y * b + z * n)

def phong_lobe_sample(axis: np.ndarray, rng: np.random.Generator, exponent: float) -> np.ndarray:
    """Phong-like lobe sampling around 'axis' with exponent k (larger = sharper)."""
    u1 = rng.random(); u2 = rng.random()
    cos_theta = u1 ** (1.0 / (exponent + 1.0))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2 * math.pi * u2
    x = sin_theta * math.cos(phi); y = sin_theta * math.sin(phi); z = cos_theta
    axis = unit(axis)
    a = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t = unit(np.cross(a, axis))
    b = unit(np.cross(axis, t))
    return unit(x * t + y * b + z * axis)

# =========================
# Spreading laws
# =========================

def pressure_spread(r: float) -> float:
    """Pressure-like amplitude ∝ 1/r."""
    return 1.0 / max(r, 1e-4)

def intensity_spread(r: float) -> float:
    """Intensity-like amplitude ∝ 1/r^2 (energy)."""
    return 1.0 / max(r * r, 1e-4)

def pick_spread_fn(scale_convention: str):
    return pressure_spread if scale_convention == "pressure" else intensity_spread

# Legacy shim (only if older code references it)
def inv_spherical_spreading(distance_m: float) -> float:
    """Legacy helper: 1/(4π r)."""
    r = max(distance_m, 1e-4)
    return 1.0 / (4.0 * math.pi * r)

# =========================
# Air absorption
# =========================

def air_lin(db_per_m: float, d_m: float) -> float:
    """Scalar linear air attenuation from dB/m over distance."""
    if db_per_m <= 0 or d_m <= 0:
        return 1.0
    return 10.0 ** (-(db_per_m * d_m) / 20.0)

def air_lin_vec(db_per_m_vec: np.ndarray, d_m: float) -> np.ndarray:
    """Vectorized linear air attenuation for per-band dB/m."""
    if d_m <= 0:
        return np.ones_like(db_per_m_vec, dtype=np.float32)
    return (10.0 ** (-(db_per_m_vec * d_m) / 20.0)).astype(np.float32)

# =========================
# IR binning / EDC / utilities
# =========================

def bin_arrivals_to_ir(arrivals, fs: int, duration_s: float) -> np.ndarray:
    """Bin scalar arrivals: list[(t_seconds, amp_scalar, bounces)] -> h[N]."""
    n = int(max(1, round(duration_s * fs)))
    h = np.zeros(n, dtype=np.float32)
    for t, a, _ in arrivals:
        if 0 <= t < duration_s:
            idx = t * fs
            i0 = int(idx); frac = float(idx - i0)
            if 0 <= i0 < n:
                h[i0] += a * (1.0 - frac)
            i1 = i0 + 1
            if 0 <= i1 < n:
                h[i1] += a * frac
    return h

def bin_arrivals_to_ir_banded(arrivals, fs: int, duration_s: float, nbands: int) -> np.ndarray:
    """Bin banded arrivals: list[(t_seconds, amp_vec[B], bounces)] -> H[B, N]."""
    n = int(max(1, round(duration_s * fs)))
    H = np.zeros((nbands, n), dtype=np.float32)
    for t, vec, _ in arrivals:
        if not (0 <= t < duration_s):
            continue
        idx = t * fs
        i0 = int(idx); frac = float(idx - i0)
        if 0 <= i0 < n:
            H[:, i0] += vec * (1.0 - frac)
        i1 = i0 + 1
        if 0 <= i1 < n:
            H[:, i1] += vec * frac
    return H

def schroeder_edc(h: np.ndarray) -> np.ndarray:
    e = h.astype(np.float64) ** 2
    edc = np.cumsum(e[::-1])[::-1]
    edc /= max(edc[0], 1e-20)
    return edc

def estimate_rt60_from_edc(edc: np.ndarray, fs: int):
    """Fit slope in dB vs time; return RT60 if slope is negative and stable."""
    if len(edc) < 10:
        return None
    db = 10 * np.log10(np.maximum(edc, 1e-20))
    t = np.arange(len(edc)) / fs

    def fit_drop(lo_db, hi_db):
        i1 = int(np.abs(db - lo_db).argmin()); i2 = int(np.abs(db - hi_db).argmin())
        if i2 <= i1:
            return None
        p = np.polyfit(t[i1:i2], db[i1:i2], 1)
        slope = float(p[0])
        if slope >= -1e-9:
            return None
        return 60.0 / (-slope)

    rt = fit_drop(-5, -35)
    if rt is None:
        rt = fit_drop(-10, -30)
    return rt

def effective_t_end_from_edc(h: np.ndarray, fs: int, drop_db: float = -60.0) -> float:
    if h.size == 0:
        return 0.0
    edc = schroeder_edc(h)
    db = 10 * np.log10(np.maximum(edc, 1e-20))
    idx = np.where(db <= drop_db)[0]
    return (idx[0] / float(fs)) if idx.size else (len(h) / float(fs))

def decimate_line(x: np.ndarray, y: np.ndarray, max_points: int = 200_000):
    n = x.size
    if n <= max_points:
        return x, y
    step = int(np.ceil(n / max_points))
    return x[::step], y[::step]
