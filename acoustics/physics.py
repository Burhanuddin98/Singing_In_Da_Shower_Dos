# acoustics/physics.py
from __future__ import annotations
import math
import numpy as np

# -----------------------------
# Vector math helpers
# -----------------------------

def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n

def reflect(dir_in: np.ndarray, n: np.ndarray) -> np.ndarray:
    d = np.asarray(dir_in, dtype=float)
    n = np.asarray(n, dtype=float)
    if float(np.dot(d, n)) > 0:
        n = -n
    return unit(d - 2.0 * float(np.dot(d, n)) * n)

def sample_unit_sphere(nrays: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(int(nrays), 3))
    return unit(v)

def jitter_direction(d: np.ndarray, sigma_rad: float, rng: np.random.Generator) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    if sigma_rad <= 0:
        return unit(d)
    return unit(d + sigma_rad * rng.normal(size=3))

# -----------------------------
# BRDF sampling
# -----------------------------

def _orthonormal_basis(n: np.ndarray):
    n = unit(n)
    if abs(n[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        a = np.array([0.0, 1.0, 0.0], dtype=float)
    t = unit(np.cross(n, a))
    b = unit(np.cross(n, t))
    return t, b, n

def cosine_hemisphere_sample(n: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Cosine-weighted hemisphere around normal n
    u1 = rng.random()
    u2 = rng.random()
    r = math.sqrt(u1)
    theta = 2.0 * math.pi * u2
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    z = math.sqrt(max(0.0, 1.0 - u1))
    t, b, nn = _orthonormal_basis(n)
    return unit(x * t + y * b + z * nn)

def phong_lobe_sample(n: np.ndarray, s: float, rng: np.random.Generator) -> np.ndarray:
    # Simple Phong-like lobe around normal, exponent s >= 0
    u1 = rng.random()
    u2 = rng.random()
    cos_theta = pow(u1, 1.0 / (s + 1.0))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * math.pi * u2
    x = sin_theta * math.cos(phi)
    y = sin_theta * math.sin(phi)
    z = cos_theta
    t, b, nn = _orthonormal_basis(n)
    return unit(x * t + y * b + z * nn)

# -----------------------------
# Spreading & air absorption
# -----------------------------

def pressure_spread(distance_m: float) -> float:
    # amplitude convention ~ 1/r   (clipped)
    r = max(float(distance_m), 1e-4)
    return 1.0 / r

def intensity_spread(distance_m: float) -> float:
    # energy convention ~ 1/(4π r^2)
    r = max(float(distance_m), 1e-4)
    return 1.0 / (4.0 * math.pi * r * r)

def pick_spread_fn(scale_convention: str):
    if str(scale_convention).lower().startswith("intens"):
        return intensity_spread
    return pressure_spread

# Legacy shim
def inv_spherical_spreading(distance_m: float) -> float:
    return intensity_spread(distance_m)

def air_lin(db_per_m: float, d_m: float) -> float:
    if db_per_m <= 0.0 or d_m <= 0.0:
        return 1.0
    return 10.0 ** (-(db_per_m * d_m) / 20.0)

def air_lin_vec(db_per_m_vec: np.ndarray, d_m: float) -> np.ndarray:
    db_per_m_vec = np.asarray(db_per_m_vec, dtype=float)
    if d_m <= 0.0:
        return np.ones_like(db_per_m_vec, dtype=np.float32)
    return (10.0 ** (-(db_per_m_vec * d_m) / 20.0)).astype(np.float32)

# -----------------------------
# Binning & decay metrics
# -----------------------------

def bin_arrivals_to_ir(arrivals, fs: int, duration_s: float) -> np.ndarray:
    """
    arrivals: list of (t, a, depth), scalar amplitude 'a'
    returns mono IR shape (N,)
    """
    n = int(max(1, round(float(duration_s) * int(fs))))
    h = np.zeros(n, dtype=np.float32)
    for t, a, _ in arrivals:
        if 0 <= t < duration_s:
            idx = t * fs
            i0 = int(idx)
            frac = float(idx - i0)
            if 0 <= i0 < n:
                h[i0] += float(a) * (1.0 - frac)
            i1 = i0 + 1
            if 0 <= i1 < n:
                h[i1] += float(a) * frac
    return h

def bin_arrivals_to_ir_banded(arrivals, fs: int, duration_s: float, nbands: int) -> np.ndarray:
    """
    arrivals: list of (t, a_vec[B], depth)
    returns IR shape (B, N)
    """
    n = int(max(1, round(float(duration_s) * int(fs))))
    H = np.zeros((int(nbands), n), dtype=np.float32)
    for t, avec, _ in arrivals:
        if 0 <= t < duration_s:
            idx = t * fs
            i0 = int(idx)
            frac = float(idx - i0)
            a = np.asarray(avec, dtype=np.float32)
            if 0 <= i0 < n:
                H[:, i0] += a * (1.0 - frac)
            i1 = i0 + 1
            if 0 <= i1 < n:
                H[:, i1] += a * frac
    return H

def schroeder_edc(h: np.ndarray) -> np.ndarray:
    e = np.asarray(h, dtype=np.float64)
    edc = np.cumsum((e * e)[::-1])[::-1]
    top = float(edc[0]) if edc.size else 1.0
    if top <= 0:
        top = 1.0
    return (edc / top)

def estimate_rt60_from_edc(edc: np.ndarray, fs: int):
    if edc is None or len(edc) < 10:
        return None
    db = 10.0 * np.log10(np.maximum(edc, 1e-20))
    t = np.arange(len(edc), dtype=float) / float(fs)

    def fit(lo_db, hi_db):
        i1 = int(np.abs(db - lo_db).argmin())
        i2 = int(np.abs(db - hi_db).argmin())
        if i2 <= i1:
            return None
        p = np.polyfit(t[i1:i2], db[i1:i2], 1)
        slope = float(p[0])
        if slope >= -1e-9:
            return None
        return 60.0 / (-slope)

    rt = fit(-5, -35)
    if rt is None:
        rt = fit(-10, -30)
    return rt

def effective_t_end_from_edc(h: np.ndarray, fs: int, drop_db: float = -60.0) -> float:
    if h is None or (getattr(h, "size", 0) == 0):
        return 0.0
    edc = schroeder_edc(h)
    db = 10.0 * np.log10(np.maximum(edc, 1e-20))
    idx = np.where(db <= float(drop_db))[0]
    if idx.size:
        return float(idx[0]) / float(fs)
    return float(len(h)) / float(fs)

def decimate_line(x: np.ndarray, y: np.ndarray, max_points: int = 200_000):
    x = np.asarray(x)
    y = np.asarray(y)
    n = int(x.size)
    if n <= int(max_points):
        return x, y
    step = int(math.ceil(n / float(max_points)))
    return x[::step], y[::step]

def _saturation_pressure_hPa(T):
    # Buck equation (good 0–50C)
    return 6.1121 * math.exp((18.678 - T/234.5) * (T/(257.14 + T)))

def air_db_per_m_iso9613(freqs_hz: np.ndarray, T_c: float, RH_pct: float, p_kPa: float) -> np.ndarray:
    """
    ISO 9613-1 approximate atmospheric absorption in dB/m.
    Good enough for room-scale sims and octave centers.
    """
    T = T_c + 273.15                      # K
    p = p_kPa * 10.0                      # kPa -> hPa
    RH = max(0.0, min(100.0, RH_pct)) / 100.0

    # Molar concentration of water vapor h
    Ps = _saturation_pressure_hPa(T_c)    # hPa
    h = RH * Ps / p

    # Relaxation frequencies (Hz)
    frO = (24.0 + 4.04e4*h*(0.02 + h)/(0.391 + h)) * (p/1013.0) * ((293.15/T)**0.5)
    frN = (T/293.15)**(-0.5) * (9.0 + 280.0*h * math.exp(-4.17*((T/293.15)**(-1.0/3.0) - 1.0)))

    f = np.asarray(freqs_hz, dtype=float)
    f2 = f*f

    # ISO 9613-1 absorption in dB/m
    # α = 8.686 * f^2 [ 1.84e-11 * (1/p)*(T/293)^0.5 + (T/293)^(-2.5) * ( 0.01275*exp(-2239.1/T)/(frO + f^2/frO) + 0.1068*exp(-3352/T)/(frN + f^2/frN) ) ]
    term0 = 1.84e-11 * (1.0/(p/1013.0)) * ((T/293.15)**0.5)
    termO = 0.01275 * math.exp(-2239.1/T) / (frO + (f2/frO))
    termN = 0.10680 * math.exp(-3352.0/T) / (frN + (f2/frN))
    alpha_np = 8.686 * f2 * ( term0 + ((T/293.15)**(-2.5)) * (termO + termN) )
    # protect very low freq / zeros
    alpha_np = np.maximum(alpha_np, 0.0)
    return alpha_np.astype(np.float32)