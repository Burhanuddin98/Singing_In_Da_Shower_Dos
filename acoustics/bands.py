# acoustics/bands.py
from __future__ import annotations
import numpy as np

def logspace_center_freqs(f_lo: float, f_hi: float, bands_per_octave: int) -> np.ndarray:
    """
    Center frequencies from f_lo..f_hi with BPO bands per octave.
    e.g., BPO=1 (octave), 3 (third), 12 (twelfth).
    """
    assert f_lo > 0 and f_hi > f_lo and bands_per_octave > 0
    octaves = np.log2(f_hi / f_lo)
    n = int(np.round(octaves * bands_per_octave)) + 1
    return f_lo * (2.0 ** (np.arange(n) / bands_per_octave))

def standard_centers(mode: str) -> np.ndarray:
    """
    Common defaults. Adjust ranges if you like.
    - octave:    125..8000
    - third:     100..10000
    - twelfth:   80..12500
    """
    mode = str(mode).lower()
    if mode == "octave":
        return logspace_center_freqs(125.0, 8000.0, bands_per_octave=1)
    if mode == "third":
        return logspace_center_freqs(100.0, 10000.0, bands_per_octave=3)
    if mode in ("twelfth", "1/12"):
        return logspace_center_freqs(80.0, 12500.0, bands_per_octave=12)
    # fallback
    return np.array([1000.0], dtype=float)

def resample_bands(src_freqs: np.ndarray, src_vals: np.ndarray, dst_freqs: np.ndarray) -> np.ndarray:
    """
    Log-frequency interpolation of any per-band array (alpha, tau, scatter)
    from material tables (src_freqs/src_vals) to current tracer bands (dst_freqs).
    src_vals shape: (B_src,) or (N, B_src). Returns shape (B_dst,) or (N, B_dst).
    Clamps at ends.
    """
    sf = np.asarray(src_freqs, float); df = np.asarray(dst_freqs, float)
    sv = np.asarray(src_vals, float)
    lx = np.log(sf); ly = np.log(df)

    if sv.ndim == 1:
        y = np.interp(ly, lx, sv, left=sv[0], right=sv[-1])
        return y.astype(np.float32)
    else:
        out = []
        for row in sv:
            out.append(np.interp(ly, lx, row, left=row[0], right=row[-1]))
        return np.asarray(out, dtype=np.float32)
