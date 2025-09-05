# acoustics/__init__.py
from __future__ import annotations

# ---- Public config / constants ----
from .config import (
    SimConfig,
    MaterialAuto,
    OCTAVE_CENTERS,
    AIR_DB_PER_M_OCT,
)

# ---- Physics helpers (math, binning, air, spreading) ----
from .physics import (
    unit,
    reflect,
    sample_unit_sphere,
    jitter_direction,
    cosine_hemisphere_sample,
    phong_lobe_sample,
    pressure_spread,
    intensity_spread,
    pick_spread_fn,
    inv_spherical_spreading,   # legacy shim (still safe to export)
    air_lin,
    air_lin_vec,
    bin_arrivals_to_ir,
    bin_arrivals_to_ir_banded,
    schroeder_edc,
    estimate_rt60_from_edc,
    effective_t_end_from_edc,
    decimate_line,
)

# ---- Geometry / intersector ----
from .geometry import (
    Intersector,
    face_connected_components,
    build_trimesh_from_arrays,
)

# ---- Tracing core ----
from .tracing import (
    Scene,
    path_trace,
    trace_preview_paths,
)

# ---- Visualization / audio helpers ----
from .viz import (
    make_fig,
    add_source_receiver,
    overlay_highlight,
    wav_bytes,
    spectrogram_figure,
)

# ---- Animation helpers ----
from .animation import (
    create_ray_animation_figure,
    render_gif_one_ray_matplotlib,
)

# ---- Streamlit caching / glue ----
from .caching import (
    mesh_hash_from_arrays,
    build_components_cached,
    build_intersector_cached,
    auto_alpha_cached,
    trace_cached,
)

__all__ = [
    # Config
    "SimConfig", "MaterialAuto", "OCTAVE_CENTERS", "AIR_DB_PER_M_OCT",
    # Physics
    "unit", "reflect", "sample_unit_sphere", "jitter_direction",
    "cosine_hemisphere_sample", "phong_lobe_sample",
    "pressure_spread", "intensity_spread", "pick_spread_fn",
    "inv_spherical_spreading", "air_lin", "air_lin_vec",
    "bin_arrivals_to_ir", "bin_arrivals_to_ir_banded",
    "schroeder_edc", "estimate_rt60_from_edc", "effective_t_end_from_edc",
    "decimate_line",
    # Geometry
    "Intersector", "face_connected_components", "build_trimesh_from_arrays",
    # Tracing
    "Scene", "path_trace", "trace_preview_paths",
    # Viz
    "make_fig", "add_source_receiver", "overlay_highlight",
    "wav_bytes", "spectrogram_figure",
    # Animation
    "create_ray_animation_figure", "render_gif_one_ray_matplotlib",
    # Caching
    "mesh_hash_from_arrays", "build_components_cached",
    "build_intersector_cached", "auto_alpha_cached", "trace_cached",
]