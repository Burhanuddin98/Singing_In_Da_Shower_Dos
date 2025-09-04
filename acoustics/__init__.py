# acoustics/__init__.py

# ---- Config & band tables ----
from .config import (
    SimConfig,
    MaterialAuto,
    OCTAVE_CENTERS,
    AIR_DB_PER_M_OCT,
)

# ---- Geometry / intersector ----
from .geometry import (
    Intersector,
    face_connected_components,
    build_trimesh_from_arrays,
)

# ---- Physics utilities & metrics ----
from .physics import (
    # vector math / sampling
    unit,
    reflect,
    sample_unit_sphere,
    jitter_direction,
    cosine_hemisphere_sample,
    # scaling / air attenuation
    pressure_spread,
    intensity_spread,
    pick_spread_fn,
    air_lin,
    air_lin_vec,
    # IR / EDC helpers
    bin_arrivals_to_ir,
    bin_arrivals_to_ir_banded,
    schroeder_edc,
    estimate_rt60_from_edc,
    effective_t_end_from_edc,
    decimate_line,
)

# ---- Tracing core ----
from .tracing import (
    Scene,
    path_trace,
    trace_preview_paths,
)

# ---- Visualization & audio I/O ----
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

# ---- Streamlit caching & glue ----
from .caching import (
    mesh_hash_from_arrays,
    build_components_cached,
    build_intersector_cached,
    auto_alpha_cached,
    trace_cached,
)

__all__ = [
    # config
    "SimConfig", "MaterialAuto", "OCTAVE_CENTERS", "AIR_DB_PER_M_OCT",
    # geometry
    "Intersector", "face_connected_components", "build_trimesh_from_arrays",
    # physics
    "unit", "reflect", "sample_unit_sphere", "jitter_direction", "cosine_hemisphere_sample",
    "pressure_spread", "intensity_spread", "pick_spread_fn", "air_lin", "air_lin_vec",
    "bin_arrivals_to_ir", "bin_arrivals_to_ir_banded",
    "schroeder_edc", "estimate_rt60_from_edc", "effective_t_end_from_edc", "decimate_line",
    # tracing
    "Scene", "path_trace", "trace_preview_paths",
    # viz
    "make_fig", "add_source_receiver", "overlay_highlight", "wav_bytes", "spectrogram_figure",
    # animation
    "create_ray_animation_figure", "render_gif_one_ray_matplotlib",
    # caching
    "mesh_hash_from_arrays", "build_components_cached", "build_intersector_cached",
    "auto_alpha_cached", "trace_cached",
]
