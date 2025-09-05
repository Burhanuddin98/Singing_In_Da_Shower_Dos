from __future__ import annotations
from acoustics.bands import standard_centers

# --- Optional banded materials (library) ---
from acoustics.materials import builtin_library, to_broadband
from acoustics.config import OCTAVE_CENTERS

import trimesh

import io, os, math, hashlib, numpy as np, streamlit as st
import plotly.graph_objects as go

# Local package (modularized)
from acoustics import (
    # Config & helpers
    SimConfig, MaterialAuto,
    # Caching / tracing glue
    mesh_hash_from_arrays, auto_alpha_cached, build_components_cached, trace_cached,
    # Physics/metrics
    schroeder_edc, estimate_rt60_from_edc, effective_t_end_from_edc, decimate_line,
    # Viz + audio
    make_fig, add_source_receiver, overlay_highlight, wav_bytes, spectrogram_figure,
)

# ===== Streamlit setup =====
st.set_page_config(page_title="Acoustic Ray Tracer", layout="wide")

# ----- Optional deps checks (UI hints only) -----
try:
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector as _RayIntersector
    EMBREE_OK = True
except Exception:
    try:
        from trimesh.ray.ray_triangle import RayMeshIntersector as _RayIntersector
        EMBREE_OK = False
    except Exception:
        st.error("Install trimesh: pip install trimesh")
        st.stop()

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    from scipy.signal import fftconvolve, spectrogram as _spec, resample_poly
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_OK = True
except Exception:
    PLOTLY_EVENTS_OK = False

if not EMBREE_OK:
    try:
        import rtree  # noqa: F401
        RTREE_OK = True
    except Exception:
        RTREE_OK = False
        st.warning(
            "Embree not found and rtree missing.\n"
            "Install either:\n• conda install -c conda-forge pyembree\n• conda install -c conda-forge rtree"
        )

# ===== Style =====
st.markdown("""
<style>
:root{ --bg:#0d0f12; --panel:#12161c; --text:#e6edf3; --muted:#a7b0b8; --line:#2a2f36; --accent:#4bd0e0; --badge-bg:#0f141a; --badge-text:#e6edf3; }
html, body, [data-testid=stAppViewContainer], [data-testid=stHeader]{ background:var(--bg)!important; color:var(--text)!important; }
[data-testid=stSidebar]{ background:var(--panel)!important; color:var(--text)!important; box-shadow: inset 0 0 0 1px var(--line); }
.stButton>button, .stDownloadButton>button{ background:#141a22; color:var(--text); border:1px solid var(--line); border-radius:8px; }
.stButton>button:hover, .stDownloadButton>button:hover{ background:#1a212b; border-color:#3a4048; }
.stSlider [data-baseweb="slider"] div{ background:#20262d!important; height:10px!important; border-radius:6px!important; box-shadow: inset 0 0 0 1px var(--line);}
.stSlider [role="slider"]{ background:var(--accent)!important; border:none!important; box-shadow:none!important; }
.js-plotly-plot .colorbar text { fill: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ===== Main UI =====
def main():
    st.title("Singing in Da Shower (SIDS) - An Acoustic Ray Tracer / RIR Estimator")
    st.caption("Interactive animation + safe GIF export. α and τ drive |R| so changes show up in IR/EDC and audio.")

    # --- Sidebar forms ---
    with st.sidebar:
        with st.form("geom_form"):
            st.subheader("1) Geometry")
            f = st.file_uploader("Mesh (STL/OBJ/DAE/GLB/GLTF/PLY)", type=["stl","obj","dae","glb","gltf","ply"])
            units_per_meter = st.number_input("Units per meter (1000 for mm)", 0.001, 1_000_000.0, 1000.0, 1.0)
            wrap_box = st.checkbox("Enclose non-watertight with padded box", value=True)
            box_padding = st.number_input("Box padding (m)", 0.0, 10.0, 0.20, 0.05)
            st.form_submit_button("Apply geometry", use_container_width=True)

        with st.form("pos_form"):
            st.subheader("2) Positions (m)")
            Sx = st.number_input("Source X", value=0.5); Sy = st.number_input("Source Y", value=0.5); Sz = st.number_input("Source Z", value=1.5)
            Rx = st.number_input("Receiver X", value=2.5); Ry = st.number_input("Receiver Y", value=1.5); Rz = st.number_input("Receiver Z", value=1.5)
            st.form_submit_button("Apply positions", use_container_width=True)

        with st.form("auto_mat_form"):
            st.subheader("3) Auto material seed")
            alpha_floor = st.slider("Floor α",0.0,0.99,0.15,0.01); alpha_ceiling = st.slider("Ceiling α",0.0,0.99,0.20,0.01)
            alpha_walls = st.slider("Walls α",0.0,0.99,0.08,0.01); alpha_default = st.slider("Default α",0.0,0.99,0.05,0.01)
            nz_thresh = st.slider("Wall |n_z| threshold",0.0,1.0,0.30,0.01); perc_margin = st.slider("Floor/Ceil percentile (%)",0.0,20.0,5.0,0.5)
            st.form_submit_button("Apply material seed", use_container_width=True)

        with st.form("sim_form"):
            st.subheader("4) Simulation")
            rays = st.slider("Rays",500,50000,8000,500); max_bounces = st.slider("Max bounces",1,30,8)
            scattering = st.slider("Scattering (° RMS)",0.0,45.0,7.0,0.5)
            air_dbm = st.slider("Air attenuation (dB/m)",0.0,1.0,0.00,0.01)
            include_direct = st.checkbox("Include direct path", value=True)
            time_budget = st.slider("Time budget (s)",1.0,40.0,8.0,0.5)
            min_amp_str = st.select_slider("Min ray amplitude (stop)",["1e-4","1e-5","1e-6"],value="1e-6")
            st.form_submit_button("Apply simulation knobs", use_container_width=True)
            min_amp = float(min_amp_str)

        with st.form("ir_form"):
            st.subheader("5) IR & output")
            fs = st.selectbox("Sample rate",[16000,22050,32000,44100,48000,96000],index=4)
            dur = st.slider("IR duration (s)",0.5,12.0,3.0,0.5)
            seed = st.number_input("Random seed",0, None, 0, 1)
            force_tail = st.checkbox("If IR is sparse, add quiet synthetic late tail", value=False)
            tail_rt = st.slider("Synthetic RT60 (s)",0.3,5.0,1.5,0.1)
            tail_lvl = st.slider("Tail splice level (dB)",-60.0,-10.0,-24.0,1.0)
            st.form_submit_button("Apply IR settings", use_container_width=True)

        with st.form("viz_form"):
            st.subheader("6) Visualization")
            vis_paths = st.slider("Ray-path preview count",0,500,150,10)
            edge_cap = st.slider("Wireframe edge cap",0,20000,8000,500)
            mesh_opacity = st.slider("Mesh opacity",0.0,1.0,0.18,0.02)
            enable_hover = st.toggle("Enable hover highlight (causes reruns)", value=False)
            st.form_submit_button("Apply viz", use_container_width=True)

        with st.form("audio_form"):
            st.subheader("7) Audio (optional)")
            st.caption("Dry → convolve with IR → Wet")
            dry_file = st.file_uploader("Dry audio", type=["wav","flac","ogg"], key="dry_audio")
            resample_mode = st.radio("Sample-rate match",["Resample audio to IR fs","Resample IR to audio fs"], index=0)
            wet_norm = st.checkbox("Normalize wet to −1 dBFS", value=False)
            wet_trim = st.slider("Wet tail trim (s)",0.0,3.0,0.0,0.1)
            st.form_submit_button("Apply audio", use_container_width=True)

        # --- Advanced (ODEON-ish) toggles ---
        with st.expander("Advanced (ODEON-ish)", expanded=False):

            air_model = st.selectbox("Air absorption model", ["flat", "iso9613"], index=0)
            air_temp_c = st.slider("Air temperature (°C)", -10.0, 40.0, 20.0, 0.5)
            air_rh_pct = st.slider("Relative humidity (%)", 0.0, 100.0, 50.0, 1.0)
            air_pressure_kpa = st.slider("Atmospheric pressure (kPa)", 90.0, 105.0, 101.325, 0.05)


            nee_all_bounces = st.checkbox("NEE at every bounce (probabilistic after N)", value=True)
            nee_bounces     = st.slider("Always sample NEE for first N bounces", 0, 10, 4)
            nee_prob        = st.slider("NEE probability after N (per bounce)", 0.0, 1.0, 0.30, 0.05)

            phys_normalization = st.checkbox("Physically normalized (MC unbiased)", value=False)
            band_mode = st.selectbox("Band mode", ["broadband", "octave", "third", "twelfth"], index=0)

            brdf_model = st.selectbox("BRDF model", ["specular+jitter", "spec+lambert"], index=0)
            scatter_ratio = st.slider("Scatter ratio (Lambert share)", 0.0, 1.0, 0.0, 0.05,
                                      help="Only used if BRDF is spec+lambert")
            transmission_paths = st.checkbox("Enable single transmission spawn", value=False)
            nee_mis = st.checkbox("Enable NEE MIS-lite", value=False)
            russian_roulette = st.checkbox("Russian roulette", value=True)
            scale_convention = st.selectbox("Scale convention", ["pressure", "intensity"], index=0)
            receiver_radius_m = st.slider("Receiver radius (m) for NEE", 0.01, 0.50, 0.10, 0.01,
                                          help="Approximates solid angle for NEE (stabilizes early reflections).")
            direct_mode = st.selectbox("Direct contribution", ["deterministic", "sampled"], index=0,
                                       help="Deterministic avoids double counting with NEE.")

        run = st.button("Run / Update simulation", type="primary", use_container_width=True)

    if f is None:
        st.info("Upload a mesh to begin.")
        return

    # --- Load mesh ---
    try:
        raw = f.read(); ext = f.name.split(".")[-1].lower()
        mesh = trimesh.load(io.BytesIO(raw), file_type=ext)
        if isinstance(mesh, trimesh.Scene):
            try:
                mesh = mesh.dump(concatenate=True)
            except Exception:
                geoms = [g for g in getattr(mesh, 'geometry', {}).values() if isinstance(g, trimesh.Trimesh)]
                mesh = trimesh.util.concatenate(tuple(geoms))
    except Exception as e:
        st.error(f"Failed to load mesh: {e}")
        return

    scale = 1.0 / float(units_per_meter)
    if abs(scale - 1.0) > 1e-12:
        mesh.apply_scale(scale)
    if (not mesh.is_watertight) and wrap_box:
        mn, mx = mesh.bounds; ext_bb = (mx - mn) + 2.0 * box_padding; ctr = (mx + mn) * 0.5
        box = trimesh.creation.box(extents=ext_bb); box.apply_translation(ctr - box.bounds.mean(axis=0))
        mesh = trimesh.util.concatenate((box, mesh))

    if not EMBREE_OK:
        try:
            import rtree  # noqa: F401
        except Exception:
            st.error("rtree required without Embree: conda install -c conda-forge rtree")
            st.stop()

    V = np.asarray(mesh.vertices); F = np.asarray(mesh.faces)
    mesh_key = mesh_hash_from_arrays(V, F)

    # --- Auto alpha & groups (cached) ---
    mats_tuple = (alpha_floor, alpha_ceiling, alpha_walls, alpha_default, nz_thresh, perc_margin)
    alpha_auto = auto_alpha_cached(V, F, mats_tuple)
    components = build_components_cached(V, F)
    group_ids = np.arange(len(components), dtype=int)
    faces_count = np.array([c.size for c in components], dtype=int)
    alpha_init = np.array([float(np.median(alpha_auto[c])) if c.size else alpha_default for c in components], dtype=float)

    # ===== Per-element materials (α, τ, scatter) =====
    st.subheader("Per-element materials")

    # ===== Tabs to avoid flooding the main UI =====
    scene_tab, materials_tab, results_tab, audio_tab = st.tabs(
        ["Scene", "Materials", "IR/EDC", "Audio"]
    )

    # Keep variables we’ll fill in the Materials tab
    alpha_face = None
    tau_face = None
    alpha_face_b_override = None
    tau_face_b_override = None
    bands_override = None

    with materials_tab:
        st.subheader("Per-element materials")

        # --- Library hook ---
        lib = builtin_library(OCTAVE_CENTERS)
        lib_names = list(lib.keys())
        default_mat_name = "Concrete" if "Concrete" in lib_names else lib_names[0]

        st.caption("Choose whether to assign **banded** library materials or use broadband sliders/table.")
        use_lib = st.checkbox(
            "Use library materials (octave bands) per element",
            value=False,
            help="Overrides the numeric α/τ table when ON. Recommended with Band mode = 'octave' (sidebar → Advanced)."
        )

        # ---- Pagination (large meshes) ----
        page_size = st.slider("Rows per page", 10, 200, 30, 10)
        num_elems = int(len(group_ids))
        num_pages = max(1, (num_elems + page_size - 1) // page_size)
        page = st.number_input("Page", min_value=1, max_value=num_pages, value=1, step=1)
        lo = (page - 1) * page_size
        hi = min(num_elems, lo + page_size)

        # ---- Build the editor table for the current page ----
        base_rows = []
        for g, c, a in zip(group_ids[lo:hi], faces_count[lo:hi], alpha_init[lo:hi]):
            base_rows.append({
                "Element ID": int(g),
                "Faces": int(c),
                "Material": default_mat_name,
                "α (absorption)": float(a),
                "τ (transmission)": 0.00,
            })

        # Column config: make Material a dropdown
        colcfg = {
            "Element ID": st.column_config.NumberColumn(disabled=True),
            "Faces": st.column_config.NumberColumn(disabled=True),
            "Material": st.column_config.SelectboxColumn(
                "Material",
                options=lib_names,
                required=True,
                help="Banded (octaves) material from library"
            ),
            "α (absorption)": st.column_config.NumberColumn(min_value=0.0, max_value=0.99, step=0.01),
            "τ (transmission)": st.column_config.NumberColumn(min_value=0.0, max_value=0.99, step=0.01),
        }

        edited = st.data_editor(
            base_rows,
            hide_index=True,
            num_rows="fixed",
            column_config=colcfg if use_lib else None,
            key=f"mat_editor_page_{page}",
            use_container_width=True,
        )

        # ---- Bulk apply helpers ----
        st.markdown("**Bulk assign**")
        colA, colB = st.columns([2,1])
        with colA:
            bulk_ids = st.multiselect(
                "Pick element IDs to assign",
                options=list(map(int, group_ids)),
                default=[],
                help="You can paste a comma-separated list too."
            )
        with colB:
            bulk_material = st.selectbox("Material to apply", lib_names, index=lib_names.index(default_mat_name))
            if st.button("Apply to selected"):
                # Save the selection to session for this run; propagated below when building arrays
                st.session_state["bulk_assign"] = {"ids": set(map(int, bulk_ids)), "mat": bulk_material}
                st.success(f"Assigned {bulk_material} to {len(bulk_ids)} elements (will apply on Run/Update).")
            else:
                st.session_state["bulk_assign"] = st.session_state.get("bulk_assign", None)

        # ---- Material Inspector (per-band view & chart) ----
        st.markdown("---")
        st.subheader("Material Inspector (bandwise)")
        mat_to_inspect = st.selectbox("Inspect material", lib_names, index=lib_names.index(default_mat_name))
        m = lib[mat_to_inspect]
        insp_df = {
            "Center (Hz)": OCTAVE_CENTERS,
            "α": [float(x) for x in m.alpha],
            "τ": [float(x) for x in m.tau],
            "s (scatter)": [float(x) for x in getattr(m, "scatter", np.zeros_like(m.alpha))],
        }
        st.dataframe(insp_df, use_container_width=True)
        # Simple band bar chart for α and τ
        fig_mat = go.Figure()
        fig_mat.add_bar(x=OCTAVE_CENTERS, y=insp_df["α"], name="α (absorption)")
        fig_mat.add_bar(x=OCTAVE_CENTERS, y=insp_df["τ"], name="τ (transmission)")
        fig_mat.update_layout(barmode="group", height=280, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig_mat, use_container_width=True)

        # ---- Build final per-face arrays (broadband always; banded optionally) ----
        # 1) base broadband from editor (or defaults)
        alpha_group = np.array([float(row["α (absorption)"]) for row in edited], dtype=float)
        tau_group   = np.array([float(row["τ (transmission)"]) for row in edited], dtype=float)

        # Clamp α+τ ≤ 0.99 (page subset only; full clamp applied when we expand to faces)
        sum_gt = alpha_group + tau_group
        if np.any(sum_gt > 0.99):
            scale_f = 0.99 / np.maximum(sum_gt, 1e-12)
            alpha_group *= scale_f
            tau_group   *= scale_f
            st.warning("Some rows had α+τ > 0.99 and were scaled.")

        # Start with whole-mesh broadband arrays
        alpha_face = np.full(len(F), float(alpha_default), dtype=np.float32)
        tau_face   = np.zeros(len(F), dtype=np.float32)

        # Apply current page’s broadband edits
        for row, g in zip(edited, group_ids[lo:hi]):
            gi = int(g)
            a = float(row["α (absorption)"])
            t = float(row["τ (transmission)"])
            # Clamp per element to keep α+τ ≤ 0.99
            if a + t > 0.99:
                s = 0.99 / (a + t + 1e-12)
                a *= s; t *= s
            alpha_face[components[gi]] = a
            tau_face[components[gi]] = t

        # Apply bulk assign (library path) if any
        bulk = st.session_state.get("bulk_assign")
        if use_lib and bulk:
            for gi in bulk["ids"]:
                if int(gi) < len(components):
                    mb = lib[bulk["mat"]]
                    a_bb, t_bb = to_broadband(mb.alpha, mb.tau, method="mean")
                    alpha_face[components[int(gi)]] = a_bb
                    tau_face[components[int(gi)]]   = t_bb

        # 2) Library band overrides if enabled
        if use_lib:
            bands_override = OCTAVE_CENTERS
            B = len(bands_override)
            alpha_face_b_override = np.zeros((len(F), B), dtype=np.float32)
            tau_face_b_override   = np.zeros((len(F), B), dtype=np.float32)

            # First: fill from current page’s “Material” column
            for row, g in zip(edited, group_ids[lo:hi]):
                gi = int(g)
                mat_name = row["Material"]
                mb = lib[mat_name]
                alpha_face_b_override[components[gi], :] = mb.alpha[None, :]
                tau_face_b_override[components[gi],   :] = mb.tau[None, :]
                # keep broadband mirror for preview/legacy
                a_bb, t_bb = to_broadband(mb.alpha, mb.tau, method="mean")
                alpha_face[components[gi]] = a_bb
                tau_face[components[gi]]   = t_bb

            # Then apply bulk (overrides page)
            if bulk:
                for gi in bulk["ids"]:
                    if int(gi) < len(components):
                        mb = lib[bulk["mat"]]
                        alpha_face_b_override[components[int(gi)], :] = mb.alpha[None, :]
                        tau_face_b_override[components[int(gi)],   :] = mb.tau[None, :]
                        a_bb, t_bb = to_broadband(mb.alpha, mb.tau, method="mean")
                        alpha_face[components[int(gi)]] = a_bb
                        tau_face[components[int(gi)]]   = t_bb
        else:
            alpha_face_b_override = None
            tau_face_b_override = None
            bands_override = None

        st.caption("Tip: change band mode in the sidebar → Advanced. Use ‘octave’ to trace with the library bands.")


    # Positions
    S = np.array([Sx, Sy, Sz], dtype=float); R = np.array([Rx, Ry, Rz], dtype=float)

    # --- 3D preview ---
    st.subheader("Scene preview")
    fig = make_fig(mesh, edge_cap=int(edge_cap), mesh_opacity=float(mesh_opacity))
    add_source_receiver(fig, S, R)

    c_left, c_right = st.columns([3, 1], gap="large")
    with c_right:
        sel_gid = st.selectbox("Select Element ID (stable)", options=list(group_ids),
                               index=0 if len(group_ids) else None, key="sel_gid")
        hover_gid = None
        if enable_hover and PLOTLY_EVENTS_OK and len(group_ids):
            hover_labels = [f"ID {int(i)} — faces {int(fc)} — α {alpha_group[int(i)]:.2f} — τ {tau_group[int(i)]:.2f}"
                            for i, fc in zip(group_ids, faces_count)]
            y = list(range(len(group_ids)))[::-1]
            px = go.Figure(go.Scatter(x=[0]*len(group_ids), y=y, mode="text", text=hover_labels,
                                      textposition="middle left", hovertext=[str(int(i)) for i in group_ids],
                                      hoverinfo="text"))
            px.update_xaxes(visible=False); px.update_yaxes(visible=False)
            px.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=10),
                             paper_bgcolor="#11151b", plot_bgcolor="#11151b",
                             font=dict(color="#e6edf3", size=12))
            events = plotly_events(px, hover_event=True, click_event=False, select_event=False,
                                   override_height=420, override_width=None, key="hover_list")
            if isinstance(events, list) and len(events) and "hovertext" in events[-1]:
                try: hover_gid = int(events[-1]["hovertext"])
                except Exception: hover_gid = None
            st.caption("Hover mode is optional. If noisy, turn off and use selectbox.")
        target_gid = hover_gid if (enable_hover and hover_gid is not None) else sel_gid
        if target_gid is not None and len(components):
            overlay_highlight(fig, V, F, components[int(target_gid)])
            a = float(alpha_group[int(target_gid)]); t = float(tau_group[int(target_gid)])
            Rmag = math.sqrt(max(0.0, 1.0 - a - t))
            st.caption(f"Highlighted Element **ID {int(target_gid)}** — Faces {int(faces_count[int(target_gid)])} — α {a:.2f} · τ {t:.2f} → |R|≈{Rmag:.2f}")
    with c_left:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Faces: {len(F):,} · Verts: {len(V):,} · Intersector: {'Embree' if EMBREE_OK else 'Triangle+Rtree'}")

    # --- Config + cache key ---
    cfg = SimConfig(

        air_model=str(air_model),
        air_temp_c=float(air_temp_c),
        air_rh_pct=float(air_rh_pct),
        air_pressure_kpa=float(air_pressure_kpa),

        nee_bounces=int(nee_bounces),
        nee_all_bounces=bool(nee_all_bounces),
        nee_prob=float(nee_prob),
        c=343.0, fs=int(fs), duration_s=float(dur), rays=int(rays), max_bounces=int(max_bounces),
        alpha_default=float(alpha_default), scattering_deg=float(scattering), air_db_per_m=float(air_dbm),
        rng_seed=int(seed), time_budget=float(time_budget), min_amp=float(min_amp), include_direct=bool(include_direct),
        synth_tail_if_sparse=bool(force_tail), synth_rt60_s=float(tail_rt), synth_tail_level_db=float(tail_lvl),
        # Advanced toggles
        phys_normalization=bool(phys_normalization),
        band_mode=str(band_mode),
        brdf_model=str(brdf_model),

        scatter_ratio=float(scatter_ratio),
        transmission_paths=bool(transmission_paths),
        nee_mis=bool(nee_mis),
        russian_roulette=bool(russian_roulette),
        scale_convention=str(scale_convention),
        receiver_radius_m=float(receiver_radius_m),
        direct_mode=str(direct_mode),
    )
    cfg_key = tuple(cfg.__dict__.values())
    alpha_crc = hashlib.sha1(alpha_face.tobytes()).hexdigest()[:12]
    tau_crc = hashlib.sha1(tau_face.tobytes()).hexdigest()[:12]
    sim_sig = (mesh_key, tuple(S.tolist()), tuple(R.tolist()), alpha_crc, tau_crc, cfg_key)

    # --- Run simulation ---
    run_clicked = run
    if run_clicked:
        h, arrivals, polylines = trace_cached(
            cfg_key, V, F, S, R,
            alpha_face.copy(), tau_face.copy(),
            band_mode=str(band_mode),
            alpha_face_b_override=alpha_face_b_override,
            tau_face_b_override=tau_face_b_override,
            bands_override=bands_override,
            scatter_face_b_override=scatter_face_b_override,
        )

        st.session_state.update({
            "h": h, "arrivals": arrivals, "polylines": polylines, "V": V, "F": F, "S": S, "R": R,
            "edge_cap": int(edge_cap), "mesh_opacity": float(mesh_opacity), "sim_sig": sim_sig,
            "band_mode": str(band_mode),
        })
        st.success(f"Tracing complete · {len(arrivals):,} arrivals")

    # --- Use cached if not run ---
    h_any = st.session_state.get("h")
    polylines = st.session_state.get("polylines", [])
    session_band_mode = st.session_state.get("band_mode", "broadband")

    def _to_broadband(h_in):
        if h_in is None:
            return None
        arr = np.asarray(h_in)
        if arr.ndim == 2:
            return arr.sum(axis=0).astype(np.float32)
        return arr.astype(np.float32)

    h_plot = _to_broadband(h_any)

    if h_plot is None:
        st.info("No IR yet — you can still audition the **dry** file below; press **Run / Update** to compute the IR and wet audio.")
    elif not run_clicked:
        st.info("Using cached results (no recompute). Press Run / Update to apply new settings.")

    # --- IR / EDC plots ---
    st.subheader("Impulse Response & Energy Decay")
    if h_plot is None or getattr(h_plot, "size", 0) == 0:
        st.info("No IR yet — press **Run / Update simulation** to compute the impulse response.")
    else:
        fs_i = int(fs)
        times = np.arange(len(h_plot)) / fs_i
        xmax = float(max(times[0], min(effective_t_end_from_edc(h_plot, fs_i, -60.0), times[-1])))
        col1, col2 = st.columns(2)
        with col1:
            x_ir, y_ir = decimate_line(times, h_plot, 200_000)
            fig_h = go.Figure(go.Scatter(x=x_ir, y=y_ir, mode="lines",
                                         line=dict(width=1.5, color="rgba(0,255,160,0.9)"),
                                         name="h(t)"))
            fig_h.update_layout(title="Impulse Response", paper_bgcolor="#000", plot_bgcolor="#000",
                                xaxis=dict(color="#cfd8dc", range=[float(times[0]), xmax]),
                                yaxis=dict(color="#cfd8dc"),
                                margin=dict(l=10, r=10, t=40, b=20), font=dict(color="#e6edf3"))
            st.plotly_chart(fig_h, use_container_width=True)
        with col2:
            edc = schroeder_edc(h_plot)
            edc_db = 10 * np.log10(np.maximum(edc, 1e-20))
            x_edc, y_edc = decimate_line(times, edc_db, 200_000)
            rt60 = estimate_rt60_from_edc(edc, fs_i)
            rt_text = f"RT60 ≈ {rt60:.2f} s" if rt60 is not None else "RT60 could not be estimated"
            fig_edc = go.Figure(go.Scatter(x=x_edc, y=y_edc, mode="lines",
                                           line=dict(width=2, color="#a8e6c7"),
                                           name="EDC [dB]"))
            fig_edc.update_layout(title=f"Energy Decay Curve ({rt_text})", paper_bgcolor="#000", plot_bgcolor="#000",
                                  xaxis=dict(color="#cfd8dc", range=[float(times[0]), xmax]),
                                  yaxis=dict(color="#cfd8dc", range=[-80, 1]),
                                  margin=dict(l=10, r=10, t=40, b=20), font=dict(color="#e6edf3"))
            st.plotly_chart(fig_edc, use_container_width=True)
        if sf is not None:
            ir_norm = h_plot / (np.max(np.abs(h_plot)) + 1e-12) * (10 ** (-3 / 20))
            st.download_button("Download IR (WAV)", data=wav_bytes(ir_norm, fs_i),
                               file_name="impulse_response.wav", mime="audio/wav")

    # --- Spectrogram & Convolution ---
    if SCIPY_OK:
        st.subheader("Spectrograms & Convolution")
        if dry_file is not None and sf is not None and SCIPY_OK:
            try:
                dry, sr_dry = sf.read(dry_file)
                if getattr(dry, "ndim", 1) > 1:
                    dry = dry.mean(axis=1)
                if dry.dtype != np.float32:
                    dry = dry.astype(np.float32, copy=False)

                c_dry_left, c_dry_right = st.columns(2, gap="large")
                with c_dry_left:
                    st.caption(f"Dry — {sr_dry} Hz"); st.audio(wav_bytes(dry, sr_dry), format="audio/wav")
                with c_dry_right:
                    st.plotly_chart(spectrogram_figure(dry, sr_dry, "Dry Spectrogram"), use_container_width=True)

                if h_plot is not None and getattr(h_plot, "size", 0) > 0:
                    fs_ir = int(fs)
                    if resample_mode == "Resample audio to IR fs":
                        dry_rs = resample_poly(dry, fs_ir, sr_dry) if sr_dry != fs_ir else dry
                        h_rs = h_plot; sr_wet = fs_ir
                    else:
                        h_rs = resample_poly(h_plot, sr_dry, fs_ir) if fs_ir != sr_dry else h_plot
                        dry_rs = dry; sr_wet = sr_dry

                    wet_full = fftconvolve(dry_rs, h_rs.astype(np.float32, copy=False), mode="full")

                    if wet_trim > 0:
                        cut = int(wet_trim * sr_wet)
                        if 0 < cut < wet_full.size:
                            wet_full = wet_full[:-cut]

                    wet = wet_full.astype(np.float32, copy=False)
                    if wet_norm:
                        peak = float(np.max(np.abs(wet))) + 1e-12
                        target = 10 ** (-1.0 / 20.0)  # -1 dBFS
                        wet *= (target / peak)

                    c_wet_left, c_wet_right = st.columns(2, gap="large")
                    with c_wet_left:
                        st.caption(f"Wet (convolved) — {sr_wet} Hz")
                        st.audio(wav_bytes(wet, sr_wet), format="audio/wav")
                    with c_wet_right:
                        st.plotly_chart(spectrogram_figure(wet, sr_wet, "Wet Spectrogram"), use_container_width=True)

                    st.download_button("Download Wet (WAV)", data=wav_bytes(wet, sr_wet),
                                       file_name="wet_convolved.wav", mime="audio/wav")
                else:
                    st.warning("No IR yet — press **Run / Update** to compute the IR and render wet audio.")
            except Exception as e:
                st.error(f"Audio processing failed: {e}")
        elif dry_file is not None:
            if not SCIPY_OK:
                st.warning("Install SciPy for convolution/spectrograms: pip install scipy")
            if sf is None:
                st.warning("Install soundfile for playback/IO: pip install soundfile")
        else:
            st.caption("Upload a dry audio file in the sidebar to enable playback and spectrograms.")

    # --- Ray-path preview (subset) ---
    if polylines:
        st.subheader("Ray-path preview (subset)")
        fig2 = make_fig(mesh, edge_cap=int(st.session_state.get("edge_cap", edge_cap)),
                        mesh_opacity=float(st.session_state.get("mesh_opacity", mesh_opacity)))
        add_source_receiver(fig2, S, R)
        xs, ys, zs = [], [], []
        for P in polylines[: int(vis_paths)]:
            for a in range(len(P) - 1):
                xs += [P[a, 0], P[a + 1, 0], None]
                ys += [P[a, 1], P[a + 1, 1], None]
                zs += [P[a, 2], P[a + 1, 2], None]
        if xs:
            fig2.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                        line=dict(width=1.5, color="rgba(0,255,160,0.8)"),
                                        name="Ray paths"))
        st.plotly_chart(fig2, use_container_width=True)

    # --- Debug / verification ---
    with st.expander("Debug & verification"):
        st.write({
            "mesh_key": mesh_key[:12],
            "alpha_crc": alpha_crc,
            "tau_crc": tau_crc,
            "alpha_mean/min/max": (float(np.mean(alpha_face)), float(np.min(alpha_face)), float(np.max(alpha_face))),
            "tau_mean/min/max": (float(np.mean(tau_face)), float(np.min(tau_face)), float(np.max(tau_face))),
            "seed": int(seed),
            "include_direct": include_direct,
            "band_mode": st.session_state.get("band_mode", "broadband"),
            "cfg_digest": hashlib.sha1(str(cfg_key).encode()).hexdigest()[:12],
            "sig_digest": hashlib.sha1(str(sim_sig).encode()).hexdigest()[:12],
        })
        st.caption("If alpha/tau CRCs change after edits + Run, your new values were used in tracing.")

if __name__ == "__main__":
    main()