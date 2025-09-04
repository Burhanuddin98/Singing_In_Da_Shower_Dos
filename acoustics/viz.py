# acoustics/viz.py
from __future__ import annotations
import io, math
import numpy as np
import plotly.graph_objects as go

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    from scipy.signal import spectrogram as _spec
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# -----------------------------
# Colors / styles
# -----------------------------

MESH_GREEN = "rgb(0,255,128)"
MESH_GREEN_FAINT = "rgba(0,255,128,0.25)"
GRID_C = "rgba(120,160,130,0.18)"
PURPLE = "rgb(200,120,255)"
RAY_NEON_ORANGE = "rgba(255,120,0,1.0)"


# -----------------------------
# Mesh wireframe utilities
# -----------------------------

def _unique_edges_for_faces(F: np.ndarray, faces_idx: np.ndarray):
    edges = set()
    for fi in faces_idx:
        i, j, k = F[fi]
        edges.add(tuple(sorted((int(i), int(j)))))
        edges.add(tuple(sorted((int(j), int(k)))))
        edges.add(tuple(sorted((int(k), int(i)))))
    return list(edges)

def _edge_lines(V: np.ndarray, edges):
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [V[a, 0], V[b, 0], None]
        ys += [V[a, 1], V[b, 1], None]
        zs += [V[a, 2], V[b, 2], None]
    return xs, ys, zs

def _sample_edges(V: np.ndarray, F: np.ndarray, max_edges: int = 8000):
    edges = set()
    for tri in F:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((j, k))))
        edges.add(tuple(sorted((k, i))))
    edges = list(edges)
    if len(edges) > max_edges:
        idx = np.random.default_rng(0).choice(
            len(edges), size=max_edges, replace=False
        )
        edges = [edges[i] for i in idx]
    return _edge_lines(V, edges)


# -----------------------------
# Plotly scene
# -----------------------------

def make_fig(mesh, edge_cap: int = 8000, mesh_opacity: float = 0.18) -> "go.Figure":
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    fig = go.Figure()

    # Mesh surface
    fig.add_trace(go.Mesh3d(
        x=V[:, 0], y=V[:, 1], z=V[:, 2],
        i=F[:, 0], j=F[:, 1], k=F[:, 2],
        color=MESH_GREEN,
        opacity=mesh_opacity,
        flatshading=True,
        name="Geometry"
    ))

    # Wireframe
    xe, ye, ze = _sample_edges(V, F, edge_cap)
    fig.add_trace(go.Scatter3d(
        x=xe, y=ye, z=ze,
        mode="lines",
        line=dict(width=1, color=MESH_GREEN_FAINT),
        name="Wireframe"
    ))

    fig.update_layout(
        paper_bgcolor="#000", plot_bgcolor="#000",
        scene=dict(
            xaxis=dict(showbackground=True, backgroundcolor="#000",
                       gridcolor=GRID_C, zerolinecolor=GRID_C, color="#cfd8dc"),
            yaxis=dict(showbackground=True, backgroundcolor="#000",
                       gridcolor=GRID_C, zerolinecolor=GRID_C, color="#cfd8dc"),
            zaxis=dict(showbackground=True, backgroundcolor="#000",
                       gridcolor=GRID_C, zerolinecolor=GRID_C, color="#cfd8dc"),
            bgcolor="#000",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(font=dict(color="#e6edf3"))
    )
    return fig


def add_source_receiver(fig: "go.Figure", S: np.ndarray, R: np.ndarray):
    fig.add_trace(go.Scatter3d(
        x=[S[0]], y=[S[1]], z=[S[2]],
        mode="markers",
        marker=dict(size=6, color="rgb(255,32,64)"),
        name="Source"
    ))
    fig.add_trace(go.Scatter3d(
        x=[R[0]], y=[R[1]], z=[R[2]],
        mode="markers",
        marker=dict(size=6, color="rgb(255,255,255)"),
        name="Receiver"
    ))


def overlay_highlight(fig: "go.Figure", V: np.ndarray, F: np.ndarray, faces_idx: np.ndarray):
    edges = _unique_edges_for_faces(F, faces_idx)
    xs, ys, zs = _edge_lines(V, edges)
    if xs:
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=5, color=PURPLE),
            name="Highlight",
            showlegend=False
        ))


# -----------------------------
# Audio utilities
# -----------------------------

def wav_bytes(y: np.ndarray, sr: int) -> io.BytesIO:
    """Convert mono float32 array to WAV in-memory buffer."""
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y /= peak
    if sf is None:
        raise RuntimeError("soundfile not available; pip install soundfile")
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


# -----------------------------
# Spectrogram
# -----------------------------

def spectrogram_figure(y: np.ndarray, sr: int, title: str,
                       nperseg: int = 1024, noverlap: int = 768,
                       max_cols: int = 1200, max_rows: int = 1024) -> "go.Figure":
    if not SCIPY_OK or y is None or y.size == 0:
        return go.Figure()

    y = np.asarray(y, dtype=np.float32)
    f, t, S = _spec(y, fs=sr, window="hann",
                    nperseg=nperseg, noverlap=noverlap,
                    mode="magnitude", detrend=False, scaling="density")
    S_db = 20 * np.log10(S + 1e-12)

    # Decimate columns if too wide
    T = S_db.shape[1]
    if T > max_cols:
        factor = int(np.ceil(T / max_cols))
        new_T = T // factor
        cut = new_T * factor
        S_db = S_db[:, :cut].reshape(S_db.shape[0], new_T, factor).mean(axis=2)
        t = t[:cut].reshape(new_T, factor).mean(axis=1)

    # Decimate rows if too tall
    F = S_db.shape[0]
    if F > max_rows:
        factor = int(np.ceil(F / max_rows))
        new_F = F // factor
        cut = new_F * factor
        S_db = S_db[:cut, :].reshape(new_F, factor, S_db.shape[1]).mean(axis=1)
        f = f[:cut].reshape(new_F, factor).mean(axis=1)

    mask = f > 0
    if np.any(mask):
        f = f[mask]
        S_db = S_db[mask, :]

    x0, x1 = (float(t[0]), float(t[-1])) if t.size else (0.0, 0.0)
    top = float(np.max(S_db)) if S_db.size else 0.0

    fig = go.Figure(data=[go.Heatmap(
        x=t, y=f, z=S_db,
        colorscale="Cividis",
        colorbar=dict(title="dB"),
        zmin=top - 80, zmax=top
    )])
    fig.update_layout(
        title=title,
        paper_bgcolor="#000", plot_bgcolor="#000",
        font=dict(color="#e6edf3"),
        margin=dict(l=50, r=20, t=40, b=40),
        xaxis=dict(title="Time (s)", color="#e6edf3", range=[x0, x1]),
        yaxis=dict(title="Freq (Hz)", type="log", color="#e6edf3")
    )
    return fig
