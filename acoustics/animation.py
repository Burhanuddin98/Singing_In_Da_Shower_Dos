# acoustics/animation.py
from __future__ import annotations
import os
import tempfile
import numpy as np

import plotly.graph_objects as go

# Optional deps for GIF export (kept safe on server-side)
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (ensures 3D backend)
    MPL_OK = True
except Exception:
    MPL_OK = False


# -----------------------------
# Internal helpers
# -----------------------------

def _interp_points_for_animation(points: np.ndarray, steps_per_segment: int) -> list[np.ndarray]:
    """Return a list of partial polylines growing along the path for animation frames."""
    frames_pts: list[np.ndarray] = []
    if points is None or points.ndim != 2 or points.shape[0] < 2:
        return frames_pts
    frames_pts.append(points[0:1, :])
    for i in range(points.shape[0] - 1):
        a, b = points[i], points[i + 1]
        for s in range(1, int(steps_per_segment) + 1):
            t = s / float(steps_per_segment)
            p = a * (1.0 - t) + b * t
            frames_pts.append(np.vstack([points[: i + 1], p[None, :]]))
    return frames_pts


def _mpl_edges(F: np.ndarray):
    """Unique undirected edge list from face index triplets."""
    edges = set()
    for tri in F:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((j, k))))
        edges.add(tuple(sorted((k, i))))
    return list(edges)


def _mpl_frame_png(V: np.ndarray, edges, seg: np.ndarray, out_png: str, bounds):
    """Render a single 3D frame (wireframe + partial ray) to a PNG (black background)."""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    fig = plt.figure(figsize=(9.6, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    # wireframe (faint green)
    for a, b in edges:
        xs = [V[a, 0], V[b, 0]]
        ys = [V[a, 1], V[b, 1]]
        zs = [V[a, 2], V[b, 2]]
        ax.plot(xs, ys, zs, linewidth=0.5, color=(0, 1, 0, 0.25))

    # ray segment (bright green)
    ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], linewidth=3.0, color=(0, 1, 0, 0.95))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    plt.tight_layout(pad=0)
    fig.savefig(out_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


# -----------------------------
# Public API
# -----------------------------

def create_ray_animation_figure(base_fig: go.Figure,
                                polyline: np.ndarray,
                                steps_per_segment: int = 6) -> go.Figure:
    """
    Returns a Plotly Figure with frames that animate a single polyline (ray).
    `base_fig` should already contain your mesh and source/receiver markers.
    """
    fig = go.Figure(base_fig)
    frames_pts = _interp_points_for_animation(polyline, int(steps_per_segment))
    if not frames_pts:
        return fig

    # Add a single line trace we will update across frames (neon orange lines)
    fig.add_trace(go.Scatter3d(
        x=frames_pts[0][:, 0], y=frames_pts[0][:, 1], z=frames_pts[0][:, 2],
        mode="lines",
        line=dict(width=8, color="rgba(255,120,0,1.0)"),
        name="Ray"
    ))
    ray_trace_index = len(fig.data) - 1

    frames = []
    for k, pts in enumerate(frames_pts):
        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="lines",
                line=dict(width=8, color="rgba(255,120,0,1.0)")
            )],
            traces=[ray_trace_index],
            name=str(k)
        ))
    fig.frames = frames

    # Controls
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 40, "redraw": True},
                                 "fromcurrent": True,
                                 "transition": {"duration": 0}}]},
                {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate",
                                   "transition": {"duration": 0}}]},
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "showactive": False,
            "x": 0.1, "xanchor": "right",
            "y": 0, "yanchor": "top",
        }],
        sliders=[{
            "steps": [{"method": "animate",
                       "args": [[str(i)], {"frame": {"duration": 0, "redraw": True},
                                           "mode": "immediate"}],
                       "label": str(i)} for i in range(len(frames))],
            "transition": {"duration": 0},
            "x": 0, "len": 1.0, "pad": {"t": 30, "b": 10},
        }]
    )
    return fig


def render_gif_one_ray_matplotlib(V: np.ndarray, F: np.ndarray,
                                  polyline: np.ndarray, out_path: str,
                                  steps_per_segment: int = 6, fps: int = 12) -> str:
    """
    Server-side safe GIF of a single ray along the mesh wireframe.
    Returns the output path. Requires imageio + matplotlib.
    """
    if imageio is None or not MPL_OK:
        raise RuntimeError("Install for GIF export: pip install imageio matplotlib")

    frames_pts = _interp_points_for_animation(polyline, int(steps_per_segment))
    if not frames_pts:
        raise RuntimeError("Selected ray has no segments.")

    edges = _mpl_edges(F)

    # Cubic bounds around the mesh for consistent framing
    V = np.asarray(V)
    mins = V.min(axis=0)
    maxs = V.max(axis=0)
    ctr = (mins + maxs) / 2.0
    ext = (maxs - mins).max() / 2.0
    bounds = ((ctr[0] - ext, ctr[0] + ext),
              (ctr[1] - ext, ctr[1] + ext),
              (ctr[2] - ext, ctr[2] + ext))

    tmpdir = tempfile.mkdtemp(prefix="rt_frames_")
    pngs = []
    for k, seg in enumerate(frames_pts):
        png = os.path.join(tmpdir, f"f_{k:05d}.png")
        _mpl_frame_png(V, edges, seg, png, bounds)
        pngs.append(png)

    imgs = [imageio.imread(p) for p in pngs]
    imageio.mimsave(out_path, imgs, duration=1.0 / max(1, int(fps)))
    return out_path
