# acoustics/geometry.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Try to use Embree; fall back to pure-triangle
try:
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector as _RayIntersector
    _RAY_BACKEND = "embree"
except Exception:
    import trimesh  # ensure trimesh is still imported
    try:
        from trimesh.ray.ray_triangle import RayMeshIntersector as _RayIntersector
        _RAY_BACKEND = "triangle"
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "trimesh ray intersector not available. Install trimesh and either pyembree or rtree.\n"
            "Conda users: conda install -c conda-forge pyembree rtree\n"
            "Pip users: pip install trimesh rtree"
        ) from e


# -----------------------------
# Mesh construction helper
# -----------------------------

def build_trimesh_from_arrays(V: np.ndarray, F: np.ndarray) -> "trimesh.Trimesh":
    """
    Construct a Trimesh from (V, F) without post-processing that could
    reorder faces/vertices. Normals and properties remain available lazily.
    """
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    return mesh


# -----------------------------
# Intersector wrapper
# -----------------------------

@dataclass
class Intersector:
    mesh: "trimesh.Trimesh"
    ray: _RayIntersector
    eps: float

    @classmethod
    def build(cls, mesh: "trimesh.Trimesh") -> "Intersector":
        """
        Build an intersector with a robust epsilon scaled to the mesh size.
        """
        ray = _RayIntersector(mesh)
        # Diagonal of AABB to scale epsilon; protects against self-hits
        diag = float(np.linalg.norm(mesh.extents))
        eps = max(1e-9, 1e-6 * (diag if diag > 0 else 1.0))
        return cls(mesh=mesh, ray=ray, eps=eps)

    def first_hit(self, origin: np.ndarray, direction: np.ndarray):
        """
        Cast a single ray and return (hit_point[3], face_index, distance).
        Returns (None, None, None) on miss.
        """
        # trimesh expects batches; ensure shape (1,3)
        o = np.asarray(origin, dtype=float).reshape(1, 3)
        d = np.asarray(direction, dtype=float).reshape(1, 3)

        # Single intersection along the ray
        locs, idx_ray, idx_tri = self.ray.intersects_location(
            o, d, multiple_hits=False
        )
        if len(locs) == 0:
            return None, None, None

        hit = np.asarray(locs[0], dtype=float)
        face = int(idx_tri[0])
        dist = float(np.linalg.norm(hit - origin))
        return hit, face, dist

    def visible(self, p: np.ndarray, q: np.ndarray) -> bool:
        """
        Visibility query between two points p->q (ignores grazing).
        Uses a small epsilon to step off the surface.
        """
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        d = q - p
        dist = float(np.linalg.norm(d))
        if dist <= self.eps:
            return True
        d_hat = d / dist
        # Nudge off p to avoid immediate self-hit
        o = p + d_hat * self.eps

        locs, *_ = self.ray.intersects_location(
            o.reshape(1, 3), d_hat.reshape(1, 3), multiple_hits=False
        )
        if len(locs) == 0:
            return True
        hit = np.asarray(locs[0], dtype=float)
        # If the first hit is beyond q (within tolerance), we consider it visible
        return np.linalg.norm(hit - o) > dist - self.eps


# -----------------------------
# Face connected components
# -----------------------------

def face_connected_components(mesh: "trimesh.Trimesh"):
    """
    Return a list of numpy arrays, each containing face indices for a
    connected component (by shared edges).
    """
    F = int(mesh.faces.shape[0])
    if F == 0:
        return []

    # Adjacency as (face_a, face_b) pairs
    adj = mesh.face_adjacency
    neighbors = [[] for _ in range(F)]
    for a, b in adj:
        a = int(a); b = int(b)
        neighbors[a].append(b)
        neighbors[b].append(a)

    # DFS over faces
    seen = np.zeros(F, dtype=bool)
    components = []
    for f in range(F):
        if seen[f]:
            continue
        stack = [f]
        seen[f] = True
        group = [f]
        while stack:
            u = stack.pop()
            for v in neighbors[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
                    group.append(v)
        components.append(np.asarray(group, dtype=np.int32))
    return components
