from __future__ import annotations
import numpy as np

# Prefer Embree; fall back to pure triangle
try:
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector as _RayIntersector
    _RAY_BACKEND = "embree"
except Exception:
    import trimesh  # ensure available
    try:
        from trimesh.ray.ray_triangle import RayMeshIntersector as _RayIntersector
        _RAY_BACKEND = "triangle"
    except Exception as e:
        raise ImportError(
            "trimesh ray intersector not available. Install trimesh and either pyembree or rtree."
        ) from e


def build_trimesh_from_arrays(V: np.ndarray, F: np.ndarray) -> "trimesh.Trimesh":
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=np.int64)
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


class Intersector:
    """Small wrapper around trimesh ray intersectors with robust epsilon."""
    def __init__(self, mesh: "trimesh.Trimesh", ray, eps: float):
        self.mesh = mesh
        self.ray = ray
        self.eps = eps

    @classmethod
    def build(cls, mesh: "trimesh.Trimesh") -> "Intersector":
        ray = _RayIntersector(mesh)
        diag = float(np.linalg.norm(mesh.extents))
        eps = max(1e-9, 1e-6 * (diag if diag > 0 else 1.0))
        return cls(mesh, ray, eps)

    def first_hit(self, origin: np.ndarray, direction: np.ndarray):
        o = np.asarray(origin, dtype=float).reshape(1, 3)
        d = np.asarray(direction, dtype=float).reshape(1, 3)
        locs, idx_ray, idx_tri = self.ray.intersects_location(o, d, multiple_hits=False)
        if len(locs) == 0:
            return None, None, None
        hit = np.asarray(locs[0], dtype=float)
        face = int(idx_tri[0])
        dist = float(np.linalg.norm(hit - origin))
        return hit, face, dist

    def visible(self, p: np.ndarray, q: np.ndarray) -> bool:
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        d = q - p
        dist = float(np.linalg.norm(d))
        if dist <= self.eps:
            return True
        d_hat = d / dist
        o = p + d_hat * self.eps
        locs, *_ = self.ray.intersects_location(o.reshape(1, 3), d_hat.reshape(1, 3), multiple_hits=False)
        if len(locs) == 0:
            return True
        hit = np.asarray(locs[0], dtype=float)
        return np.linalg.norm(hit - o) > dist - self.eps


def face_connected_components(mesh: "trimesh.Trimesh"):
    F = int(mesh.faces.shape[0])
    if F == 0:
        return []
    adj = mesh.face_adjacency
    neighbors = [[] for _ in range(F)]
    for a, b in adj:
        a = int(a); b = int(b)
        neighbors[a].append(b); neighbors[b].append(a)
    seen = np.zeros(F, dtype=bool)
    comps = []
    for f in range(F):
        if seen[f]:
            continue
        stack = [f]; seen[f] = True; group = [f]
        while stack:
            u = stack.pop()
            for v in neighbors[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
                    group.append(v)
        comps.append(np.asarray(group, dtype=np.int32))
    return comps