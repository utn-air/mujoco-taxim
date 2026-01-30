import numpy as np
import trimesh


def _unique_edges_with_face_adjacency(faces: np.ndarray):
    """
    Build unique undirected edges and (up to) 2 adjacent face indices per edge.

    Returns:
      edges_u:      (E,2) int64 unique undirected edges (v0 < v1)
      edge_faces:   (E,2) int64 adjacent faces, -1 if none (boundary)
      edge_counts:  (E,)  int64 number of incident faces (1 => boundary, 2 => interior, >2 non-manifold)
    """
    faces = np.asarray(faces, dtype=np.int64)
    F = faces.shape[0]
    f_idx = np.arange(F, dtype=np.int64)

    # All directed edges per face (3 edges per face)
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    edges = np.vstack((e01, e12, e20))  # (3F,2)

    # Undirected: sort vertices within each edge
    edges = np.sort(edges, axis=1)      # (3F,2), now v0 <= v1

    # Face index per edge record
    face_of_edge = np.concatenate((f_idx, f_idx, f_idx), axis=0)  # (3F,)

    # Unique edges
    edges_u, inv, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    E = edges_u.shape[0]

    # For each unique edge, store up to two adjacent faces
    edge_faces = np.full((E, 2), -1, dtype=np.int64)

    # Fill adjacency: stable approach by sorting inv, then taking first two faces per group
    order = np.argsort(inv)
    inv_s = inv[order]
    face_s = face_of_edge[order]

    # Walk groups
    start = 0
    while start < len(inv_s):
        e_id = inv_s[start]
        end = start + 1
        while end < len(inv_s) and inv_s[end] == e_id:
            end += 1
        # faces incident to this edge:
        # there may be duplicates if mesh is degenerate, so unique them
        fs = face_s[start:end]
        # unique but keep order
        # (small group, python unique is fine)
        uniq = []
        for f in fs:
            if f not in uniq:
                uniq.append(int(f))
            if len(uniq) == 2:
                break
        if len(uniq) > 0:
            edge_faces[e_id, 0] = uniq[0]
        if len(uniq) > 1:
            edge_faces[e_id, 1] = uniq[1]
        start = end

    return edges_u, edge_faces, counts


def _boundary_vertex_mask(mesh: trimesh.Trimesh) -> np.ndarray:
    edges_u, edge_faces, counts = _unique_edges_with_face_adjacency(mesh.faces)
    boundary_edges = edges_u[counts == 1]
    mask = np.zeros(len(mesh.vertices), dtype=bool)
    if len(boundary_edges) > 0:
        mask[boundary_edges[:, 0]] = True
        mask[boundary_edges[:, 1]] = True
    return mask


def _crease_vertex_mask(mesh: trimesh.Trimesh, feature_angle_deg: float) -> np.ndarray:
    """
    Mark vertices that belong to edges with dihedral angle > feature_angle_deg.
    Uses only faces + face_normals (no trimesh edge helpers).
    """
    edges_u, edge_faces, counts = _unique_edges_with_face_adjacency(mesh.faces)

    # Only interior-ish edges with at least 2 adjacent faces
    interior = counts >= 2
    if not np.any(interior):
        return np.zeros(len(mesh.vertices), dtype=bool)

    f0 = edge_faces[interior, 0]
    f1 = edge_faces[interior, 1]
    ok = (f0 >= 0) & (f1 >= 0)
    if not np.any(ok):
        return np.zeros(len(mesh.vertices), dtype=bool)

    # Ensure normals exist
    fn = mesh.face_normals
    n0 = fn[f0[ok]]
    n1 = fn[f1[ok]]

    dot = np.einsum("ij,ij->i", n0, n1)
    dot = np.clip(dot, -1.0, 1.0)
    ang = np.arccos(dot)

    sharp = ang > np.deg2rad(feature_angle_deg)

    sharp_edges = edges_u[interior][ok][sharp]
    mask = np.zeros(len(mesh.vertices), dtype=bool)
    if len(sharp_edges) > 0:
        mask[sharp_edges[:, 0]] = True
        mask[sharp_edges[:, 1]] = True
    return mask


def smooth_mesh(
    mesh: trimesh.Trimesh,
    iterations: int = 10,
    method: str = "taubin",          # "laplacian" or "taubin"
    lam: float = 0.5,
    mu: float = -0.53,
    preserve_creases: bool = True,
    feature_angle_deg: float = 35.0,
    freeze_boundary: bool = True,
) -> trimesh.Trimesh:
    """
    Version-independent vertex-position smoothing.

    NOTE: This moves vertices (changes geometry). For rendering-only smoothing,
    apply to a copy and keep the original mesh for physics/contact if needed.
    """
    if iterations <= 0:
        return mesh.copy()

    m = mesh.copy()
    m.remove_duplicate_faces()
    m.remove_degenerate_faces()
    m.remove_unreferenced_vertices()
    m.fix_normals()  # ensure face_normals exist

    # Neighbor list (this one is widely supported across trimesh versions)
    nbrs = m.vertex_neighbors

    frozen = np.zeros(len(m.vertices), dtype=bool)
    if freeze_boundary:
        frozen |= _boundary_vertex_mask(m)
    if preserve_creases:
        frozen |= _crease_vertex_mask(m, feature_angle_deg)

    V = m.vertices.astype(np.float64)

    def laplacian_step(Vcur: np.ndarray, step: float) -> np.ndarray:
        Vnew = Vcur.copy()
        for i in range(Vcur.shape[0]):
            if frozen[i]:
                continue
            nb = nbrs[i]
            if len(nb) == 0:
                continue
            avg = Vcur[np.asarray(nb, dtype=np.int64)].mean(axis=0)
            Vnew[i] = Vcur[i] + step * (avg - Vcur[i])
        return Vnew

    if method.lower() == "laplacian":
        for _ in range(iterations):
            V = laplacian_step(V, lam)

    elif method.lower() == "taubin":
        for _ in range(iterations):
            V = laplacian_step(V, lam)
            V = laplacian_step(V, mu)
    else:
        raise ValueError("method must be 'laplacian' or 'taubin'")

    m.vertices = V.astype(np.float32)
    m.fix_normals()
    return m


def invert_homogeneous_matrix(T):
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3]  = -R.T @ t
    return T_inv