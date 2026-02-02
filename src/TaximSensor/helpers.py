import numpy as np
import trimesh
import cv2


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

def smooth_heightmap_mm(height_mm: np.ndarray,
                        valid_mask: np.ndarray,
                        inpaint_radius_px: int = 3,
                        d: int = 7,
                        sigma_space_px: float = 3.0,
                        sigma_height_mm: float = 0.15):
    """
    height_mm: (H,W) float32/float64 height map in mm, 0 outside contact (or invalid)
    valid_mask: (H,W) bool, True where height is valid/contact
    sigma_height_mm: how much depth variation to smooth across (in mm!)
    """
    h = height_mm.astype(np.float32, copy=False)

    # 1) inpaint invalid regions to prevent boundary shrink
    inv = (~valid_mask).astype(np.uint8) * 255
    h_fill = cv2.inpaint(h, inv, inpaint_radius_px, cv2.INPAINT_TELEA)

    # 2) edge-preserving smoothing in depth (mm)
    h_smooth = cv2.bilateralFilter(h_fill, d=d,
                                   sigmaColor=float(sigma_height_mm),
                                   sigmaSpace=float(sigma_space_px))

    # 3) restore invalid region (keep background as 0)
    out = np.zeros_like(h_smooth)
    out[valid_mask] = h_smooth[valid_mask]
    return out


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
    # m.remove_duplicate_faces() # Deprecated function
    m.update_faces(m.unique_faces())
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

def build_trimesh_from_mujoco_mesh(model, mesh_id):
    """
    Construct a trimesh.Mesh from a MuJoCo mesh using trimesh utilities.

    Parameters
    ----------
    model : mjModel
    mesh_id : int
    n_points : int
        Number of points to sample on the surface.
    seed : int | None
        Random seed for deterministic sampling.

    Returns
    -------
    np.ndarray
        (n_points, 3) float array of sampled points.
    """
    # --- build trimesh from MuJoCo mesh buffers (same as before) ---
    start_vert = model.mesh_vertadr[mesh_id]
    num_vert = model.mesh_vertnum[mesh_id]
    vertices = model.mesh_vert[start_vert : start_vert + num_vert].reshape(-1, 3)

    start_face = model.mesh_faceadr[mesh_id]
    num_face = model.mesh_facenum[mesh_id]
    faces = model.mesh_face[start_face : start_face + num_face].reshape(-1, 3)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    return mesh


def build_trimesh_from_mujoco_primitive(model, geom_id, geom_type):
    """
    Construct a trimesh.Mesh from a MuJoCo primitive geom.
    Parameters
    ----------
    model : mjModel
    geom_id : int
    geom_type : int
        model.geom_type[geom_id]
    n_points : int
        Number of points to sample on the surface.
    seed : int | None
        Random seed for deterministic sampling.

    Returns
    -------
    np.ndarray
        (n_points, 3) float array of sampled points.
    """
    geom_type_map = {
        2: "sphere",     # mjGEOM_SPHERE
        3: "capsule",    # mjGEOM_CAPSULE
        4: "ellipsoid",  # mjGEOM_ELLIPSOID
        5: "cylinder",   # mjGEOM_CYLINDER
        6: "box",        # mjGEOM_BOX
    }

    kind = geom_type_map.get(geom_type, None)
    size = model.geom_size[geom_id]

    if kind == "sphere":
        radius = float(size[0])
        mesh = trimesh.creation.icosphere(radius=radius, subdivisions=6)

    elif kind == "cylinder":
        radius = float(size[0])
        height = float(2.0 * size[1])  # MuJoCo uses half-length
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=1000)

    elif kind == "box":
        extents = (2.0 * size[:3]).astype(float)  # MuJoCo uses half-extents
        mesh = trimesh.creation.box(extents=extents)

    elif kind == "capsule":
        radius = float(size[0])
        height = float(2.0 * size[1])  # MuJoCo uses half-length (cyl part)
        mesh = trimesh.creation.capsule(radius=radius, height=height, count=[32, 16])

    elif kind == "ellipsoid":
        # Approximate ellipsoid by scaling a unit sphere
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        mesh.apply_scale(size[:3])

    else:
        raise NotImplementedError(
            f"Primitive geom_type '{kind}' not supported or unknown (type id: {geom_type})"
        )

    return mesh