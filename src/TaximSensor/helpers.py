import mujoco
from dataclasses import dataclass

import numpy as np
import trimesh
import cv2


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    rgb = bgr[..., ::-1]  # BGR -> RGB
    return rgb

def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    bgr = rgb[..., ::-1]  # RGB -> BGR
    return bgr

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


@dataclass(frozen=True)
class IndexedRasterData:
    """Compact geometry with face-corner attributes for software rasterization."""

    vertices_h: np.ndarray
    faces: np.ndarray
    uv_tris: np.ndarray
    normal_tris: np.ndarray


def build_trimesh_with_uvs_from_mujoco_mesh(
    model,
    mesh_id,
    *,
    return_raster_data: bool = False,
):
    """
    Construct a trimesh mesh directly from MuJoCo mesh buffers, preserving the
    same local-frame orientation as `build_trimesh_from_mujoco_mesh` while also
    attaching UVs from MuJoCo's texcoord buffers.

    MuJoCo stores texture coordinates per face-corner, not strictly per unique
    vertex, so we duplicate vertices per triangle corner to keep geometry and UVs
    exactly aligned.

    Returns
    -------
    mesh : trimesh.Trimesh
        Mesh in the same MuJoCo mesh-local coordinates as `model.mesh_vert`.
    uvs : np.ndarray
        Per-vertex UVs aligned with `mesh.vertices`.
    raster_data : IndexedRasterData, optional
        Returned when ``return_raster_data=True``. Geometric vertices remain
        indexed while UVs and normals are stored per face-corner.
    """
    start_vert = model.mesh_vertadr[mesh_id]
    num_vert = model.mesh_vertnum[mesh_id]
    vertices = np.asarray(
        model.mesh_vert[start_vert : start_vert + num_vert].reshape(-1, 3),
        dtype=np.float64,
    )

    start_face = model.mesh_faceadr[mesh_id]
    num_face = model.mesh_facenum[mesh_id]
    faces = np.asarray(
        model.mesh_face[start_face : start_face + num_face].reshape(-1, 3),
        dtype=np.int32,
    )

    texcoord_adr = int(model.mesh_texcoordadr[mesh_id])
    texcoord_num = int(model.mesh_texcoordnum[mesh_id])
    if texcoord_adr < 0 or texcoord_num <= 0:
        raise ValueError(f"MuJoCo mesh id {mesh_id} does not contain texture coordinates.")

    texcoords = np.asarray(
        model.mesh_texcoord[texcoord_adr : texcoord_adr + texcoord_num].reshape(-1, 2),
        dtype=np.float64,
    )

    face_texcoords = np.asarray(
        model.mesh_facetexcoord[start_face : start_face + num_face].reshape(-1, 3),
        dtype=np.int32,
    )

    uv_tris = texcoords[face_texcoords]
    expanded_uvs = uv_tris.reshape(-1, 2)
    if return_raster_data:
        # Taxim consumes UVs and normals from IndexedRasterData, so its
        # Trimesh can retain the original indexed geometry as well.
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        vertices_mm = vertices.astype(np.float32) * np.float32(1000.0)
        vertices_h = np.concatenate(
            (vertices_mm, np.ones((len(vertices_mm), 1), dtype=np.float32)),
            axis=1,
        )
        face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
        normal_tris = np.repeat(face_normals[:, None, :], 3, axis=1)
        raster_data = IndexedRasterData(
            vertices_h=np.ascontiguousarray(vertices_h),
            faces=np.ascontiguousarray(faces, dtype=np.int32),
            uv_tris=np.ascontiguousarray(uv_tris, dtype=np.float32),
            normal_tris=np.ascontiguousarray(normal_tris, dtype=np.float32),
        )
        return mesh, expanded_uvs, raster_data

    # Preserve the original helper behavior for callers that need one UV per
    # Trimesh vertex rather than the indexed raster representation.
    expanded_vertices = vertices[faces.reshape(-1)]
    expanded_faces = np.arange(len(expanded_vertices), dtype=np.int32).reshape(-1, 3)
    visual = trimesh.visual.texture.TextureVisuals(uv=expanded_uvs)
    mesh = trimesh.Trimesh(
        vertices=expanded_vertices,
        faces=expanded_faces,
        visual=visual,
        process=False,
    )
    return mesh, expanded_uvs


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


# Contact checking utilities
def _geom_ids_of_body(model, body_name):
    """Return a Python set of geom ids belonging to a given body name."""
    if type(body_name) is str:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    else:
        bid = body_name
    # model.geom_bodyid is an array mapping geom -> body id
    return {i for i in range(model.ngeom) if model.geom_bodyid[i] == bid}

def _bodies_in_contact(model, data, body_a, body_b):
    """
    True if any contact exists between any geoms belonging to body_a and body_b.
    Call after mj_step/mj_forward.
    """
    # cache sets on first call
    if not hasattr(_bodies_in_contact, "_cache"):
        _bodies_in_contact._cache = {}
    key_a = ("geomset", body_a)
    key_b = ("geomset", body_b)
    if key_a not in _bodies_in_contact._cache:
        _bodies_in_contact._cache[key_a] = _geom_ids_of_body(model, body_a)
    if key_b not in _bodies_in_contact._cache:
        _bodies_in_contact._cache[key_b] = _geom_ids_of_body(model, body_b)
    set_a = _bodies_in_contact._cache[key_a]
    set_b = _bodies_in_contact._cache[key_b]

    for k in range(data.ncon):
        con = data.contact[k]
        g1, g2 = con.geom1, con.geom2
        if (g1 in set_a and g2 in set_b) or (g1 in set_b and g2 in set_a):
            return True
    return False

def _body_ids_in_contact(model, data, body_a, body_b):
    """
    True if any contact exists between any geoms belonging to body_a and body_b.
    Call after mj_step/mj_forward.
    """
    # cache sets on first call
    if not hasattr(_body_ids_in_contact, "_cache"):
        _body_ids_in_contact._cache = {}
    key_a = ("geomset", body_a)
    key_b = ("geomset", body_b)
    if key_a not in _body_ids_in_contact._cache:
        _body_ids_in_contact._cache[key_a] = _geom_ids_of_body(model, body_a)
    if key_b not in _body_ids_in_contact._cache:
        _body_ids_in_contact._cache[key_b] = _geom_ids_of_body(model, body_b)
    set_a = _body_ids_in_contact._cache[key_a]
    set_b = _body_ids_in_contact._cache[key_b]

    for k in range(data.ncon):
        con = data.contact[k]
        g1, g2 = con.geom1, con.geom2
        if (g1 in set_a and g2 in set_b) or (g1 in set_b and g2 in set_a):
            return True
    return False

def _check_for_geom_contact(model, data, geom_name="digit_pad"):
    """True if any contact exists involving the named pad geom."""
    if not hasattr(_check_for_geom_contact, "_cache"):
        _check_for_geom_contact._cache = {}
    if geom_name not in _check_for_geom_contact._cache:
        try:
            _check_for_geom_contact._cache[geom_name] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )
        except Exception:
            raise RuntimeError(f'No geom named "{geom_name}" in the model.')

    geom_id = _check_for_geom_contact._cache[geom_name]

    for k in range(data.ncon):
        con = data.contact[k]
        if con.geom1 == geom_id or con.geom2 == geom_id:
            return True
    return False

def _penetration_stats_between_body_and_geom(model, data, body, geom):
    """
    Compute penetration statistics between any geoms of `body` and the named `geom`.
    Returns:
        count          : number of active contact points between these bodies
        max_penetration: largest (-dist) among contacts, clipped at 0 (meters)
        min_dist       : smallest raw signed distance (can be >0 if margin is used)
    Notes:
        - Requires that you have called mj_forward or mj_step before (to populate contacts).
        - If geom margins are >0, you may get contacts with dist > 0 (no real penetration).
    """
    # Get the list of geoms for the given body
    geoms_a = _geom_ids_of_body(model, body)
    try:
        geom_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    except Exception:
        raise RuntimeError(f'No geom named "{geom}" in the model.')
    geoms_b = {geom_b}

    count = 0
    min_dist = np.inf
    max_pen = 0.0

    for k in range(data.ncon):
        c = data.contact[k]
        g1, g2 = c.geom1, c.geom2
        if (g1 in geoms_a and g2 in geoms_b) or (g1 in geoms_b and g2 in geoms_a):
            count += 1
            d = float(c.dist)  # signed meters
            min_dist = min(min_dist, d)
            max_pen = max(max_pen, max(0.0, -d))  # penetration depth (meters)

    if count == 0:
        min_dist = np.inf
        max_pen = 0.0
    return count, max_pen, min_dist

def _penetration_stats_between_geoms(model, data, geom_a, geom_b):
    """
    Compute penetration statistics between any geoms of body_a and body_b.
    Returns:
        count          : number of active contact points between these bodies
        max_penetration: largest (-dist) among contacts, clipped at 0 (meters)
        min_dist       : smallest raw signed distance (can be >0 if margin is used)
    Notes:
        - Requires that you have called mj_forward or mj_step before (to populate contacts).
        - If geom margins are >0, you may get contacts with dist > 0 (no real penetration).
    """
    geom_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_a)
    geom_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_b)

    count = 0
    min_dist = np.inf
    max_pen = 0.0

    for k in range(data.ncon):
        c = data.contact[k]
        g1, g2 = c.geom1, c.geom2
        if (g1 == geom_a and g2 == geom_b) or (g1 == geom_b and g2 == geom_a):
            count += 1
            d = float(c.dist)  # signed meters
            min_dist = min(min_dist, d)
            max_pen = max(max_pen, max(0.0, -d))  # penetration depth (meters)

    if count == 0:
        min_dist = np.inf
        max_pen = 0.0
    return count, max_pen, min_dist

def _body_and_geom_penetrating(model, data, body, geom, penetration_tol=0.0):
    """
    True if body and geom interpenetrate by more than penetration_tol (meters).
    Set penetration_tol > 0 to 'relax' sensitivity.
    """
    _, max_pen, _ = _penetration_stats_between_body_and_geom(model, data, body, geom)
    return max_pen > penetration_tol
