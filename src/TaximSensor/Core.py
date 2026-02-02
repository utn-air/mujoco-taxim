import numpy as np
import trimesh
import cv2 
from numba import njit
import TaximSensor.Basics.params as pr
import TaximSensor.Basics.sensorParams as psp

def padding(img):
    """ pad one row & one col on each side """
    return np.pad(img, ((1, 1), (1, 1)), 'symmetric')

def generate_normals(height_map):
    """
    get the gradient (magnitude & direction) map from the height map
    """
    [h,w] = height_map.shape
    top = height_map[0:h-2,1:w-1] # z(x-1,y)
    bot = height_map[2:h,1:w-1] # z(x+1,y)
    left = height_map[1:h-1,0:w-2] # z(x,y-1)
    right = height_map[1:h-1,2:w] # z(x,y+1)
    dzdx = (bot-top)/2.0
    dzdy = (right-left)/2.0

    mag_tan = np.sqrt(dzdx**2 + dzdy**2)
    grad_mag = np.arctan(mag_tan)
    invalid_mask = mag_tan == 0
    valid_mask = ~invalid_mask
    grad_dir = np.zeros((h-2,w-2))
    grad_dir[valid_mask] = np.arctan2(dzdx[valid_mask]/mag_tan[valid_mask], dzdy[valid_mask]/mag_tan[valid_mask])

    grad_mag = padding(grad_mag)
    grad_dir = padding(grad_dir)
    return grad_mag, grad_dir

def interpolate(img):
    """
    fill the zero value holes with interpolation
    """
    x = np.arange(0, img.shape[1])
    y = np.arange(0, img.shape[0])
    # mask invalid values
    array = np.ma.masked_where(img == 0, img)
    xx, yy = np.meshgrid(x, y)
    # get the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = img[~array.mask]

    GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                                (xx, yy),
                                    method='linear', fill_value = 0) # cubic # nearest # linear
    
    return GD1

def deformApprox(pressing_height_mm, height_map, gel_map, contact_mask):
    zq = height_map.copy()
    zq_back = zq.copy()
    pressing_height_pix = pressing_height_mm/psp.pixmm
    # contact mask which is a little smaller than the real contact mask
    mask = (zq-(gel_map)) > pressing_height_pix * pr.contact_scale
    mask = mask & contact_mask

    # approximate soft body deformation with pyramid gaussian_filter
    for i in range(len(pr.pyramid_kernel_size)):
        zq = cv2.GaussianBlur(zq.astype(np.float32),(pr.pyramid_kernel_size[i],pr.pyramid_kernel_size[i]),0)
        zq[mask] = zq_back[mask]
    zq = cv2.GaussianBlur(zq.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)
    contact_height = zq - gel_map
    return zq, mask, contact_height

@njit(cache=True)
def zbuf_rasterize_numba(us, vs, zs, H, W):
    """
    us,vs,zs: (F,3) float32 arrays of triangle vertex coords in pixel space (u,v) and depth z (mm)
    Returns zbuf (H,W) float32 min-z per pixel.
    """
    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    F = us.shape[0]
    for f in range(F):
        u0, u1, u2 = us[f, 0], us[f, 1], us[f, 2]
        v0, v1, v2 = vs[f, 0], vs[f, 1], vs[f, 2]
        z0, z1, z2 = zs[f, 0], zs[f, 1], zs[f, 2]

        # quick reject if triangle is entirely above gel and you only care about z<0
        # (optional, but often helps)
        if z0 >= 0.0 and z1 >= 0.0 and z2 >= 0.0:
            continue

        # bbox in pixel space
        umin = int(np.floor(min(u0, u1, u2)))
        umax = int(np.ceil (max(u0, u1, u2)))
        vmin = int(np.floor(min(v0, v1, v2)))
        vmax = int(np.ceil (max(v0, v1, v2)))

        if umax < 0 or umin >= W or vmax < 0 or vmin >= H:
            continue

        if umin < 0: umin = 0
        if umax > W - 1: umax = W - 1
        if vmin < 0: vmin = 0
        if vmax > H - 1: vmax = H - 1

        denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
        if denom == 0.0:
            continue
        inv_denom = 1.0 / denom

        for v in range(vmin, vmax + 1):
            py = v + 0.5
            for u in range(umin, umax + 1):
                px = u + 0.5

                w0 = ((v1 - v2) * (px - u2) + (u2 - u1) * (py - v2)) * inv_denom
                w1 = ((v2 - v0) * (px - u2) + (u0 - u2) * (py - v2)) * inv_denom
                w2 = 1.0 - w0 - w1

                if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                    z = w0 * z0 + w1 * z1 + w2 * z2
                    if z < zbuf[v, u]:
                        zbuf[v, u] = z

    return zbuf

def rasterize_depth_from_trimesh(
    mesh: trimesh.Trimesh,
    sTo: np.ndarray,
    H: int,
    W: int,
    pixmm: float,
) -> np.ndarray:
    """
    Returns zbuf (H,W) storing minimum z in *sensor frame* for each pixel.
    z is in the same units as mesh coordinates (typically meters or mm).
    """
    # --- Transform mesh vertices into sensor frame ---

    V = (mesh.vertices.astype(np.float32) * 1000)      # (V,3)
    F = mesh.faces.astype(np.int32)                    # (F,3)

    Vh = np.c_[V, np.ones((len(V), 1), dtype=np.float32)]
    Vs = (sTo @ Vh.T).T[:, :3]                         # (V,3) in sensor frame

    # Gather triangles in sensor frame: (F,3,3)
    tris = Vs[F]

    # --- Project triangle vertices to pixel coords (float) ---
    us = tris[..., 0] / pixmm + (W * 0.5)
    vs = tris[..., 1] / pixmm + (H * 0.5)
    zs = tris[..., 2].astype(np.float32)

    # --- Cull irrelevant triangles ---
    u_min = np.min(us, axis=1); u_max = np.max(us, axis=1)
    v_min = np.min(vs, axis=1); v_max = np.max(vs, axis=1)

    in_img = (u_max >= 0) & (u_min < psp.w) & (v_max >= 0) & (v_min < psp.h)
    penetrates = np.min(zs, axis=1) < 0.0  # at least one vertex below gel plane

    keep = in_img & penetrates
    us_k = us[keep]; vs_k = vs[keep]; zs_k = zs[keep]
    us = us_k; vs = vs_k; zs = zs_k

    zbuf = zbuf_rasterize_numba(us, vs, zs, H, W)
    
    return zbuf

def heightmap_from_zbuf(zbuf: np.ndarray, pixmm: float) -> np.ndarray:
    """
    Matches your convention: keep only points below gel surface (z < 0),
    and height = -z / pixmm.
    """
    height = np.zeros_like(zbuf, dtype=np.float32)
    hit = np.isfinite(zbuf) & (zbuf < 0.0)
    height[hit] = -zbuf[hit] / pixmm
    return height

import numpy as np

def pointcloud_from_zbuf(
    zbuf,
    pixmm,
    *,
    contact_only=True,
    n_points=None,
    rng=None,
    normalize = True,
    z_max = 3.0,
    include_border=False,
    border_thickness=1,
):
    """
    Build a point cloud (sensor frame, mm) from a z-buffer.

    Args:
      zbuf: (H,W) depth in sensor frame (mm), +inf for invalid
      pixmm: mm per pixel
      contact_only: if True, keep only z < 0 (penetration/contact)
      n_points: if set, randomly downsample to this many points (applied AFTER border merge)
      rng: np.random.Generator or None
      include_border: if True, also include points along the image boundary (a 'frame' of points)
      border_thickness: boundary thickness in pixels (>=1)
      border_z_fill: z value (mm) used for border pixels where zbuf is invalid (inf).
                     Common choices:
                       0.0  -> gel plane
                       min(zbuf[finite]) -> lowest visible depth (more conservative)
    Returns:
      pts: (N,3) float32 point cloud in sensor frame (mm)
    """
    H, W = zbuf.shape
    zbuf_f = zbuf.astype(np.float32, copy=False)

    # --- main valid mask ---
    valid = np.isfinite(zbuf_f)
    if contact_only:
        valid &= (zbuf_f < 0.0)

    vv, uu = np.nonzero(valid)
    if vv.size > 0:
        z = zbuf_f[vv, uu]
        x = (uu.astype(np.float32) - (W * 0.5)) * float(pixmm)
        y = (vv.astype(np.float32) - (H * 0.5)) * float(pixmm)
        pts_main = np.stack([x, y, z], axis=1)
    else:
        pts_main = np.zeros((0, 3), dtype=np.float32)

    # --- optional downsample (after border merge) ---
    if n_points is not None and pts_main.shape[0] > int(n_points):
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(pts_main.shape[0], size=int(n_points), replace=False)
        pts_main = pts_main[idx]

    # --- optional border points ---
    if include_border:
        t = int(border_thickness)
        if t < 1:
            t = 1
        t = min(t, H // 2, W // 2)  # avoid weird cases

        border = np.zeros((H, W), dtype=bool)
        border[:t, :] = True
        border[-t:, :] = True
        border[:, :t] = True
        border[:, -t:] = True

        bv, bu = np.nonzero(border)
        zb = zbuf_f[bv, bu]

        # Fill all border depths with 0
        zb_filled = np.zeros_like(zb, dtype=np.float32)

        xb = (bu.astype(np.float32) - (W * 0.5)) * float(pixmm)
        yb = (bv.astype(np.float32) - (H * 0.5)) * float(pixmm)
        pts_border = np.stack([xb, yb, zb_filled.astype(np.float32)], axis=1)

        pts = np.vstack([pts_main, pts_border]) if pts_main.size else pts_border
    else:
        pts = pts_main
    
    if normalize:
        w_min = (0 - (W*0.5)) * float(pixmm)
        w_max = (W - (W*0.5)) * float(pixmm)
        h_min = (0 - (H*0.5)) * float(pixmm)
        h_max = (H - (H*0.5)) * float(pixmm)
        z_min = 0.0
        z_max = z_max
        pts[:,0] = (pts[:,0] - w_min) / (w_max - w_min)
        pts[:,1] = (pts[:,1] - h_min) / (h_max - h_min)
        pts[:,2] = (pts[:,2] - z_min) / (z_max - z_min)

    
    return pts.astype(np.float32)
