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

import numpy as np
import cv2

def pointcloud_from_zbuf_with_normals(
    zbuf,
    pixmm,
    *,
    normalize=True,
    z_max=3.0,
    contact_only=True,
    n_points=None,
    rng=None,
    include_border=False,
    border_thickness=1,
    border_z_fill=0.0,
    inpaint_for_normals=True,
    inpaint_radius_px=3,
    # ----------------------------
    # Reconstructed-mesh roughness tuning knobs
    # ----------------------------
    roughness_enable=True,
    roughness_alpha=0.1,
    # spatial scales (in pixels) for multi-scale correlated noise
    roughness_sigmas_px=(4.0, 6.0, 8.0, 12.0, 16.0),
    # weights per sigma; if None, uses 1/(2^k) style and normalizes to unit std
    roughness_weights=None,
    # how noise amplitude scales with depth
    # "constant" | "linear" | "quadratic" | "clamped_linear"
    roughness_depth_scale="clamped_linear",
    # clamp range for "clamped_linear" (multipliers relative to median depth)
    roughness_depth_clamp=(0.5, 2.0),
    # reduce noise near steep gradients to avoid tearing at discontinuities
    roughness_edge_attenuate=True,
    roughness_edge_gamma=1.0,       # higher -> stronger attenuation with slope
    roughness_edge_eps=1e-3,        # stability term in attenuation
    # add sparse outliers (optional)
    roughness_outlier_p=0.0,        # e.g. 0.002
    roughness_outlier_scale=4.0,    # multiplier on base amplitude for outliers
):
    """
    Build a point cloud + per-point normals from a z-buffer.

    Normal computation is done in the SAME coordinate system you output:
      - x,y normalized to [-1, 1]
      - z normalized to [0, 1] (using z_max in mm)

    The order of operations matches your preference:
      1) fill/inpaint depth
      2) compute normals
      3) collect main points
      4) optional downsample main points
      5) append border points at the end (for debugging visualization)
      6) (points are already normalized if normalize=True)

    Args:
      zbuf: (H,W) depth in sensor frame (mm), +inf for invalid
      pixmm: mm per pixel
      normalize: if True, output x,y in [-1,1], z in [0,1]
      z_max: maximum indentation depth in mm for normalization
      contact_only: if True, keep only z < 0 for the main points
      n_points: if set, randomly downsample main points to this many (before border append)
      include_border: if True, append border "frame" points at the end
      border_thickness: border thickness in pixels
      border_z_fill: z (mm) used to fill invalid z before inpainting AND for border point z values
      inpaint_for_normals: if True, inpaint invalid z before computing normals
      inpaint_radius_px: inpainting radius

      roughness_*: controls correlated “reconstructed mesh” bumpiness injected into depth.

    Returns:
      ret: (N,6) float16 where columns are [x,y,z,nx,ny,nz]
    """
    H, W = zbuf.shape
    zbuf_f = zbuf.astype(np.float32, copy=False)

    # ----------------------------
    # 1) Fill/inpaint depth for stable gradients
    # ----------------------------
    z_fill = zbuf_f.copy()
    invalid = ~np.isfinite(z_fill)
    if np.any(invalid):
        z_fill[invalid] = float(border_z_fill)

    if inpaint_for_normals and np.any(invalid):
        inv_mask = invalid.astype(np.uint8) * 255
        z_fill = cv2.inpaint(z_fill, inv_mask, int(inpaint_radius_px), cv2.INPAINT_TELEA).astype(np.float32)

    # Convert depth (z) to indentation height in mm for consistent [0..] handling:
    # Your convention: penetrating points have z<0, height_mm = -z
    height_fill_mm = np.maximum(-z_fill, 0.0).astype(np.float32)

    # ----------------------------
    # 1.5) OPTIONAL: Inject reconstructed-mesh roughness noise into height_fill_mm
    #      (only on valid/contact pixels, and without bleeding across holes)
    # ----------------------------
    if roughness_enable:
        if rng is None:
            rng = np.random.default_rng()

        # mask where we want to perturb the surface (non-zero / contact-like region)
        # If contact_only=True, match your main point selection; otherwise use any finite depth.
        M = np.isfinite(zbuf_f)
        if contact_only:
            M &= (zbuf_f < 0.0)

        if np.any(M):
            # --- multi-scale correlated noise in image space ---
            base = np.zeros((H, W), dtype=np.float32)

            sigmas = tuple(float(s) for s in roughness_sigmas_px)
            if roughness_weights is None:
                # 1/(2^k) falloff
                ws = np.array([1.0 / (2.0 ** i) for i in range(len(sigmas))], dtype=np.float32)
            else:
                ws = np.asarray(roughness_weights, dtype=np.float32)
                if ws.size != len(sigmas):
                    raise ValueError("roughness_weights must match length of roughness_sigmas_px")

            # build and sum bands
            for w_i, s_i in zip(ws, sigmas):
                n = rng.standard_normal((H, W), dtype=np.float32)
                # Gaussian blur (border reflect avoids edge darkening)
                n = cv2.GaussianBlur(n, ksize=(0, 0), sigmaX=s_i, sigmaY=s_i, borderType=cv2.BORDER_REFLECT101)
                base += w_i * n

            # normalize to unit std on masked region (so roughness_alpha is meaningful)
            std = float(np.std(base[M])) if np.any(M) else 1.0
            if std > 1e-8:
                base = base / std

            # --- depth-dependent amplitude ---
            # Use local "height" (indentation) as the depth proxy here.
            h = height_fill_mm
            h_med = float(np.median(h[M])) if np.any(M) else 0.0

            if roughness_depth_scale == "constant" or h_med <= 1e-8:
                depth_mult = 1.0
            elif roughness_depth_scale == "linear":
                depth_mult = (h / (h_med + 1e-8))
            elif roughness_depth_scale == "quadratic":
                depth_mult = (h / (h_med + 1e-8)) ** 2
            elif roughness_depth_scale == "clamped_linear":
                dm = (h / (h_med + 1e-8))
                lo, hi = float(roughness_depth_clamp[0]), float(roughness_depth_clamp[1])
                depth_mult = np.clip(dm, lo, hi)
            else:
                raise ValueError(
                    "roughness_depth_scale must be one of: "
                    "'constant'|'linear'|'quadratic'|'clamped_linear'"
                )

            # --- edge attenuation (avoid tearing at discontinuities / steep slopes) ---
            if roughness_edge_attenuate:
                # slope magnitude in mm/pixel (simple, robust)
                gy, gx = np.gradient(h, 1.0, 1.0)
                slope = np.sqrt(gx * gx + gy * gy).astype(np.float32)
                # attenuation in (0,1]; higher slope -> smaller multiplier
                edge_mult = 1.0 / (1.0 + (slope / float(roughness_edge_eps)) ** float(roughness_edge_gamma))
            else:
                edge_mult = 1.0

            # --- sparse outliers (optional) ---
            if roughness_outlier_p and roughness_outlier_p > 0.0:
                out = (rng.random((H, W), dtype=np.float32) < float(roughness_outlier_p)).astype(np.float32)
                out *= rng.standard_normal((H, W), dtype=np.float32)
                base = base + float(roughness_outlier_scale) * out

            # --- apply only on mask ---
            # roughness_alpha is in *mm* if normalize=False, or still in mm here because
            # we're perturbing height_fill_mm before normalization.
            delta_mm = float(roughness_alpha) * base
            if not isinstance(depth_mult, (float, int)):
                delta_mm = delta_mm * depth_mult.astype(np.float32)
            delta_mm = delta_mm * (edge_mult if isinstance(edge_mult, (float, int)) else edge_mult.astype(np.float32))

            height_fill_mm = height_fill_mm.copy()
            height_fill_mm[M] = np.maximum(height_fill_mm[M] + delta_mm[M], 0.0)
    # debug_height_fill_mm = height_fill_mm.copy()
    # debug_height_fill_mm = np.repeat(debug_height_fill_mm[:, :, np.newaxis], 3, axis=2)  # (H,W,3)
    
    # div = float(z_max) if z_max > 1e-8 else 1.0
    # cv2.imwrite("debug_height_fill.png", (debug_height_fill_mm / div * 255.0).clip(0, 255).astype(np.uint8))
    # breakpoint()

    # ----------------------------
    # 2) Define coordinate mapping + compute normals in output space
    # ----------------------------
    if normalize:
        dx = 2.0 / float(W)
        dy = 2.0 / float(H)

        z_for_grad = (height_fill_mm / float(z_max)).astype(np.float32)

        dz_dy, dz_dx = np.gradient(z_for_grad, dy, dx)  # axis0=y, axis1=x

        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(z_for_grad, dtype=np.float32)

        nrm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
        nx /= nrm
        ny /= nrm
        nz /= nrm

    else:
        dz_dy, dz_dx = np.gradient(height_fill_mm, float(pixmm), float(pixmm))

        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(height_fill_mm, dtype=np.float32)

        nrm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
        nx /= nrm
        ny /= nrm
        nz /= nrm

    # ----------------------------
    # 3) Main point mask + point extraction
    # ----------------------------
    valid = np.isfinite(zbuf_f)
    if contact_only:
        valid &= (zbuf_f < 0.0)

    vv, uu = np.nonzero(valid)

    if vv.size > 0:
        if normalize:
            x = (2.0 * uu.astype(np.float32) / float(W)) - 1.0
            y = (2.0 * vv.astype(np.float32) / float(H)) - 1.0
            # IMPORTANT: use the *same* (possibly roughened) height field for z output
            z = (height_fill_mm[vv, uu] / float(z_max)).astype(np.float32)
        else:
            x = (uu.astype(np.float32) - (W * 0.5)) * float(pixmm)
            y = (vv.astype(np.float32) - (H * 0.5)) * float(pixmm)
            # Keep original signed z convention, but use roughened height for contact points:
            z = (-height_fill_mm[vv, uu]).astype(np.float32)

        pts_main = np.stack([x, y, z], axis=1).astype(np.float32)
        n_main = np.stack([nx[vv, uu], ny[vv, uu], nz[vv, uu]], axis=1).astype(np.float32)
    else:
        pts_main = np.zeros((0, 3), dtype=np.float32)
        n_main = np.zeros((0, 3), dtype=np.float32)

    # ----------------------------
    # 4) Optional downsample main points (before border append)
    # ----------------------------
    if n_points is not None and pts_main.shape[0] > int(n_points):
        if rng is None:
            rng = np.random.default_rng()
        
        idx = rng.choice(pts_main.shape[0], size=int(n_points), replace=False)
        pts_main = pts_main[idx]
        n_main = n_main[idx]
    elif n_points is not None and pts_main.shape[0] > 3000:
        if rng is None:
            rng = np.random.default_rng()
        
        idx = rng.choice(pts_main.shape[0], size=int(n_points), replace=True)
        pts_main = pts_main[idx]
        n_main = n_main[idx]
    # ----------------------------
    # 5) Append border points at the end (for debug visualization)
    # ----------------------------
    if include_border:
        t = max(1, int(border_thickness))
        t = min(t, H // 2, W // 2)

        border = np.zeros((H, W), dtype=bool)
        border[:t, :] = True
        border[-t:, :] = True
        border[:, :t] = True
        border[:, -t:] = True

        bv, bu = np.nonzero(border)

        if normalize:
            xb = (2.0 * bu.astype(np.float32) / float(W)) - 1.0
            yb = (2.0 * bv.astype(np.float32) / float(H)) - 1.0
            zb = np.zeros_like(xb, dtype=np.float32)
        else:
            xb = (bu.astype(np.float32) - (W * 0.5)) * float(pixmm)
            yb = (bv.astype(np.float32) - (H * 0.5)) * float(pixmm)
            zb = np.full_like(xb, float(border_z_fill), dtype=np.float32)

        pts_border = np.stack([xb, yb, zb], axis=1).astype(np.float32)
        n_border = np.stack([nx[bv, bu], ny[bv, bu], nz[bv, bu]], axis=1).astype(np.float32)

        if pts_main.size:
            pts = np.vstack([pts_main, pts_border]).astype(np.float32)
            normals = np.vstack([n_main, n_border]).astype(np.float32)
        else:
            pts = pts_border
            normals = n_border
    else:
        pts = pts_main
        normals = n_main

    ret = np.hstack([pts, normals]).astype(np.float16)
    return ret

