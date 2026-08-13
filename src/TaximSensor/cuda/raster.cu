__device__ __forceinline__ unsigned int float_to_ordered(float value) {
    const unsigned int bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ float ordered_to_float(unsigned int ordered) {
    const unsigned int bits =
        (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
    return __uint_as_float(bits);
}

__device__ __forceinline__ void barycentrics(
    float px,
    float py,
    float u0,
    float v0,
    float u1,
    float v1,
    float u2,
    float v2,
    float inv_denom,
    float *w0,
    float *w1,
    float *w2
) {
    *w0 = ((v1 - v2) * (px - u2) + (u2 - u1) * (py - v2)) * inv_denom;
    *w1 = ((v2 - v0) * (px - u2) + (u0 - u2) * (py - v2)) * inv_denom;
    *w2 = 1.0f - *w0 - *w1;
}

extern "C" __global__ void transform_vertices(
    const float *vertices_h,
    const float *transform,
    float *projected,
    int vertex_count,
    float inv_pixmm,
    float half_width,
    float half_height
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= vertex_count) {
        return;
    }

    const float x = vertices_h[idx * 4 + 0];
    const float y = vertices_h[idx * 4 + 1];
    const float z = vertices_h[idx * 4 + 2];
    const float w = vertices_h[idx * 4 + 3];

    const float sx = transform[0] * x + transform[1] * y
        + transform[2] * z + transform[3] * w;
    const float sy = transform[4] * x + transform[5] * y
        + transform[6] * z + transform[7] * w;
    const float sz = transform[8] * x + transform[9] * y
        + transform[10] * z + transform[11] * w;

    projected[idx * 3 + 0] = sx * inv_pixmm + half_width;
    projected[idx * 3 + 1] = sy * inv_pixmm + half_height;
    projected[idx * 3 + 2] = sz;
}

extern "C" __global__ void rasterize_faces(
    const float *projected,
    const int *faces,
    unsigned long long *winner,
    int face_count,
    int height,
    int width
) {
    const unsigned int face_idx = blockIdx.x;
    if (face_idx >= (unsigned int)face_count) {
        return;
    }

    const int i0 = faces[face_idx * 3 + 0];
    const int i1 = faces[face_idx * 3 + 1];
    const int i2 = faces[face_idx * 3 + 2];

    const float u0 = projected[i0 * 3 + 0];
    const float v0 = projected[i0 * 3 + 1];
    const float z0 = projected[i0 * 3 + 2];
    const float u1 = projected[i1 * 3 + 0];
    const float v1 = projected[i1 * 3 + 1];
    const float z1 = projected[i1 * 3 + 2];
    const float u2 = projected[i2 * 3 + 0];
    const float v2 = projected[i2 * 3 + 1];
    const float z2 = projected[i2 * 3 + 2];

    if (z0 >= 0.0f && z1 >= 0.0f && z2 >= 0.0f) {
        return;
    }

    int umin = (int)floorf(fminf(u0, fminf(u1, u2)));
    int umax = (int)ceilf(fmaxf(u0, fmaxf(u1, u2)));
    int vmin = (int)floorf(fminf(v0, fminf(v1, v2)));
    int vmax = (int)ceilf(fmaxf(v0, fmaxf(v1, v2)));

    if (umax < 0 || umin >= width || vmax < 0 || vmin >= height) {
        return;
    }
    umin = umin < 0 ? 0 : umin;
    umax = umax > width - 1 ? width - 1 : umax;
    vmin = vmin < 0 ? 0 : vmin;
    vmax = vmax > height - 1 ? height - 1 : vmax;

    const float denom =
        (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2);
    if (denom == 0.0f) {
        return;
    }
    const float inv_denom = 1.0f / denom;
    const int bbox_width = umax - umin + 1;
    const int bbox_height = vmax - vmin + 1;
    const int bbox_pixels = bbox_width * bbox_height;

    for (int offset = threadIdx.x; offset < bbox_pixels; offset += blockDim.x) {
        const int x = umin + offset % bbox_width;
        const int y = vmin + offset / bbox_width;
        float w0, w1, w2;
        barycentrics(
            x + 0.5f,
            y + 0.5f,
            u0,
            v0,
            u1,
            v1,
            u2,
            v2,
            inv_denom,
            &w0,
            &w1,
            &w2
        );
        if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) {
            continue;
        }

        const float z = w0 * z0 + w1 * z1 + w2 * z2;
        if (!(z < 0.0f)) {
            continue;
        }

        const unsigned long long depth_key =
            (unsigned long long)float_to_ordered(z) << 32;
        const unsigned long long face_key = 0xffffffffu - face_idx;
        const unsigned long long key = depth_key | face_key;
        atomicMax(&winner[y * width + x], key);
    }
}

extern "C" __global__ void resolve_base(
    const unsigned long long *winner,
    float *zbuf,
    float *height_map,
    float *overlay,
    int pixel_count,
    float inv_pixmm
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= pixel_count) {
        return;
    }
    const unsigned long long key = winner[idx];
    if (key == 0ull) {
        zbuf[idx] = __int_as_float(0x7f800000);
        height_map[idx] = 0.0f;
        overlay[idx] = 0.0f;
        return;
    }

    const float z = ordered_to_float((unsigned int)(key >> 32));
    zbuf[idx] = z;
    height_map[idx] = -z * inv_pixmm;
    overlay[idx] = 0.0f;
}

extern "C" __global__ void resolve_textured(
    const unsigned long long *winner,
    const float *projected,
    const int *faces,
    const float *uv_tris,
    const float *normal_tris,
    const float *pseudo_height,
    const float *transform,
    float *zbuf,
    float *base_height,
    float *displaced_height,
    unsigned char *displaced_valid,
    int height,
    int width,
    int texture_height,
    int texture_width,
    float bump_scale_mm,
    float inv_pixmm
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int pixel_count = height * width;
    if (idx >= pixel_count) {
        return;
    }

    const unsigned long long key = winner[idx];
    if (key == 0ull) {
        zbuf[idx] = __int_as_float(0x7f800000);
        base_height[idx] = 0.0f;
        displaced_height[idx] = 0.0f;
        displaced_valid[idx] = 0;
        return;
    }

    const unsigned int face_key = (unsigned int)key;
    const unsigned int face_idx = 0xffffffffu - face_key;
    const float z = ordered_to_float((unsigned int)(key >> 32));
    zbuf[idx] = z;
    base_height[idx] = -z * inv_pixmm;

    const int i0 = faces[face_idx * 3 + 0];
    const int i1 = faces[face_idx * 3 + 1];
    const int i2 = faces[face_idx * 3 + 2];
    const float u0 = projected[i0 * 3 + 0];
    const float v0 = projected[i0 * 3 + 1];
    const float u1 = projected[i1 * 3 + 0];
    const float v1 = projected[i1 * 3 + 1];
    const float u2 = projected[i2 * 3 + 0];
    const float v2 = projected[i2 * 3 + 1];
    const float denom =
        (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2);
    if (denom == 0.0f) {
        displaced_height[idx] = 0.0f;
        displaced_valid[idx] = 0;
        return;
    }

    const int x = idx % width;
    const int y = idx / width;
    float w0, w1, w2;
    barycentrics(
        x + 0.5f,
        y + 0.5f,
        u0,
        v0,
        u1,
        v1,
        u2,
        v2,
        1.0f / denom,
        &w0,
        &w1,
        &w2
    );

    const int uv_base = face_idx * 6;
    float tex_u = w0 * uv_tris[uv_base + 0]
        + w1 * uv_tris[uv_base + 2]
        + w2 * uv_tris[uv_base + 4];
    float tex_v = w0 * uv_tris[uv_base + 1]
        + w1 * uv_tris[uv_base + 3]
        + w2 * uv_tris[uv_base + 5];
    tex_u = fminf(fmaxf(tex_u, 0.0f), 1.0f);
    tex_v = fminf(fmaxf(tex_v, 0.0f), 1.0f);

    const float tx = tex_u * (texture_width - 1);
    const float ty = tex_v * (texture_height - 1);
    const int x0 = (int)floorf(tx);
    const int y0 = (int)floorf(ty);
    const int x1 = x0 + 1 < texture_width ? x0 + 1 : texture_width - 1;
    const int y1 = y0 + 1 < texture_height ? y0 + 1 : texture_height - 1;
    const float dx = tx - x0;
    const float dy = ty - y0;
    const float top = (1.0f - dx) * pseudo_height[y0 * texture_width + x0]
        + dx * pseudo_height[y0 * texture_width + x1];
    const float bottom = (1.0f - dx) * pseudo_height[y1 * texture_width + x0]
        + dx * pseudo_height[y1 * texture_width + x1];
    const float sampled_height = (1.0f - dy) * top + dy * bottom;

    const int normal_base = face_idx * 9;
    const float nx = w0 * normal_tris[normal_base + 0]
        + w1 * normal_tris[normal_base + 3]
        + w2 * normal_tris[normal_base + 6];
    const float ny = w0 * normal_tris[normal_base + 1]
        + w1 * normal_tris[normal_base + 4]
        + w2 * normal_tris[normal_base + 7];
    const float nz = w0 * normal_tris[normal_base + 2]
        + w1 * normal_tris[normal_base + 5]
        + w2 * normal_tris[normal_base + 8];
    const float norm = sqrtf(nx * nx + ny * ny + nz * nz);
    if (!(norm > 1e-8f)) {
        displaced_height[idx] = 0.0f;
        displaced_valid[idx] = 0;
        return;
    }

    const float nz_sensor =
        (transform[8] * nx + transform[9] * ny + transform[10] * nz) / norm;
    const float displaced_z =
        z + sampled_height * bump_scale_mm * nz_sensor;
    if (displaced_z < 0.0f) {
        displaced_height[idx] = -displaced_z * inv_pixmm;
        displaced_valid[idx] = 1;
    } else {
        displaced_height[idx] = 0.0f;
        displaced_valid[idx] = 0;
    }
}

extern "C" __global__ void erode_and_merge(
    const float *base_height,
    const float *displaced_height,
    const unsigned char *displaced_valid,
    float *height_map,
    float *overlay,
    int height,
    int width
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int pixel_count = height * width;
    if (idx >= pixel_count) {
        return;
    }

    const float base = base_height[idx];
    const float result = displaced_valid[idx] != 0
        ? displaced_height[idx]
        : base;
    height_map[idx] = result;
    overlay[idx] = result - base;
}
