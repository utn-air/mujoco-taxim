__device__ __forceinline__ float atomic_min_float(float *address, float value) {
    int *address_i = (int *)address;
    int old = *address_i;
    while (value < __int_as_float(old)) {
        const int assumed = old;
        old = atomicCAS(address_i, assumed, __float_as_int(value));
        if (old == assumed) {
            break;
        }
    }
    return __int_as_float(old);
}

extern "C" __global__ void simulate_pixels(
    const float *height_map,
    const float *calibration,
    const float *background,
    float *gradient_direction,
    float *raw_image,
    float *lit_image,
    int height,
    int width,
    int bins
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int pixel_count = height * width;
    if (idx >= pixel_count) {
        return;
    }

    const int x = idx % width;
    const int y = idx / width;
    const int cx = x < 1 ? 1 : (x > width - 2 ? width - 2 : x);
    const int cy = y < 1 ? 1 : (y > height - 2 ? height - 2 : y);
    const float top = height_map[(cy - 1) * width + cx];
    const float bottom = height_map[(cy + 1) * width + cx];
    const float left = height_map[cy * width + cx - 1];
    const float right = height_map[cy * width + cx + 1];
    const float dzdx = 0.5f * (bottom - top);
    const float dzdy = 0.5f * (right - left);
    const float tangent = sqrtf(dzdx * dzdx + dzdy * dzdy);
    const float magnitude = atanf(tangent);
    const float direction = tangent == 0.0f
        ? 0.0f
        : atan2f(dzdx / tangent, dzdy / tangent);
    gradient_direction[idx] = direction;

    const float pi = 3.14159265358979323846f;
    const float x_bin_width = 0.5f * pi / (bins - 1);
    const float y_bin_width = 2.0f * pi / (bins - 1);
    int ix = (int)floorf(magnitude / x_bin_width);
    int iy = (int)floorf((direction + pi) / y_bin_width);
    ix = ix < 0 ? 0 : (ix >= bins ? bins - 1 : ix);
    iy = iy < 0 ? 0 : (iy >= bins ? bins - 1 : iy);

    const float px = (float)x;
    const float py = (float)y;
    const float design[6] = {
        px * px, py * py, px * py, px, py, 1.0f
    };
    for (int channel = 0; channel < 3; ++channel) {
        const int calibration_base =
            (((channel * bins + ix) * bins + iy) * 6);
        float value = 0.0f;
        for (int coefficient = 0; coefficient < 6; ++coefficient) {
            value += design[coefficient]
                * calibration[calibration_base + coefficient];
        }
        const int output_idx = idx * 3 + channel;
        raw_image[output_idx] = value;
        lit_image[output_idx] = value + background[output_idx];
    }
}

extern "C" __global__ void cast_shadows(
    const unsigned char *shadow_boundary,
    const float *contact_height,
    const float *height_map,
    const float *fan_cosines,
    const float *fan_sines,
    const int *fan_lengths,
    const float *profiles,
    const int *profile_lengths,
    float *shadow_image,
    int height,
    int width,
    int direction_count,
    int height_count,
    int max_steps,
    int max_fans,
    float pixmm,
    float shadow_depth_min,
    float height_precision,
    float direction_precision,
    float shadow_step
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int pixel_count = height * width;
    if (idx >= pixel_count || shadow_boundary[idx] == 0) {
        return;
    }

    const float pi = 3.14159265358979323846f;
    const int x = idx % width;
    const int y = idx / width;
    const int cx = x < 1 ? 1 : (x > width - 2 ? width - 2 : x);
    const int cy = y < 1 ? 1 : (y > height - 2 ? height - 2 : y);
    const float top = height_map[(cy - 1) * width + cx];
    const float bottom = height_map[(cy + 1) * width + cx];
    const float left = height_map[cy * width + cx - 1];
    const float right = height_map[cy * width + cx + 1];
    const float dzdx = 0.5f * (bottom - top);
    const float dzdy = 0.5f * (right - left);
    const float tangent = sqrtf(dzdx * dzdx + dzdy * dzdy);
    const float gradient_direction = tangent == 0.0f
        ? 0.0f
        : atan2f(dzdx / tangent, dzdy / tangent);
    const int normal_idx = (int)floorf(
        (gradient_direction + pi) / direction_precision
    );
    const int height_idx = (int)floorf(
        (contact_height[idx] * pixmm - shadow_depth_min) / height_precision
    ) + 6;
    if (normal_idx < 0 || normal_idx >= direction_count
        || height_idx < 0 || height_idx >= height_count) {
        return;
    }

    const int origin_x = x;
    const int origin_y = y;
    const float origin_height = height_map[idx];
    const int fan_count = fan_lengths[normal_idx];

    for (int channel = 0; channel < 3; ++channel) {
        const int profile_idx =
            (channel * direction_count + normal_idx) * height_count + height_idx;
        const int step_count = profile_lengths[profile_idx];
        if (step_count <= 1) {
            continue;
        }
        const int profile_base = profile_idx * max_steps;
        for (int fan_idx = 0; fan_idx < fan_count; ++fan_idx) {
            const float ct = fan_cosines[normal_idx * max_fans + fan_idx];
            const float st = fan_sines[normal_idx * max_fans + fan_idx];
            for (int step = 1; step < step_count; ++step) {
                const int target_x = (int)(
                    origin_x + shadow_step * step * ct
                );
                const int target_y = (int)(
                    origin_y + shadow_step * step * st
                );
                if (target_x < 0 || target_x >= width
                    || target_y < 0 || target_y >= height) {
                    continue;
                }
                const int target_idx = target_y * width + target_x;
                if (origin_height > height_map[target_idx]) {
                    atomic_min_float(
                        &shadow_image[target_idx * 3 + channel],
                        profiles[profile_base + step]
                    );
                }
            }
        }
    }
}
