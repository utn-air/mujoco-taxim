from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from time import perf_counter
from typing import Any

import numpy as np

from norm2tex.timing import record_timing


def cuda_device_available(device: int = 0) -> tuple[bool, str | None]:
    """Return whether CuPy can create a context for the requested CUDA device."""
    try:
        device_index = int(device)
    except (TypeError, ValueError):
        return False, f"invalid CUDA device index {device!r}"
    if device_index < 0:
        return False, f"invalid CUDA device index {device_index}"

    try:
        import cupy as cp
        from cupyx.scipy import ndimage as _cuda_ndimage  # noqa: F401
    except (ImportError, OSError) as exc:
        return False, f"CuPy is unavailable ({exc})"

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_index >= device_count:
            return False, (
                f"CUDA device {device_index} is unavailable "
                f"({device_count} device(s) detected)"
            )
        cuda_device = cp.cuda.Device(device_index)
        cuda_device.use()
        _ = cuda_device.compute_capability
    except Exception as exc:
        return False, f"CUDA runtime initialization failed ({exc})"
    return True, None


@dataclass
class _CudaRasterAsset:
    vertices_h: Any
    faces: Any
    projected: Any
    uv_tris: Any | None
    normal_tris: Any | None
    pseudo_height: Any | None

    @property
    def textured(self) -> bool:
        return self.pseudo_height is not None


class CudaRasterBackend:
    """Persistent CUDA state for rasterization, deformation, and TAXIM shading."""

    _TRANSFORM_THREADS = 256
    _RASTER_THREADS = 128
    _PIXEL_THREADS = 256

    def __init__(self, height: int, width: int, pixmm: float, device: int = 0):
        try:
            import cupy as cp
            from cupyx.scipy import ndimage as cuda_ndimage
        except ImportError as exc:
            raise RuntimeError(
                "The CUDA backend requires CuPy. Install the CuPy wheel "
                "matching the system CUDA toolkit, for example cupy-cuda12x."
            ) from exc

        self.cp = cp
        self.cuda_ndimage = cuda_ndimage
        self.height = int(height)
        self.width = int(width)
        self.pixel_count = self.height * self.width
        self.pixmm = np.float32(pixmm)
        device_index = int(device)
        if device_index < 0 or cp.cuda.runtime.getDeviceCount() <= device_index:
            raise RuntimeError(f"CUDA device {device} is not available")
        self.device = cp.cuda.Device(device_index)
        self.device.use()

        raster_source = files("TaximSensor.cuda").joinpath("raster.cu").read_text()
        self.module = cp.RawModule(code=raster_source, options=("--std=c++11",))
        self.transform_kernel = self.module.get_function("transform_vertices")
        self.raster_kernel = self.module.get_function("rasterize_faces")
        self.resolve_base_kernel = self.module.get_function("resolve_base")
        self.resolve_textured_kernel = self.module.get_function("resolve_textured")
        self.merge_kernel = self.module.get_function("erode_and_merge")

        frame_source = files("TaximSensor.cuda").joinpath("frame.cu").read_text()
        self.frame_module = cp.RawModule(code=frame_source, options=("--std=c++11",))
        self.simulate_kernel = self.frame_module.get_function("simulate_pixels")
        self.shadow_kernel = self.frame_module.get_function("cast_shadows")

        self.stream = cp.cuda.Stream(non_blocking=True)
        self.transform = cp.empty((4, 4), dtype=cp.float32)
        self.winner = cp.empty(self.pixel_count, dtype=cp.uint64)
        self.zbuf = cp.empty(self.pixel_count, dtype=cp.float32)
        self.base_height = cp.empty(self.pixel_count, dtype=cp.float32)
        self.displaced_height = cp.empty(self.pixel_count, dtype=cp.float32)
        self.displaced_valid = cp.empty(self.pixel_count, dtype=cp.uint8)
        self.height_map = cp.empty(self.pixel_count, dtype=cp.float32)
        self.overlay = cp.empty(self.pixel_count, dtype=cp.float32)
        self.gradient_direction = cp.empty(self.pixel_count, dtype=cp.float32)
        self.raw_image = cp.empty((self.height, self.width, 3), dtype=cp.float32)
        self.lit_image = cp.empty_like(self.raw_image)
        self.shadow_image = cp.empty_like(self.raw_image)
        self.assets: dict[str, _CudaRasterAsset] = {}
        self._frame_configured = False
        self._gaussian_kernels: dict[int, Any] = {}

    def configure_frame_pipeline(
        self,
        *,
        gel_map: np.ndarray,
        background: np.ndarray,
        grad_r: np.ndarray,
        grad_g: np.ndarray,
        grad_b: np.ndarray,
        shadow_directions: np.ndarray,
        shadow_table: np.ndarray,
        fan_angle: float,
        fan_precision: float,
    ) -> None:
        """Upload static deformation, lighting, and shadow calibration data."""
        cp = self.cp
        self.device.use()
        gel_map = np.ascontiguousarray(gel_map, dtype=np.float32)
        background = np.ascontiguousarray(background, dtype=np.float32)
        if gel_map.shape != (self.height, self.width):
            raise ValueError("gel_map shape does not match the CUDA frame size")
        if background.shape != (self.height, self.width, 3):
            raise ValueError("background shape does not match the CUDA frame size")

        calibration = np.ascontiguousarray(
            np.stack((grad_r, grad_g, grad_b), axis=0), dtype=np.float32
        )
        if calibration.ndim != 4 or calibration.shape[0] != 3 or calibration.shape[-1] != 6:
            raise ValueError("calibration tables must have shape (bins, bins, 6)")
        self.bins = int(calibration.shape[1])
        if calibration.shape[2] != self.bins:
            raise ValueError("calibration tables must use the same bin count on both axes")

        shadow_directions = np.ascontiguousarray(shadow_directions, dtype=np.float32)
        if shadow_table.shape[0] != 3 or shadow_table.shape[1] != len(shadow_directions):
            raise ValueError("shadow table dimensions do not match shadow directions")
        direction_count = int(shadow_table.shape[1])
        height_count = int(shadow_table.shape[2])
        max_steps = max(
            (len(shadow_table[c, n, h])
             for c in range(3)
             for n in range(direction_count)
             for h in range(height_count)),
            default=0,
        )
        dense_profiles = np.zeros(
            (3, direction_count, height_count, max_steps), dtype=np.float32
        )
        profile_lengths = np.zeros(
            (3, direction_count, height_count), dtype=np.int32
        )
        for channel in range(3):
            for normal_idx in range(direction_count):
                for height_idx in range(height_count):
                    profile = np.asarray(
                        shadow_table[channel, normal_idx, height_idx], dtype=np.float32
                    )
                    profile_lengths[channel, normal_idx, height_idx] = len(profile)
                    dense_profiles[channel, normal_idx, height_idx, :len(profile)] = profile

        fan_angles = [
            np.arange(
                float(direction) - float(fan_angle),
                float(direction) + float(fan_angle),
                float(fan_precision),
                dtype=np.float32,
            )
            for direction in shadow_directions
        ]
        max_fans = max((len(angles) for angles in fan_angles), default=0)
        fan_cosines = np.zeros((direction_count, max_fans), dtype=np.float32)
        fan_sines = np.zeros((direction_count, max_fans), dtype=np.float32)
        fan_lengths = np.asarray([len(angles) for angles in fan_angles], dtype=np.int32)
        for normal_idx, angles in enumerate(fan_angles):
            fan_cosines[normal_idx, :len(angles)] = np.cos(angles).astype(np.float32)
            fan_sines[normal_idx, :len(angles)] = np.sin(angles).astype(np.float32)

        with self.stream:
            self.gel_map = cp.asarray(gel_map)
            self.background = cp.asarray(background)
            self.calibration = cp.asarray(calibration)
            self.shadow_profiles = cp.asarray(dense_profiles)
            self.shadow_profile_lengths = cp.asarray(profile_lengths)
            self.shadow_fan_cosines = cp.asarray(fan_cosines)
            self.shadow_fan_sines = cp.asarray(fan_sines)
            self.shadow_fan_lengths = cp.asarray(fan_lengths)
        self.max_gel_height = np.float32(gel_map.max())
        self.shadow_direction_count = direction_count
        self.shadow_height_count = height_count
        self.shadow_max_steps = max_steps
        self.shadow_max_fans = max_fans
        self._frame_configured = True

    def update_background(self, background: np.ndarray) -> None:
        if not self._frame_configured:
            return
        background = np.ascontiguousarray(background, dtype=np.float32)
        if background.shape != (self.height, self.width, 3):
            raise ValueError("background shape does not match the CUDA frame size")
        self.device.use()
        self.background.set(background, stream=self.stream)

    def register_object(
        self,
        name: str,
        vertices_h: np.ndarray,
        faces: np.ndarray,
        *,
        uv_tris: np.ndarray | None = None,
        normal_tris: np.ndarray | None = None,
        pseudo_height: np.ndarray | None = None,
    ) -> None:
        cp = self.cp
        self.device.use()
        vertices_h = np.ascontiguousarray(vertices_h, dtype=np.float32)
        faces = np.ascontiguousarray(faces, dtype=np.int32)
        if vertices_h.ndim != 2 or vertices_h.shape[1] != 4:
            raise ValueError("vertices_h must have shape (V, 4)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if faces.size and (faces.min() < 0 or faces.max() >= len(vertices_h)):
            raise ValueError("faces contain an out-of-range compact vertex index")

        texture_args = (uv_tris, normal_tris, pseudo_height)
        if any(value is not None for value in texture_args) and not all(
            value is not None for value in texture_args
        ):
            raise ValueError(
                "uv_tris, normal_tris, and pseudo_height must be provided together"
            )

        uv_device = normal_device = texture_device = None
        if pseudo_height is not None:
            uv_tris = np.ascontiguousarray(uv_tris, dtype=np.float32)
            normal_tris = np.ascontiguousarray(normal_tris, dtype=np.float32)
            pseudo_height = np.ascontiguousarray(pseudo_height, dtype=np.float32)
            if uv_tris.shape != (len(faces), 3, 2):
                raise ValueError("uv_tris must have shape (F, 3, 2)")
            if normal_tris.shape != (len(faces), 3, 3):
                raise ValueError("normal_tris must have shape (F, 3, 3)")
            if pseudo_height.ndim != 2:
                raise ValueError("pseudo_height must be a two-dimensional texture")
            uv_device = cp.asarray(uv_tris)
            normal_device = cp.asarray(normal_tris)
            texture_device = cp.asarray(pseudo_height)

        self.assets[name] = _CudaRasterAsset(
            vertices_h=cp.asarray(vertices_h),
            faces=cp.asarray(faces),
            projected=cp.empty((len(vertices_h), 3), dtype=cp.float32),
            uv_tris=uv_device,
            normal_tris=normal_device,
            pseudo_height=texture_device,
        )

    def _rasterize_device(
        self,
        name: str,
        sTo: np.ndarray,
        bump_scale_mm: float,
    ) -> tuple[_CudaRasterAsset, list[Any]]:
        cp = self.cp
        try:
            asset = self.assets[name]
        except KeyError as exc:
            raise KeyError(f"CUDA raster object {name!r} has not been registered") from exc
        transform_host = np.ascontiguousarray(sTo, dtype=np.float32)
        if transform_host.shape != (4, 4):
            raise ValueError("sTo must have shape (4, 4)")

        events = [cp.cuda.Event() for _ in range(6)]
        transform_blocks = (
            len(asset.vertices_h) + self._TRANSFORM_THREADS - 1
        ) // self._TRANSFORM_THREADS
        pixel_blocks = (
            self.pixel_count + self._PIXEL_THREADS - 1
        ) // self._PIXEL_THREADS

        events[0].record(self.stream)
        self.transform.set(transform_host, stream=self.stream)
        events[1].record(self.stream)
        if transform_blocks:
            self.transform_kernel(
                (transform_blocks,),
                (self._TRANSFORM_THREADS,),
                (
                    asset.vertices_h,
                    self.transform,
                    asset.projected,
                    np.int32(len(asset.vertices_h)),
                    np.float32(1.0 / self.pixmm),
                    np.float32(self.width * 0.5),
                    np.float32(self.height * 0.5),
                ),
                stream=self.stream,
            )
        events[2].record(self.stream)

        self.winner.fill(0)
        if len(asset.faces):
            self.raster_kernel(
                (len(asset.faces),),
                (self._RASTER_THREADS,),
                (
                    asset.projected,
                    asset.faces,
                    self.winner,
                    np.int32(len(asset.faces)),
                    np.int32(self.height),
                    np.int32(self.width),
                ),
                stream=self.stream,
            )
        events[3].record(self.stream)

        if asset.textured:
            texture_height, texture_width = asset.pseudo_height.shape
            self.resolve_textured_kernel(
                (pixel_blocks,),
                (self._PIXEL_THREADS,),
                (
                    self.winner,
                    asset.projected,
                    asset.faces,
                    asset.uv_tris,
                    asset.normal_tris,
                    asset.pseudo_height,
                    self.transform,
                    self.zbuf,
                    self.base_height,
                    self.displaced_height,
                    self.displaced_valid,
                    np.int32(self.height),
                    np.int32(self.width),
                    np.int32(texture_height),
                    np.int32(texture_width),
                    np.float32(bump_scale_mm),
                    np.float32(1.0 / self.pixmm),
                ),
                stream=self.stream,
            )
        else:
            self.resolve_base_kernel(
                (pixel_blocks,),
                (self._PIXEL_THREADS,),
                (
                    self.winner,
                    self.zbuf,
                    self.height_map,
                    self.overlay,
                    np.int32(self.pixel_count),
                    np.float32(1.0 / self.pixmm),
                ),
                stream=self.stream,
            )
        events[4].record(self.stream)

        if asset.textured:
            self.merge_kernel(
                (pixel_blocks,),
                (self._PIXEL_THREADS,),
                (
                    self.base_height,
                    self.displaced_height,
                    self.displaced_valid,
                    self.height_map,
                    self.overlay,
                    np.int32(self.height),
                    np.int32(self.width),
                ),
                stream=self.stream,
            )
        events[5].record(self.stream)
        return asset, events

    def _record_raster_timings(self, asset: _CudaRasterAsset, events: list[Any]) -> None:
        cp = self.cp
        labels = (
            "hm_upload_transform",
            "hm_transform_vertices",
            "hm_rasterize",
            "hm_resolve_texture" if asset.textured else "hm_resolve_base",
        )
        for index, label in enumerate(labels):
            record_timing(
                label,
                cp.cuda.get_elapsed_time(events[index], events[index + 1]) * 1e-3,
            )
        if asset.textured:
            record_timing(
                "hm_erode_merge",
                cp.cuda.get_elapsed_time(events[4], events[5]) * 1e-3,
            )

    def _gaussian_blur(self, image: Any, kernel_size: int) -> Any:
        import cv2

        cp = self.cp
        kernel = self._gaussian_kernels.get(kernel_size)
        if kernel is None:
            host_kernel = cv2.getGaussianKernel(kernel_size, 0, cv2.CV_32F).reshape(-1)
            kernel = cp.asarray(host_kernel)
            self._gaussian_kernels[kernel_size] = kernel
        result = self.cuda_ndimage.convolve1d(image, kernel, axis=1, mode="mirror")
        return self.cuda_ndimage.convolve1d(result, kernel, axis=0, mode="mirror")

    def rasterize(
        self,
        name: str,
        sTo: np.ndarray,
        bump_scale_mm: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compatibility API that downloads the three raster outputs."""
        cp = self.cp
        self.device.use()
        with self.stream:
            asset, events = self._rasterize_device(name, sTo, bump_scale_mm)
        download_start = perf_counter()
        with self.stream:
            height_map = cp.asnumpy(self.height_map, stream=self.stream).reshape(
                self.height, self.width
            )
            overlay = cp.asnumpy(self.overlay, stream=self.stream).reshape(
                self.height, self.width
            )
            zbuf = cp.asnumpy(self.zbuf, stream=self.stream).reshape(
                self.height, self.width
            )
        self.stream.synchronize()
        record_timing("hm_download", perf_counter() - download_start)
        self._record_raster_timings(asset, events)
        return height_map, overlay, zbuf

    def render_frame(
        self,
        name: str,
        sTo: np.ndarray,
        *,
        bump_scale_mm: float,
        pressing_mm_max: float,
        flat_contact_curvature_mm: float,
        flat_contact_slope_threshold: float,
        contact_scale: float,
        pyramid_kernel_sizes: tuple[int, ...],
        final_kernel_size: int,
        shadow: bool,
        shadow_depth_min: float,
        height_precision: float,
        direction_precision: float,
        shadow_step: float,
        shadow_sigma: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render one TAXIM frame without intermediate host transfers."""
        if not self._frame_configured:
            raise RuntimeError("CUDA frame pipeline has not been configured")
        cp = self.cp
        self.device.use()
        pixel_blocks = (
            self.pixel_count + self._PIXEL_THREADS - 1
        ) // self._PIXEL_THREADS
        stage_events = [cp.cuda.Event() for _ in range(4)]

        with self.stream:
            stage_events[0].record(self.stream)
            asset, raster_events = self._rasterize_device(name, sTo, bump_scale_mm)

            height_2d = self.height_map.reshape(self.height, self.width)
            zbuf_2d = self.zbuf.reshape(self.height, self.width)
            overlay_2d = self.overlay.reshape(self.height, self.width)
            base_height_2d = height_2d - overlay_2d
            min_z = cp.min(zbuf_2d)
            pressing_height_mm = cp.minimum(
                np.float32(pressing_mm_max), cp.maximum(np.float32(0.0), -min_z)
            )
            pressing_height_pix = pressing_height_mm / self.pixmm
            max_object_height = cp.max(base_height_2d)
            gel_interacted = -self.gel_map + (
                self.max_gel_height + max_object_height - pressing_height_pix
            )
            # Contact and deformation geometry come from the undisplaced mesh.
            # The normal map changes appearance only; it must not change the
            # footprint or any of the shadow calculations.
            contact_mask = base_height_2d > gel_interacted
            deformed = cp.where(contact_mask, height_2d, gel_interacted)
            shadow_deformed = cp.where(
                contact_mask, base_height_2d, gel_interacted
            )
            if flat_contact_curvature_mm > 0.0:
                # Test the undisplaced object surface so that norm2tex details
                # do not make a flat face appear geometrically curved.
                interior_mask = self.cuda_ndimage.minimum_filter(
                    contact_mask.astype(cp.uint8),
                    size=3,
                    mode="constant",
                    cval=0,
                ) != 0
                grad_y = self.cuda_ndimage.sobel(
                    base_height_2d, axis=0, mode="mirror"
                ) * np.float32(0.125)
                grad_x = self.cuda_ndimage.sobel(
                    base_height_2d, axis=1, mode="mirror"
                ) * np.float32(0.125)
                surface_slope = cp.hypot(grad_x, grad_y)
                has_interior = cp.any(interior_mask)
                max_contact_slope = cp.max(
                    cp.where(interior_mask, surface_slope, np.float32(0.0))
                )
                flat_contact = has_interior & (
                    max_contact_slope
                    <= np.float32(flat_contact_slope_threshold)
                )

                penetration_weight = cp.clip(
                    (base_height_2d - gel_interacted)
                    / cp.maximum(pressing_height_pix, np.float32(1e-8)),
                    np.float32(0.0),
                    np.float32(1.0),
                )
                penetration_weight = (
                    penetration_weight
                    * penetration_weight
                    * (np.float32(3.0) - np.float32(2.0) * penetration_weight)
                )
                cap_amplitude_mm = cp.minimum(
                    np.float32(flat_contact_curvature_mm), pressing_height_mm
                )
                cap_height = (
                    penetration_weight * cap_amplitude_mm / self.pixmm
                )
                curved_height = height_2d + cp.where(
                    flat_contact, cap_height, np.float32(0.0)
                )
                shadow_curved_height = base_height_2d + cp.where(
                    flat_contact, cap_height, np.float32(0.0)
                )
                deformed = cp.where(
                    contact_mask, curved_height, gel_interacted
                )
                shadow_deformed = cp.where(
                    contact_mask, shadow_curved_height, gel_interacted
                )
            normalized_height = height_2d / cp.maximum(
                max_object_height, np.float32(1e-8)
            )
            ground_truth = self._gaussian_blur(normalized_height, 5)
            stage_events[1].record(self.stream)

            original = deformed.copy()
            shadow_original = shadow_deformed.copy()
            deformation_mask = (
                (shadow_deformed - gel_interacted)
                > pressing_height_pix * np.float32(contact_scale)
            ) & contact_mask
            for kernel_size in pyramid_kernel_sizes:
                deformed = self._gaussian_blur(deformed, int(kernel_size))
                deformed = cp.where(deformation_mask, original, deformed)
                shadow_deformed = self._gaussian_blur(
                    shadow_deformed, int(kernel_size)
                )
                shadow_deformed = cp.where(
                    deformation_mask, shadow_original, shadow_deformed
                )
            deformed = self._gaussian_blur(deformed, int(final_kernel_size))
            shadow_deformed = self._gaussian_blur(
                shadow_deformed, int(final_kernel_size)
            )
            shadow_contact_height = shadow_deformed - gel_interacted
            stage_events[2].record(self.stream)

            self.simulate_kernel(
                (pixel_blocks,),
                (self._PIXEL_THREADS,),
                (
                    deformed,
                    self.calibration,
                    self.background,
                    self.gradient_direction,
                    self.raw_image,
                    self.lit_image,
                    np.int32(self.height),
                    np.int32(self.width),
                    np.int32(self.bins),
                ),
                stream=self.stream,
            )
            output_image = self.lit_image
            if shadow:
                enlarged_mask = self.cuda_ndimage.maximum_filter(
                    deformation_mask.astype(cp.uint8),
                    size=9,
                    mode="constant",
                    cval=0,
                )
                shadow_boundary = (enlarged_mask != 0) & ~deformation_mask
                cp.copyto(self.shadow_image, self.raw_image)
                self.shadow_kernel(
                    (pixel_blocks,),
                    (self._PIXEL_THREADS,),
                    (
                        shadow_boundary,
                        shadow_contact_height,
                        shadow_deformed,
                        self.shadow_fan_cosines,
                        self.shadow_fan_sines,
                        self.shadow_fan_lengths,
                        self.shadow_profiles,
                        self.shadow_profile_lengths,
                        self.shadow_image,
                        np.int32(self.height),
                        np.int32(self.width),
                        np.int32(self.shadow_direction_count),
                        np.int32(self.shadow_height_count),
                        np.int32(self.shadow_max_steps),
                        np.int32(self.shadow_max_fans),
                        np.float32(self.pixmm),
                        np.float32(shadow_depth_min),
                        np.float32(height_precision),
                        np.float32(direction_precision),
                        np.float32(shadow_step),
                    ),
                    stream=self.stream,
                )
                shadow_blurred = self.cuda_ndimage.gaussian_filter(
                    self.shadow_image,
                    sigma=(float(shadow_sigma), float(shadow_sigma), 0.0),
                    order=0,
                    mode="reflect",
                )
                output_image = self._gaussian_blur(
                    shadow_blurred + self.background, int(final_kernel_size)
                )
            stage_events[3].record(self.stream)

        download_start = perf_counter()
        with self.stream:
            sim_image_host = cp.asnumpy(output_image, stream=self.stream)
            ground_truth_host = cp.asnumpy(ground_truth, stream=self.stream)
            overlay_host = cp.asnumpy(overlay_2d, stream=self.stream)
        self.stream.synchronize()
        record_timing("cuda_frame_download", perf_counter() - download_start)
        self._record_raster_timings(asset, raster_events)
        record_timing(
            "hm_total",
            cp.cuda.get_elapsed_time(stage_events[0], stage_events[1]) * 1e-3,
        )
        record_timing(
            "deform_total",
            cp.cuda.get_elapsed_time(stage_events[1], stage_events[2]) * 1e-3,
        )
        record_timing(
            "sim_total",
            cp.cuda.get_elapsed_time(stage_events[2], stage_events[3]) * 1e-3,
        )
        return sim_image_host, ground_truth_host, overlay_host
