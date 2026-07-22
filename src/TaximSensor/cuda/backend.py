from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from time import perf_counter
from typing import Any

import numpy as np

from norm2tex.timing import record_timing


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
    """Persistent CUDA raster state for one Taxim sensor."""

    _TRANSFORM_THREADS = 256
    _RASTER_THREADS = 128
    _PIXEL_THREADS = 256

    def __init__(self, height: int, width: int, pixmm: float, device: int = 0):
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                "The CUDA raster backend requires CuPy. Install the CuPy wheel "
                "matching the system CUDA toolkit, for example cupy-cuda12x."
            ) from exc

        self.cp = cp
        self.height = int(height)
        self.width = int(width)
        self.pixel_count = self.height * self.width
        self.pixmm = np.float32(pixmm)
        device_index = int(device)
        if device_index < 0 or cp.cuda.runtime.getDeviceCount() <= device_index:
            raise RuntimeError(f"CUDA device {device} is not available")
        self.device = cp.cuda.Device(device_index)
        self.device.use()

        source = files("TaximSensor.cuda").joinpath("raster.cu").read_text()
        self.module = cp.RawModule(
            code=source,
            options=("--std=c++11",),
        )
        self.transform_kernel = self.module.get_function("transform_vertices")
        self.raster_kernel = self.module.get_function("rasterize_faces")
        self.resolve_base_kernel = self.module.get_function("resolve_base")
        self.resolve_textured_kernel = self.module.get_function("resolve_textured")
        self.merge_kernel = self.module.get_function("erode_and_merge")

        self.stream = cp.cuda.Stream(non_blocking=True)
        self.transform = cp.empty((4, 4), dtype=cp.float32)
        self.winner = cp.empty(self.pixel_count, dtype=cp.uint64)
        self.zbuf = cp.empty(self.pixel_count, dtype=cp.float32)
        self.base_height = cp.empty(self.pixel_count, dtype=cp.float32)
        self.displaced_height = cp.empty(self.pixel_count, dtype=cp.float32)
        self.displaced_valid = cp.empty(self.pixel_count, dtype=cp.uint8)
        self.height_map = cp.empty(self.pixel_count, dtype=cp.float32)
        self.overlay = cp.empty(self.pixel_count, dtype=cp.float32)
        self.assets: dict[str, _CudaRasterAsset] = {}

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

        vertices_device = cp.asarray(vertices_h)
        self.assets[name] = _CudaRasterAsset(
            vertices_h=vertices_device,
            faces=cp.asarray(faces),
            projected=cp.empty((len(vertices_h), 3), dtype=cp.float32),
            uv_tris=uv_device,
            normal_tris=normal_device,
            pseudo_height=texture_device,
        )

    def rasterize(
        self,
        name: str,
        sTo: np.ndarray,
        bump_scale_mm: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cp = self.cp
        self.device.use()
        try:
            asset = self.assets[name]
        except KeyError as exc:
            raise KeyError(f"CUDA raster object {name!r} has not been registered") from exc

        transform_host = np.ascontiguousarray(sTo, dtype=np.float32)
        if transform_host.shape != (4, 4):
            raise ValueError("sTo must have shape (4, 4)")

        events = [cp.cuda.Event() for _ in range(6)]
        transform_blocks = (
            (len(asset.vertices_h) + self._TRANSFORM_THREADS - 1)
            // self._TRANSFORM_THREADS
        )
        pixel_blocks = (
            self.pixel_count + self._PIXEL_THREADS - 1
        ) // self._PIXEL_THREADS

        with self.stream:
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

        download_start = perf_counter()
        with self.stream:
            height_map = cp.asnumpy(
                self.height_map,
                stream=self.stream,
            ).reshape(self.height, self.width)
            overlay = cp.asnumpy(
                self.overlay,
                stream=self.stream,
            ).reshape(self.height, self.width)
            zbuf = cp.asnumpy(
                self.zbuf,
                stream=self.stream,
            ).reshape(self.height, self.width)
        self.stream.synchronize()
        record_timing("hm_download", perf_counter() - download_start)

        record_timing(
            "hm_upload_transform",
            cp.cuda.get_elapsed_time(events[0], events[1]) * 1e-3,
        )
        record_timing(
            "hm_transform_vertices",
            cp.cuda.get_elapsed_time(events[1], events[2]) * 1e-3,
        )
        record_timing(
            "hm_rasterize",
            cp.cuda.get_elapsed_time(events[2], events[3]) * 1e-3,
        )
        record_timing(
            "hm_resolve_texture" if asset.textured else "hm_resolve_base",
            cp.cuda.get_elapsed_time(events[3], events[4]) * 1e-3,
        )
        if asset.textured:
            record_timing(
                "hm_erode_merge",
                cp.cuda.get_elapsed_time(events[4], events[5]) * 1e-3,
            )

        return height_map, overlay, zbuf
