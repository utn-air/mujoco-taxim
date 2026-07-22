# MuJoCo-Taxim: A MuJoCo implementation of the Taxim Sensor

This package implements [Taxim](https://github.com/Robo-Touch/Taxim) inside MuJoCo. 
Currently, only the DIGIT sensor version is implemented.
Changes have been made to the original codebase to minimize the package's size, only keeping the bare minimum for what is necessary to make Taxim work in MuJoCo.

## Installation and Prerequisites
Basic dependencies: numpy, scipy, matplotlib, mujoco, trimesh

To install dependencies: `pip install -r requirements.txt`

The package can also be installed via `pip install .`.

## Usage
MuJoCo-Taxim can be easily dropped into an existing MuJoCo application/scene.
Check `examples/mujoco` for a detailed use case on how to achieve this.
Currently, MuJoCo-Taxim can only simulate the tactile image of one object in a given step, but doing multi-geom simulation is planned.

## Normal map-based texture support
This branch now supports using normal maps to inject additional texture into the rendered Taxim image, allowing for richer tactile image simulation. 

To aid in this, it is shipped with a blender-based procedural normal map and color texture generation pipeline, along with the corresponding `.blend` file that contains the procedural materials.

For a given mesh in `.obj` format, you can easily generate the corresponding texture with the `examples/blender/bake_procedural_material.py` script with the following script:

```bash
blender --background --python examples/blender/bake_procedural_material.py -- \
  --obj /path/to/mesh.obj \
  --blend examples/blender/proc_materials.blend \
  --material-config examples/blender/rocky.json \
  --output-dir /path/to/output \
  --uv-angle-limit 66 \
  --samples 128 \
  --image-size 2048
```

At the moment, we support 6 different materials:
- `Wood`
- `Rocky`
- `Clay`
- `Fabric`
- `3DPrint`
- `Uniform` (Voronoi texture for debugging purposes)

The script does the following:
- import an `.obj`
- append a procedural material from the `proc_materials.blend` chosen by the JSON file passed to `--material-config`
- apply configurable `Smart UV Project` unwrapping
- bake color and tangent-space normal maps with Cycles
- export the resulting color, normal and UV-baked `.obj` file to the output directory.

The baked outputs are written as:
- `{obj_name}_{material}_color.png`
- `{obj_name}_{material}_normal.png`
- `{obj_name}_{material}_uv.obj`

These can then be used in the MuJoCo program. See the `eamples/mujoco/normal_map_test.py` for a demonstration.

### CUDA backend

The optional CUDA backend accelerates the complete TAXIM contact-frame path:
mesh transformation, triangle rasterization, UV texture resolution,
height-map finalization, soft-body deformation, calibrated illumination, and
optional shadow casting. Intermediate height, mask, and deformation arrays
remain on the GPU; only the rendered image and requested depth/debug outputs
are downloaded. Install the CuPy package matching the system CUDA toolkit. For
CUDA 12, the project extra can be used:

```bash
pip install -e '.[cuda12]'
```

Select the backend when constructing the sensor:

```python
sensor = TaximSensor(
    sensor_type="digit",
    raster_backend="cuda",
    cuda_device=0,
)
```

Use `raster_backend="cpu"` for the reference implementation. CUDA kernels are
compiled during sensor construction. Static gel, lighting, shadow, and
background calibration is uploaded once, while object geometry is uploaded by
`add_geom_mujoco`. Benchmark several warm-up frames before resetting the timing
counters and collecting results. The timing summary reports `hm_total`,
`deform_total`, and `sim_total` from CUDA events, `cuda_frame_download` for the
final transfers, and `cuda_pipeline_wall` for the complete host-visible call.

The indexed geometry counts used by either backend can be inspected after
object registration:

```python
print(sensor.get_raster_stats("object_geom"))
```

## Operating System
MuJoCo-Taxim has been tested on Ubuntu 22.04. Chances are, the package will work fine in almost any environment so long as it is capable of installing and running MuJoCo 3.

Configuration for Ubuntu:
python 3.11
numpy 2.4.1,
scipy 1.17.0
opencv-python 4.13
mujoco 3.2.6
trimesh 4.8.3

## License
MuJoCo-Taxim is licensed under [MIT license](LICENSE).

## Citating Taxim
If you use Taxim in your research, please cite:
```BibTeX
@article{si2021taxim,
  title={Taxim: An Example-based Simulation Model for GelSight Tactile Sensors},
  author={Si, Zilin and Yuan, Wenzhen},
  journal={arXiv preprint arXiv:2109.04027},
  year={2021}
}
```
