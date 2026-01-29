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


