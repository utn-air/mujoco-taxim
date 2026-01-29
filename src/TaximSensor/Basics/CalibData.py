from __future__ import annotations

from importlib import resources
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_PACKAGE = "TaximSensor"
_CALIB_DIR = "calibs"

import numpy as np

@contextmanager
def calib_file_path(name: str) -> Iterator[Path]:
    """
    Yields a real filesystem Path to a calibration file.
    Works even if package is in a zip/wheel (uses extraction if needed).
    """
    traversable = resources.files(_PACKAGE) / _CALIB_DIR / name
    with resources.as_file(traversable) as p:
        yield p


def read_calib_bytes(name: str) -> bytes:
    return (resources.files(_PACKAGE) / _CALIB_DIR / name).read_bytes()

def read_calib_np(name: str) -> np.ndarray:
    with calib_file_path(name) as p:
        return np.load(p, allow_pickle=True)
    
def read_calib_text(name: str, encoding: str = "utf-8") -> str:
    return (resources.files("TaximSensor") / "calibs" / name).read_text(encoding=encoding)

class CalibData:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        data = read_calib_np(dataPath)

        self.numBins = data['bins']
        self.grad_r = data['grad_r']
        self.grad_g = data['grad_g']
        self.grad_b = data['grad_b']