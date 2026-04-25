from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml


def _load_yaml_dict(path: str) -> Dict[str, Any]:
    yaml_path = Path(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Board calibration file not found: {path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at top level: {path}")
    return data


@dataclass
class BoardYZCalibration:
    """
    Pixel (u, v) in full image coordinates -> world (y, z) in base_link,
    with board plane at fixed world X.
    """

    homography: np.ndarray
    homography_inv: np.ndarray
    base_frame: str = "yumi_base_link"
    tcp_frame: str = ""

    @classmethod
    def from_yaml_path(cls, path: str) -> "BoardYZCalibration":
        data = _load_yaml_dict(path)
        h = data.get("homography_matrix_3x3")
        if h is None:
            raise KeyError(f"'homography_matrix_3x3' missing in board calibration: {path}")
        homography = np.asarray(h, dtype=float).reshape(3, 3)
        det = float(np.linalg.det(homography))
        if abs(det) < 1e-12:
            raise ValueError(f"Singular homography in {path}")
        return cls(
            homography=homography,
            homography_inv=np.linalg.inv(homography),
            base_frame=str(data.get("base_frame", "yumi_base_link")),
            tcp_frame=str(data.get("tcp_frame", "")),
        )

    def pixel_to_yz(self, u: float, v: float) -> Tuple[float, float]:
        point = self.homography @ np.array([float(u), float(v), 1.0], dtype=float)
        if abs(point[2]) < 1e-12:
            raise ValueError(f"Degenerate homogeneous coordinate for pixel ({u}, {v})")
        return float(point[0] / point[2]), float(point[1] / point[2])

    def yz_to_pixel(self, y: float, z: float) -> Tuple[float, float]:
        point = self.homography_inv @ np.array([float(y), float(z), 1.0], dtype=float)
        if abs(point[2]) < 1e-12:
            raise ValueError(f"Degenerate homogeneous coordinate for yz ({y}, {z})")
        return float(point[0] / point[2]), float(point[1] / point[2])

    def pixel_to_world(self, u: float, v: float, board_plane_x_m: float) -> np.ndarray:
        y, z = self.pixel_to_yz(u, v)
        return np.array([float(board_plane_x_m), y, z], dtype=float)


def load_board_yz_calibration_optional(path: Optional[str]) -> Optional[BoardYZCalibration]:
    if path is None or str(path).strip() == "":
        return None
    try:
        return BoardYZCalibration.from_yaml_path(str(path))
    except Exception as exc:
        print(f"Warning: could not load board YZ calibration from {path!r}: {exc}")
        return None
