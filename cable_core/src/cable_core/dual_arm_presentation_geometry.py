"""
Tool orientations for dual-arm cable presentation (world frame).
"""

from __future__ import annotations

import numpy as np

from cable_core.planes import RoutingPlane


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        raise ValueError("zero vector")
    return vec / norm


def rotation_carrier_cable_vertical_world(plane: RoutingPlane) -> np.ndarray:
    n = _unit(np.asarray(plane.normal, dtype=float).reshape(3))
    x_axis = np.array([0.0, 0.0, -1.0], dtype=float)
    z_axis = -n
    if abs(float(np.dot(x_axis, z_axis))) > 0.995:
        raise RuntimeError(
            "Cable-down axis nearly parallel to board normal; check routing plane."
        )
    y_axis = _unit(np.cross(z_axis, x_axis))
    z_axis = _unit(np.cross(x_axis, y_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def rotation_world_ry_deg(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(theta_deg))
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rotation_second_arm_side_grasp_world(second_arm_is_right: bool) -> np.ndarray:
    x_axis = np.array([0.0, 0.0, -1.0], dtype=float)
    z_axis = np.array([0.0, 1.0, 0.0], dtype=float) if second_arm_is_right else np.array([0.0, -1.0, 0.0], dtype=float)
    y_axis = _unit(np.cross(z_axis, x_axis))
    z_axis = _unit(np.cross(x_axis, y_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1)
