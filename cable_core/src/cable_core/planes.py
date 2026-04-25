from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RoutingPlane:
    origin: np.ndarray
    normal: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray

    @staticmethod
    def from_config_entry(entry: Dict[str, Any]) -> "RoutingPlane":
        origin = np.asarray(entry.get("origin", [0.0, 0.0, 0.15]), dtype=float).reshape(3)
        normal = np.asarray(entry.get("normal", [0.0, 0.0, 1.0]), dtype=float).reshape(3)
        u_axis = np.asarray(entry.get("u_axis", [1.0, 0.0, 0.0]), dtype=float).reshape(3)

        normal /= np.linalg.norm(normal) + 1e-8
        u_axis = u_axis - np.dot(u_axis, normal) * normal
        u_axis /= np.linalg.norm(u_axis) + 1e-8
        v_axis = np.cross(normal, u_axis)
        v_axis /= np.linalg.norm(v_axis) + 1e-8

        return RoutingPlane(origin=origin, normal=normal, u_axis=u_axis, v_axis=v_axis)


def get_routing_plane(config: Any, clip_id: Optional[int] = None) -> RoutingPlane:
    if config is None:
        raise RuntimeError("Config not available.")

    plane_id = getattr(config, "routing_plane_default_id", "main")
    clip_assignments = getattr(config, "clip_plane_assignments", {}) or {}
    if clip_id is not None and clip_id in clip_assignments:
        plane_id = clip_assignments[clip_id]

    planes = getattr(config, "routing_planes", None)
    if not planes or plane_id not in planes:
        raise RuntimeError(f"Routing plane '{plane_id}' not defined in config.")

    return RoutingPlane.from_config_entry(planes[plane_id])


def project_to_plane(point_world: np.ndarray, plane: RoutingPlane) -> np.ndarray:
    point = np.asarray(point_world, dtype=float).reshape(3)
    vec = point - plane.origin
    signed = float(np.dot(vec, plane.normal))
    return point - signed * plane.normal


def point_at_plane_height(
    point_world: np.ndarray,
    plane: RoutingPlane,
    height_above_plane_m: float,
) -> np.ndarray:
    point_on_plane = project_to_plane(point_world, plane)
    return point_on_plane + float(height_above_plane_m) * plane.normal


def ensure_min_plane_height(
    point_world: np.ndarray,
    plane: RoutingPlane,
    min_height_above_plane_m: float,
) -> np.ndarray:
    point = np.asarray(point_world, dtype=float).reshape(3)
    signed = float(np.dot(point - plane.origin, plane.normal))
    if signed < float(min_height_above_plane_m):
        return point + (float(min_height_above_plane_m) - signed) * plane.normal
    return point


def routing_plane_is_world_yz(plane: RoutingPlane, cos_threshold: float = 0.85) -> bool:
    normal = np.asarray(plane.normal, dtype=float).reshape(3)
    normal /= np.linalg.norm(normal) + 1e-8
    ex = np.array([1.0, 0.0, 0.0], dtype=float)
    return float(abs(np.dot(normal, ex))) >= float(cos_threshold)
