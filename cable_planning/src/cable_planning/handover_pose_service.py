"""
Handover helpers: small tool-frame adjustments relative to the current grasp rotation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from cable_core.board_projection import world_from_pixel_debug


def routing_clip_world_m(
    state: Any,
    clip_index: int,
    arm: str = "right",
) -> np.ndarray:
    """World position for clip centre from board projection."""
    if state.env is None:
        raise RuntimeError("Environment not available.")
    clips = state.clips
    if clips is None:
        raise RuntimeError("clips not available.")
    clip = clips[int(clip_index)]
    img_shape = state.rgb_image.shape if state.rgb_image is not None else None
    return world_from_pixel_debug(
        state.env,
        state.config,
        (float(clip.x), float(clip.y)),
        arm=arm,
        is_clip=True,
        image_shape=img_shape,
    ).reshape(3)


def resolve_handover_arm(state: Any, override: Optional[str]) -> str:
    if override in ("left", "right"):
        return str(override)
    first = getattr(state, "descend_first_arm", None)
    if first in ("left", "right"):
        return str(first)
    if hasattr(state, "grasp_poses") and state.grasp_poses:
        arm = state.grasp_poses[0].get("arm", "right")
        if arm in ("left", "right"):
            return str(arm)
    raise RuntimeError(
        "Cannot resolve handover arm: set handover_arm to 'left'/'right' or run descend/grasp first."
    )


def grasp_pose_for_arm(grasp_poses: List[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    for pose in grasp_poses:
        if pose.get("arm") == arm:
            return pose
    if len(grasp_poses) == 1:
        return grasp_poses[0]
    raise RuntimeError(f"No grasp pose for arm '{arm}'.")


def lift_offset_along_plane_normal(
    plane: Any,
    lift_distance_m: float,
) -> np.ndarray:
    normal = np.asarray(plane.normal, dtype=float).reshape(3)
    normal /= np.linalg.norm(normal) + 1e-8
    return float(lift_distance_m) * normal


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        raise RuntimeError("Cannot normalize near-zero vector.")
    return arr / norm


def _rotation_about_world_axis(axis: np.ndarray, theta_rad: float) -> np.ndarray:
    axis = _unit(axis)
    x, y, z = axis
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def _tool_axis_world(rotation: np.ndarray, axis_name: str) -> np.ndarray:
    name = str(axis_name).strip().lower().replace("tool_", "")
    sign = -1.0
    if name.startswith("+"):
        sign = 1.0
        name = name[1:]
    elif name.startswith("-"):
        name = name[1:]
    else:
        sign = 1.0

    if name == "x":
        idx = 0
    elif name == "y":
        idx = 1
    else:
        raise RuntimeError(
            f"Unsupported handover exchange alignment axis '{axis_name}'. Use tool_x or tool_y."
        )
    return sign * np.asarray(rotation, dtype=float).reshape(3, 3)[:, idx]


def align_tool_axis_to_direction_about_tool_z(
    rotation: np.ndarray,
    direction_world: np.ndarray,
    tool_axis: str = "tool_x",
    yaw_offset_deg: float = 0.0,
    allow_axis_flip: bool = True,
) -> np.ndarray:
    """
    Preserve tool-z and yaw the TCP so a tool x/y axis is parallel to direction_world.
    """
    rot = np.asarray(rotation, dtype=float).reshape(3, 3)
    tool_z = _unit(rot[:, 2])

    target = np.asarray(direction_world, dtype=float).reshape(3)
    target = target - float(np.dot(target, tool_z)) * tool_z
    target = _unit(target)

    axis = _tool_axis_world(rot, tool_axis)
    axis = axis - float(np.dot(axis, tool_z)) * tool_z
    axis = _unit(axis)

    if bool(allow_axis_flip) and float(np.dot(axis, target)) < 0.0:
        target = -target

    yaw = float(np.arctan2(np.dot(tool_z, np.cross(axis, target)), np.dot(axis, target)))
    yaw += float(np.deg2rad(yaw_offset_deg))
    return _rotation_about_world_axis(tool_z, yaw) @ rot


def _rot_axis_deg(axis: str, deg: float) -> np.ndarray:
    theta = np.deg2rad(float(deg))
    c, s = np.cos(theta), np.sin(theta)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    raise ValueError(axis)


def fine_orient_on_grasp_rotation(
    grasp_rotation: np.ndarray,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
) -> np.ndarray:
    rotation = np.asarray(grasp_rotation, dtype=float).reshape(3, 3)
    return (
        rotation
        @ _rot_axis_deg("z", rz_deg)
        @ _rot_axis_deg("y", ry_deg)
        @ _rot_axis_deg("x", rx_deg)
    )


class HandoverPoseService:
    """Thin facade for tests and future extensions."""

    def routing_clip_world_m(self, state: Any, clip_index: int, arm: str) -> np.ndarray:
        return routing_clip_world_m(state, clip_index, arm=arm)

    def fine_orient_on_grasp_rotation(
        self,
        grasp_rotation: np.ndarray,
        rx_deg: float,
        ry_deg: float,
        rz_deg: float,
    ) -> np.ndarray:
        return fine_orient_on_grasp_rotation(grasp_rotation, rx_deg, ry_deg, rz_deg)
