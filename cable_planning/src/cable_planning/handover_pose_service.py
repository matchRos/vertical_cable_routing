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
