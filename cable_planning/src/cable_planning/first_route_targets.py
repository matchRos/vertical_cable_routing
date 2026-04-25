"""
World-frame targets for executing the first-route preview.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from cable_core.board_projection import world_from_pixel_debug
from cable_core.clip_types import CLIP_TYPE_C_CLIP, CLIP_TYPE_PEG
from cable_core.motion_primitives.c_clip import build_c_clip_center_pixels
from cable_core.planes import ensure_min_plane_height, get_routing_plane
from cable_planning.pose_validation import is_dual_arm_grasp, validate_min_distance


def _grasp_pose_for_arm_or_fallback(state: Any, arm_name: str) -> Dict[str, Any]:
    for pose in state.grasp_poses:
        if pose.get("arm") == arm_name:
            return pose
    if len(state.grasp_poses) == 1:
        return state.grasp_poses[0]
    raise RuntimeError(f"No grasp pose for arm '{arm_name}'.")


def _pixel_to_world_clip(uv: np.ndarray, state: Any, arm: str) -> np.ndarray:
    return world_from_pixel_debug(
        state.env,
        state.config,
        (float(uv[0]), float(uv[1])),
        arm=arm,
        is_clip=True,
        image_shape=state.rgb_image.shape,
    ).reshape(3)


def _move_pixel_along_route(primary_px: np.ndarray, state: Any) -> np.ndarray:
    extra_px = float(getattr(state.config, "first_route_primary_extra_along_route_px", 0.0))
    if extra_px <= 1e-6:
        return primary_px

    curr_idx = getattr(state, "first_route_curr_clip_id", None)
    next_idx = getattr(state, "first_route_next_clip_id", None)
    if state.clips is None or curr_idx is None or next_idx is None:
        return primary_px

    curr_clip = state.clips[curr_idx]
    next_clip = state.clips[next_idx]
    p_curr = np.array([float(curr_clip.x), float(curr_clip.y)], dtype=float)
    p_next = np.array([float(next_clip.x), float(next_clip.y)], dtype=float)
    direction = p_next - p_curr
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return primary_px
    return primary_px + direction / norm * extra_px


def _route_height_for_state(state: Any) -> float:
    return float(
        getattr(
            state,
            "first_route_route_height_m",
            float(state.config.routing_height_above_plane_m),
        )
    )


def _execute_secondary_arm(state: Any) -> bool:
    return bool(
        getattr(state.config, "first_route_execute_secondary_arm", True)
        and getattr(state, "first_route_secondary_shown", False)
        and getattr(state, "first_route_secondary_target_px", None) is not None
    )


def build_first_route_execution_poses(
    state: Any,
    min_dist_xyz: float = 0.08,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    if state.env is None:
        raise RuntimeError("Environment not available.")
    if getattr(state.env, "board_yz_calibration", None) is None:
        if state.env.camera is None or not hasattr(state.env, "T_CAM_BASE"):
            raise RuntimeError("Camera / T_CAM_BASE not available (no board homography).")
    if state.rgb_image is None:
        raise RuntimeError("No rgb_image in state.")
    if not hasattr(state, "grasp_poses"):
        raise RuntimeError("No grasp_poses in state.")
    if getattr(state, "first_route_target_px", None) is None:
        raise RuntimeError(
            "No first_route_target_px. Run plan_first_route before execute_first_route."
        )

    primary_arm = getattr(state, "current_primary_arm", None) or "left"
    secondary_arm = "right" if primary_arm == "left" else "left"

    curr_idx = getattr(state, "first_route_curr_clip_id", None)
    if curr_idx is None or state.clips is None:
        raise RuntimeError("Missing first_route_curr_clip_id or clips.")
    curr_clip = state.clips[curr_idx]
    clip_type = int(curr_clip.clip_type)
    plane = get_routing_plane(state.config, clip_id=curr_idx)
    routing_height = _route_height_for_state(state)
    mode = getattr(state, "first_route_mode", None)

    primary_px = np.asarray(state.first_route_target_px, dtype=float).reshape(2)
    if mode not in {"c_clip_entry", "u_clip_entry"}:
        primary_px = _move_pixel_along_route(primary_px, state)
    primary_pos = ensure_min_plane_height(
        _pixel_to_world_clip(primary_px, state, primary_arm),
        plane,
        routing_height,
    )
    primary_pose = {
        "position": primary_pos,
        "rotation": np.asarray(_grasp_pose_for_arm_or_fallback(state, primary_arm)["rotation"]),
    }

    secondary_pose = _grasp_pose_for_arm_or_fallback(state, secondary_arm)
    secondary_hold_pos = ensure_min_plane_height(
        np.asarray(secondary_pose["position"], dtype=float).copy(),
        plane,
        routing_height,
    )
    secondary_hold = {
        "position": secondary_hold_pos,
        "rotation": np.asarray(secondary_pose["rotation"]),
    }

    if clip_type == CLIP_TYPE_PEG:
        left, right = (
            (primary_pose, secondary_hold)
            if primary_arm == "left"
            else (secondary_hold, primary_pose)
        )
        if is_dual_arm_grasp(state.config):
            validate_min_distance(left, right, min_dist_xyz, label="First route (peg)")
        return left, right, "peg_hold"

    if not getattr(state, "first_route_secondary_shown", False):
        raise RuntimeError(
            "first_route_secondary_shown is False but clip is not a peg; re-run plan_first_route."
        )
    sec_px = getattr(state, "first_route_secondary_target_px", None)
    if sec_px is None:
        raise RuntimeError("Missing first_route_secondary_target_px for dual first route.")

    secondary_pos = ensure_min_plane_height(
        _pixel_to_world_clip(np.asarray(sec_px, dtype=float).reshape(2), state, secondary_arm),
        plane,
        routing_height,
    )
    secondary_target = {
        "position": secondary_pos,
        "rotation": np.asarray(secondary_pose["rotation"]),
    }

    left, right = (
        (primary_pose, secondary_target)
        if primary_arm == "left"
        else (secondary_target, primary_pose)
    )
    if is_dual_arm_grasp(state.config) or _execute_secondary_arm(state):
        validate_min_distance(left, right, min_dist_xyz, label=f"First route ({mode or 'dual'})")
    return left, right, mode or "dual_slide"


def build_c_clip_centering_poses(
    state: Any,
    min_dist_xyz: float = 0.08,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    if state.env is None:
        raise RuntimeError("Environment not available.")
    if getattr(state.env, "board_yz_calibration", None) is None:
        if state.env.camera is None or not hasattr(state.env, "T_CAM_BASE"):
            raise RuntimeError("Camera / T_CAM_BASE not available (no board homography).")
    if state.rgb_image is None:
        raise RuntimeError("No rgb_image in state.")
    if not hasattr(state, "grasp_poses"):
        raise RuntimeError("No grasp_poses in state.")

    primary_arm = getattr(state, "current_primary_arm", None) or "left"
    secondary_arm = "right" if primary_arm == "left" else "left"

    curr_idx = getattr(state, "first_route_curr_clip_id", None)
    if curr_idx is None or state.clips is None:
        raise RuntimeError("Missing first_route_curr_clip_id or clips.")
    curr_clip = state.clips[curr_idx]
    if int(curr_clip.clip_type) != CLIP_TYPE_C_CLIP:
        raise RuntimeError("Centering phase is only valid for C-clip.")

    plane = get_routing_plane(state.config, clip_id=curr_idx)
    routing_height = _route_height_for_state(state)
    primary_px, secondary_px = build_c_clip_center_pixels(
        curr_clip=curr_clip,
        primary_arm=primary_arm,
        config=state.config,
    )

    primary_pose = {
        "position": ensure_min_plane_height(
            _pixel_to_world_clip(primary_px, state, primary_arm),
            plane,
            routing_height,
        ),
        "rotation": np.asarray(_grasp_pose_for_arm_or_fallback(state, primary_arm)["rotation"]),
    }
    secondary_pose = {
        "position": ensure_min_plane_height(
            _pixel_to_world_clip(secondary_px, state, secondary_arm),
            plane,
            routing_height,
        ),
        "rotation": np.asarray(_grasp_pose_for_arm_or_fallback(state, secondary_arm)["rotation"]),
    }

    left, right = (
        (primary_pose, secondary_pose)
        if primary_arm == "left"
        else (secondary_pose, primary_pose)
    )
    if is_dual_arm_grasp(state.config) or _execute_secondary_arm(state):
        validate_min_distance(left, right, min_dist_xyz, label="C-clip centering")
    return left, right, "c_clip_center"
