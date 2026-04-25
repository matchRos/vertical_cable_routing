from typing import Any, Mapping, Tuple

import numpy as np

from cable_core.board_projection import pixel_from_world_debug, world_from_pixel_debug
from cable_core.motion_primitives.c_clip import clip_forward_axis_px


def build_u_clip_entry_pixels(
    state: Any,
    curr_clip: Any,
    primary_arm: str,
    secondary_arm: str,
    clip_type_config: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build clip-local approach targets for a U-clip.
    """
    center = np.array([float(curr_clip.x), float(curr_clip.y)], dtype=float)
    forward = clip_forward_axis_px(float(curr_clip.orientation))

    entry_offset_m = float(clip_type_config.get("entry_offset_m", 0.05))
    exit_offset_m = float(clip_type_config.get("exit_offset_m", 0.05))

    center_world = world_from_pixel_debug(
        state.env,
        state.config,
        (float(center[0]), float(center[1])),
        arm=primary_arm,
        is_clip=True,
        image_shape=state.rgb_image.shape,
    ).reshape(3)
    forward_probe_world = world_from_pixel_debug(
        state.env,
        state.config,
        (float(center[0] + forward[0]), float(center[1] + forward[1])),
        arm=primary_arm,
        is_clip=True,
        image_shape=state.rgb_image.shape,
    ).reshape(3)

    clip_forward_world = forward_probe_world - center_world
    clip_forward_world[0] = 0.0
    norm = float(np.linalg.norm(clip_forward_world))
    if norm < 1e-9:
        raise RuntimeError("Could not determine U-clip forward direction in world space.")
    clip_forward_world /= norm

    primary_world = center_world - clip_forward_world * entry_offset_m
    secondary_world = center_world + clip_forward_world * exit_offset_m

    primary_uv = pixel_from_world_debug(state.env, state.config, primary_world, arm=primary_arm)
    secondary_uv = pixel_from_world_debug(
        state.env,
        state.config,
        secondary_world,
        arm=secondary_arm,
    )
    if primary_uv is None or secondary_uv is None:
        raise RuntimeError("Could not project U-clip world targets back to image pixels.")

    primary_px = np.array([float(primary_uv[0]), float(primary_uv[1])], dtype=float)
    secondary_px = np.array([float(secondary_uv[0]), float(secondary_uv[1])], dtype=float)
    return primary_px, secondary_px
