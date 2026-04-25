"""
Project between image pixels and world frame for board-centric tooling.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from cable_core.camera_projection import (
    get_world_coord_from_pixel_coord,
    project_world_to_pixel,
)


def _apply_world_z_offset_from_config(point_world: np.ndarray, config: Any) -> np.ndarray:
    dz = float(getattr(config, "world_from_pixel_z_offset_m", 0.0))
    point = np.asarray(point_world, dtype=float).reshape(3).copy()
    point[2] += dz
    return point


def world_from_pixel_debug(
    env: Any,
    config: Any,
    pixel_xy: Tuple[float, float],
    arm: str = "right",
    is_clip: bool = False,
    image_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    cal = getattr(env, "board_yz_calibration", None)
    if cal is not None:
        u, v = float(pixel_xy[0]), float(pixel_xy[1])
        bx = float(getattr(config, "board_plane_x_m", 0.56))
        point = cal.pixel_to_world(u, v, bx)
        return _apply_world_z_offset_from_config(point, config)

    if not hasattr(env, "T_CAM_BASE") or arm not in env.T_CAM_BASE:
        raise RuntimeError(f"T_CAM_BASE missing for arm '{arm}' in pinhole projection mode.")

    intrinsic = env.camera.intrinsic
    t_cam_base = env.T_CAM_BASE[arm]
    shape = image_shape
    if shape is None and getattr(env, "camera", None) is not None:
        img = None
        for name in ("get_rgb", "get_rgb_image", "get_image", "get_frame", "read"):
            if hasattr(env.camera, name):
                try:
                    img = getattr(env.camera, name)()
                    break
                except Exception:
                    pass
        if img is not None:
            shape = img.shape

    point = get_world_coord_from_pixel_coord(
        (float(pixel_xy[0]), float(pixel_xy[1])),
        intrinsic,
        t_cam_base,
        image_shape=shape,
        is_clip=is_clip,
        arm=arm,
    )
    return _apply_world_z_offset_from_config(np.asarray(point, dtype=float).reshape(3), config)


def pixel_from_world_debug(
    env: Any,
    config: Any,
    world_xyz: np.ndarray,
    arm: str = "right",
    intrinsic: Any = None,
    t_cam_base: Any = None,
) -> Optional[Tuple[int, int]]:
    cal = getattr(env, "board_yz_calibration", None)
    if cal is not None:
        point = np.asarray(world_xyz, dtype=float).reshape(3).copy()
        point[2] -= float(getattr(config, "world_from_pixel_z_offset_m", 0.0))
        u, v = cal.yz_to_pixel(float(point[1]), float(point[2]))
        return int(round(u)), int(round(v))

    if intrinsic is None:
        intrinsic = env.camera.intrinsic
    if t_cam_base is None:
        if not hasattr(env, "T_CAM_BASE") or arm not in env.T_CAM_BASE:
            return None
        t_cam_base = env.T_CAM_BASE[arm]

    point = np.asarray(world_xyz, dtype=float).reshape(3).copy()
    point[2] -= float(getattr(config, "world_from_pixel_z_offset_m", 0.0))
    return project_world_to_pixel(point, intrinsic, t_cam_base)
