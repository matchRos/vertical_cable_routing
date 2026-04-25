from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np


def pixel_to_3d_world(
    pixel_coord: Sequence[float],
    depth: float,
    cam_intrinsics: Any,
    cam_extrinsics: Any,
) -> np.ndarray:
    pixel_homogeneous = np.array([pixel_coord[0], pixel_coord[1], 1.0], dtype=float)
    point_3d_cam = np.linalg.inv(cam_intrinsics._K).dot(pixel_homogeneous) * float(depth)
    return cam_extrinsics.rotation.dot(point_3d_cam) + cam_extrinsics.translation


def get_world_coord_from_pixel_coord(
    pixel_coord: Sequence[float],
    cam_intrinsics: Any,
    cam_extrinsics: Any,
    image_shape: Optional[Tuple[int, int]] = None,
    table_depth: float = 0.83615,
    depth_map: Any = None,
    neighborhood_radius: int = 10,
    display: bool = False,
    is_clip: bool = False,
    table_depth_compensation: bool = False,
    arm: str = "right",
    depth_compensation_for_c_shape_clip: bool = False,
    depth_compensation_for_crossing_case: bool = False,
    grasp_depth_compensation: bool = False,
) -> np.ndarray:
    """
    Minimal local copy of the pixel->world projection helper used by the legacy stack.

    This keeps `cable_core` independent from the old repository while preserving the
    current projection behavior for the migration period.
    """
    pixel = np.array(pixel_coord, dtype=np.float32)

    if image_shape and (
        cam_intrinsics.width != image_shape[1]
        or cam_intrinsics.height != image_shape[0]
    ):
        pixel[0] *= cam_intrinsics.width / image_shape[1]
        pixel[1] *= cam_intrinsics.height / image_shape[0]

    if is_clip:
        depth = 0.81
    else:
        depth = float(table_depth)

        if depth_map is not None:
            try:
                import cv2  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "OpenCV is required for depth-map based pixel projection."
                ) from exc

            x, y = int(pixel[0]), int(pixel[1])
            y_min, y_max = max(0, y - neighborhood_radius), min(
                depth_map.shape[0], y + neighborhood_radius + 1
            )
            x_min, x_max = max(0, x - neighborhood_radius), min(
                depth_map.shape[1], x + neighborhood_radius + 1
            )
            focus_region = depth_map[y_min:y_max, x_min:x_max].copy()
            mask = np.zeros_like(focus_region, dtype=np.uint8)
            cv2.circle(
                mask,
                (neighborhood_radius, neighborhood_radius),
                neighborhood_radius,
                1,
                -1,
            )
            focus_region *= mask
            valid_depths = focus_region[focus_region > 0]
            if valid_depths.size > 0:
                depth = float(np.min(valid_depths))
        else:
            if table_depth_compensation and arm == "right":
                depth += 0.002
            elif table_depth_compensation and arm == "left":
                depth += 0.009

            if grasp_depth_compensation and arm == "left":
                depth += 0.004

        depth = min(depth, float(table_depth))

        if depth_compensation_for_c_shape_clip:
            depth += 0.001

        if depth_compensation_for_crossing_case:
            depth -= 0.003

    if display:
        print(
            "camera_projection.get_world_coord_from_pixel_coord display output is not "
            "implemented in cable_core; continuing without visualization."
        )

    return pixel_to_3d_world(pixel, depth, cam_intrinsics, cam_extrinsics)


def project_world_to_pixel(point_world: Sequence[float], intrinsic: Any, t_cam_base: Any):
    """
    Project a world-frame 3D point into the camera image.
    """
    t_base_cam = t_cam_base.inverse()
    point = np.asarray(point_world, dtype=float).reshape(3)
    point_cam = t_base_cam.rotation @ point + t_base_cam.translation
    x, y, z = point_cam
    if z <= 0:
        return None
    k_mat = intrinsic._K
    u = k_mat[0, 0] * x / z + k_mat[0, 2]
    v = k_mat[1, 1] * y / z + k_mat[1, 2]
    return int(u), int(v)
