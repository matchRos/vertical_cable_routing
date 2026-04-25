from __future__ import annotations

import numpy as np

from cable_core.board_projection import world_from_pixel_debug
from cable_core.camera_projection import get_world_coord_from_pixel_coord


class PathProjectionService:
    def convert_path_to_world(self, env, path_pixels, arm: str = "right", config=None):
        """
        Convert a pixel path to world coordinates using either:
        - board YZ homography, or
        - pinhole camera model + `T_CAM_BASE`.
        """
        if path_pixels is None or len(path_pixels) == 0:
            raise RuntimeError("No pixel path available for projection.")
        if env is None:
            raise RuntimeError("Environment not initialized.")

        world_path = []
        for pixel_coord in path_pixels:
            pixel_coord = np.asarray(pixel_coord).squeeze().reshape(-1)
            pixel_xy = (float(pixel_coord[0]), float(pixel_coord[1]))

            if getattr(env, "board_yz_calibration", None) is not None:
                if config is None:
                    raise RuntimeError(
                        "config is required for board YZ homography path projection."
                    )
                world_coord = world_from_pixel_debug(
                    env,
                    config,
                    pixel_xy,
                    arm=arm,
                    is_clip=False,
                    image_shape=None,
                )
            else:
                if env.camera is None:
                    raise RuntimeError("Camera not available in debug context.")
                if not hasattr(env, "T_CAM_BASE"):
                    raise RuntimeError("DebugContext has no T_CAM_BASE transform.")
                if arm not in env.T_CAM_BASE:
                    raise RuntimeError(f"Arm '{arm}' not found in T_CAM_BASE.")

                intrinsic = env.camera.intrinsic
                t_cam_base = env.T_CAM_BASE[arm]
                world_coord = get_world_coord_from_pixel_coord(
                    (int(pixel_xy[0]), int(pixel_xy[1])),
                    intrinsic,
                    t_cam_base,
                )

            world_path.append(world_coord)

        return np.array(world_path)
