import cv2
import numpy as np

from cable_core.board_projection import pixel_from_world_debug


class VisualizationService:
    def draw_grasps(self, image, poses, env, config, arm="right"):
        img = image.copy()
        intrinsic = env.camera.intrinsic if env.camera is not None else None
        t_cam_base = None
        if hasattr(env, "T_CAM_BASE") and env.T_CAM_BASE and arm in env.T_CAM_BASE:
            t_cam_base = env.T_CAM_BASE[arm]

        for idx, pose in enumerate(poses):
            pos = pose["position"]
            rot = pose["rotation"]

            px = pixel_from_world_debug(
                env,
                config,
                np.asarray(pos, dtype=float),
                arm=arm,
                intrinsic=intrinsic,
                T_cam_base=t_cam_base,
            )
            if px is None:
                continue

            u, v = px
            cv2.circle(img, (u, v), 6, (0, 0, 255), -1)

            direction = pose.get("tangent_world")
            if direction is None:
                direction = rot[:, 0]
            else:
                direction = np.asarray(direction, dtype=float).reshape(3)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            tip = np.asarray(pos) + np.asarray(direction) * 0.05

            tip_px = pixel_from_world_debug(
                env,
                config,
                np.asarray(tip, dtype=float),
                arm=arm,
                intrinsic=intrinsic,
                T_cam_base=t_cam_base,
            )
            if tip_px is not None:
                cv2.arrowedLine(img, (u, v), tip_px, (255, 0, 0), 2)

            cv2.putText(
                img,
                f"{pose.get('arm', arm)}_{idx}",
                (u + 5, v - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        return img
