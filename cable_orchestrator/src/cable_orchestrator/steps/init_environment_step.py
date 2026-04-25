from typing import Any, Dict, Optional, Tuple

import numpy as np

from cable_core.board_models import DebugBoard
from cable_core.board_yz_calibration import load_board_yz_calibration_optional
from cable_orchestrator.base_step import BaseStep
from cable_perception.camera_adapter import create_camera_subscriber
from cable_perception.tracer_adapter import create_cable_tracer
from cable_studio.debug_config import DebugConfig, load_debug_config
from cable_studio.debug_context import DebugContext


class SimpleRigidTransform:
    def __init__(self, rotation, translation, from_frame="zed", to_frame="base_link"):
        self.rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        self.translation = np.asarray(translation, dtype=float).reshape(3)
        self.from_frame = from_frame
        self.to_frame = to_frame

    def as_frames(self, from_frame=None, to_frame=None):
        return SimpleRigidTransform(
            self.rotation,
            self.translation,
            from_frame=from_frame or self.from_frame,
            to_frame=to_frame or self.to_frame,
        )

    def inverse(self):
        inv_rotation = self.rotation.T
        inv_translation = -inv_rotation @ self.translation
        return SimpleRigidTransform(
            inv_rotation,
            inv_translation,
            from_frame=self.to_frame,
            to_frame=self.from_frame,
        )


class InitEnvironmentStep(BaseStep):
    name = "init_environment"
    description = "Initialize config, debug board, and optional camera preview."

    def _create_debug_config(self) -> DebugConfig:
        return load_debug_config()

    def _try_create_board(self, config: DebugConfig) -> Tuple[Optional[Any], Optional[str]]:
        try:
            return DebugBoard(config_path=config.board_cfg_path), None
        except Exception as exc:
            return None, str(exc)

    def _try_create_camera(self, config: DebugConfig) -> Tuple[Optional[Any], Optional[str]]:
        try:
            import rospy

            if not rospy.core.is_initialized():
                rospy.init_node("cable_studio_camera", anonymous=True, disable_signals=True)
            rospy.sleep(1.0)
            return create_camera_subscriber(
                topic_rgb=getattr(config, "camera_rgb_topic", None),
                topic_depth=getattr(config, "camera_depth_topic", None),
                topic_camera_info=getattr(config, "camera_info_topic", None),
                require_depth=bool(getattr(config, "camera_require_depth", False)),
                wait_timeout_sec=float(getattr(config, "camera_wait_timeout_sec", 5.0)),
            ), None
        except Exception as exc:
            return None, str(exc)

    def _try_create_tracer(self) -> Tuple[Optional[Any], Optional[str]]:
        try:
            return create_cable_tracer(), None
        except Exception as exc:
            return None, str(exc)

    def _safe_get_camera_image(self, camera: Any) -> Optional[np.ndarray]:
        if camera is None:
            return None
        for method_name in ("get_rgb", "get_rgb_image", "get_image", "get_frame", "read"):
            if hasattr(camera, method_name):
                try:
                    result = getattr(camera, method_name)()
                    if isinstance(result, np.ndarray):
                        return result
                except Exception:
                    pass
        return None

    def _load_rigid_transform_file(self, path: str) -> SimpleRigidTransform:
        with open(path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.readlines() if line.strip()]
        if len(lines) < 5:
            raise RuntimeError(f"Rigid transform file is incomplete: {path}")
        from_frame = lines[0]
        to_frame = lines[1]
        translation = np.fromstring(lines[2], sep=" ")
        rotation = np.vstack([np.fromstring(line, sep=" ") for line in lines[3:6]])
        return SimpleRigidTransform(rotation, translation, from_frame=from_frame, to_frame=to_frame)

    def run(self, state) -> Dict[str, Any]:
        config = self._create_debug_config()
        board, board_error = self._try_create_board(config)
        camera, camera_error = self._try_create_camera(config)
        tracer, tracer_error = self._try_create_tracer()

        context = DebugContext(config=config, robot=None, camera=camera, board=board, tracer=tracer)
        context.camera_error = camera_error
        context.tracer_error = tracer_error
        context.board_yz_calibration = load_board_yz_calibration_optional(
            getattr(config, "board_calibration_yaml", None)
        )
        context.T_CAM_BASE = {}

        try:
            if config.cam_to_robot_left_trans_path:
                context.T_CAM_BASE["left"] = self._load_rigid_transform_file(config.cam_to_robot_left_trans_path).as_frames(from_frame="zed", to_frame="base_link")
            if config.cam_to_robot_right_trans_path:
                context.T_CAM_BASE["right"] = self._load_rigid_transform_file(config.cam_to_robot_right_trans_path).as_frames(from_frame="zed", to_frame="base_link")
        except Exception as exc:
            print(f"Warning: Failed to load T_CAM_BASE: {exc}")

        state.config = config
        state.env = context
        preview = self._safe_get_camera_image(camera)
        if preview is not None:
            state.rgb_image = preview

        return {
            "config_loaded": config is not None,
            "board_loaded": board is not None,
            "camera_loaded": camera is not None,
            "tracer_loaded": tracer is not None,
            "robot_loaded": False,
            "camera_preview_available": preview is not None,
            "board_cfg_path": config.board_cfg_path,
            "board_num_clips": board.num_clips() if board is not None else 0,
            "board_error": board_error,
            "camera_error": camera_error,
            "tracer_error": tracer_error,
            "debug_image_path": config.debug_image_path,
            "trace_start_points": config.trace_start_points,
            "trace_end_points": config.trace_end_points,
            "board_yz_calibration_loaded": context.board_yz_calibration is not None,
        }
