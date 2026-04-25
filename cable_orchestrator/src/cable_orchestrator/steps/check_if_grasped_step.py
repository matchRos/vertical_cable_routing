from typing import Any, Dict, Optional

import numpy as np
import cv2
from std_srvs.srv import Trigger
from cable_perception.image_utils import is_cable_in_gripper

from cable_orchestrator.base_step import BaseStep


class CheckIfGraspedStep(BaseStep):
    name = "check_if_grasped"
    description = "Check if the cable is in either gripper."

    def __init__(self):
        super().__init__()
    

    def run(self, state) -> Dict[str, object]:
        service_name = "/yumi/camera/is_grasped"
        
        if state.env is None:
            raise RuntimeError("Debug context not initialized. Run init_environment first.")

        left_img, _ = self.get_image_from_camera(
            camera=state.env.left_camera
        )
        if left_img is None:
            raise RuntimeError(
                "No left image available. Neither camera nor debug_image_path provided a valid image."
            )
        
        right_img, _ = self.get_image_from_camera(
            camera=state.env.right_camera
        )
        if right_img is None:
            raise RuntimeError(
                "No right image available. Neither camera nor debug_image_path provided a valid image."
            )

        state.left_rgb_image = left_img
        state.right_rgb_image = right_img
        
        left_gripper_bbox = None # TODO: set left gripper bounding box
        right_gripper_bbox = None # TODO: set right gripper bounding box
        left_anchor_point = None # TODO: set left anchor point based on bounding box
        right_anchor_point = None # TODO: set right anchor point based on bounding box


        state.left_grasped = is_cable_in_gripper(
            state.left_rgb_image,
            left_gripper_bbox,
            clip={"x": left_anchor_point[0], "y": left_anchor_point[1]} if left_anchor_point is not None else None
        )
        state.right_grasped = is_cable_in_gripper(
            state.right_rgb_image,
            right_gripper_bbox,
            clip={"x": right_anchor_point[0], "y": right_anchor_point[1]} if right_anchor_point is not None else None
        )
        return {
            "left_gripper_has_cable": state.left_grasped,
            "right_gripper_has_cable": state.right_grasped,
            "service": service_name,
        }
    
    def get_image_from_camera(self, camera: Any) -> Optional[np.ndarray]:
        if camera is None:
            return None

        for method_name in ("get_rgb", "get_rgb_image", "get_image", "get_frame", "read"):
            if hasattr(camera, method_name):
                try:
                    result = getattr(camera, method_name)()
                    if isinstance(result, np.ndarray):
                        return self._ensure_rgb_uint8(result)
                except Exception:
                    pass
        return None
    
    def _ensure_rgb_uint8(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image