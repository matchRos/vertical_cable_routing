from typing import Any, Dict, Optional

import numpy as np
import cv2
from std_srvs.srv import Trigger
from cable_perception.image_utils import is_cable_too_lax

from cable_orchestrator.base_step import BaseStep


class CheckIfLaxStep(BaseStep):
    name = "check_if_lax"
    description = "Check if the cable is too lax."

    MODEL_WEIGHTS = "path/to/your/model/weights.pt"  # TODO: set the correct path to model weights once the model is trained
    MODEL_TYPE = None  # TODO: set the correct model type, e.g., YourModelClass

    def __init__(self):
        super().__init__()    

    def run(self, state) -> Dict[str, object]:
        service_name = "/yumi/camera/is_lax"
        
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


        state.is_lax_from_left = is_cable_too_lax(
            state.left_rgb_image,
            MODEL_TYPE,
            MODEL_WEIGHTS
        )
        state.is_lax_from_right = is_cable_too_lax(
            state.right_rgb_image,
            MODEL_TYPE,
            MODEL_WEIGHTS
        )
        return {
            "is_lax_from_left": state.is_lax_from_left,
            "is_lax_from_right": state.is_lax_from_right,
            "is_lax": state.is_lax_from_left or state.is_lax_from_right,
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