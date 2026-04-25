from typing import Dict

from cable_orchestrator.base_step import BaseStep
from cable_studio.visualization_service import VisualizationService


class VisualizeGraspsStep(BaseStep):
    name = "visualize_grasps"
    description = "Overlay grasp poses on image."

    def __init__(self):
        super().__init__()
        self.service = VisualizationService()

    def run(self, state) -> Dict[str, object]:
        if state.rgb_image is None:
            raise RuntimeError("No image available.")
        if not hasattr(state, "grasp_poses"):
            raise RuntimeError("No grasp poses available.")

        env = state.env
        img = state.rgb_image.copy()
        for pose in state.grasp_poses:
            arm = pose.get("arm", "right")
            img = self.service.draw_grasps(
                image=img,
                poses=[pose],
                env=env,
                config=state.config,
                arm=arm,
            )

        state.grasp_overlay = img
        return {
            "overlay_available": True,
            "num_grasps": len(state.grasp_poses),
        }
