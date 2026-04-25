from typing import Dict

from cable_orchestrator.base_step import BaseStep
from cable_perception.path_projection_service import PathProjectionService


class TraceToWorldStep(BaseStep):
    name = "trace_to_world"
    description = "Convert traced pixel path to world coordinates."

    def __init__(self):
        super().__init__()
        self.service = PathProjectionService()

    def run(self, state) -> Dict[str, object]:
        if state.env is None:
            raise RuntimeError("Environment not initialized.")
        if state.path_in_pixels is None:
            raise RuntimeError("No pixel path available. Run trace_cable first.")

        if getattr(state.env, "board_yz_calibration", None) is not None:
            arm = "right"
        else:
            available_arms = list(getattr(state.env, "T_CAM_BASE", {}).keys())
            if not available_arms:
                raise RuntimeError("T_CAM_BASE is empty or missing.")
            if "right" in available_arms:
                arm = "right"
            elif "left" in available_arms:
                arm = "left"
            else:
                raise RuntimeError(f"No valid arm found in T_CAM_BASE. Available keys: {available_arms}")

        path_world = self.service.convert_path_to_world(
            env=state.env,
            path_pixels=state.path_in_pixels,
            arm=arm,
            config=state.config,
        )
        state.path_in_world = path_world

        return {
            "path_world_available": True,
            "num_world_points": len(path_world),
            "first_point": path_world[0].tolist(),
            "last_point": path_world[-1].tolist(),
            "arm": arm,
        }
