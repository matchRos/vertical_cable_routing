from typing import Dict

import numpy as np

from cable_orchestrator.base_step import BaseStep
from cable_core.planes import get_routing_plane, routing_plane_is_world_yz
from cable_planning.grasp_pose_service import GraspPoseService


class GraspPoseStep(BaseStep):
    name = "grasp_pose"
    description = "Compute grasp poses (position + rotation)."

    def __init__(self):
        super().__init__()
        self.service = GraspPoseService()

    def _apply_initial_left_grasp_offset(self, state, poses) -> np.ndarray:
        offset = np.asarray(
            getattr(state.config, "initial_left_grasp_position_offset_m", (0.0, 0.0, 0.0)),
            dtype=float,
        ).reshape(3)
        if not np.any(np.abs(offset) > 1e-9):
            return offset

        for pose in poses:
            if pose.get("arm") == "left":
                pose["position"] = np.asarray(pose["position"], dtype=float).reshape(3) + offset
        return offset

    def run(self, state) -> Dict[str, object]:
        if not hasattr(state, "grasps"):
            raise RuntimeError("No grasps available.")

        plane = get_routing_plane(state.config)
        poses = self.service.compute_grasp_poses(
            state.grasps,
            plane=plane,
            grasp_height_above_plane_m=float(state.config.grasp_height_above_plane_m),
            extra_world_rx_deg=float(getattr(state.config, "grasp_extra_world_rx_deg", 0.0)),
        )

        dual = bool(getattr(state.config, "dual_arm_grasp", True))
        if dual:
            if len(poses) != 2:
                raise RuntimeError("Dual-arm grasp requires exactly 2 grasp poses.")
            if poses[0]["position"][1] > poses[1]["position"][1]:
                poses[0]["arm"] = "left"
                poses[1]["arm"] = "right"
            else:
                poses[0]["arm"] = "right"
                poses[1]["arm"] = "left"

            if not routing_plane_is_world_yz(plane):
                theta = np.deg2rad(180.0)
                rz_bias = np.array(
                    [
                        [np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                for pose in poses:
                    if pose["arm"] == "left":
                        pose["rotation"] = pose["rotation"] @ rz_bias
        else:
            if len(poses) != 1:
                raise RuntimeError("Single-arm grasp requires exactly 1 grasp pose.")
            gpos = np.asarray(poses[0]["position"], dtype=float).reshape(3)
            left_nom = np.asarray(
                getattr(state.config, "single_arm_nominal_tcp_left_m", (0.35, 0.22, 0.14)),
                dtype=float,
            ).reshape(3)
            right_nom = np.asarray(
                getattr(state.config, "single_arm_nominal_tcp_right_m", (0.35, -0.22, 0.14)),
                dtype=float,
            ).reshape(3)
            poses[0]["arm"] = "left" if float(np.linalg.norm(gpos - left_nom)) <= float(np.linalg.norm(gpos - right_nom)) else "right"

        left_grasp_offset = self._apply_initial_left_grasp_offset(state, poses)
        state.grasp_poses = poses
        print("Assigned arms:", [p["arm"] for p in poses])
        if np.any(np.abs(left_grasp_offset) > 1e-9):
            print("Applied initial left grasp position offset:", left_grasp_offset.tolist())

        out: Dict[str, object] = {
            "poses_available": True,
            "num_poses": len(poses),
            "arms": [p["arm"] for p in poses],
            "initial_left_grasp_position_offset_m": left_grasp_offset.tolist(),
            "first_pose_pos": poses[0]["position"].tolist(),
        }
        if len(poses) > 1:
            out["second_pose_pos"] = poses[1]["position"].tolist()
        return out
