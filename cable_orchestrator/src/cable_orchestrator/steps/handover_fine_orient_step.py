from typing import Any, Dict

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped

from cable_motion.arm_motion_utils import pose_to_msg, wait_for_cartesian_motion_result
from cable_orchestrator.base_step import BaseStep
from cable_planning.handover_pose_service import (
    fine_orient_on_grasp_rotation,
    grasp_pose_for_arm,
    lift_offset_along_plane_normal,
    resolve_handover_arm,
)
from cable_core.planes import get_routing_plane


class HandoverFineOrientStep(BaseStep):
    name = "handover_fine_orient"
    description = "Fine-tune gripper orientation at lift pose (small deg from grasp frame)."

    def __init__(self) -> None:
        super().__init__()
        if not rospy.core.is_initialized():
            rospy.init_node("cable_studio_handover_fine_orient", anonymous=True)
        self.pub_left = rospy.Publisher("/yumi/robl/slowly_approach_pose", PoseStamped, queue_size=1)
        self.pub_right = rospy.Publisher("/yumi/robr/slowly_approach_pose", PoseStamped, queue_size=1)

    def _publish(self, arm: str, msg: PoseStamped) -> None:
        msg.header.stamp = rospy.Time.now()
        if arm == "left":
            self.pub_left.publish(msg)
        elif arm == "right":
            self.pub_right.publish(msg)
        else:
            raise RuntimeError(f"Invalid arm: {arm}")

    def run(self, state) -> Dict[str, Any]:
        if state.env is None:
            raise RuntimeError("Environment not initialized.")
        if not hasattr(state, "grasp_poses") or not state.grasp_poses:
            raise RuntimeError("No grasp_poses; run grasp_pose first.")
        if state.routing is None or len(state.routing) < 1:
            raise RuntimeError("routing required for routing plane selection.")

        arm = resolve_handover_arm(state, getattr(state.config, "handover_arm", None))
        ridx = int(getattr(state.config, "handover_clip_routing_index", 0))
        if ridx < 0 or ridx >= len(state.routing):
            raise RuntimeError(f"handover_clip_routing_index {ridx} invalid for routing length {len(state.routing)}.")
        clip_board_idx = int(state.routing[ridx])
        plane = get_routing_plane(state.config, clip_id=clip_board_idx)

        gpose = grasp_pose_for_arm(state.grasp_poses, arm)
        r_grasp = np.asarray(gpose["rotation"], dtype=float).reshape(3, 3)
        lift_m = float(getattr(state.config, "handover_lift_along_normal_m", 0.02))
        pos_lift = np.asarray(gpose["position"], dtype=float).reshape(3) + lift_offset_along_plane_normal(plane, lift_m)

        rx = float(getattr(state.config, "handover_fine_tool_rx_deg", 0.0))
        ry = float(getattr(state.config, "handover_fine_tool_ry_deg", 0.0))
        rz = float(getattr(state.config, "handover_fine_tool_rz_deg", 0.0))
        rot = fine_orient_on_grasp_rotation(r_grasp, rx, ry, rz)
        state.handover_tcp_rotation_world = rot.copy()
        state.handover_fine_orient_done = True

        msg, quat = pose_to_msg(pos_lift, rot, config=state.config)
        self._publish(arm, msg)
        motion_result = wait_for_cartesian_motion_result([arm])
        return {
            "arm": arm,
            "motion_result": motion_result,
            "reference_clip_board_idx": clip_board_idx,
            "position_m": pos_lift.tolist(),
            "fine_tool_rx_deg": rx,
            "fine_tool_ry_deg": ry,
            "fine_tool_rz_deg": rz,
            "quaternion_xyzw": [float(quat[i]) for i in range(4)],
        }
