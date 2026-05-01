from typing import Any, Dict

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped

from cable_motion.arm_motion_utils import (
    enforce_pose_min_height,
    pose_to_msg,
    wait_for_cartesian_motion_result,
)
from cable_orchestrator.base_step import BaseStep
from cable_planning.handover_pose_service import (
    align_tool_axis_to_direction_about_tool_z,
    fine_orient_on_grasp_rotation,
    grasp_pose_for_arm,
    resolve_handover_arm,
    routing_clip_world_m,
)


class HandoverMoveExchangeStep(BaseStep):
    name = "handover_move_exchange"
    description = "Move to handover exchange position with cable-aligned gripper yaw."

    def __init__(self) -> None:
        super().__init__()
        if not rospy.core.is_initialized():
            rospy.init_node("cable_studio_handover_move_exchange", anonymous=True)
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
            raise RuntimeError("routing required to align handover orientation to the previous clip.")

        arm = resolve_handover_arm(state, getattr(state.config, "handover_arm", None))
        rot = getattr(state, "handover_tcp_rotation_world", None)
        if rot is None:
            gpose = grasp_pose_for_arm(state.grasp_poses, arm)
            r_grasp = np.asarray(gpose["rotation"], dtype=float).reshape(3, 3)
            rx = float(getattr(state.config, "handover_fine_tool_rx_deg", 0.0))
            ry = float(getattr(state.config, "handover_fine_tool_ry_deg", 0.0))
            rz = float(getattr(state.config, "handover_fine_tool_rz_deg", 0.0))
            rot = fine_orient_on_grasp_rotation(r_grasp, rx, ry, rz)
        rot = np.asarray(rot, dtype=float).reshape(3, 3)

        pos_goal = np.asarray(getattr(state.config, "handover_goal_world_m", (0.4, 0.0, 0.4)), dtype=float).reshape(3)
        if bool(getattr(state.config, "handover_enforce_min_plane_height", False)):
            floor = float(state.config.routing_height_above_plane_m)
            pos_goal = enforce_pose_min_height({"position": pos_goal, "rotation": np.eye(3)}, state, floor)["position"]

        ridx = int(getattr(state.config, "handover_clip_routing_index", 0))
        if ridx < 0 or ridx >= len(state.routing):
            raise RuntimeError(f"handover_clip_routing_index {ridx} invalid for routing length {len(state.routing)}.")
        reference_clip_board_idx = int(state.routing[ridx])
        reference_clip_world = routing_clip_world_m(state, reference_clip_board_idx, arm=arm)
        cable_direction_world = pos_goal - reference_clip_world

        align_axis = str(getattr(state.config, "handover_exchange_align_axis", "tool_x"))
        yaw_offset_deg = float(getattr(state.config, "handover_exchange_tool_z_yaw_offset_deg", 0.0))
        allow_axis_flip = bool(getattr(state.config, "handover_exchange_allow_axis_flip", True))
        rot = align_tool_axis_to_direction_about_tool_z(
            rot,
            cable_direction_world,
            tool_axis=align_axis,
            yaw_offset_deg=yaw_offset_deg,
            allow_axis_flip=allow_axis_flip,
        )
        state.handover_tcp_rotation_world = rot.copy()

        msg, quat = pose_to_msg(pos_goal, rot, config=state.config)
        self._publish(arm, msg)
        motion_result = wait_for_cartesian_motion_result([arm])

        state.handover_carrier_tcp_world = np.asarray(pos_goal, dtype=float).reshape(3).copy()
        state.handover_repark_done = True
        state.handover_exchange_done = True
        return {
            "arm": arm,
            "motion_result": motion_result,
            "reference_clip_board_idx": reference_clip_board_idx,
            "reference_clip_world_m": reference_clip_world.tolist(),
            "cable_direction_world": cable_direction_world.tolist(),
            "handover_exchange_align_axis": align_axis,
            "handover_exchange_tool_z_yaw_offset_deg": yaw_offset_deg,
            "exchange_position_m": pos_goal.tolist(),
            "quaternion_xyzw": [float(quat[i]) for i in range(4)],
        }
