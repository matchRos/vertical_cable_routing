from typing import Any, Dict

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped

from cable_core.dual_arm_presentation_geometry import (
    rotation_second_arm_side_grasp_world,
    rotation_world_ry_deg,
)
from cable_motion.arm_motion_utils import pose_to_msg, wait_for_cartesian_motion_result
from cable_orchestrator.base_step import BaseStep
from cable_planning.handover_pose_service import resolve_handover_arm


class SecondArmSideApproachStep(BaseStep):
    name = "second_arm_side_approach"
    description = "Prepose (moveit) from the side, then slow approach; side-grasp orientation (±world Y)."

    def __init__(self) -> None:
        super().__init__()
        if not rospy.core.is_initialized():
            rospy.init_node("cable_studio_second_arm_side_approach", anonymous=True)
        self.pub_left_slow = rospy.Publisher("/yumi/robl/slowly_approach_pose", PoseStamped, queue_size=1)
        self.pub_right_slow = rospy.Publisher("/yumi/robr/slowly_approach_pose", PoseStamped, queue_size=1)
        self.pub_left_moveit = rospy.Publisher("/yumi/robl/cartesian_pose_command", PoseStamped, queue_size=1)
        self.pub_right_moveit = rospy.Publisher("/yumi/robr/cartesian_pose_command", PoseStamped, queue_size=1)

    def _publish_moveit(self, arm: str, msg: PoseStamped) -> None:
        msg.header.stamp = rospy.Time.now()
        if arm == "left":
            self.pub_left_moveit.publish(msg)
        elif arm == "right":
            self.pub_right_moveit.publish(msg)
        else:
            raise RuntimeError(f"Invalid arm: {arm}")

    def _publish_slow(self, arm: str, msg: PoseStamped) -> None:
        msg.header.stamp = rospy.Time.now()
        if arm == "left":
            self.pub_left_slow.publish(msg)
        elif arm == "right":
            self.pub_right_slow.publish(msg)
        else:
            raise RuntimeError(f"Invalid arm: {arm}")

    def run(self, state) -> Dict[str, Any]:
        if state.env is None:
            raise RuntimeError("Environment not initialized.")

        carrier = resolve_handover_arm(state, getattr(state.config, "handover_arm", None))
        second = "right" if carrier == "left" else "left"

        pos_c = getattr(state, "handover_carrier_tcp_world", None)
        if pos_c is None:
            pos_c = np.asarray(getattr(state.config, "handover_goal_world_m", (0.4, 0.0, 0.4)), dtype=float).reshape(3)
        else:
            pos_c = np.asarray(pos_c, dtype=float).reshape(3)

        dz = float(getattr(state.config, "dual_side_second_arm_delta_z_m", -0.1))
        pos_final = pos_c + np.array([0.0, 0.0, dz], dtype=float)
        extra_y = float(getattr(state.config, "dual_side_second_arm_slow_approach_extra_y_m", 0.01))
        if second == "right":
            pos_final += np.array([0.0, abs(extra_y), 0.0], dtype=float)
        else:
            pos_final += np.array([0.0, -abs(extra_y), 0.0], dtype=float)

        lateral = float(getattr(state.config, "dual_side_second_arm_prepose_offset_y_m", 0.08))
        dy_pre = -abs(lateral) if second == "right" else abs(lateral)
        pos_pre = pos_final + np.array([0.0, dy_pre, 0.0], dtype=float)

        rot = rotation_second_arm_side_grasp_world(second_arm_is_right=(second == "right"))
        ry_extra = float(getattr(state.config, "second_arm_extra_world_ry_deg", 90.0))
        if abs(ry_extra) > 1e-9:
            rot = rotation_world_ry_deg(ry_extra) @ rot

        msg_pre, _ = pose_to_msg(pos_pre, rot, config=state.config)
        self._publish_moveit(second, msg_pre)
        prepose_result = wait_for_cartesian_motion_result([second])

        msg_fin, quat = pose_to_msg(pos_final, rot, config=state.config)
        self._publish_slow(second, msg_fin)
        slow_result = wait_for_cartesian_motion_result([second])

        state.descend_second_arm = second
        state.second_arm_side_approach_done = True
        return {
            "carrier_arm": carrier,
            "second_arm": second,
            "prepose_result": prepose_result,
            "slow_result": slow_result,
            "second_arm_tool_z_world": ("+Y" if second == "right" else "-Y"),
            "prepose_position_m": pos_pre.tolist(),
            "final_position_m": pos_final.tolist(),
            "prepose_delta_y_m": float(dy_pre),
            "delta_z_m": dz,
            "slow_approach_extra_y_m": float(extra_y),
            "second_arm_extra_world_ry_deg": ry_extra,
            "quaternion_xyzw": [float(quat[i]) for i in range(4)],
        }
