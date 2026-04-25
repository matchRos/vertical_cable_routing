from typing import Any, Dict

import rospy
from geometry_msgs.msg import PoseStamped

from cable_motion.arm_motion_utils import (
    enforce_pose_min_height,
    is_dual_arm_grasp,
    pose_to_msg,
    wait_for_cartesian_motion_result,
)
from cable_orchestrator.base_step import BaseStep
from cable_planning.first_route_targets import (
    build_c_clip_centering_poses,
    build_first_route_execution_poses,
)


class ExecuteFirstRouteStep(BaseStep):
    name = "execute_first_route"
    description = "Execute planned first-route targets with slowly_approach_pose."

    def __init__(self) -> None:
        super().__init__()
        if not rospy.core.is_initialized():
            rospy.init_node("cable_studio_execute_first_route", anonymous=True)
        self.pub_left = rospy.Publisher("/yumi/robl/slowly_approach_pose", PoseStamped, queue_size=1)
        self.pub_right = rospy.Publisher("/yumi/robr/slowly_approach_pose", PoseStamped, queue_size=1)

    def _publish_route_pair(self, state, left_msg: PoseStamped, right_msg: PoseStamped) -> str:
        if is_dual_arm_grasp(state.config):
            self.pub_left.publish(left_msg)
            self.pub_right.publish(right_msg)
            return "both"
        primary = getattr(state, "current_primary_arm", None)
        if primary is None and hasattr(state, "grasp_poses") and state.grasp_poses:
            primary = state.grasp_poses[0].get("arm", "left")
        if primary == "right":
            self.pub_right.publish(right_msg)
            return "right"
        self.pub_left.publish(left_msg)
        return "left"

    def _arms_from_publish_mode(self, publish_mode: str) -> list[str]:
        if publish_mode == "both":
            return ["left", "right"]
        if publish_mode == "right":
            return ["right"]
        return ["left"]

    def run(self, state) -> Dict[str, Any]:
        left_pose, right_pose, mode = build_first_route_execution_poses(state)
        routing_floor = float(state.config.routing_height_above_plane_m)
        left_pose = enforce_pose_min_height(left_pose, state, routing_floor)
        right_pose = enforce_pose_min_height(right_pose, state, routing_floor)

        left_msg, _ = pose_to_msg(left_pose["position"], left_pose["rotation"], config=state.config)
        right_msg, _ = pose_to_msg(right_pose["position"], right_pose["rotation"], config=state.config)
        now = rospy.Time.now()
        left_msg.header.stamp = now
        right_msg.header.stamp = now

        publish_mode = self._publish_route_pair(state, left_msg, right_msg)
        motion_result = wait_for_cartesian_motion_result(self._arms_from_publish_mode(publish_mode))

        second_phase_executed = False
        second_phase_mode = None
        if mode == "c_clip_entry":
            left_center, right_center, second_phase_mode = build_c_clip_centering_poses(state)
            left_center = enforce_pose_min_height(left_center, state, routing_floor)
            right_center = enforce_pose_min_height(right_center, state, routing_floor)
            left_center_msg, _ = pose_to_msg(left_center["position"], left_center["rotation"], config=state.config)
            right_center_msg, _ = pose_to_msg(right_center["position"], right_center["rotation"], config=state.config)
            now2 = rospy.Time.now()
            left_center_msg.header.stamp = now2
            right_center_msg.header.stamp = now2
            second_publish_mode = self._publish_route_pair(state, left_center_msg, right_center_msg)
            second_phase_result = wait_for_cartesian_motion_result(self._arms_from_publish_mode(second_publish_mode))
            second_phase_executed = True
        else:
            second_phase_result = None

        state.first_route_executed = True
        return {
            "executed": True,
            "mode": mode,
            "publish_mode": publish_mode,
            "motion_result": motion_result,
            "second_phase_executed": second_phase_executed,
            "second_phase_mode": second_phase_mode,
            "second_phase_result": second_phase_result,
            "left_position": [left_msg.pose.position.x, left_msg.pose.position.y, left_msg.pose.position.z],
            "right_position": [right_msg.pose.position.x, right_msg.pose.position.y, right_msg.pose.position.z],
        }
