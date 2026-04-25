from typing import Any, Dict

import rospy
from geometry_msgs.msg import PoseStamped

from cable_core.planes import get_routing_plane, point_at_plane_height
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

    def _execute_secondary_arm(self, state) -> bool:
        return bool(
            getattr(state.config, "first_route_execute_secondary_arm", True)
            and getattr(state, "first_route_secondary_shown", False)
            and getattr(state, "first_route_secondary_target_px", None) is not None
        )

    def _publish_route_pair(self, state, left_msg: PoseStamped, right_msg: PoseStamped) -> str:
        if is_dual_arm_grasp(state.config) or self._execute_secondary_arm(state):
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

    def _insert_height_for_state(self, state) -> float | None:
        if not bool(getattr(state.config, "first_route_insert_after_route", True)):
            return None
        clip_cfg = getattr(state, "first_route_clip_type_config", None) or {}
        if "insert_height_above_plane_m" not in clip_cfg:
            return None
        return float(
            clip_cfg.get(
                "insert_height_above_plane_m",
                getattr(state.config, "first_route_default_insert_height_above_plane_m", 0.01),
            )
        )

    def _build_insert_pose(self, pose: dict, state, height_above_plane_m: float) -> dict:
        curr_idx = getattr(state, "first_route_curr_clip_id", None)
        plane = get_routing_plane(state.config, clip_id=curr_idx)
        out = dict(pose)
        out["position"] = point_at_plane_height(
            pose["position"],
            plane,
            float(height_above_plane_m),
        )
        return out

    def _publish_and_wait(self, state, left_pose: dict, right_pose: dict) -> tuple[str, dict, PoseStamped, PoseStamped]:
        left_msg, _ = pose_to_msg(left_pose["position"], left_pose["rotation"], config=state.config)
        right_msg, _ = pose_to_msg(right_pose["position"], right_pose["rotation"], config=state.config)
        now = rospy.Time.now()
        left_msg.header.stamp = now
        right_msg.header.stamp = now
        publish_mode = self._publish_route_pair(state, left_msg, right_msg)
        result = wait_for_cartesian_motion_result(self._arms_from_publish_mode(publish_mode))
        return publish_mode, result, left_msg, right_msg

    def run(self, state) -> Dict[str, Any]:
        left_pose, right_pose, mode = build_first_route_execution_poses(state)
        routing_floor = float(
            getattr(
                state,
                "first_route_route_height_m",
                float(state.config.routing_height_above_plane_m),
            )
        )
        left_pose = enforce_pose_min_height(left_pose, state, routing_floor)
        right_pose = enforce_pose_min_height(right_pose, state, routing_floor)

        publish_mode, motion_result, left_msg, right_msg = self._publish_and_wait(
            state,
            left_pose,
            right_pose,
        )
        final_left_pose = left_pose
        final_right_pose = right_pose
        final_publish_mode = publish_mode

        second_phase_executed = False
        second_phase_mode = None
        if mode == "c_clip_entry":
            left_center, right_center, second_phase_mode = build_c_clip_centering_poses(state)
            left_center = enforce_pose_min_height(left_center, state, routing_floor)
            right_center = enforce_pose_min_height(right_center, state, routing_floor)
            second_publish_mode, second_phase_result, _, _ = self._publish_and_wait(
                state,
                left_center,
                right_center,
            )
            second_phase_executed = True
            final_left_pose = left_center
            final_right_pose = right_center
            final_publish_mode = second_publish_mode
        else:
            second_phase_result = None

        insert_height = self._insert_height_for_state(state)
        insert_executed = False
        insert_result = None
        insert_publish_mode = None
        insert_left_msg = None
        insert_right_msg = None
        if insert_height is not None:
            left_insert = self._build_insert_pose(final_left_pose, state, insert_height)
            right_insert = self._build_insert_pose(final_right_pose, state, insert_height)
            insert_publish_mode, insert_result, insert_left_msg, insert_right_msg = self._publish_and_wait(
                state,
                left_insert,
                right_insert,
            )
            final_left_pose = left_insert
            final_right_pose = right_insert
            final_publish_mode = insert_publish_mode
            insert_executed = True

        state.current_arm_poses_world = {
            "left": {
                "position": final_left_pose["position"],
                "rotation": final_left_pose["rotation"],
            },
            "right": {
                "position": final_right_pose["position"],
                "rotation": final_right_pose["rotation"],
            },
        }
        route = list(getattr(state, "routing", []) or [])
        curr_clip_id = getattr(state, "first_route_curr_clip_id", None)
        if curr_clip_id is not None:
            try:
                state.next_route_routing_index = route.index(int(curr_clip_id)) + 1
            except ValueError:
                state.next_route_routing_index = 2

        state.first_route_executed = True
        return {
            "executed": True,
            "mode": mode,
            "publish_mode": publish_mode,
            "motion_result": motion_result,
            "second_phase_executed": second_phase_executed,
            "second_phase_mode": second_phase_mode,
            "second_phase_result": second_phase_result,
            "insert_executed": insert_executed,
            "insert_height_above_plane_m": insert_height,
            "insert_publish_mode": insert_publish_mode,
            "insert_result": insert_result,
            "final_publish_mode": final_publish_mode,
            "left_position": [left_msg.pose.position.x, left_msg.pose.position.y, left_msg.pose.position.z],
            "right_position": [right_msg.pose.position.x, right_msg.pose.position.y, right_msg.pose.position.z],
            "insert_left_position": (
                None
                if insert_left_msg is None
                else [
                    insert_left_msg.pose.position.x,
                    insert_left_msg.pose.position.y,
                    insert_left_msg.pose.position.z,
                ]
            ),
            "insert_right_position": (
                None
                if insert_right_msg is None
                else [
                    insert_right_msg.pose.position.x,
                    insert_right_msg.pose.position.y,
                    insert_right_msg.pose.position.z,
                ]
            ),
        }
