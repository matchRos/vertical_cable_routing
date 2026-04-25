from typing import Dict

import numpy as np
import rospy
from geometry_msgs.msg import PoseArray

from cable_motion.arm_motion_utils import pose_to_msg, wait_for_moveit_motion_result
from cable_orchestrator.base_step import BaseStep


class ExecuteNextPegRouteStep(BaseStep):
    name = "execute_next_peg_route"
    description = "Execute the planned next-peg Cartesian waypoint route."

    def __init__(self):
        super().__init__()
        if not rospy.core.is_initialized():
            rospy.init_node("cable_studio_execute_next_peg_route", anonymous=True)
        self.pub_left = rospy.Publisher("/yumi/robl/moveit_waypoints", PoseArray, queue_size=1)
        self.pub_right = rospy.Publisher("/yumi/robr/moveit_waypoints", PoseArray, queue_size=1)

    def _build_pose_array(self, state, plan: dict) -> PoseArray:
        arr = PoseArray()
        arr.header.stamp = rospy.Time.now()
        arr.header.frame_id = getattr(state.config, "cartesian_targets_world_frame_id", "world")
        for pose in plan["poses"]:
            msg, _ = pose_to_msg(pose["position"], pose["rotation"], config=state.config)
            arr.poses.append(msg.pose)
        return arr

    def run(self, state) -> Dict[str, object]:
        plan = getattr(state, "peg_route_plan", None)
        if not plan:
            raise RuntimeError("No peg_route_plan. Run plan_next_peg_route first.")

        arm = str(plan["arm"])
        if bool(plan.get("needs_handover", False)):
            return {
                "executed": False,
                "skipped": "needs_handover",
                "arm": arm,
                "curr_clip_idx": plan.get("curr_clip_idx"),
            }
        if bool(plan.get("other_arm_should_move_aside", False)):
            return {
                "executed": False,
                "skipped": "other_arm_should_move_aside",
                "arm": arm,
                "other_arm": plan.get("other_arm"),
                "min_other_arm_distance_m": plan.get("min_other_arm_distance_m"),
            }

        waypoints = list(plan.get("poses", []))
        if not waypoints:
            raise RuntimeError("peg_route_plan has no poses.")

        msg = self._build_pose_array(state, plan)
        if arm == "left":
            self.pub_left.publish(msg)
        else:
            self.pub_right.publish(msg)

        motion_result = wait_for_moveit_motion_result([arm])
        last_pose = waypoints[-1]
        if getattr(state, "current_arm_poses_world", None) is None:
            state.current_arm_poses_world = {}
        state.current_arm_poses_world[arm] = {
            "position": np.asarray(last_pose["position"], dtype=float).reshape(3),
            "rotation": np.asarray(last_pose["rotation"], dtype=float).reshape(3, 3),
        }
        state.next_route_routing_index = int(plan.get("route_after_idx", int(plan["route_idx"]) + 1))
        state.peg_route_executed = True
        return {
            "executed": True,
            "arm": arm,
            "waypoint_count": len(waypoints),
            "curr_clip_idx": plan["curr_clip_idx"],
            "next_route_routing_index": state.next_route_routing_index,
            "motion_result": motion_result,
            "final_position": np.asarray(last_pose["position"], dtype=float).reshape(3).tolist(),
        }
