from typing import Dict

import numpy as np
import rospy
from geometry_msgs.msg import PoseArray

from cable_core.planes import get_routing_plane
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

    def _log_waypoint_preview(self, state, plan: dict, msg: PoseArray, waypoints) -> None:
        max_count = min(len(waypoints), 8)
        plane = get_routing_plane(state.config, clip_id=int(plan["curr_clip_idx"]))
        plane_origin = np.asarray(plane.origin, dtype=float).reshape(3)
        plane_normal = np.asarray(plane.normal, dtype=float).reshape(3)
        route_height = float(
            getattr(
                state,
                "first_route_route_height_m",
                float(state.config.routing_height_above_plane_m),
            )
        )
        rospy.logwarn(
            "[execute_next_peg_route] publishing %d waypoint(s) for %s on frame '%s'; "
            "routing_plane origin=%s normal=%s target_height=%.4f",
            len(waypoints),
            plan["arm"],
            msg.header.frame_id,
            np.round(plane_origin, 4).tolist(),
            np.round(plane_normal, 4).tolist(),
            route_height,
        )
        for idx in range(max_count):
            planned = waypoints[idx]
            planned_pos = np.asarray(planned["position"], dtype=float).reshape(3)
            planned_rot = np.asarray(planned["rotation"], dtype=float).reshape(3, 3)
            published = msg.poses[idx]
            quat = np.array(
                [
                    published.orientation.x,
                    published.orientation.y,
                    published.orientation.z,
                    published.orientation.w,
                ],
                dtype=float,
            )
            published_pos = np.array(
                [
                    published.position.x,
                    published.position.y,
                    published.position.z,
                ],
                dtype=float,
            )
            plane_distance = float(np.dot(planned_pos - plane_origin, plane_normal))
            rospy.logwarn(
                "[execute_next_peg_route] wp[%02d] plan_pos=%s published_pos=%s "
                "plane_dist=%.4f quat_xyzw=%s rot_rows=%s",
                idx,
                np.round(planned_pos, 4).tolist(),
                np.round(published_pos, 4).tolist(),
                plane_distance,
                np.round(quat, 4).tolist(),
                np.round(planned_rot, 4).tolist(),
            )
        if len(waypoints) > max_count:
            rospy.logwarn(
                "[execute_next_peg_route] ... %d additional waypoint(s) omitted",
                len(waypoints) - max_count,
            )

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
        self._log_waypoint_preview(state, plan, msg, waypoints)
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
