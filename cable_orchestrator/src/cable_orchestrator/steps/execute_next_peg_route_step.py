from typing import Dict, List

import numpy as np
import rospy
import tf
from geometry_msgs.msg import PoseArray, PoseStamped
from scipy.spatial.transform import Rotation as R

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
        self.pub_left_pose = rospy.Publisher("/yumi/robl/moveit_target_pose", PoseStamped, queue_size=1)
        self.pub_right_pose = rospy.Publisher("/yumi/robr/moveit_target_pose", PoseStamped, queue_size=1)
        self.tf_listener = tf.TransformListener()

    def _target_plane_distance_delta(self, state) -> float:
        if hasattr(state.config, "execute_next_peg_route_target_plane_distance_delta_m"):
            return float(getattr(state.config, "execute_next_peg_route_target_plane_distance_delta_m"))
        return float(getattr(state.config, "execute_next_peg_route_plane_normal_offset_m", 0.0))

    def _min_plane_distance_delta(self, state) -> float:
        if hasattr(state.config, "execute_next_peg_route_min_plane_distance_delta_m"):
            return float(getattr(state.config, "execute_next_peg_route_min_plane_distance_delta_m"))
        return self._target_plane_distance_delta(state)

    def _execution_offsets(self, state, plan: dict) -> tuple[np.ndarray, np.ndarray, float]:
        plane = get_routing_plane(state.config, clip_id=int(plan["curr_clip_idx"]))
        plane_normal = np.asarray(plane.normal, dtype=float).reshape(3)
        world_offset = np.asarray(
            getattr(state.config, "execute_next_peg_route_world_offset_m", (0.0, 0.0, 0.0)),
            dtype=float,
        ).reshape(3)
        target_delta = self._target_plane_distance_delta(state)
        total_offset = world_offset + target_delta * plane_normal
        return total_offset, world_offset, target_delta

    def _execution_waypoints(self, state, plan: dict, waypoints) -> List[dict]:
        total_offset, _, _ = self._execution_offsets(state, plan)
        out = []
        for waypoint in waypoints:
            adjusted = dict(waypoint)
            adjusted["position"] = (
                np.asarray(waypoint["position"], dtype=float).reshape(3) + total_offset
            )
            adjusted["rotation"] = np.asarray(waypoint["rotation"], dtype=float).reshape(3, 3)
            out.append(adjusted)
        return out

    def _build_pose_array(self, state, waypoints) -> PoseArray:
        arr = PoseArray()
        arr.header.stamp = rospy.Time.now()
        arr.header.frame_id = getattr(state.config, "cartesian_targets_world_frame_id", "world")
        for pose in waypoints:
            msg, _ = pose_to_msg(pose["position"], pose["rotation"], config=state.config)
            arr.poses.append(msg.pose)
        return arr

    def _publish_pose_array(self, arm: str, msg: PoseArray) -> None:
        if arm == "left":
            self.pub_left.publish(msg)
        else:
            self.pub_right.publish(msg)

    def _publish_pose(self, arm: str, msg: PoseStamped) -> None:
        msg.header.stamp = rospy.Time.now()
        if arm == "left":
            self.pub_left_pose.publish(msg)
        else:
            self.pub_right_pose.publish(msg)

    def _pose_stamped_from_waypoint(self, state, waypoint: dict) -> PoseStamped:
        msg, _ = pose_to_msg(waypoint["position"], waypoint["rotation"], config=state.config)
        msg.header.stamp = rospy.Time.now()
        return msg

    def _plane_coord(self, point, origin, axis_x, axis_y) -> List[float]:
        pos = np.asarray(point, dtype=float).reshape(3)
        delta = pos - origin
        return [
            float(np.dot(delta, axis_x)),
            float(np.dot(delta, axis_y)),
        ]

    def _tip_link_for_arm(self, state, arm: str) -> str:
        if arm == "left":
            return str(getattr(state.config, "execute_next_peg_route_left_tip_link", "yumi_tcp_l"))
        return str(getattr(state.config, "execute_next_peg_route_right_tip_link", "yumi_tcp_r"))

    def _pose_from_msg(self, pose_msg) -> tuple[np.ndarray, np.ndarray]:
        pos = np.array(
            [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z],
            dtype=float,
        )
        quat = np.array(
            [
                pose_msg.orientation.x,
                pose_msg.orientation.y,
                pose_msg.orientation.z,
                pose_msg.orientation.w,
            ],
            dtype=float,
        )
        return pos, quat

    def _current_pose(self, state, arm: str, frame_id: str) -> tuple[np.ndarray, np.ndarray, str]:
        tip_link = self._tip_link_for_arm(state, arm)
        try:
            self.tf_listener.waitForTransform(
                frame_id,
                tip_link,
                rospy.Time(0),
                rospy.Duration(0.5),
            )
            trans, quat = self.tf_listener.lookupTransform(frame_id, tip_link, rospy.Time(0))
            return np.asarray(trans, dtype=float), np.asarray(quat, dtype=float), frame_id
        except Exception as exc:
            stored = (getattr(state, "current_arm_poses_world", None) or {}).get(arm)
            if stored is None:
                rospy.logwarn(
                    "[execute_next_peg_route] current pose unavailable via TF (%s -> %s): %s",
                    frame_id,
                    tip_link,
                    exc,
                )
                return np.full(3, np.nan), np.full(4, np.nan), frame_id
            quat = R.from_matrix(np.asarray(stored["rotation"], dtype=float).reshape(3, 3)).as_quat()
            rospy.logwarn(
                "[execute_next_peg_route] TF current pose unavailable (%s -> %s); "
                "using stored pipeline pose instead",
                frame_id,
                tip_link,
            )
            return np.asarray(stored["position"], dtype=float).reshape(3), quat, "pipeline_state"

    def _pose_text(self, pos: np.ndarray, quat: np.ndarray) -> str:
        return "pos=%s quat_xyzw=%s" % (
            np.round(np.asarray(pos, dtype=float), 4).tolist(),
            np.round(np.asarray(quat, dtype=float), 4).tolist(),
        )

    def _log_route_header(self, state, plan: dict, waypoints, msg: PoseArray) -> None:
        plane = get_routing_plane(state.config, clip_id=int(plan["curr_clip_idx"]))
        plane_origin = np.asarray(plane.origin, dtype=float).reshape(3)
        plane_normal = np.asarray(plane.normal, dtype=float).reshape(3)
        positions = [np.asarray(wp["position"], dtype=float).reshape(3) for wp in waypoints]
        distances = [float(np.dot(pos - plane_origin, plane_normal)) for pos in positions]
        total_offset, world_offset, target_delta = self._execution_offsets(state, plan)
        min_delta = self._min_plane_distance_delta(state)
        rospy.logwarn(
            "[execute_next_peg_route] route %s -> %s -> %s | arm=%s | frame=%s | "
            "waypoints=%d | plane_dist[min/max]=%.4f/%.4f | "
            "execution_offset=%s world_offset=%s target_plane_delta=%.4f min_plane_delta=%.4f",
            plan.get("prev_clip_label", plan.get("prev_clip_idx")),
            ",".join(str(v) for v in plan.get("peg_clip_labels", plan.get("peg_clip_indices", []))),
            plan.get("terminal_clip_label", plan.get("next_clip_idx")),
            plan["arm"],
            msg.header.frame_id,
            len(positions),
            min(distances),
            max(distances),
            np.round(total_offset, 4).tolist(),
            np.round(world_offset, 4).tolist(),
            target_delta,
            min_delta,
        )

    def _log_waypoint_preview(self, state, plan: dict, msg: PoseArray, waypoints) -> None:
        self._log_route_header(state, plan, waypoints, msg)
        if bool(getattr(state.config, "execute_next_peg_route_verbose_geometry", False)):
            self._log_verbose_geometry(state, plan, msg, waypoints)

    def _log_verbose_geometry(self, state, plan: dict, msg: PoseArray, waypoints) -> None:
        max_count = min(
            len(waypoints),
            int(getattr(state.config, "execute_next_peg_route_verbose_waypoint_count", 8)),
        )
        plane = get_routing_plane(state.config, clip_id=int(plan["curr_clip_idx"]))
        plane_origin = np.asarray(plane.origin, dtype=float).reshape(3)
        plane_normal = np.asarray(plane.normal, dtype=float).reshape(3)
        axis_x = np.asarray(plane.v_axis, dtype=float).reshape(3)
        axis_y = np.asarray(plane.u_axis, dtype=float).reshape(3)
        start_pos = np.asarray(plan.get("start_position", waypoints[0]["position"]), dtype=float).reshape(3)
        start_2d = self._plane_coord(start_pos, plane_origin, axis_x, axis_y)
        route_height = float(
            getattr(
                state,
                "first_route_route_height_m",
                float(state.config.routing_height_above_plane_m),
            )
        )
        rospy.logwarn(
            "[execute_next_peg_route] geometry: publishing %d waypoint(s) for %s on frame '%s'; "
            "routing_plane origin=%s normal=%s axis_x(v)=%s axis_y(u)=%s target_height=%.4f "
            "start_pos=%s start_2d=%s",
            len(waypoints),
            plan["arm"],
            msg.header.frame_id,
            np.round(plane_origin, 4).tolist(),
            np.round(plane_normal, 4).tolist(),
            np.round(axis_x, 4).tolist(),
            np.round(axis_y, 4).tolist(),
            route_height,
            np.round(start_pos, 4).tolist(),
            np.round(start_2d, 4).tolist(),
        )
        for side_idx, side in enumerate(plan.get("side_debug", []) or []):
            rospy.logwarn(
                "[execute_next_peg_route] geometry side[%02d] prev=%s curr=%s next=%s side=%s "
                "reason=%s side_vec_2d=%s",
                side_idx,
                side.get("prev_clip"),
                side.get("curr_clip"),
                side.get("next_clip"),
                side.get("side"),
                side.get("reason"),
                np.round(np.asarray(side.get("side_vector_2d", [0.0, 0.0]), dtype=float), 4).tolist(),
            )
        prev_pos = start_pos
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
            segment = planned_pos - prev_pos
            segment_len = float(np.linalg.norm(segment))
            segment_2d = [
                float(np.dot(segment, axis_x)),
                float(np.dot(segment, axis_y)),
            ]
            segment_normal = float(np.dot(segment, plane_normal))
            planned_2d = self._plane_coord(planned_pos, plane_origin, axis_x, axis_y)
            rospy.logwarn(
                "[execute_next_peg_route] geometry wp[%02d] plan_pos=%s published_pos=%s "
                "plane_2d=%s plane_dist=%.4f seg_len=%.4f seg_2d=%s seg_normal=%.4f "
                "quat_xyzw=%s rot_rows=%s",
                idx,
                np.round(planned_pos, 4).tolist(),
                np.round(published_pos, 4).tolist(),
                np.round(planned_2d, 4).tolist(),
                plane_distance,
                segment_len,
                np.round(segment_2d, 4).tolist(),
                segment_normal,
                np.round(quat, 4).tolist(),
                np.round(planned_rot, 4).tolist(),
            )
            prev_pos = planned_pos
        if len(waypoints) > max_count:
            rospy.logwarn(
                "[execute_next_peg_route] ... %d additional waypoint(s) omitted",
                len(waypoints) - max_count,
            )

    def _validate_plane_clearance(self, state, plan: dict, waypoints) -> None:
        configured_min_distance = getattr(
            state.config,
            "execute_next_peg_route_min_plane_distance_m",
            None,
        )
        route_height = float(
            getattr(
                state,
                "first_route_route_height_m",
                float(state.config.routing_height_above_plane_m),
            )
        )
        min_distance_delta = self._min_plane_distance_delta(state)
        if configured_min_distance is None:
            tolerance = max(
                0.0,
                float(
                    getattr(
                        state.config,
                        "execute_next_peg_route_plane_distance_tolerance_m",
                        0.005,
                    )
                ),
            )
            min_distance = route_height + min_distance_delta - tolerance
        else:
            min_distance = float(configured_min_distance)
        if min_distance <= -1e6:
            return
        plane = get_routing_plane(state.config, clip_id=int(plan["curr_clip_idx"]))
        plane_origin = np.asarray(plane.origin, dtype=float).reshape(3)
        plane_normal = np.asarray(plane.normal, dtype=float).reshape(3)
        distances = [
            float(np.dot(np.asarray(wp["position"], dtype=float).reshape(3) - plane_origin, plane_normal))
            for wp in waypoints
        ]
        min_observed = min(distances)
        rospy.logwarn(
            "[execute_next_peg_route] plane clearance check: min_dist=%.4f m, "
            "required>=%.4f m, target_route_height=%.4f m, min_plane_delta=%.4f m",
            min_observed,
            min_distance,
            route_height,
            min_distance_delta,
        )
        if min_observed < min_distance:
            stale_plan_hint = ""
            if min_observed <= 0.001 and route_height > 0.001:
                stale_plan_hint = (
                    " The waypoints are on the plane although a positive route height is configured; "
                    "this usually means state.peg_route_plan came from an old checkpoint. "
                    "Run plan_next_peg_route again after loading the checkpoint, then execute."
                )
            raise RuntimeError(
                "Peg route waypoint is below the allowed routing plane distance: "
                f"min_dist={min_observed:.4f} m, allowed={min_distance:.4f} m. "
                "Check routing plane normal/height before executing."
                f"{stale_plan_hint}"
            )

    def _execute_stepwise(self, state, arm: str, msg: PoseArray, pause_s: float) -> dict:
        results = []
        for idx, pose in enumerate(msg.poses):
            current_pos, current_quat, current_frame = self._current_pose(state, arm, msg.header.frame_id)
            target_pos, target_quat = self._pose_from_msg(pose)
            rospy.logwarn(
                "[execute_next_peg_route] waypoint %d/%d target %s frame=%s",
                idx + 1,
                len(msg.poses),
                self._pose_text(target_pos, target_quat),
                msg.header.frame_id,
            )
            rospy.logwarn(
                "[execute_next_peg_route] waypoint %d/%d current_before %s frame=%s",
                idx + 1,
                len(msg.poses),
                self._pose_text(current_pos, current_quat),
                current_frame,
            )
            single = PoseArray()
            single.header = msg.header
            single.header.stamp = rospy.Time.now()
            single.poses.append(pose)
            self._publish_pose_array(arm, single)
            result = wait_for_moveit_motion_result([arm])
            results.append(result)
            arrived_pos, arrived_quat, arrived_frame = self._current_pose(state, arm, msg.header.frame_id)
            pos_err = float(np.linalg.norm(arrived_pos - target_pos)) if np.all(np.isfinite(arrived_pos)) else float("nan")
            rospy.logwarn(
                "[execute_next_peg_route] waypoint %d/%d arrived %s frame=%s pos_err=%.4f status=%s",
                idx + 1,
                len(msg.poses),
                self._pose_text(arrived_pos, arrived_quat),
                arrived_frame,
                pos_err,
                result.get("status", {}).get(arm),
            )
            if pause_s > 0.0 and idx + 1 < len(msg.poses):
                rospy.sleep(pause_s)
        return {"stepwise": True, "steps": results}

    def _preposition_first_waypoint(self, state, arm: str, waypoint: dict) -> dict:
        pose_msg = self._pose_stamped_from_waypoint(state, waypoint)
        target_pos, target_quat = self._pose_from_msg(pose_msg.pose)
        current_pos, current_quat, current_frame = self._current_pose(state, arm, pose_msg.header.frame_id)
        rospy.logwarn(
            "[execute_next_peg_route] preposition first waypoint target %s frame=%s",
            self._pose_text(target_pos, target_quat),
            pose_msg.header.frame_id,
        )
        rospy.logwarn(
            "[execute_next_peg_route] preposition current_before %s frame=%s",
            self._pose_text(current_pos, current_quat),
            current_frame,
        )
        self._publish_pose(arm, pose_msg)
        result = wait_for_moveit_motion_result([arm])
        arrived_pos, arrived_quat, arrived_frame = self._current_pose(state, arm, pose_msg.header.frame_id)
        pos_err = float(np.linalg.norm(arrived_pos - target_pos)) if np.all(np.isfinite(arrived_pos)) else float("nan")
        rospy.logwarn(
            "[execute_next_peg_route] preposition arrived %s frame=%s pos_err=%.4f status=%s",
            self._pose_text(arrived_pos, arrived_quat),
            arrived_frame,
            pos_err,
            result.get("status", {}).get(arm),
        )
        return result

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

        execution_waypoints = self._execution_waypoints(state, plan, waypoints)
        msg = self._build_pose_array(state, execution_waypoints)
        self._log_waypoint_preview(state, plan, msg, execution_waypoints)
        self._validate_plane_clearance(state, plan, execution_waypoints)

        dry_run = bool(getattr(state.config, "execute_next_peg_route_dry_run", False))
        if dry_run:
            rospy.logwarn("[execute_next_peg_route] dry-run enabled; not publishing waypoints")
            return {
                "executed": False,
                "dry_run": True,
                "arm": arm,
                "waypoint_count": len(waypoints),
                "curr_clip_idx": plan["curr_clip_idx"],
            }

        stepwise = bool(getattr(state.config, "execute_next_peg_route_stepwise", False))
        if stepwise:
            pause_s = max(0.0, float(getattr(state.config, "execute_next_peg_route_pause_s", 0.2)))
            motion_result = self._execute_stepwise(state, arm, msg, pause_s)
        else:
            preposition_result = None
            if bool(getattr(state.config, "execute_next_peg_route_preposition_first_waypoint", False)):
                preposition_result = self._preposition_first_waypoint(state, arm, execution_waypoints[0])
            self._publish_pose_array(arm, msg)
            motion_result = wait_for_moveit_motion_result([arm])
            if preposition_result is not None:
                motion_result = {
                    "preposition": preposition_result,
                    "waypoints": motion_result,
                }
        debug_partial = bool(getattr(state.config, "execute_next_peg_route_debug_partial_execution", False))
        if debug_partial and not stepwise:
            current_pos, current_quat, current_frame = self._current_pose(
                state,
                arm,
                msg.header.frame_id,
            )
            if getattr(state, "current_arm_poses_world", None) is None:
                state.current_arm_poses_world = {}
            if np.all(np.isfinite(current_pos)) and np.all(np.isfinite(current_quat)):
                state.current_arm_poses_world[arm] = {
                    "position": current_pos,
                    "rotation": R.from_quat(current_quat).as_matrix(),
                }
            rospy.logwarn(
                "[execute_next_peg_route] debug partial execution enabled; "
                "not marking peg route complete and not advancing route index. "
                "current_after %s frame=%s",
                self._pose_text(current_pos, current_quat),
                current_frame,
            )
            return {
                "executed": False,
                "route_complete": False,
                "debug_partial_execution": True,
                "arm": arm,
                "waypoint_count": len(waypoints),
                "curr_clip_idx": plan["curr_clip_idx"],
                "motion_result": motion_result,
                "current_after_position": np.asarray(current_pos, dtype=float).reshape(3).tolist(),
            }
        last_pose = execution_waypoints[-1]
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
            "stepwise": stepwise,
            "curr_clip_idx": plan["curr_clip_idx"],
            "next_route_routing_index": state.next_route_routing_index,
            "motion_result": motion_result,
            "final_position": np.asarray(last_pose["position"], dtype=float).reshape(3).tolist(),
        }
