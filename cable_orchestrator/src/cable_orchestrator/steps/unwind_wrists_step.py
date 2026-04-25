from typing import Dict, Optional

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from cable_core.planes import get_routing_plane
from cable_motion.arm_motion_utils import is_dual_arm_grasp
from cable_orchestrator.base_step import BaseStep


class UnwindWristsStep(BaseStep):
    name = "unwind_wrists"
    description = "Unwind wrist joint q7 if the end-effector is approximately vertical."

    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node("cable_studio_unwind_wrists", anonymous=True)

        self.pub = rospy.Publisher(
            "/yumi/joint_group_velocity_command",
            Float64MultiArray,
            queue_size=1,
        )

        self._last_joint_state: Optional[JointState] = None
        self._joint_sub = rospy.Subscriber(
            "/joint_states",
            JointState,
            self._joint_state_cb,
            queue_size=1,
        )

    def _joint_state_cb(self, msg: JointState):
        self._last_joint_state = msg

    def _wait_for_joint_state(self, timeout=2.0) -> JointState:
        start = rospy.Time.now().to_sec()
        rate = rospy.Rate(50)

        while not rospy.is_shutdown():
            if self._last_joint_state is not None:
                return self._last_joint_state

            if rospy.Time.now().to_sec() - start > timeout:
                raise RuntimeError("Did not receive /joint_states in time.")

            rate.sleep()

        raise RuntimeError("ROS shutdown while waiting for /joint_states.")

    def _get_pose_for_arm(self, poses, arm_name):
        for pose in poses:
            if pose.get("arm") == arm_name:
                return pose
        return None

    def _is_tool_vertical_enough(
        self, pose, plane_normal: np.ndarray, cosine_threshold=0.9
    ):
        rotation = np.asarray(pose["rotation"], dtype=float).reshape(3, 3)

        tool_z = rotation[:, 2]
        tool_z = tool_z / (np.linalg.norm(tool_z) + 1e-8)

        normal = np.asarray(plane_normal, dtype=float).reshape(3)
        normal /= np.linalg.norm(normal) + 1e-8

        alignment = float(abs(np.dot(tool_z, normal)))
        return alignment >= cosine_threshold, alignment

    def _find_joint_index(self, joint_state: JointState, candidates):
        for candidate in candidates:
            if candidate in joint_state.name:
                return joint_state.name.index(candidate), candidate
        return None, None

    def _publish_velocity_for_duration(self, velocities, duration_s):
        rate_hz = 50.0
        rate = rospy.Rate(rate_hz)

        msg = Float64MultiArray()
        msg.data = list(velocities)

        n_steps = int(duration_s * rate_hz)
        for _ in range(n_steps):
            self.pub.publish(msg)
            rate.sleep()

        stop_msg = Float64MultiArray()
        stop_msg.data = [0.0] * len(velocities)
        self.pub.publish(stop_msg)

    def run(self, state) -> Dict[str, object]:
        if not hasattr(state, "pregrasp_poses"):
            raise RuntimeError("No pregrasp poses available.")

        joint_state = self._wait_for_joint_state(timeout=2.0)

        poses = state.pregrasp_poses
        plane = get_routing_plane(state.config)
        plane_normal = np.asarray(plane.normal, dtype=float).reshape(3)

        left_pose = self._get_pose_for_arm(poses, "left")
        right_pose = self._get_pose_for_arm(poses, "right")

        if is_dual_arm_grasp(state.config):
            if left_pose is None or right_pose is None:
                raise RuntimeError("Need exactly one left and one right pregrasp pose.")
        else:
            if left_pose is None and right_pose is None:
                raise RuntimeError("No pregrasp pose found for unwind.")

        left_ok, left_align = (
            self._is_tool_vertical_enough(left_pose, plane_normal)
            if left_pose is not None
            else (False, 0.0)
        )
        right_ok, right_align = (
            self._is_tool_vertical_enough(right_pose, plane_normal)
            if right_pose is not None
            else (False, 0.0)
        )

        left_idx, left_name = self._find_joint_index(
            joint_state,
            ["yumi_robl_joint_7"],
        )
        right_idx, right_name = self._find_joint_index(
            joint_state,
            ["yumi_robr_joint_7"],
        )

        if left_idx is None or right_idx is None:
            raise RuntimeError(
                f"Could not find q7 joints in /joint_states names: {joint_state.name}"
            )

        q = np.asarray(joint_state.position, dtype=float)
        num_joints = len(q)

        q7_left = float(q[left_idx])
        q7_right = float(q[right_idx])

        threshold = np.deg2rad(90.0)
        target_speed = np.deg2rad(25.0)
        duration_s = 1.2

        velocities = np.zeros(num_joints, dtype=float)

        left_action = "none"
        right_action = "none"

        if left_pose is not None:
            if left_ok and q7_left > threshold:
                velocities[left_idx] = -target_speed
                left_action = "negative_unwind"
            elif left_ok and q7_left < -threshold:
                velocities[left_idx] = target_speed
                left_action = "positive_unwind"

        if right_pose is not None:
            if right_ok and q7_right > threshold:
                velocities[right_idx] = -target_speed
                right_action = "negative_unwind"
            elif right_ok and q7_right < -threshold:
                velocities[right_idx] = target_speed
                right_action = "positive_unwind"

        if np.allclose(velocities, 0.0):
            return {
                "unwind_executed": False,
                "reason": "no_arm_needed_unwind",
                "left_q7_rad": q7_left,
                "right_q7_rad": q7_right,
                "left_vertical_ok": left_ok,
                "right_vertical_ok": right_ok,
                "left_alignment": left_align,
                "right_alignment": right_align,
                "left_joint_name": left_name,
                "right_joint_name": right_name,
            }

        self._publish_velocity_for_duration(velocities, duration_s)

        state.wrist_unwind_done = True

        return {
            "unwind_executed": True,
            "duration_s": duration_s,
            "target_speed_rad_s": target_speed,
            "left_q7_rad": q7_left,
            "right_q7_rad": q7_right,
            "left_vertical_ok": left_ok,
            "right_vertical_ok": right_ok,
            "left_alignment": left_align,
            "right_alignment": right_align,
            "left_joint_name": left_name,
            "right_joint_name": right_name,
            "left_action": left_action,
            "right_action": right_action,
        }
