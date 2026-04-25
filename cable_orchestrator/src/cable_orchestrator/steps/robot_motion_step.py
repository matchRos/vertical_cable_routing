from typing import Dict

import rospy
from geometry_msgs.msg import PoseStamped

from cable_motion.arm_motion_utils import (
    is_dual_arm_grasp,
    pose_to_msg,
    wait_for_moveit_motion_result,
)
from cable_orchestrator.base_step import BaseStep


class RobotMotionStep(BaseStep):
    name = "robot_motion"
    description = "Send pre-grasp pose to YuMi via ROS."

    def __init__(self):
        super().__init__()
        if not rospy.core.is_initialized():
            rospy.init_node("cable_studio_robot_motion", anonymous=True)
        self.pub_left = rospy.Publisher("/yumi/robl/moveit_target_pose", PoseStamped, queue_size=1)
        self.pub_right = rospy.Publisher("/yumi/robr/moveit_target_pose", PoseStamped, queue_size=1)

    def run(self, state) -> Dict[str, object]:
        if not hasattr(state, "pregrasp_poses"):
            raise RuntimeError("No pregrasp poses available.")
        poses = state.pregrasp_poses

        if is_dual_arm_grasp(state.config):
            if len(poses) != 2:
                raise RuntimeError("Dual-arm motion requires exactly 2 pregrasp poses.")
            left_pose = None
            right_pose = None
            for pose in poses:
                arm = pose.get("arm", None)
                if arm == "left":
                    left_pose = pose
                elif arm == "right":
                    right_pose = pose
            if left_pose is None or right_pose is None:
                raise RuntimeError("Need exactly one left pose and one right pose.")

            min_dist_xyz = 0.1
            dist_xyz = float(((left_pose["position"] - right_pose["position"]) ** 2).sum() ** 0.5)
            if dist_xyz < min_dist_xyz:
                raise RuntimeError(
                    f"Pregrasp poses too close: distance={dist_xyz:.3f} m < {min_dist_xyz:.3f} m"
                )

            left_msg, left_quat = pose_to_msg(left_pose["position"], left_pose["rotation"], config=state.config)
            right_msg, right_quat = pose_to_msg(right_pose["position"], right_pose["rotation"], config=state.config)
            now = rospy.Time.now()
            left_msg.header.stamp = now
            right_msg.header.stamp = now
            left_msg.pose.position.z += 0.1
            right_msg.pose.position.z += 0.1
            self.pub_left.publish(left_msg)
            rospy.sleep(1.0)
            self.pub_right.publish(right_msg)
            motion_result = wait_for_moveit_motion_result(["left", "right"])
            state.robot_target_sent = True
            return {
                "target_sent": True,
                "motion_result": motion_result,
                "arms": ["left", "right"],
                "distance_xyz": dist_xyz,
                "left_position": [left_msg.pose.position.x, left_msg.pose.position.y, left_msg.pose.position.z],
                "right_position": [right_msg.pose.position.x, right_msg.pose.position.y, right_msg.pose.position.z],
                "left_quaternion": [float(left_quat[0]), float(left_quat[1]), float(left_quat[2]), float(left_quat[3])],
                "right_quaternion": [float(right_quat[0]), float(right_quat[1]), float(right_quat[2]), float(right_quat[3])],
            }

        if len(poses) != 1:
            raise RuntimeError("Single-arm motion requires exactly 1 pregrasp pose.")
        only = poses[0]
        arm = only.get("arm", "right")
        msg, quat = pose_to_msg(only["position"], only["rotation"], config=state.config)
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.z += 0.1
        if arm == "left":
            self.pub_left.publish(msg)
        else:
            self.pub_right.publish(msg)
        motion_result = wait_for_moveit_motion_result([arm])
        state.robot_target_sent = True
        return {
            "target_sent": True,
            "motion_result": motion_result,
            "arms": [arm],
            "distance_xyz": 0.0,
            "left_position": [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z] if arm == "left" else None,
            "right_position": [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z] if arm == "right" else None,
            "left_quaternion": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])] if arm == "left" else None,
            "right_quaternion": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])] if arm == "right" else None,
        }
