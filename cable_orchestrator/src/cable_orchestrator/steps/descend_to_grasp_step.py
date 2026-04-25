from typing import Dict

import rospy
from geometry_msgs.msg import PoseStamped

from cable_motion.arm_motion_utils import (
    enforce_pose_min_height,
    is_dual_arm_grasp,
    pose_to_msg,
    wait_for_cartesian_motion_result,
)
from cable_orchestrator.base_step import BaseStep


class DescendToGraspStep(BaseStep):
    name = "descend_to_grasp"
    description = "Descend first with the arm farther away from the cable start."

    def __init__(self):
        super().__init__()
        if not rospy.core.is_initialized():
            rospy.init_node("cable_studio_descend_to_grasp", anonymous=True)
        self.pub_left = rospy.Publisher("/yumi/robl/slowly_approach_pose", PoseStamped, queue_size=1)
        self.pub_right = rospy.Publisher("/yumi/robr/slowly_approach_pose", PoseStamped, queue_size=1)

    def _split_by_arm(self, poses):
        left_pose = None
        right_pose = None
        for pose in poses:
            if pose.get("arm") == "left":
                left_pose = pose
            elif pose.get("arm") == "right":
                right_pose = pose
        if left_pose is None or right_pose is None:
            raise RuntimeError("Need exactly one left pose and one right pose.")
        return left_pose, right_pose

    def run(self, state) -> Dict[str, object]:
        if not hasattr(state, "grasp_poses"):
            raise RuntimeError("No grasp poses available.")
        if not hasattr(state, "pregrasp_poses"):
            raise RuntimeError("No pregrasp poses available.")

        grasp_poses = state.grasp_poses
        pregrasp_poses = state.pregrasp_poses

        if is_dual_arm_grasp(state.config):
            if len(grasp_poses) != 2 or len(pregrasp_poses) != 2:
                raise RuntimeError("Sequential descend requires exactly 2 grasp and 2 pregrasp poses.")

            left_grasp, right_grasp = self._split_by_arm(grasp_poses)
            self._split_by_arm(pregrasp_poses)

            if "path_index" not in left_grasp or "path_index" not in right_grasp:
                raise RuntimeError("Both grasp poses need 'path_index'.")

            left_progress = float(left_grasp["path_index"])
            right_progress = float(right_grasp["path_index"])

            if left_progress > right_progress:
                first_arm, first_pose = "left", left_grasp
                second_arm, second_pose = "right", right_grasp
            else:
                first_arm, first_pose = "right", right_grasp
                second_arm, second_pose = "left", left_grasp

            grasp_floor = float(state.config.grasp_height_above_plane_m)
            first_pose = enforce_pose_min_height(first_pose, state, grasp_floor)
            second_pose = enforce_pose_min_height(second_pose, state, grasp_floor)
            first_msg, first_quat = pose_to_msg(first_pose["position"], first_pose["rotation"], config=state.config)
            first_msg.header.stamp = rospy.Time.now()
            if first_arm == "left":
                self.pub_left.publish(first_msg)
            else:
                self.pub_right.publish(first_msg)
            motion_result = wait_for_cartesian_motion_result([first_arm])

            state.descend_first_arm = first_arm
            state.descend_second_arm = second_arm
            state.first_grasp_pose = first_pose
            state.second_grasp_pose = second_pose
            state.descend_target_sent = True
            return {
                "descend_sent": True,
                "mode": "sequential",
                "first_arm_sent": first_arm,
                "second_arm_pending": second_arm,
                "left_progress": left_progress,
                "right_progress": right_progress,
                "motion_result": motion_result,
                "first_position": [first_msg.pose.position.x, first_msg.pose.position.y, first_msg.pose.position.z],
                "first_quaternion": [float(first_quat[0]), float(first_quat[1]), float(first_quat[2]), float(first_quat[3])],
            }

        if len(grasp_poses) != 1:
            raise RuntimeError("Single-arm descend requires exactly 1 grasp pose.")
        first_pose = enforce_pose_min_height(grasp_poses[0], state, float(state.config.grasp_height_above_plane_m))
        first_arm = first_pose.get("arm", "right")
        first_msg, first_quat = pose_to_msg(first_pose["position"], first_pose["rotation"], config=state.config)
        first_msg.header.stamp = rospy.Time.now()
        if first_arm == "left":
            self.pub_left.publish(first_msg)
        else:
            self.pub_right.publish(first_msg)
        motion_result = wait_for_cartesian_motion_result([first_arm])
        state.descend_first_arm = first_arm
        state.descend_second_arm = None
        state.first_grasp_pose = first_pose
        state.second_grasp_pose = None
        state.descend_target_sent = True
        return {
            "descend_sent": True,
            "mode": "single_arm",
            "first_arm_sent": first_arm,
            "motion_result": motion_result,
            "second_arm_pending": None,
            "first_position": [first_msg.pose.position.x, first_msg.pose.position.y, first_msg.pose.position.z],
            "first_quaternion": [float(first_quat[0]), float(first_quat[1]), float(first_quat[2]), float(first_quat[3])],
        }
