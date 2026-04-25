from typing import List, Optional

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Bool, String

from cable_core.planes import ensure_min_plane_height, get_routing_plane


MOTION_FRAME_ID = "yumi_base_link"


def is_dual_arm_grasp(config) -> bool:
    return bool(getattr(config, "dual_arm_grasp", True))


def pose_to_msg(position, rotation, frame_id=None, config=None):
    pos = np.asarray(position, dtype=float).reshape(3)
    rot = np.asarray(rotation, dtype=float).reshape(3, 3)
    quat = R.from_matrix(rot).as_quat()

    if config is not None:
        if not getattr(config, "publish_cartesian_targets_in_world_frame", True):
            frame_id = MOTION_FRAME_ID
        else:
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = MOTION_FRAME_ID
            msg.pose.position.x = float(pos[0])
            msg.pose.position.y = float(pos[1])
            msg.pose.position.z = float(pos[2])
            msg.pose.orientation.x = float(quat[0])
            msg.pose.orientation.y = float(quat[1])
            msg.pose.orientation.z = float(quat[2])
            msg.pose.orientation.w = float(quat[3])

            target = getattr(config, "cartesian_targets_world_frame_id", "world")
            off = np.asarray(
                getattr(config, "cartesian_targets_world_position_offset_m", (0.0, 0.0, 0.0)),
                dtype=float,
            ).reshape(3)
            out = PoseStamped()
            out.header.stamp = rospy.Time.now()
            out.header.frame_id = target
            p = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                dtype=float,
            ) + off
            out.pose.position.x = float(p[0])
            out.pose.position.y = float(p[1])
            out.pose.position.z = float(p[2])
            out.pose.orientation = msg.pose.orientation
            quat_out = np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ],
                dtype=float,
            )
            return out, quat_out

    if frame_id is None:
        frame_id = MOTION_FRAME_ID

    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(pos[0])
    msg.pose.position.y = float(pos[1])
    msg.pose.position.z = float(pos[2])
    msg.pose.orientation.x = float(quat[0])
    msg.pose.orientation.y = float(quat[1])
    msg.pose.orientation.z = float(quat[2])
    msg.pose.orientation.w = float(quat[3])
    return msg, quat


def wait_for_moveit_motion_result(
    arms: List[str],
    timeout_sec: float = 45.0,
    poll_rate_hz: float = 20.0,
) -> dict:
    arm_suffix = {"left": "l", "right": "r"}
    latest_status = {arm: "" for arm in arms}
    latest_arrived = {arm: False for arm in arms}
    subs = []

    def _make_status_cb(arm_name: str):
        def _cb(msg: String) -> None:
            latest_status[arm_name] = str(msg.data)
        return _cb

    def _make_arrived_cb(arm_name: str):
        def _cb(msg: Bool) -> None:
            latest_arrived[arm_name] = bool(msg.data)
        return _cb

    for arm in arms:
        suffix = arm_suffix[arm]
        subs.append(rospy.Subscriber(f"/yumi/rob{suffix}/moveit_status", String, _make_status_cb(arm), queue_size=1))
        subs.append(rospy.Subscriber(f"/yumi/rob{suffix}/moveit_arrived", Bool, _make_arrived_cb(arm), queue_size=1))

    deadline = rospy.Time.now().to_sec() + timeout_sec
    rate = rospy.Rate(poll_rate_hz)
    try:
        while not rospy.is_shutdown():
            if rospy.Time.now().to_sec() > deadline:
                raise RuntimeError(f"Timeout ({timeout_sec}s) waiting for moveit motion result for arms {arms}.")

            all_done = True
            for arm in arms:
                status = latest_status[arm].lower().strip()
                arrived = latest_arrived[arm]
                if status.startswith("timeout") or "failed" in status or status.startswith("error"):
                    raise RuntimeError(f"MoveIt {arm} arm failed: {latest_status[arm]}")
                if not arrived and status != "succeeded":
                    all_done = False
            if all_done:
                return {
                    "arms": list(arms),
                    "status": {arm: latest_status[arm] for arm in arms},
                    "arrived": {arm: latest_arrived[arm] for arm in arms},
                }
            rate.sleep()
    finally:
        for sub in subs:
            sub.unregister()


def wait_for_cartesian_motion_result(
    arms: List[str],
    timeout_sec: float = 45.0,
    poll_rate_hz: float = 20.0,
) -> dict:
    arm_suffix = {"left": "l", "right": "r"}
    latest_status = {arm: "" for arm in arms}
    latest_arrived = {arm: False for arm in arms}
    subs = []

    def _make_status_cb(arm_name: str):
        def _cb(msg: String) -> None:
            latest_status[arm_name] = str(msg.data)
        return _cb

    def _make_arrived_cb(arm_name: str):
        def _cb(msg: Bool) -> None:
            latest_arrived[arm_name] = bool(msg.data)
        return _cb

    for arm in arms:
        suffix = arm_suffix[arm]
        subs.append(rospy.Subscriber(f"/yumi/rob{suffix}/cartesian_status", String, _make_status_cb(arm), queue_size=1))
        subs.append(rospy.Subscriber(f"/yumi/rob{suffix}/cartesian_arrived", Bool, _make_arrived_cb(arm), queue_size=1))

    deadline = rospy.Time.now().to_sec() + timeout_sec
    rate = rospy.Rate(poll_rate_hz)
    try:
        while not rospy.is_shutdown():
            if rospy.Time.now().to_sec() > deadline:
                raise RuntimeError(f"Timeout ({timeout_sec}s) waiting for cartesian motion result for arms {arms}.")
            all_done = True
            for arm in arms:
                status = latest_status[arm].lower().strip()
                arrived = latest_arrived[arm]
                if status.startswith("timeout") or status.startswith("error"):
                    raise RuntimeError(f"Cartesian {arm} arm failed: {latest_status[arm]}")
                if not arrived and status != "succeeded":
                    all_done = False
            if all_done:
                return {
                    "arms": list(arms),
                    "status": {arm: latest_status[arm] for arm in arms},
                    "arrived": {arm: latest_arrived[arm] for arm in arms},
                }
            rate.sleep()
    finally:
        for sub in subs:
            sub.unregister()


def enforce_pose_min_height(
    pose: dict,
    state,
    min_height_above_plane_m: float,
    clip_id=None,
) -> dict:
    plane = get_routing_plane(state.config, clip_id=clip_id)
    out = dict(pose)
    out["position"] = ensure_min_plane_height(
        np.asarray(pose["position"], dtype=float).copy(),
        plane,
        float(min_height_above_plane_m),
    )
    return out
