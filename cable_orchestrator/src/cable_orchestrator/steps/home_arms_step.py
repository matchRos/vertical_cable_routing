from typing import Dict

import rospy
from std_srvs.srv import Trigger

from cable_orchestrator.base_step import BaseStep


class HomeArmsStep(BaseStep):
    name = "home_arms"
    description = "Send both YuMi arms to home before taking an image."

    def __init__(self) -> None:
        super().__init__()
        if not rospy.core.is_initialized():
            rospy.init_node("cable_studio_home_arms", anonymous=True, disable_signals=True)

    def run(self, state) -> Dict[str, object]:
        rospy.wait_for_service("/yumi/gripper_l/open", timeout=5.0)
        open_left_srv = rospy.ServiceProxy("/yumi/gripper_l/open", Trigger)
        resp_left = open_left_srv()

        rospy.wait_for_service("/yumi/gripper_r/open", timeout=5.0)
        open_right_srv = rospy.ServiceProxy("/yumi/gripper_r/open", Trigger)
        resp_right = open_right_srv()

        rospy.sleep(1.0)

        rospy.wait_for_service("/yumi/home_both_arms", timeout=5.0)
        home_srv = rospy.ServiceProxy("/yumi/home_both_arms", Trigger)
        resp = home_srv()
        if not resp.success:
            raise RuntimeError(f"/yumi/home_both_arms failed: {resp.message}")
        if not resp_right.success:
            raise RuntimeError(f"Right gripper open failed: {resp_right.message}")
        if not resp_left.success:
            raise RuntimeError(f"Left gripper open failed: {resp_left.message}")

        rospy.sleep(3.0)
        return {"home_called": True, "service": "/yumi/home_both_arms", "message": resp.message}
