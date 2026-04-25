from typing import Dict

import rospy
from std_srvs.srv import Trigger

from cable_orchestrator.base_step import BaseStep


class CloseSecondGripperStep(BaseStep):
    name = "close_second_gripper"
    description = "Close the gripper of the second arm."

    def __init__(self):
        super().__init__()
        if not rospy.core.is_initialized():
            rospy.init_node("cable_studio_close_second_gripper", anonymous=True)

    def run(self, state) -> Dict[str, object]:
        second_arm = state.descend_second_arm
        if second_arm is None:
            raise RuntimeError("No second arm stored in state.")
        if second_arm == "left":
            service_name = "/yumi/gripper_l/close"
        elif second_arm == "right":
            service_name = "/yumi/gripper_r/close"
        else:
            raise RuntimeError(f"Invalid second arm: {second_arm}")

        rospy.wait_for_service(service_name, timeout=5.0)
        close_srv = rospy.ServiceProxy(service_name, Trigger)
        resp = close_srv()
        if not resp.success:
            raise RuntimeError(f"{service_name} failed: {resp.message}")

        rospy.sleep(1.0)
        state.second_gripper_closed = True
        return {
            "gripper_closed": True,
            "arm": second_arm,
            "service": service_name,
            "message": resp.message,
        }
