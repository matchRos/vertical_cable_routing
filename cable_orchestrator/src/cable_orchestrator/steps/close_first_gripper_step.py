from typing import Dict

import rospy
from std_srvs.srv import Trigger

from cable_orchestrator.base_step import BaseStep


class CloseFirstGripperStep(BaseStep):
    name = "close_first_gripper"
    description = "Close the gripper of the arm that descended first."

    def __init__(self):
        super().__init__()
        if not rospy.core.is_initialized():
            rospy.init_node("cable_studio_close_first_gripper", anonymous=True)

    def run(self, state) -> Dict[str, object]:
        if not hasattr(state, "descend_first_arm"):
            raise RuntimeError("No first descend arm stored in state.")
        first_arm = state.descend_first_arm
        if first_arm == "left":
            service_name = "/yumi/gripper_l/close"
        elif first_arm == "right":
            service_name = "/yumi/gripper_r/close"
        else:
            raise RuntimeError(f"Invalid first arm: {first_arm}")

        rospy.wait_for_service(service_name, timeout=5.0)
        close_srv = rospy.ServiceProxy(service_name, Trigger)
        resp = close_srv()
        if not resp.success:
            raise RuntimeError(f"{service_name} failed: {resp.message}")

        rospy.sleep(1.0)
        state.first_gripper_closed = True
        return {
            "gripper_closed": True,
            "arm": first_arm,
            "service": service_name,
            "message": resp.message,
        }
