from __future__ import annotations

from cable_orchestrator.pipeline_orchestrator import PipelineOrchestrator
from cable_orchestrator.step_action import StepBackedAction
from cable_orchestrator.steps.compute_orientation_step import ComputeOrientationStep
from cable_orchestrator.steps.grasp_planning_step import GraspPlanningStep
from cable_orchestrator.steps.grasp_pose_step import GraspPoseStep
from cable_orchestrator.steps.pregrasp_pose_step import PreGraspPoseStep
from cable_orchestrator.steps.prepare_routing_step import PrepareRoutingStep
from cable_orchestrator.steps.trace_cable_step import TraceCableStep
from cable_orchestrator.steps.trace_to_world_step import TraceToWorldStep
from cable_orchestrator.steps.visualize_grasps_step import VisualizeGraspsStep
from cable_routing.debug_gui.pipeline.steps.close_first_gripper_step import CloseFirstGripperStep
from cable_routing.debug_gui.pipeline.steps.close_second_gripper_step import CloseSecondGripperStep
from cable_routing.debug_gui.pipeline.steps.descend_to_grasp_step import DescendToGraspStep
from cable_routing.debug_gui.pipeline.steps.execute_first_route_step import ExecuteFirstRouteStep
from cable_routing.debug_gui.pipeline.steps.handover_fine_orient_step import HandoverFineOrientStep
from cable_routing.debug_gui.pipeline.steps.handover_move_exchange_step import HandoverMoveExchangeStep
from cable_routing.debug_gui.pipeline.steps.home_arms_step import HomeArmsStep
from cable_routing.debug_gui.pipeline.steps.init_environment_step import InitEnvironmentStep
from cable_routing.debug_gui.pipeline.steps.plan_first_route_step import PlanFirstRouteStep
from cable_routing.debug_gui.pipeline.steps.present_cable_vertical_step import PresentCableVerticalStep
from cable_routing.debug_gui.pipeline.steps.robot_motion_step import RobotMotionStep
from cable_routing.debug_gui.pipeline.steps.second_arm_side_approach_step import SecondArmSideApproachStep
from cable_routing.debug_gui.pipeline.unwind_wrists_step import UnwindWristsStep


def build_default_orchestrator() -> PipelineOrchestrator:
    steps = [
        HomeArmsStep(),
        InitEnvironmentStep(),
        PrepareRoutingStep(),
        TraceCableStep(),
        TraceToWorldStep(),
        ComputeOrientationStep(),
        GraspPlanningStep(),
        GraspPoseStep(),
        VisualizeGraspsStep(),
        PreGraspPoseStep(),
        RobotMotionStep(),
        UnwindWristsStep(),
        DescendToGraspStep(),
        CloseFirstGripperStep(),
        HandoverFineOrientStep(),
        HandoverMoveExchangeStep(),
        PresentCableVerticalStep(),
        SecondArmSideApproachStep(),
        CloseSecondGripperStep(),
        PlanFirstRouteStep(),
        ExecuteFirstRouteStep(),
    ]
    actions = [StepBackedAction(step) for step in steps]
    return PipelineOrchestrator(actions)
