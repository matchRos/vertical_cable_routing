from __future__ import annotations

from cable_orchestrator.pipeline_orchestrator import PipelineOrchestrator
from cable_orchestrator.step_action import StepBackedAction
from cable_orchestrator.steps.check_if_grasped_step import CheckIfGraspedStep
from cable_orchestrator.steps.check_if_lax_step import CheckIfLaxStep
from cable_orchestrator.steps.close_first_gripper_step import CloseFirstGripperStep
from cable_orchestrator.steps.close_second_gripper_step import CloseSecondGripperStep
from cable_orchestrator.steps.compute_orientation_step import ComputeOrientationStep
from cable_orchestrator.steps.descend_to_grasp_step import DescendToGraspStep
from cable_orchestrator.steps.execute_first_route_step import ExecuteFirstRouteStep
from cable_orchestrator.steps.grasp_planning_step import GraspPlanningStep
from cable_orchestrator.steps.grasp_pose_step import GraspPoseStep
from cable_orchestrator.steps.handover_fine_orient_step import HandoverFineOrientStep
from cable_orchestrator.steps.handover_move_exchange_step import HandoverMoveExchangeStep
from cable_orchestrator.steps.home_arms_step import HomeArmsStep
from cable_orchestrator.steps.init_environment_step import InitEnvironmentStep
from cable_orchestrator.steps.plan_first_route_step import PlanFirstRouteStep
from cable_orchestrator.steps.pregrasp_pose_step import PreGraspPoseStep
from cable_orchestrator.steps.prepare_routing_step import PrepareRoutingStep
from cable_orchestrator.steps.present_cable_vertical_step import PresentCableVerticalStep
from cable_orchestrator.steps.robot_motion_step import RobotMotionStep
from cable_orchestrator.steps.second_arm_side_approach_step import SecondArmSideApproachStep
from cable_orchestrator.steps.trace_cable_step import TraceCableStep
from cable_orchestrator.steps.trace_to_world_step import TraceToWorldStep
from cable_orchestrator.steps.unwind_wrists_step import UnwindWristsStep
from cable_orchestrator.steps.visualize_grasps_step import VisualizeGraspsStep


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
        CheckIfGraspedStep(),
        HandoverFineOrientStep(),
        CheckIfGraspedStep(),
        CheckIfLaxStep(),
        HandoverMoveExchangeStep(),
        CheckIfGraspedStep(),
        PresentCableVerticalStep(),
        SecondArmSideApproachStep(),
        CloseSecondGripperStep(),
        CheckIfGraspedStep(),
        CheckIfLaxStep(),
        PlanFirstRouteStep(),
        ExecuteFirstRouteStep(),
    ]
    actions = [StepBackedAction(step) for step in steps]
    return PipelineOrchestrator(actions)
