from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PipelineState:
    """
    Shared state object passed between pipeline steps.
    """

    env: Optional[Any] = None
    config: Optional[Any] = None

    routing: Optional[List[int]] = None
    clips: Optional[Any] = None
    crossing_fixture_id_list: Optional[List[int]] = None

    rgb_image: Optional[np.ndarray] = None
    routing_overlay: Optional[np.ndarray] = None
    trace_overlay: Optional[np.ndarray] = None
    grasp_overlay: Optional[np.ndarray] = None
    first_route_overlay: Optional[np.ndarray] = None
    peg_route_overlay: Optional[np.ndarray] = None
    first_route_executed: bool = False

    path_in_pixels: Optional[np.ndarray] = None
    path_in_world: Optional[np.ndarray] = None
    cable_orientations: Optional[np.ndarray] = None
    loaded_trace_path: Optional[str] = None

    grasp_preview: Optional[Dict[str, Any]] = None
    grasps: Optional[Any] = None
    grasp_poses: Optional[Any] = None
    path_tangents: Optional[np.ndarray] = None

    robot_target_sent: bool = False
    descend_first_arm: Optional[str] = None
    descend_second_arm: Optional[str] = None
    handover_tcp_rotation_world: Optional[np.ndarray] = None
    handover_carrier_tcp_world: Optional[np.ndarray] = None
    handover_fine_orient_done: bool = False
    handover_exchange_done: bool = False
    handover_repark_done: bool = False
    present_cable_vertical_done: bool = False
    second_arm_side_approach_done: bool = False
    first_route_arm_top_side_signs: Optional[Dict[str, float]] = None
    current_primary_arm: Optional[str] = None
    first_route_prev_clip_id: Optional[int] = None
    first_route_curr_clip_id: Optional[int] = None
    first_route_next_clip_id: Optional[int] = None
    first_route_clockwise: Optional[int] = None
    first_route_sequence: Optional[Any] = None
    first_route_start_px: Optional[np.ndarray] = None
    first_route_target_px: Optional[np.ndarray] = None
    first_route_mode: Optional[str] = None
    first_route_route_height_m: Optional[float] = None
    first_route_clip_type_config: Optional[Dict[str, Any]] = None
    first_route_secondary_arm: Optional[str] = None
    first_route_secondary_start_px: Optional[np.ndarray] = None
    first_route_secondary_target_px: Optional[np.ndarray] = None
    first_route_secondary_shown: bool = False
    current_arm_poses_world: Optional[Dict[str, Dict[str, Any]]] = None
    next_route_routing_index: Optional[int] = None
    peg_route_plan: Optional[Dict[str, Any]] = None
    peg_route_executed: bool = False

    logs: List[str] = field(default_factory=list)
    finished_steps: List[str] = field(default_factory=list)
    step_results: Dict[str, Any] = field(default_factory=dict)
    action_feedback: Dict[str, Any] = field(default_factory=dict)
    action_history: List[Any] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.logs.append(message)

    def reset_runtime_data(self) -> None:
        self.routing = None
        self.clips = None
        self.crossing_fixture_id_list = None

        self.rgb_image = None
        self.routing_overlay = None
        self.trace_overlay = None
        self.grasp_overlay = None
        self.first_route_overlay = None
        self.peg_route_overlay = None
        self.first_route_executed = False

        self.path_in_pixels = None
        self.path_in_world = None
        self.cable_orientations = None
        self.loaded_trace_path = None

        self.grasp_preview = None
        self.grasps = None
        self.grasp_poses = None
        self.path_tangents = None

        self.handover_tcp_rotation_world = None
        self.handover_carrier_tcp_world = None
        self.descend_first_arm = None
        self.descend_second_arm = None
        self.handover_fine_orient_done = False
        self.handover_exchange_done = False
        self.handover_repark_done = False
        self.present_cable_vertical_done = False
        self.second_arm_side_approach_done = False
        self.first_route_arm_top_side_signs = None
        self.current_primary_arm = None
        self.first_route_prev_clip_id = None
        self.first_route_curr_clip_id = None
        self.first_route_next_clip_id = None
        self.first_route_clockwise = None
        self.first_route_sequence = None
        self.first_route_start_px = None
        self.first_route_target_px = None
        self.first_route_mode = None
        self.first_route_route_height_m = None
        self.first_route_clip_type_config = None
        self.first_route_secondary_arm = None
        self.first_route_secondary_start_px = None
        self.first_route_secondary_target_px = None
        self.first_route_secondary_shown = False
        self.current_arm_poses_world = None
        self.next_route_routing_index = None
        self.peg_route_plan = None
        self.peg_route_executed = False

        self.logs.clear()
        self.finished_steps.clear()
        self.step_results.clear()
        self.action_feedback.clear()
        self.action_history.clear()
