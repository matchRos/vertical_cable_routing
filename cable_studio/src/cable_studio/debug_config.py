"""
Studio configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


PACKAGE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_DIR.parent
PARTS_DIR = PACKAGE_DIR / "config" / "parts"
PART_FILES = ("core.yaml", "routing_plane.yaml", "trace.yaml", "first_route.yaml")
CAMERA_CONFIG_DIR = PACKAGE_DIR / "config" / "cameras"
CORE_CONFIG_DIR = REPO_ROOT / "cable_core" / "config"
CLIP_TYPE_CONFIG_DIR = CORE_CONFIG_DIR / "clip_types"


def _load_yaml_merged(parts_dir: Path) -> Dict[str, Any]:
    import yaml

    merged: Dict[str, Any] = {}
    for name in PART_FILES:
        path = parts_dir / name
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if isinstance(data, dict):
            merged.update(data)
    return merged


def _expand_paths_if_relative(cfg: Dict[str, Any]) -> None:
    keys = (
        "board_cfg_path",
        "cam_to_robot_left_trans_path",
        "cam_to_robot_right_trans_path",
        "board_calibration_yaml",
        "clip_type_config_dir",
    )
    for key in keys:
        val = cfg.get(key)
        if not val or not isinstance(val, str) or os.path.isabs(val):
            continue
        if val.startswith("config/"):
            cfg[key] = os.path.normpath(os.path.join(str(PACKAGE_DIR), val))
        elif val.startswith("cable_core/"):
            cfg[key] = os.path.normpath(os.path.join(str(REPO_ROOT), val))
        else:
            cfg[key] = os.path.normpath(os.path.join(str(REPO_ROOT), val))


def _coerce_for_dataclass(name: str, value: Any) -> Any:
    if name == "trace_end_points" and value is None:
        return None
    if name == "default_routing" and isinstance(value, list):
        return tuple(int(x) for x in value)
    if name in ("trace_start_points", "trace_end_points") and isinstance(value, list):
        return tuple(tuple(int(x) for x in row) for row in value)
    if name in (
        "single_arm_nominal_tcp_left_m",
        "single_arm_nominal_tcp_right_m",
        "cartesian_targets_world_position_offset_m",
        "handover_goal_world_m",
    ) and isinstance(value, list):
        return tuple(float(x) for x in value)
    if name == "routing_planes" and isinstance(value, dict):
        return {str(k): dict(v) for k, v in value.items()}
    if name == "clip_plane_assignments" and isinstance(value, dict):
        return {int(k): str(v) for k, v in value.items()}
    if name == "trace_white_ring_k_candidates" and isinstance(value, list):
        return tuple(float(x) for x in value)
    return value


def load_debug_config(parts_dir: Optional[Path] = None) -> "DebugConfig":
    parts_dir = parts_dir or PARTS_DIR
    merged = _load_yaml_merged(parts_dir)
    _expand_paths_if_relative(merged)

    base = DebugConfig()
    kwargs: Dict[str, Any] = {}
    for f in fields(DebugConfig):
        if f.name in merged:
            kwargs[f.name] = _coerce_for_dataclass(f.name, merged[f.name])
        else:
            kwargs[f.name] = getattr(base, f.name)
    return DebugConfig(**kwargs)


@dataclass
class DebugConfig:
    board_cfg_path: str = str(CORE_CONFIG_DIR / "board" / "board_config.json")
    default_routing: tuple = (0, 1, 2, 3)
    fallback_image_width: int = 1500
    fallback_image_height: int = 800
    debug_image_path: Optional[str] = "/ABSOLUTER/PFAD/ZU/DEINEM/BILD.png"
    camera_rgb_topic: str = "/zedm/zed_node/left/image_rect_color"
    camera_depth_topic: str = "/zedm/zed_node/depth/depth_registered"
    camera_info_topic: str = "/zedm/zed_node/left/camera_info"
    camera_require_depth: bool = False
    camera_wait_timeout_sec: float = 5.0
    checkpoint_joint_tolerance_rad: float = 0.15
    trace_start_points: Tuple[Tuple[int, int], ...] = ((100, 100),)
    trace_end_points: Optional[Tuple[Tuple[int, int], ...]] = None
    trace_start_mode: str = "auto_from_config"
    trace_anchor_max_start_dist_px: float = 90.0
    trace_candidate_min_route_dot: float = 0.25
    trace_anchor_outward_min_delta_px: float = 8.0
    trace_auto_clip_a_p1_offset_px: float = 20.0
    trace_auto_clip_a_p2_offset_px: float = 40.0
    trace_white_ring_step_px: float = 20.0
    trace_white_ring_k_candidates: Tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0)
    trace_min_path_points: int = 40
    trace_analytic_min_path_points: int = 25
    trace_min_end_to_start_px: float = 100.0
    trace_model_path_len: int = 200
    trace_analytic_path_len: int = 90
    trace_analytic_timeout_sec: float = 8.0
    trace_seed_order_descending_from_anchor: bool = True
    routing_plane_default_id: str = "main"
    routing_planes: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "main": {
                "origin": [0.56, 0.0, 0.15],
                "normal": [-1.0, 0.0, 0.0],
                "u_axis": [0.0, 0.0, 1.0],
                "v_axis": [0.0, 1.0, 0.0],
            }
        }
    )
    clip_plane_assignments: Dict[int, str] = field(default_factory=dict)
    routing_height_above_plane_m: float = 0.025
    grasp_height_above_plane_m: float = 0.025
    grasp_min_clearance_from_first_peg_m: float = 0.05
    grasp_second_min_arc_from_first_grasp_m: float = 0.08
    pregrasp_offset_from_grasp_m: float = 0.08
    grasp_extra_world_rx_deg: float = 90.0
    publish_cartesian_targets_in_world_frame: bool = True
    cartesian_targets_world_frame_id: str = "world"
    cartesian_targets_world_position_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    detangle_offset_from_routing_m: float = 0.03
    first_route_primary_extra_along_route_px: float = 60.0
    first_route_execute_secondary_arm: bool = True
    first_route_insert_after_route: bool = True
    first_route_default_insert_height_above_plane_m: float = 0.01
    peg_route_clearance_radius_m: float = 0.035
    peg_route_workspace_cross_y_m: float = 0.2
    peg_route_min_other_arm_distance_m: float = 0.1
    peg_route_arc_samples: int = 10
    peg_route_arc_max_step_deg: float = 8.0
    peg_route_moveit_waypoint_min_fraction: float = 0.85
    peg_route_collinear_tolerance_m: float = 0.02
    peg_route_height_tolerance_m: float = 0.005
    peg_route_lateral_tolerance_m: float = 0.01
    c_clip_primary_lateral_px: float = 90.0
    c_clip_secondary_lateral_px: float = 70.0
    c_clip_primary_forward_px: float = 30.0
    c_clip_secondary_forward_px: float = -15.0
    c_clip_swap_sides_when_primary_right: bool = False
    c_clip_center_primary_lateral_px: float = 20.0
    c_clip_center_secondary_lateral_px: float = 45.0
    c_clip_center_primary_forward_px: float = 5.0
    c_clip_center_secondary_forward_px: float = -10.0
    cam_to_robot_left_trans_path: str = str(CAMERA_CONFIG_DIR / "zed_to_world_left.tf")
    cam_to_robot_right_trans_path: str = str(CAMERA_CONFIG_DIR / "zed_to_world_right.tf")
    board_calibration_yaml: str = str(CAMERA_CONFIG_DIR / "camera_robot_2d_calibration.yaml")
    board_plane_x_m: float = 0.56
    world_from_pixel_z_offset_m: float = 0.1
    dual_arm_grasp: bool = False
    single_arm_nominal_tcp_left_m: Tuple[float, float, float] = (0.35, 0.22, 0.14)
    single_arm_nominal_tcp_right_m: Tuple[float, float, float] = (0.35, -0.22, 0.14)
    handover_arm: Optional[str] = None
    handover_clip_routing_index: int = 0
    handover_goal_world_m: Tuple[float, float, float] = (0.4, 0.0, 0.4)
    handover_lift_along_normal_m: float = 0.02
    handover_enforce_min_plane_height: bool = False
    handover_fine_tool_rx_deg: float = 0.0
    handover_fine_tool_ry_deg: float = 0.0
    handover_fine_tool_rz_deg: float = 0.0
    dual_side_second_arm_delta_z_m: float = -0.1
    present_cable_extra_world_rx_deg: float = 90.0
    second_arm_extra_world_ry_deg: float = 90.0
    second_arm_flip_world_y_180: bool = True
    dual_side_second_arm_prepose_offset_y_m: float = 0.08
    dual_side_second_arm_prepose_pause_s: float = 0.5
    dual_side_second_arm_slow_approach_extra_y_m: float = 0.01
    clip_type_config_dir: str = str(CLIP_TYPE_CONFIG_DIR)
