from __future__ import annotations

import pickle
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cable_core.board_models import DebugBoard
from cable_core.board_yz_calibration import load_board_yz_calibration_optional
from cable_orchestrator.steps.init_environment_step import SimpleRigidTransform
from cable_studio.debug_config import load_debug_config
from cable_studio.debug_context import DebugContext


CHECKPOINT_VERSION = 1


STATE_KEYS = (
    "routing",
    "clips",
    "crossing_fixture_id_list",
    "rgb_image",
    "routing_overlay",
    "trace_overlay",
    "grasp_overlay",
    "first_route_overlay",
    "peg_route_overlay",
    "first_route_executed",
    "path_in_pixels",
    "path_in_world",
    "cable_orientations",
    "loaded_trace_path",
    "grasp_preview",
    "grasps",
    "grasp_poses",
    "path_tangents",
    "robot_target_sent",
    "descend_first_arm",
    "descend_second_arm",
    "handover_tcp_rotation_world",
    "handover_carrier_tcp_world",
    "handover_fine_orient_done",
    "handover_exchange_done",
    "handover_repark_done",
    "present_cable_vertical_done",
    "second_arm_side_approach_done",
    "first_route_arm_top_side_signs",
    "current_primary_arm",
    "first_route_prev_clip_id",
    "first_route_curr_clip_id",
    "first_route_next_clip_id",
    "first_route_clockwise",
    "first_route_sequence",
    "first_route_start_px",
    "first_route_target_px",
    "first_route_mode",
    "first_route_route_height_m",
    "first_route_clip_type_config",
    "first_route_secondary_arm",
    "first_route_secondary_start_px",
    "first_route_secondary_target_px",
    "first_route_secondary_shown",
    "current_arm_poses_world",
    "next_route_routing_index",
    "peg_route_plan",
    "peg_route_executed",
    "logs",
    "finished_steps",
    "step_results",
    "action_feedback",
    "action_history",
)


class StudioCheckpointIO:
    def _read_transform_file(self, path: str) -> SimpleRigidTransform:
        with open(path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.readlines() if line.strip()]
        if len(lines) < 5:
            raise RuntimeError(f"Rigid transform file is incomplete: {path}")
        translation = np.fromstring(lines[2], sep=" ")
        rotation = np.vstack([np.fromstring(line, sep=" ") for line in lines[3:6]])
        return SimpleRigidTransform(rotation, translation, from_frame=lines[0], to_frame=lines[1])

    def capture_joint_snapshot(self, timeout_sec: float = 1.0) -> Optional[Dict[str, float]]:
        try:
            import rospy
            from sensor_msgs.msg import JointState

            if not rospy.core.is_initialized():
                rospy.init_node("cable_studio_checkpoint", anonymous=True, disable_signals=True)

            last_exc = None
            for topic in ("/yumi/egm/joint_states", "/joint_states"):
                try:
                    msg = rospy.wait_for_message(topic, JointState, timeout=float(timeout_sec))
                    return {
                        str(name): float(pos)
                        for name, pos in zip(msg.name, msg.position)
                        if str(name).startswith("yumi_")
                    }
                except Exception as exc:
                    last_exc = exc
            if last_exc is not None:
                raise last_exc
        except Exception:
            return None
        return None

    def compare_joint_snapshot(
        self,
        saved: Optional[Dict[str, float]],
        current: Optional[Dict[str, float]],
    ) -> Tuple[bool, str, float]:
        if not saved:
            return True, "No saved joint snapshot in checkpoint.", 0.0
        if not current:
            return False, "Could not read current YuMi joint state.", float("inf")

        common = sorted(set(saved.keys()) & set(current.keys()))
        if not common:
            return False, "No overlapping YuMi joints between checkpoint and current state.", float("inf")

        diffs = {name: abs(float(saved[name]) - float(current[name])) for name in common}
        worst_name = max(diffs, key=diffs.get)
        worst = float(diffs[worst_name])
        return True, f"Max joint delta: {worst:.3f} rad at {worst_name}.", worst

    def _state_payload(self, state: Any) -> Dict[str, Any]:
        payload = {key: getattr(state, key, None) for key in STATE_KEYS}
        payload["config"] = getattr(state, "config", None)
        if is_dataclass(payload["config"]):
            payload["config_asdict"] = asdict(payload["config"])
        return payload

    def save(self, path: str, state: Any, runner: Any) -> None:
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "created_unix": time.time(),
            "runner_current_idx": int(getattr(runner, "current_idx", 0)),
            "state": self._state_payload(state),
            "joint_snapshot": self.capture_joint_snapshot(),
        }
        with Path(path).open("wb") as handle:
            pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def read(self, path: str) -> Dict[str, Any]:
        with Path(path).open("rb") as handle:
            checkpoint = pickle.load(handle)
        if int(checkpoint.get("version", 0)) != CHECKPOINT_VERSION:
            raise RuntimeError(f"Unsupported checkpoint version: {checkpoint.get('version')}")
        return checkpoint

    def _rehydrate_context(self, config: Any) -> DebugContext:
        board = None
        try:
            board = DebugBoard(config_path=config.board_cfg_path)
        except Exception:
            board = None

        context = DebugContext(config=config, robot=None, camera=None, board=board, tracer=None)
        context.board_yz_calibration = load_board_yz_calibration_optional(
            getattr(config, "board_calibration_yaml", None)
        )
        context.T_CAM_BASE = {}
        for arm, attr in (
            ("left", "cam_to_robot_left_trans_path"),
            ("right", "cam_to_robot_right_trans_path"),
        ):
            path = getattr(config, attr, None)
            if path:
                try:
                    context.T_CAM_BASE[arm] = self._read_transform_file(path).as_frames(
                        from_frame="zed",
                        to_frame="base_link",
                    )
                except Exception:
                    pass
        return context

    def apply(self, checkpoint: Dict[str, Any], state: Any, runner: Any) -> None:
        payload = checkpoint["state"]
        # Checkpoints capture pipeline data, but config is intentionally refreshed
        # from the current YAML so code/config changes take effect after reload.
        config = load_debug_config()
        state.reset_runtime_data()
        state.config = config
        state.env = self._rehydrate_context(config)
        for key in STATE_KEYS:
            if key in payload:
                setattr(state, key, payload[key])
        if not hasattr(state, "step_results") or state.step_results is None:
            state.step_results = {}
        if not hasattr(state, "logs") or state.logs is None:
            state.logs = []
        if not hasattr(state, "action_feedback") or state.action_feedback is None:
            state.action_feedback = {}
        if not hasattr(state, "action_history") or state.action_history is None:
            state.action_history = []

        runner.current_idx = int(checkpoint.get("runner_current_idx", 0))

    def load(self, path: str, state: Any, runner: Any) -> Dict[str, Any]:
        checkpoint = self.read(path)
        self.apply(checkpoint, state, runner)
        return checkpoint
