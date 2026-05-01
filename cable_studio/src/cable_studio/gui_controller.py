from typing import Any, Dict, Optional
import traceback

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

from cable_perception.tracing_service import TracingService

from .cable_trace_io import CableTraceIO
from .checkpoint_io import StudioCheckpointIO


class GuiController:
    """
    Qt controller for the migrated studio application.
    """

    def __init__(self, state, runner) -> None:
        self.state = state
        self.runner = runner
        self.window = None
        self.trace_io = CableTraceIO()
        self.checkpoint_io = StudioCheckpointIO()
        self.tracing_service = TracingService()
        self._trace_start_mode_override = "auto_from_config"

    def set_window(self, window: Any) -> None:
        self.window = window
        self._populate_step_list()
        self._sync_trace_mode_combo_from_config()
        self._append_log("GUI controller initialized.")
        self._append_log(f"Current step: {self.runner.get_current_step_name()}")

    def _sync_trace_mode_combo_from_config(self) -> None:
        if self.window is None:
            return
        combo = self.window.trace_mode_combo
        current_mode = self._trace_start_mode_override
        if self.state.config is not None and hasattr(self.state.config, "trace_start_mode"):
            current_mode = self.state.config.trace_start_mode
        self._trace_start_mode_override = str(current_mode)
        idx = combo.findData(current_mode)
        if idx >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _apply_trace_mode_to_config_if_ready(self) -> None:
        if self.state.config is None:
            return
        if getattr(self.state.config, "trace_start_mode", None) != self._trace_start_mode_override:
            self.state.config.trace_start_mode = self._trace_start_mode_override

    def on_trace_start_mode_changed(self, _index: int) -> None:
        if self.window is None:
            return
        mode = self.window.trace_mode_combo.currentData()
        if mode is None:
            return
        self._trace_start_mode_override = str(mode)
        self._apply_trace_mode_to_config_if_ready()
        self._append_log(f"Trace start mode set to: {self._trace_start_mode_override}")

    def _populate_step_list(self) -> None:
        if self.window is not None:
            self.window.populate_step_table(self.runner.get_step_names())

    def _append_log(self, message: str) -> None:
        self.state.log(message)
        if self.window is not None:
            self.window.log_box.append(message)

    def _append_latest_action_result(self) -> None:
        history = getattr(self.state, "action_history", None)
        if not history:
            return
        action_result = history[-1]
        self._append_log(f"[action] {action_result.action_name} -> {action_result.status.value}")
        if action_result.duration_s is not None:
            self._append_log(f"[action] duration_s: {action_result.duration_s:.3f}")
        if action_result.message:
            self._append_log(f"[action] message: {action_result.message}")
        if action_result.error_type:
            self._append_log(f"[action] error_type: {action_result.error_type}")

    def _update_step_highlight(self) -> None:
        if self.window is not None:
            self.window.set_current_step(self.runner.get_current_step_name())

    def _remember_step_result(self, step_name: str, result: Dict[str, Any]) -> None:
        status_text, status_color = self._classify_step_result(step_name, result=result)
        if not hasattr(self.state, "step_results") or self.state.step_results is None:
            self.state.step_results = {}
        self.state.step_results[step_name] = {
            "status_text": status_text,
            "status_color": status_color,
            "result": result or {},
        }

    def _restore_step_results_view(self) -> None:
        if self.window is None:
            return
        self.window.clear_step_results()
        stored = getattr(self.state, "step_results", None) or {}
        for step_name, entry in stored.items():
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("status_text", "succeeded"))
            color = str(entry.get("status_color", "#cfeec2"))
            self.window.set_step_result(step_name, text, color)

        if stored:
            return
        for step_name in getattr(self.state, "finished_steps", []) or []:
            self.window.set_step_result(step_name, "succeeded", "#cfeec2")

    def _restore_log_view(self) -> None:
        if self.window is None:
            return
        self.window.log_box.clear()
        for message in getattr(self.state, "logs", []) or []:
            self.window.log_box.append(str(message))

    def _classify_step_result(
        self,
        step_name: str,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> tuple[str, str]:
        if error_message:
            return f"error: {error_message}", "#f6c7c7"
        result = result or {}
        action_status = str(result.get("action_status", "")).lower()
        action_message = str(result.get("action_message", "")).strip()
        warning_keys = ("warning", "warnings", "skipped", "retry", "fallback")
        has_warning = any(key in result for key in warning_keys)
        warning_text = ""
        for key in warning_keys:
            if key in result and result[key]:
                warning_text = str(result[key])
                break
        if action_status == "succeeded" and has_warning:
            return f"warning: {warning_text or action_message or 'completed with warning'}", "#ffd8a8"
        if action_status == "succeeded":
            return action_message or "succeeded", "#cfeec2"
        if action_status == "failed":
            return f"error: {action_message or 'failed'}", "#f6c7c7"
        return "succeeded", "#cfeec2"

    def _numpy_to_pixmap(self, image) -> Optional[QPixmap]:
        if image is None:
            return None
        if len(image.shape) != 3 or image.shape[2] != 3:
            self._append_log("Image conversion skipped: expected RGB image with shape HxWx3.")
            return None
        height, width, channels = image.shape
        qimage = QImage(image.data, width, height, channels * width, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage.copy())

    def _refresh_image_view(self) -> None:
        if self.window is None:
            return
        image = None
        if self.state.grasp_overlay is not None:
            image = self.state.grasp_overlay
        elif getattr(self.state, "peg_route_overlay", None) is not None:
            image = self.state.peg_route_overlay
        elif self.state.first_route_overlay is not None:
            image = self.state.first_route_overlay
        elif self.state.trace_overlay is not None:
            image = self.state.trace_overlay
        elif self.state.routing_overlay is not None:
            image = self.state.routing_overlay
        elif self.state.rgb_image is not None:
            image = self.state.rgb_image
        pixmap = self._numpy_to_pixmap(image)
        if pixmap is None:
            self.window.image_label.setText("No image available")
            return
        scaled = pixmap.scaled(
            self.window.image_label.width(),
            self.window.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.window.image_label.setPixmap(scaled)

    def _handle_step_result(self, step_name: str, result: Dict[str, Any]) -> None:
        self._append_log(f"Finished step: {step_name}")
        self._append_latest_action_result()
        self._remember_step_result(step_name, result)
        if self.window is not None:
            stored = self.state.step_results.get(step_name, {})
            self.window.set_step_result(
                step_name,
                stored.get("status_text", "succeeded"),
                stored.get("status_color", "#cfeec2"),
            )
        if result:
            for key, value in result.items():
                self._append_log(f"  {key}: {value}")
        self._append_log(f"Next step: {self.runner.get_current_step_name()}")
        self._update_step_highlight()
        self._refresh_image_view()

    def on_next_step(self) -> None:
        self._apply_trace_mode_to_config_if_ready()
        self._append_log(f"Running action: {self.runner.get_current_step_name()}")
        try:
            step_name, result = self.runner.run_next(self.state)
            self._handle_step_result(step_name, result)
            self._sync_trace_mode_combo_from_config()
        except Exception as exc:
            traceback.print_exc()
            self._append_latest_action_result()
            self._append_log(f"ERROR while running next step: {exc}")

    def on_auto_run_to_selected(self) -> None:
        if self.window is None:
            return
        step_name = self.window.selected_step_name()
        if not step_name:
            self._append_log("No step selected.")
            return
        step_names = self.runner.get_step_names()
        if step_name not in step_names:
            self._append_log(f"ERROR: Unknown step '{step_name}'.")
            return
        target_idx = step_names.index(step_name)
        if target_idx < self.runner.current_idx:
            self._append_log(f"Selected step '{step_name}' is behind current pointer. Resetting pipeline first.")
            self.runner.reset()
            self.state.reset_runtime_data()
        self._append_log(f"Auto-running steps up to '{step_name}'...")
        while self.runner.has_next() and self.runner.current_idx <= target_idx:
            try:
                self._apply_trace_mode_to_config_if_ready()
                self._append_log(f"Running action: {self.runner.get_current_step_name()}")
                executed_name, result = self.runner.run_next(self.state)
                self._handle_step_result(executed_name, result)
                self._sync_trace_mode_combo_from_config()
            except Exception as exc:
                traceback.print_exc()
                self._append_latest_action_result()
                if self.window is not None:
                    status_text, status_color = self._classify_step_result(
                        self.runner.get_current_step_name(),
                        error_message=str(exc),
                    )
                    self.window.set_step_result(self.runner.get_current_step_name(), status_text, status_color)
                self._append_log(f"ERROR while auto-running: {exc}")
                break

    def on_run_selected(self) -> None:
        if self.window is None:
            return
        step_name = self.window.selected_step_name()
        if not step_name:
            self._append_log("No step selected.")
            return
        self._apply_trace_mode_to_config_if_ready()
        self._append_log(f"Running selected step: {step_name}")
        try:
            executed_name, result = self.runner.run_step_by_name(self.state, step_name)
            self._handle_step_result(executed_name, result)
            self._sync_trace_mode_combo_from_config()
        except Exception as exc:
            traceback.print_exc()
            self._append_latest_action_result()
            if self.window is not None:
                status_text, status_color = self._classify_step_result(step_name, error_message=str(exc))
                self.window.set_step_result(step_name, status_text, status_color)
            self._append_log(f"ERROR while running selected step '{step_name}': {exc}")

    def on_reset(self) -> None:
        self.runner.reset()
        self.state.reset_runtime_data()
        if self.window is not None:
            self.window.clear_step_results()
        self._sync_trace_mode_combo_from_config()
        self._update_step_highlight()
        self._refresh_image_view()
        self._append_log("Pipeline reset.")

    def on_save_trace(self) -> None:
        if self.state.path_in_world is None:
            self._append_log("No cable trace available to save.")
            return
        path = self.window.ask_save_trace_path()
        if not path:
            return
        self.trace_io.save_csv(path, self.state.path_in_world)
        self._append_log(f"Saved cable trace to: {path}")

    def on_load_trace(self) -> None:
        path = self.window.ask_load_trace_path()
        if not path:
            return
        world_path = self.trace_io.load_csv(path)
        self.state.path_in_world = world_path
        self._append_log(f"Loaded cable trace from: {path}")

    def on_save_checkpoint(self) -> None:
        if self.window is None:
            return
        path = self.window.ask_save_checkpoint_path()
        if not path:
            return
        try:
            self.checkpoint_io.save(path, self.state, self.runner)
            self._append_log(f"Saved studio checkpoint to: {path}")
        except Exception as exc:
            traceback.print_exc()
            self._append_log(f"ERROR while saving checkpoint: {exc}")

    def on_load_checkpoint(self) -> None:
        if self.window is None:
            return
        path = self.window.ask_load_checkpoint_path()
        if not path:
            return
        try:
            checkpoint = self.checkpoint_io.read(path)
            saved_joints = checkpoint.get("joint_snapshot")
            current_joints = self.checkpoint_io.capture_joint_snapshot()
            readable, msg, worst = self.checkpoint_io.compare_joint_snapshot(saved_joints, current_joints)
            checkpoint_config = checkpoint.get("state", {}).get("config")
            tolerance = float(getattr(checkpoint_config, "checkpoint_joint_tolerance_rad", 0.15))
            if (not readable) or worst > tolerance:
                warning = (
                    f"{msg}\n"
                    f"Tolerance: {tolerance:.3f} rad.\n"
                    "The robot may not be in the saved physical state."
                )
                if not self.window.confirm_checkpoint_joint_mismatch(warning):
                    self._append_log("Checkpoint load cancelled after YuMi state check.")
                    return
                joint_check_message = f"Checkpoint YuMi state warning accepted: {msg}"
            else:
                joint_check_message = f"Checkpoint YuMi state check OK: {msg}"

            self.checkpoint_io.apply(checkpoint, self.state, self.runner)
            self._sync_trace_mode_combo_from_config()
            self._restore_log_view()
            self._restore_step_results_view()
            self._update_step_highlight()
            self._refresh_image_view()
            self._append_log(joint_check_message)
            self._append_log(f"Loaded studio checkpoint from: {path}")
            self._append_log("Checkpoint state loaded with current YAML config.")
            self._append_log(f"Current step: {self.runner.get_current_step_name()}")
        except Exception as exc:
            traceback.print_exc()
            self._append_log(f"ERROR while loading checkpoint: {exc}")
