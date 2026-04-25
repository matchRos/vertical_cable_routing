#!/usr/bin/env python3
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
legacy_root = repo_root.parent / "cable_routing"
if str(legacy_root) not in sys.path:
    sys.path.insert(0, str(legacy_root))

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from PyQt5.QtWidgets import QApplication

from cable_orchestrator.default_pipeline import build_default_orchestrator
from cable_orchestrator.pipeline_state import PipelineState
from cable_orchestrator.step_runner import StepRunner
from cable_studio.gui_controller import GuiController
from cable_studio.main_window import MainWindow


def build_runner() -> StepRunner:
    orchestrator = build_default_orchestrator()
    return StepRunner(orchestrator.build_steps())


def main() -> None:
    app = QApplication(sys.argv)
    state = PipelineState()
    runner = build_runner()
    controller = GuiController(state=state, runner=runner)
    window = MainWindow(controller=controller)
    controller.set_window(window)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
