from __future__ import annotations

from typing import Any, Dict

from cable_orchestrator.action_types import ActionStatus

from cable_routing.debug_gui.pipeline.base_step import BaseStep

from .base_action import BasePipelineAction


class ActionStep(BaseStep):
    """
    Bridge so the existing step runner can execute actions as if they were steps.
    """

    def __init__(self, action: BasePipelineAction) -> None:
        self.action = action
        self.name = action.name
        self.description = action.description

    def run(self, state) -> Dict[str, Any]:
        result = self.action.execute(state)
        if not hasattr(state, "action_history"):
            state.action_history = []
        state.action_history.append(result)

        if result.status != ActionStatus.SUCCEEDED:
            raise RuntimeError(
                f"Action '{result.action_name}' failed"
                + (f": {result.message}" if result.message else "")
            )

        outputs = dict(result.outputs)
        outputs["action_status"] = result.status.value
        outputs["action_duration_s"] = result.duration_s
        outputs["action_message"] = result.message
        outputs["action_error_type"] = result.error_type
        return outputs
