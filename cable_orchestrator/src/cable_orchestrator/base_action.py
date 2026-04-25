from __future__ import annotations

from abc import ABC, abstractmethod

from cable_orchestrator.action_types import ActionFeedback, ActionResult


class BasePipelineAction(ABC):
    name = "unnamed_action"
    description = "No description provided."

    def emit_feedback(self, state, feedback: ActionFeedback) -> None:
        if not hasattr(state, "action_feedback"):
            state.action_feedback = {}
        state.action_feedback[self.name] = feedback
        state.log(
            f"[action:{self.name}] stage={feedback.stage}"
            + (f" msg={feedback.message}" if feedback.message else "")
        )

    @abstractmethod
    def execute(self, state) -> ActionResult:
        raise NotImplementedError
