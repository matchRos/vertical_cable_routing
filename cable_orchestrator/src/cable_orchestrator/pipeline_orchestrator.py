from __future__ import annotations

from typing import Iterable, List

from cable_orchestrator.base_step import BaseStep

from .action_step import ActionStep
from .base_action import BasePipelineAction


class PipelineOrchestrator:
    def __init__(self, actions: Iterable[BasePipelineAction]) -> None:
        self.actions = list(actions)

    def get_action_names(self) -> List[str]:
        return [action.name for action in self.actions]

    def build_steps(self) -> List[BaseStep]:
        return [ActionStep(action) for action in self.actions]
