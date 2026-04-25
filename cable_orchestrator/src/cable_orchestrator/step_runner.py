from typing import Any, Dict, List, Tuple


class StepRunner:
    """
    Executes configured steps one-by-one.
    """

    def __init__(self, steps: List[Any]) -> None:
        self.steps = steps
        self.current_idx = 0

    def reset(self) -> None:
        self.current_idx = 0

    def has_next(self) -> bool:
        return self.current_idx < len(self.steps)

    def get_step_names(self) -> List[str]:
        return [step.name for step in self.steps]

    def get_current_step_name(self) -> str:
        if not self.has_next():
            return "finished"
        return self.steps[self.current_idx].name

    def run_next(self, state) -> Tuple[str, Dict[str, Any]]:
        if not self.has_next():
            raise RuntimeError("No more steps available.")
        step = self.steps[self.current_idx]
        result = step.run(state)
        state.finished_steps.append(step.name)
        self.current_idx += 1
        return step.name, result

    def run_step_by_name(self, state, step_name: str) -> Tuple[str, Dict[str, Any]]:
        for idx, step in enumerate(self.steps):
            if step.name == step_name:
                result = step.run(state)
                if step.name not in state.finished_steps:
                    state.finished_steps.append(step.name)
                if idx >= self.current_idx:
                    self.current_idx = idx + 1
                return step.name, result
        raise ValueError(f"Unknown step name: {step_name}")

    def set_pointer_to_step_name(self, step_name: str) -> None:
        for idx, step in enumerate(self.steps):
            if step.name == step_name:
                self.current_idx = idx
                return
        raise ValueError(f"Unknown step name: {step_name}")
