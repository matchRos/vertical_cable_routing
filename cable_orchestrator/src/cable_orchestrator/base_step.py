from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseStep(ABC):
    """
    Abstract base class for one step in the pipeline.
    """

    name = "unnamed_step"
    description = "No description provided."

    @abstractmethod
    def run(self, state) -> Dict[str, Any]:
        raise NotImplementedError
