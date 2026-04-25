from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ActionStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class ActionFeedback:
    stage: str
    message: str = ""
    progress: Optional[float] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    status: ActionStatus
    action_name: str
    message: str = ""
    outputs: Dict[str, Any] = field(default_factory=dict)
    error_type: Optional[str] = None
    started_at_s: Optional[float] = None
    finished_at_s: Optional[float] = None
    duration_s: Optional[float] = None
