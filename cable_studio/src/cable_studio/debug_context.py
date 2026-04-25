from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class DebugContext:
    config: Any
    robot: Optional[Any] = None
    camera: Optional[Any] = None
    board: Optional[Any] = None
    tracer: Optional[Any] = None
    t_cam_base_left: Optional[np.ndarray] = None
    t_cam_base_right: Optional[np.ndarray] = None
    T_CAM_BASE: Optional[dict] = None
    board_yz_calibration: Optional[Any] = None
