from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-8)


def calculate_sequence(
    curr_clip: Dict[str, int],
    prev_clip: Dict[str, int],
    next_clip: Dict[str, int],
) -> Tuple[List[str], int]:
    """
    Determine how the route approaches and leaves the current clip.

    Returns:
        sequence: list of three symbolic directions
        clockwise_direction: -1 for clockwise, +1 for counter-clockwise
    """
    curr_x, curr_y = curr_clip["x"], curr_clip["y"]
    prev_x, prev_y = prev_clip["x"], prev_clip["y"]
    next_x, next_y = next_clip["x"], next_clip["y"]

    num2dir = {0: "up", 1: "right", 2: "down", 3: "left"}
    dir2num = {val: key for key, val in num2dir.items()}

    prev2curr = _normalize(np.array([curr_x - prev_x, -(curr_y - prev_y), 0.0]))
    curr2prev = -prev2curr
    curr2next = _normalize(np.array([next_x - curr_x, -(next_y - curr_y), 0.0]))
    is_clockwise = np.cross(prev2curr, curr2next)[-1] > 0

    net_vector = curr2prev + curr2next
    if abs(net_vector[0]) > abs(net_vector[1]):
        middle_node = dir2num["left"] if net_vector[0] > 0 else dir2num["right"]
    else:
        middle_node = dir2num["down"] if net_vector[1] > 0 else dir2num["up"]

    if is_clockwise:
        sequence = [
            num2dir[(middle_node + 1) % 4],
            num2dir[middle_node],
            num2dir[(middle_node - 1) % 4],
        ]
    else:
        sequence = [
            num2dir[(middle_node - 1) % 4],
            num2dir[middle_node],
            num2dir[(middle_node + 1) % 4],
        ]

    return sequence, -1 if is_clockwise else 1
