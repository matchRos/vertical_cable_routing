from __future__ import annotations

from typing import Any, Dict

import numpy as np


def is_dual_arm_grasp(config: Any) -> bool:
    return bool(getattr(config, "dual_arm_grasp", True))


def compute_pose_distance(pose_a: Dict[str, Any], pose_b: Dict[str, Any]) -> float:
    pos_a = np.asarray(pose_a["position"], dtype=float).reshape(3)
    pos_b = np.asarray(pose_b["position"], dtype=float).reshape(3)
    return float(np.linalg.norm(pos_a - pos_b))


def validate_min_distance(
    left_pose: Dict[str, Any],
    right_pose: Dict[str, Any],
    min_dist_xyz: float,
    label: str = "Poses",
) -> float:
    dist_xyz = compute_pose_distance(left_pose, right_pose)
    if dist_xyz < min_dist_xyz:
        raise RuntimeError(
            f"{label} too close: distance={dist_xyz:.3f} m < {min_dist_xyz:.3f} m"
        )
    return dist_xyz
