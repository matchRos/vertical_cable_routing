import numpy as np


class PreGraspPoseService:
    def compute_pregrasp_poses(
        self,
        grasp_poses,
        pregrasp_offset_from_grasp_m: float = 0.08,
    ):
        """
        Pre-grasp is the same pose as grasp, retracted along -X only.
        """
        pregrasp_poses = []
        dx = float(pregrasp_offset_from_grasp_m)

        for pose in grasp_poses:
            pos = np.asarray(pose["position"], dtype=float).reshape(3).copy()
            rot = np.asarray(pose["rotation"], dtype=float)
            pos[0] -= dx
            pregrasp_poses.append(
                {
                    "position": pos,
                    "rotation": rot.copy(),
                    "arm": pose.get("arm", "right"),
                    "path_index": int(pose["path_index"]),
                }
            )

        return pregrasp_poses
