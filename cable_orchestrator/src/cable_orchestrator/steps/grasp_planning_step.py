from typing import Dict

import numpy as np

from cable_core.board_projection import world_from_pixel_debug
from cable_orchestrator.base_step import BaseStep


def _first_routing_clip_world_xy(state, clip_index: int) -> np.ndarray:
    if state.env is None:
        raise RuntimeError("Environment not available for peg world position.")
    clips = state.clips
    if clips is None:
        raise RuntimeError("clips not available.")
    clip = clips[clip_index]
    img_shape = state.rgb_image.shape if state.rgb_image is not None else None
    return world_from_pixel_debug(
        state.env,
        state.config,
        (float(clip.x), float(clip.y)),
        arm="right",
        is_clip=True,
        image_shape=img_shape,
    ).reshape(3)


def _first_path_index_clear_of_anchor(
    path_world: np.ndarray,
    anchor_world: np.ndarray,
    min_dist_m: float,
) -> int:
    for idx in range(len(path_world)):
        if float(np.linalg.norm(path_world[idx] - anchor_world)) >= min_dist_m:
            return idx
    return int(len(path_world) - 1)


def _path_index_after_arc_length(
    path_world: np.ndarray,
    start_idx: int,
    min_arc_m: float,
) -> int:
    if start_idx >= len(path_world) - 1:
        return int(len(path_world) - 1)
    arc = 0.0
    for idx in range(start_idx, len(path_world) - 1):
        arc += float(np.linalg.norm(path_world[idx + 1] - path_world[idx]))
        if arc >= min_arc_m:
            return int(idx + 1)
    return int(len(path_world) - 1)


class GraspPlanningStep(BaseStep):
    name = "grasp_planning"
    description = "Choose grasp point(s) along the traced cable in world frame."

    def run(self, state) -> Dict[str, object]:
        if state.path_in_world is None:
            raise RuntimeError("No world path available.")
        if not hasattr(state, "path_tangents"):
            raise RuntimeError("No tangents available.")
        if state.path_in_pixels is None:
            raise RuntimeError("No pixel path available.")
        if state.routing is None or len(state.routing) < 2:
            raise RuntimeError("Routing not available or too short.")

        path_w = np.asarray(state.path_in_world, dtype=float)
        peg_id = int(state.routing[0])
        peg_world = _first_routing_clip_world_xy(state, peg_id)
        min_clear = float(getattr(state.config, "grasp_min_clearance_from_first_peg_m", 0.05))
        grasp_idx1 = _first_path_index_clear_of_anchor(path_w, peg_world, min_clear)

        dual = bool(getattr(state.config, "dual_arm_grasp", True))
        if dual:
            second_arc = float(getattr(state.config, "grasp_second_min_arc_from_first_grasp_m", 0.08))
            grasp_idx2 = _path_index_after_arc_length(path_w, grasp_idx1, second_arc)
            if grasp_idx2 <= grasp_idx1 and len(path_w) > grasp_idx1 + 1:
                grasp_idx2 = min(len(path_w) - 1, grasp_idx1 + 1)
            grasps = [
                {"position": path_w[grasp_idx1], "tangent": state.path_tangents[grasp_idx1], "index": grasp_idx1},
                {"position": path_w[grasp_idx2], "tangent": state.path_tangents[grasp_idx2], "index": grasp_idx2},
            ]
            print(
                f"Peg clip index {peg_id} world (for clearance): {peg_world.tolist()}, "
                f"min_clearance_m={min_clear}, grasp1_idx={grasp_idx1}, grasp2_idx={grasp_idx2}"
            )
        else:
            grasps = [
                {"position": path_w[grasp_idx1], "tangent": state.path_tangents[grasp_idx1], "index": grasp_idx1}
            ]
            print(
                f"Peg clip index {peg_id} world (for clearance): {peg_world.tolist()}, "
                f"min_clearance_m={min_clear}, single_grasp_idx={grasp_idx1}"
            )

        state.grasps = grasps
        out = {
            "grasps_available": True,
            "num_grasps": len(grasps),
            "grasp_indices": [g["index"] for g in grasps],
            "peg_world_for_grasp_clearance": peg_world.tolist(),
            "first_grasp": {
                "position": grasps[0]["position"].tolist(),
                "tangent": grasps[0]["tangent"].tolist(),
            },
        }
        if len(grasps) > 1:
            out["second_grasp"] = {
                "position": grasps[1]["position"].tolist(),
                "tangent": grasps[1]["tangent"].tolist(),
            }
        return out
