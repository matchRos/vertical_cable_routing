from typing import Dict, Optional

import cv2
import numpy as np

from cable_core.board_projection import pixel_from_world_debug
from cable_orchestrator.base_step import BaseStep
from cable_planning.peg_route_planner import PegRoutePlanner


class PlanNextPegRouteStep(BaseStep):
    name = "plan_next_peg_route"
    description = "Plan tangent-arc-tangent routing around the next peg."

    def __init__(self):
        super().__init__()
        self.planner = PegRoutePlanner()

    def _world_to_pixel(self, world_point, arm, state) -> Optional[np.ndarray]:
        env = state.env
        if env is None:
            return None
        intrinsic = env.camera.intrinsic if env.camera is not None else None
        t_cam_base = None
        if hasattr(env, "T_CAM_BASE") and env.T_CAM_BASE and arm in env.T_CAM_BASE:
            t_cam_base = env.T_CAM_BASE[arm]
        if getattr(env, "board_yz_calibration", None) is None and (intrinsic is None or t_cam_base is None):
            return None

        uv = pixel_from_world_debug(
            env,
            state.config,
            np.asarray(world_point, dtype=float),
            arm=arm,
            intrinsic=intrinsic,
            T_cam_base=t_cam_base,
        )
        if uv is None:
            return None
        return np.array([float(uv[0]), float(uv[1])], dtype=float)

    def _clip_px(self, clip):
        return np.array([float(clip.x), float(clip.y)], dtype=float)

    def _draw_overlay(self, state, plan: Dict[str, object]):
        image = state.rgb_image
        if image is None:
            return None
        overlay = image.copy()
        arm = str(plan["arm"])
        clips = state.clips
        for clip in clips:
            p = self._clip_px(clip).astype(int)
            cv2.circle(overlay, tuple(p), 5, (170, 170, 170), -1)
            cv2.putText(
                overlay,
                str(clip.clip_id),
                (int(p[0]) + 7, int(p[1]) - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        prev_id = int(plan["prev_clip_idx"])
        curr_id = int(plan["curr_clip_idx"])
        next_raw = plan.get("next_clip_idx")
        next_id = int(next_raw) if next_raw is not None else None
        for idx, color, radius in (
            (prev_id, (255, 0, 0), 9),
            (curr_id, (255, 0, 255), 12),
        ):
            p = self._clip_px(clips[idx]).astype(int)
            cv2.circle(overlay, tuple(p), radius, color, 2)
        if next_id is not None:
            p = self._clip_px(clips[next_id]).astype(int)
            cv2.circle(overlay, tuple(p), 9, (0, 255, 0), 2)

        for peg_id in plan.get("peg_clip_indices", []) or []:
            p = self._clip_px(clips[int(peg_id)]).astype(int)
            cv2.circle(overlay, tuple(p), 15, (255, 0, 255), 1)

        points = []
        start_px = self._world_to_pixel(plan["start_position"], arm, state)
        if start_px is not None:
            points.append(start_px.astype(int))
        for p_world in plan["waypoints_world"]:
            p = self._world_to_pixel(p_world, arm, state)
            if p is not None:
                points.append(p.astype(int))

        if len(points) >= 2:
            for a, b in zip(points[:-1], points[1:]):
                cv2.line(overlay, tuple(a), tuple(b), (255, 120, 120), 2)
        for idx, p in enumerate(points):
            color = (0, 255, 255) if idx == 0 else (255, 120, 120)
            cv2.circle(overlay, tuple(p), 5, color, -1)

        continuation = []
        for p_world in plan.get("continuation_world", []) or []:
            p = self._world_to_pixel(p_world, arm, state)
            if p is not None:
                continuation.append(p.astype(int))
        if len(continuation) >= 2:
            for a, b in zip(continuation[:-1], continuation[1:]):
                cv2.line(overlay, tuple(a), tuple(b), (80, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(
                overlay,
                "next",
                (int(continuation[-1][0]) + 8, int(continuation[-1][1]) + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (80, 220, 255),
                1,
                cv2.LINE_AA,
            )

        info = (
            f"peg route {prev_id}->{','.join(str(i) for i in plan.get('peg_clip_indices', [curr_id]))}->{next_id if next_id is not None else 'end'} | arm={arm} | "
            f"{plan['side']} {plan.get('arc_direction', '')} | handover={bool(plan['needs_handover'])} | "
            f"move_aside={bool(plan['other_arm_should_move_aside'])}"
        )
        cv2.putText(overlay, info, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 120), 2, cv2.LINE_AA)
        return overlay

    def run(self, state) -> Dict[str, object]:
        plan = self.planner.plan(state)
        state.peg_route_plan = plan
        state.peg_route_overlay = self._draw_overlay(state, plan)
        state.first_route_overlay = None
        state.grasp_overlay = None
        return {
            "planned": True,
            "arm": plan["arm"],
            "route_idx": plan["route_idx"],
            "prev_clip_idx": plan["prev_clip_idx"],
            "curr_clip_idx": plan["curr_clip_idx"],
            "next_clip_idx": plan["next_clip_idx"],
            "peg_clip_indices": plan["peg_clip_indices"],
            "prev_clip_label": plan.get("prev_clip_label"),
            "curr_clip_label": plan.get("curr_clip_label"),
            "peg_clip_labels": plan.get("peg_clip_labels"),
            "terminal_clip_label": plan.get("terminal_clip_label"),
            "terminal_clip_idx": plan["terminal_clip_idx"],
            "side": plan["side"],
            "side_reasons": plan.get("side_reasons"),
            "side_vectors_2d": plan.get("side_vectors_2d"),
            "side_debug": plan.get("side_debug"),
            "arc_direction": plan["arc_direction"],
            "arc_direction_score": plan["arc_direction_score"],
            "arc_tangent_scores": plan.get("arc_tangent_scores"),
            "arc_tangent_violation_count": plan.get("arc_tangent_violation_count"),
            "min_arc_tangent_score": plan.get("min_arc_tangent_score"),
            "arc_side_flow_scores": plan.get("arc_side_flow_scores"),
            "arc_sample_counts": plan.get("arc_sample_counts"),
            "arc_spans_deg": plan.get("arc_spans_deg"),
            "side_scores": plan.get("side_scores"),
            "side_violation_count": plan.get("side_violation_count"),
            "min_side_score": plan.get("min_side_score"),
            "side_flow_violation_count": plan.get("side_flow_violation_count"),
            "min_side_flow_score": plan.get("min_side_flow_score"),
            "waypoint_count": len(plan["poses"]),
            "clearance_radius_m": plan["clearance_radius_m"],
            "reachable": plan["reachable"],
            "needs_handover": plan["needs_handover"],
            "other_arm_should_move_aside": plan["other_arm_should_move_aside"],
            "min_other_arm_distance_m": plan["min_other_arm_distance_m"],
            "overlay_updated": state.peg_route_overlay is not None,
        }
