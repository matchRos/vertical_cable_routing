from typing import Dict, Optional

import cv2
import numpy as np

from cable_core.board_projection import pixel_from_world_debug
from cable_core.clip_types import CLIP_TYPE_PEG
from cable_orchestrator.base_step import BaseStep
from cable_planning.first_route_clip_target_service import FirstRouteClipTargetService


class PlanFirstRouteStep(BaseStep):
    name = "plan_first_route"
    description = "Prepare and visualize the first routing move around the first target clip."

    def __init__(self):
        super().__init__()
        self.clip_target_service = FirstRouteClipTargetService()

    def _clip_px(self, clip):
        return np.array([float(clip.x), float(clip.y)], dtype=float)

    def _routing_index(self, value) -> int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid routing index: {value!r}") from exc

    def _pose_for_arm(self, poses, arm_name) -> Optional[dict]:
        for pose in poses:
            if pose.get("arm") == arm_name:
                return pose
        return None

    def _world_to_pixel(self, world_point, arm, state):
        env = state.env
        if env is None:
            raise RuntimeError("Environment not available for world->pixel projection.")
        intrinsic = env.camera.intrinsic if env.camera is not None else None
        t_cam_base = None
        if hasattr(env, "T_CAM_BASE") and env.T_CAM_BASE and arm in env.T_CAM_BASE:
            t_cam_base = env.T_CAM_BASE[arm]
        if getattr(env, "board_yz_calibration", None) is None and (intrinsic is None or t_cam_base is None):
            raise RuntimeError("Camera / T_CAM_BASE not available for pinhole world->pixel projection.")

        uv = pixel_from_world_debug(
            env,
            state.config,
            np.asarray(world_point, dtype=float),
            arm=arm,
            intrinsic=intrinsic,
            T_cam_base=t_cam_base,
        )
        if uv is None:
            raise RuntimeError(f"Grasp point projects behind camera or invalid for arm '{arm}'.")
        return np.array([float(uv[0]), float(uv[1])], dtype=float)

    def _draw_overlay(
        self,
        image,
        clips,
        prev_clip_id,
        curr_clip_id,
        next_clip_id,
        clockwise_direction,
        primary_arm,
        start_px,
        target_px,
        secondary_arm=None,
        secondary_start_px=None,
        secondary_target_px=None,
        secondary_anchor_px=None,
        show_secondary=False,
        secondary_skipped_peg=False,
    ):
        overlay = image.copy()
        for clip in clips:
            p = self._clip_px(clip).astype(int)
            cv2.circle(overlay, tuple(p), 6, (180, 180, 180), -1)
            cv2.putText(overlay, str(clip.clip_id), (int(p[0]) + 8, int(p[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        prev_clip = clips[prev_clip_id]
        curr_clip = clips[curr_clip_id]
        next_clip = clips[next_clip_id]

        p_prev = self._clip_px(prev_clip).astype(int)
        p_curr = self._clip_px(curr_clip).astype(int)
        p_next = self._clip_px(next_clip).astype(int)

        cv2.line(overlay, tuple(p_prev), tuple(p_curr), (255, 255, 0), 2)
        cv2.line(overlay, tuple(p_curr), tuple(p_next), (255, 255, 0), 2)

        cv2.circle(overlay, tuple(p_prev), 10, (255, 0, 0), 2)
        cv2.circle(overlay, tuple(p_curr), 12, (255, 0, 255), 2)
        cv2.circle(overlay, tuple(p_next), 10, (0, 255, 0), 2)

        s = np.asarray(start_px, dtype=float).astype(int)
        t = np.asarray(target_px, dtype=float).astype(int)
        cv2.circle(overlay, tuple(s), 8, (0, 255, 255), -1)
        cv2.circle(overlay, tuple(t), 8, (255, 100, 100), -1)
        cv2.line(overlay, tuple(s), tuple(t), (255, 120, 120), 2)
        cv2.putText(overlay, "start", (int(s[0]) + 8, int(s[1]) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, "target", (int(t[0]) + 8, int(t[1]) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 120), 1, cv2.LINE_AA)

        if show_secondary and secondary_target_px is not None:
            st = np.asarray(secondary_target_px, dtype=float).astype(int)
            cv2.circle(overlay, tuple(st), 8, (120, 200, 255), -1)
            cv2.putText(overlay, "2nd target", (int(st[0]) + 8, int(st[1]) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 200, 255), 1, cv2.LINE_AA)
            if secondary_start_px is not None:
                ss = np.asarray(secondary_start_px, dtype=float).astype(int)
                cv2.circle(overlay, tuple(ss), 8, (180, 255, 120), -1)
                cv2.line(overlay, tuple(ss), tuple(st), (120, 220, 180), 2)
                cv2.putText(overlay, "2nd start", (int(ss[0]) + 8, int(ss[1]) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 120), 1, cv2.LINE_AA)
            elif secondary_anchor_px is not None:
                sa = np.asarray(secondary_anchor_px, dtype=float).astype(int)
                cv2.circle(overlay, tuple(sa), 6, (120, 220, 180), 2)
                cv2.line(overlay, tuple(sa), tuple(st), (120, 220, 180), 1)
                cv2.putText(overlay, "2nd plan", (int(sa[0]) + 8, int(sa[1]) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 220, 180), 1, cv2.LINE_AA)

        direction_text = "CW" if clockwise_direction < 0 else "CCW"
        if show_secondary and secondary_arm:
            sec_note = f" | 2nd={secondary_arm}"
        elif secondary_skipped_peg:
            sec_note = " | 2nd=— (peg)"
        else:
            sec_note = " | 2nd=—"
        info_text = f"{prev_clip_id} -> {curr_clip_id} -> {next_clip_id} | {direction_text} | arm={primary_arm}{sec_note}"
        cv2.putText(overlay, info_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2, cv2.LINE_AA)
        return overlay

    def run(self, state) -> Dict[str, object]:
        if state.routing is None or len(state.routing) < 3:
            raise RuntimeError("Routing must contain at least 3 clip indices.")
        if state.clips is None:
            raise RuntimeError("Clip data not available.")
        if state.rgb_image is None:
            raise RuntimeError("No RGB image available for visualization.")
        if not hasattr(state, "grasp_poses"):
            raise RuntimeError("No grasp poses available.")

        prev_clip_id = self._routing_index(state.routing[0])
        curr_clip_id = self._routing_index(state.routing[1])
        next_clip_id = self._routing_index(state.routing[2])

        clips = state.clips
        curr_clip = clips[curr_clip_id]
        primary_arm = getattr(state, "descend_first_arm", None) or "left"

        try:
            route_plan = self.clip_target_service.plan_first_route_targets(state=state, primary_arm=primary_arm)
            sequence = route_plan["sequence"]
            clockwise_direction = route_plan["clockwise_direction"]
            primary_pose = self._pose_for_arm(state.grasp_poses, primary_arm)
            if primary_pose is None:
                raise RuntimeError(f"No grasp pose found for primary arm '{primary_arm}'.")
            start_px = self._world_to_pixel(primary_pose["position"], primary_arm, state)
            target_px = np.asarray(route_plan["primary_target_px"], dtype=float).reshape(2)

            secondary_arm = route_plan["secondary_arm"]
            secondary_skipped_peg = curr_clip.clip_type == CLIP_TYPE_PEG
            show_secondary = bool(route_plan["secondary_shown"])
            secondary_start_px = None
            secondary_target_px = None
            secondary_anchor_px = self._clip_px(curr_clip)
            if show_secondary:
                secondary_pose = self._pose_for_arm(state.grasp_poses, secondary_arm)
                if secondary_pose is None:
                    secondary_target_px = route_plan["secondary_target_px"]
                    if secondary_target_px is None:
                        show_secondary = False
                else:
                    secondary_start_px = self._world_to_pixel(secondary_pose["position"], secondary_arm, state)
                    secondary_target_px = route_plan["secondary_target_px"]
                    if secondary_target_px is None:
                        show_secondary = False
        except Exception as exc:
            raise RuntimeError(
                "plan_first_route failed after clip selection: "
                f"routing={list(state.routing)}, "
                f"clip_indices=({prev_clip_id}, {curr_clip_id}, {next_clip_id}), "
                f"clips_type={type(clips).__name__}, "
                f"clips_len={len(clips) if hasattr(clips, '__len__') else 'n/a'}, "
                f"primary_arm={primary_arm}"
            ) from exc

        state.current_primary_arm = primary_arm
        state.first_route_prev_clip_id = prev_clip_id
        state.first_route_curr_clip_id = curr_clip_id
        state.first_route_next_clip_id = next_clip_id
        state.first_route_clockwise = clockwise_direction
        state.first_route_sequence = sequence
        state.first_route_start_px = start_px
        state.first_route_target_px = target_px
        state.first_route_mode = route_plan["mode"]
        state.first_route_route_height_m = route_plan["route_height_m"]
        state.first_route_clip_type_config = route_plan["clip_type_config"]
        state.first_route_secondary_arm = secondary_arm if show_secondary else None
        state.first_route_secondary_start_px = secondary_start_px
        state.first_route_secondary_target_px = secondary_target_px
        state.first_route_secondary_shown = show_secondary

        overlay = self._draw_overlay(
            image=state.rgb_image,
            clips=clips,
            prev_clip_id=prev_clip_id,
            curr_clip_id=curr_clip_id,
            next_clip_id=next_clip_id,
            clockwise_direction=clockwise_direction,
            primary_arm=primary_arm,
            start_px=start_px,
            target_px=target_px,
            secondary_arm=secondary_arm,
            secondary_start_px=secondary_start_px,
            secondary_target_px=secondary_target_px,
            secondary_anchor_px=secondary_anchor_px,
            show_secondary=show_secondary,
            secondary_skipped_peg=secondary_skipped_peg,
        )

        state.routing_overlay = overlay
        state.first_route_overlay = overlay
        state.grasp_overlay = None
        return {
            "route_ready": True,
            "overlay_updated": True,
            "primary_arm": primary_arm,
            "prev_clip_id": prev_clip_id,
            "curr_clip_id": curr_clip_id,
            "next_clip_id": next_clip_id,
            "clockwise_direction": clockwise_direction,
            "sequence": sequence,
            "mode": route_plan["mode"],
            "curr_clip_type": curr_clip.clip_type,
            "curr_clip_orientation": curr_clip.orientation,
            "start_px": [float(start_px[0]), float(start_px[1])],
            "target_px": [float(target_px[0]), float(target_px[1])],
            "secondary_arm": secondary_arm if show_secondary else None,
            "secondary_shown": show_secondary,
            "secondary_start_px": ([float(secondary_start_px[0]), float(secondary_start_px[1])] if show_secondary and secondary_start_px is not None else None),
            "secondary_target_px": ([float(secondary_target_px[0]), float(secondary_target_px[1])] if show_secondary and secondary_target_px is not None else None),
        }
