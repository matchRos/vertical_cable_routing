from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from cable_core.clip_types import CLIP_TYPE_C_CLIP, CLIP_TYPE_PEG, CLIP_TYPE_U_CLIP
from cable_core.motion_primitives.c_clip import build_c_clip_entry_pixels
from cable_core.motion_primitives.u_clip import build_u_clip_entry_pixels
from cable_planning.sequence import calculate_sequence


class FirstRouteClipTargetService:
    """
    Resolve clip-type specific first-route targets from the routing triplet.
    """

    def _clip_px(self, clip: Any) -> np.ndarray:
        return np.array([float(clip.x), float(clip.y)], dtype=float)

    def _clip_to_dict(self, clip: Any) -> Dict[str, int]:
        return {
            "x": int(clip.x),
            "y": int(clip.y),
            "type": int(clip.clip_type),
            "orientation": int(clip.orientation),
        }

    def _routing_index(self, value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid routing index: {value!r}") from exc

    def _clip_at_index(self, clips: Any, idx: int, label: str):
        try:
            return clips[idx]
        except Exception as exc:
            raise RuntimeError(
                f"Failed to access {label} clip at index {idx}. "
                f"clips_type={type(clips).__name__}, "
                f"clips_len={len(clips) if hasattr(clips, '__len__') else 'n/a'}"
            ) from exc

    def _compute_secondary_support_px(
        self,
        prev_clip: Any,
        curr_clip: Any,
        clockwise_direction: int,
        img_shape,
        extension_factor: float = 50.0,
        secondary_along_prev_normal: float = 0.5,
    ):
        curr_clip_pos = self._clip_px(curr_clip)
        prev_clip_pos = self._clip_px(prev_clip)
        clip_vector = curr_clip_pos - prev_clip_pos
        norm_cv = np.linalg.norm(clip_vector)
        if norm_cv < 1e-6:
            return None

        if clockwise_direction < 0:
            normal = np.array([-clip_vector[1], clip_vector[0]], dtype=float)
        else:
            normal = np.array([clip_vector[1], -clip_vector[0]], dtype=float)

        normal = normal / np.linalg.norm(normal)
        normal_point = curr_clip_pos + normal * extension_factor

        prev_to_normal = normal_point - prev_clip_pos
        prev_to_normal = prev_to_normal / (np.linalg.norm(prev_to_normal) + 1e-8)
        clip_distance = float(np.linalg.norm(curr_clip_pos - prev_clip_pos))
        target_secondary = prev_clip_pos + prev_to_normal * (
            clip_distance * secondary_along_prev_normal
        )

        h, w = int(img_shape[0]), int(img_shape[1])
        target_secondary[0] = float(np.clip(target_secondary[0], 0, w - 1))
        target_secondary[1] = float(np.clip(target_secondary[1], 0, h - 1))
        return target_secondary

    def _compute_generic_primary_target_px(
        self,
        prev_clip: Any,
        curr_clip: Any,
        next_clip: Any,
        clockwise_direction: int,
        offset_px: float = 60.0,
    ) -> np.ndarray:
        p_prev = self._clip_px(prev_clip)
        p_curr = self._clip_px(curr_clip)
        p_next = self._clip_px(next_clip)

        direction = p_next - p_curr
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = p_curr - p_prev
            norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([1.0, 0.0], dtype=float)
            norm = 1.0

        direction = direction / norm
        perp = np.array([-direction[1], direction[0]], dtype=float)
        side_sign = -1.0 if clockwise_direction else 1.0
        return p_curr + direction * offset_px + perp * side_sign * (0.25 * offset_px)

    def _load_clip_type_config(self, state: Any, clip_type: int) -> Dict[str, Any]:
        base_dir = Path(getattr(state.config, "clip_type_config_dir", ""))
        if not base_dir.is_dir():
            return {}

        file_map = {CLIP_TYPE_U_CLIP: "u_clip.yaml"}
        name = file_map.get(int(clip_type))
        if name is None:
            return {}

        path = base_dir / name
        if not path.is_file():
            return {}

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data if isinstance(data, dict) else {}

    def plan_first_route_targets(self, state: Any, primary_arm: str) -> Dict[str, Any]:
        if state.routing is None or len(state.routing) < 3:
            raise RuntimeError("Routing must contain at least 3 clip indices.")
        if state.clips is None:
            raise RuntimeError("Clip data not available.")

        prev_clip_id = self._routing_index(state.routing[0])
        curr_clip_id = self._routing_index(state.routing[1])
        next_clip_id = self._routing_index(state.routing[2])

        clips = state.clips
        prev_clip = self._clip_at_index(clips, prev_clip_id, "prev")
        curr_clip = self._clip_at_index(clips, curr_clip_id, "curr")
        next_clip = self._clip_at_index(clips, next_clip_id, "next")

        sequence, clockwise_direction = calculate_sequence(
            self._clip_to_dict(curr_clip),
            self._clip_to_dict(prev_clip),
            self._clip_to_dict(next_clip),
        )

        clip_type = int(curr_clip.clip_type)
        clip_type_config = self._load_clip_type_config(state, clip_type)
        secondary_arm = "right" if primary_arm == "left" else "left"

        mode = "dual_slide"
        secondary_shown = True
        target_px = None
        secondary_target_px = None

        if clip_type == CLIP_TYPE_PEG:
            mode = "peg_hold"
            secondary_shown = False
            target_px = self._compute_generic_primary_target_px(
                prev_clip, curr_clip, next_clip, clockwise_direction
            )
        elif clip_type == CLIP_TYPE_C_CLIP:
            mode = "c_clip_entry"
            target_px, secondary_target_px = build_c_clip_entry_pixels(
                curr_clip=curr_clip,
                primary_arm=primary_arm,
                config=state.config,
            )
        elif clip_type == CLIP_TYPE_U_CLIP:
            mode = "u_clip_entry"
            target_px, secondary_target_px = build_u_clip_entry_pixels(
                state=state,
                curr_clip=curr_clip,
                primary_arm=primary_arm,
                secondary_arm=secondary_arm,
                clip_type_config=clip_type_config,
            )
        else:
            target_px = self._compute_generic_primary_target_px(
                prev_clip, curr_clip, next_clip, clockwise_direction
            )
            secondary_target_px = self._compute_secondary_support_px(
                prev_clip=prev_clip,
                curr_clip=curr_clip,
                clockwise_direction=int(clockwise_direction),
                img_shape=state.rgb_image.shape,
            )
            secondary_shown = secondary_target_px is not None

        route_height_m = float(
            clip_type_config.get(
                "route_height_above_plane_m",
                float(state.config.routing_height_above_plane_m),
            )
        )

        return {
            "prev_clip_id": prev_clip_id,
            "curr_clip_id": curr_clip_id,
            "next_clip_id": next_clip_id,
            "prev_clip": prev_clip,
            "curr_clip": curr_clip,
            "next_clip": next_clip,
            "clip_type": clip_type,
            "clip_type_config": clip_type_config,
            "sequence": sequence,
            "clockwise_direction": int(clockwise_direction),
            "primary_arm": primary_arm,
            "secondary_arm": secondary_arm,
            "mode": mode,
            "primary_target_px": target_px,
            "secondary_target_px": secondary_target_px,
            "secondary_shown": secondary_shown,
            "route_height_m": route_height_m,
        }
