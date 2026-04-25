from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cable_core.board_projection import world_from_pixel_debug
from cable_core.clip_types import CLIP_TYPE_PEG
from cable_core.planes import get_routing_plane, point_at_plane_height


def _normalize(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        raise RuntimeError("Cannot normalize near-zero vector.")
    return arr / norm


def _clip_center_world(state: Any, clip_idx: int, arm: str) -> np.ndarray:
    clip = state.clips[int(clip_idx)]
    return world_from_pixel_debug(
        state.env,
        state.config,
        (float(clip.x), float(clip.y)),
        arm=arm,
        is_clip=True,
        image_shape=state.rgb_image.shape,
    ).reshape(3)


def _plane_coords(point: np.ndarray, center: np.ndarray, axis_x: np.ndarray, axis_y: np.ndarray) -> np.ndarray:
    delta = np.asarray(point, dtype=float).reshape(3) - np.asarray(center, dtype=float).reshape(3)
    return np.array([float(np.dot(delta, axis_x)), float(np.dot(delta, axis_y))], dtype=float)


def _from_plane_coords(coords: np.ndarray, center: np.ndarray, axis_x: np.ndarray, axis_y: np.ndarray) -> np.ndarray:
    c = np.asarray(coords, dtype=float).reshape(2)
    return np.asarray(center, dtype=float).reshape(3) + c[0] * axis_x + c[1] * axis_y


def _tangent_points_from_external(point_2d: np.ndarray, radius: float) -> List[np.ndarray]:
    p = np.asarray(point_2d, dtype=float).reshape(2)
    dist = float(np.linalg.norm(p))
    if dist <= radius + 1e-6:
        raise RuntimeError(
            f"Cannot build tangent to peg clearance circle: point is inside radius "
            f"(dist={dist:.3f}, radius={radius:.3f})."
        )
    base = float(np.arctan2(p[1], p[0]))
    delta = float(np.arccos(radius / dist))
    return [
        radius * np.array([np.cos(base + delta), np.sin(base + delta)], dtype=float),
        radius * np.array([np.cos(base - delta), np.sin(base - delta)], dtype=float),
    ]


def _common_tangents_equal_radius(
    center_a_2d: np.ndarray,
    center_b_2d: np.ndarray,
    radius: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    a = np.asarray(center_a_2d, dtype=float).reshape(2)
    b = np.asarray(center_b_2d, dtype=float).reshape(2)
    delta = b - a
    distance = float(np.linalg.norm(delta))
    if distance < 1e-9:
        raise RuntimeError("Cannot build tangents between coincident peg centers.")
    direction = delta / distance
    normal = np.array([-direction[1], direction[0]], dtype=float)
    tangents = [
        (a + float(sign) * float(radius) * normal, b + float(sign) * float(radius) * normal)
        for sign in (1.0, -1.0)
    ]
    if distance > 2.0 * float(radius) + 1e-6:
        cos_theta = np.clip(2.0 * float(radius) / distance, -1.0, 1.0)
        sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
        for sign in (1.0, -1.0):
            circle_normal = cos_theta * direction + float(sign) * sin_theta * normal
            tangents.append((a + float(radius) * circle_normal, b - float(radius) * circle_normal))
    return tangents


def _choose_tangent_by_side(candidates: List[np.ndarray], side_axis_2d: np.ndarray, side_sign: float) -> np.ndarray:
    side = _normalize(side_axis_2d)
    scored = [
        (float(side_sign) * float(np.dot(c, side)), idx, c)
        for idx, c in enumerate(candidates)
    ]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def _angle_of(point_2d: np.ndarray) -> float:
    p = np.asarray(point_2d, dtype=float).reshape(2)
    return float(np.arctan2(p[1], p[0]))


def _circle_tangent(theta: float, direction: str) -> np.ndarray:
    radial = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    if direction == "ccw":
        return np.array([-radial[1], radial[0]], dtype=float)
    return np.array([radial[1], -radial[0]], dtype=float)


def _arc_angles(
    theta_start: float,
    theta_end: float,
    incoming_direction_2d: np.ndarray,
    outgoing_direction_2d: np.ndarray,
    samples: int,
    preferred_side_2d: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, str, float]:
    samples = max(int(samples), 2)
    two_pi = 2.0 * np.pi
    incoming = _normalize(incoming_direction_2d)
    outgoing = _normalize(outgoing_direction_2d)
    preferred_side = _normalize(preferred_side_2d) if preferred_side_2d is not None else None
    candidates = []
    for direction, delta in (
        ("ccw", (theta_end - theta_start) % two_pi),
        ("cw", -((theta_start - theta_end) % two_pi)),
    ):
        angles = theta_start + np.linspace(0.0, delta, samples)
        start_tangent = _circle_tangent(theta_start, direction)
        end_tangent = _circle_tangent(theta_end, direction)
        tangent_score = (
            2.0 * float(np.dot(start_tangent, incoming))
            + float(np.dot(end_tangent, outgoing))
        )
        score = tangent_score - 0.05 * abs(float(delta))
        if preferred_side is not None:
            side_score = max(
                float(
                    np.dot(
                        np.array([np.cos(float(angle)), np.sin(float(angle))], dtype=float),
                        preferred_side,
                    )
                )
                for angle in angles
            )
            score += 4.0 * side_score
            if side_score < -0.02:
                score -= 10.0
        candidates.append((score, direction, angles))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2], candidates[0][1], float(candidates[0][0])


def _distance_from_line(point_2d: np.ndarray, line_a_2d: np.ndarray, line_b_2d: np.ndarray) -> float:
    point = np.asarray(point_2d, dtype=float).reshape(2)
    line_a = np.asarray(line_a_2d, dtype=float).reshape(2)
    line_b = np.asarray(line_b_2d, dtype=float).reshape(2)
    line = line_b - line_a
    line_len = float(np.linalg.norm(line))
    if line_len < 1e-9:
        return float("inf")
    return float(abs(np.cross(line, point - line_a)) / line_len)


def _side_label(side_2d: np.ndarray) -> str:
    side = _normalize(side_2d)
    x = float(side[0])
    y = float(side[1])
    if abs(y) >= abs(x) * 1.25:
        return "up" if y > 0.0 else "down"
    if abs(x) >= abs(y) * 1.25:
        return "right" if x < 0.0 else "left"
    vertical = "up" if y > 0.0 else "down"
    lateral = "right" if x < 0.0 else "left"
    return f"{vertical}-{lateral}"


def _side_away_from_successor(
    center_2d: np.ndarray,
    successor_2d: np.ndarray,
    lateral_tol_m: float,
) -> np.ndarray:
    center = np.asarray(center_2d, dtype=float).reshape(2)
    successor = np.asarray(successor_2d, dtype=float).reshape(2)
    dx = float(successor[0] - center[0])
    if abs(dx) > float(lateral_tol_m):
        return np.array([-np.sign(dx), 0.0], dtype=float)
    return -_normalize(successor - center)


def _choose_collinear_side(
    prev_2d: np.ndarray,
    next_2d: np.ndarray,
    previous_side_2d: Optional[np.ndarray],
) -> np.ndarray:
    line = _normalize(np.asarray(next_2d, dtype=float).reshape(2) - np.asarray(prev_2d, dtype=float).reshape(2))
    normal = np.array([-line[1], line[0]], dtype=float)
    candidates = [normal, -normal]
    if previous_side_2d is not None:
        previous = _normalize(previous_side_2d)
        candidates.sort(key=lambda cand: float(np.dot(cand, previous)))
        return candidates[0]
    candidates.sort(key=lambda cand: (float(cand[1]), -abs(float(cand[0]))), reverse=True)
    return candidates[0]


def _preferred_side_for_peg(
    prev_2d: np.ndarray,
    center_2d: np.ndarray,
    next_2d: np.ndarray,
    previous_side_2d: Optional[np.ndarray],
    successor_is_real_peg: bool,
    collinear_tol_m: float,
    height_tol_m: float,
    lateral_tol_m: float,
) -> Tuple[np.ndarray, str]:
    prev = np.asarray(prev_2d, dtype=float).reshape(2)
    center = np.asarray(center_2d, dtype=float).reshape(2)
    nxt = np.asarray(next_2d, dtype=float).reshape(2)

    if successor_is_real_peg and _distance_from_line(center, prev, nxt) <= float(collinear_tol_m):
        side = _choose_collinear_side(prev, nxt, previous_side_2d)
        return side, "collinear"

    center_h = float(center[1])
    if center_h > max(float(prev[1]), float(nxt[1])) + float(height_tol_m):
        return np.array([0.0, 1.0], dtype=float), "height_extreme"
    if center_h < min(float(prev[1]), float(nxt[1])) - float(height_tol_m):
        return np.array([0.0, -1.0], dtype=float), "height_extreme"

    return _side_away_from_successor(center, nxt, lateral_tol_m), "away_from_successor"


def _rotation_from_pull_direction(
    plane_normal: np.ndarray,
    pull_direction: np.ndarray,
    pull_side_sign: float,
) -> np.ndarray:
    tool_z = -_normalize(plane_normal)
    pull = np.asarray(pull_direction, dtype=float).reshape(3)
    pull = pull - float(np.dot(pull, tool_z)) * tool_z
    pull = _normalize(pull)
    tool_y = pull * float(pull_side_sign)
    tool_x = _normalize(np.cross(tool_y, tool_z))
    tool_z = _normalize(np.cross(tool_x, tool_y))
    return np.stack([tool_x, tool_y, tool_z], axis=1)


def _arm_reaches_y(arm: str, target_y: float, cross_y_m: float) -> bool:
    if arm == "right":
        return float(target_y) <= float(cross_y_m)
    return float(target_y) >= -float(cross_y_m)


class PegRoutePlanner:
    def _peg_chain_from_route(self, state: Any, route_idx: int) -> Tuple[List[int], Optional[int], int]:
        routing = [int(v) for v in state.routing]
        peg_clip_indices: List[int] = []
        scan_idx = int(route_idx)
        while scan_idx < len(routing):
            clip_idx = int(routing[scan_idx])
            if int(state.clips[clip_idx].clip_type) != CLIP_TYPE_PEG:
                break
            peg_clip_indices.append(clip_idx)
            scan_idx += 1
        terminal_clip_idx = int(routing[scan_idx]) if scan_idx < len(routing) else None
        return peg_clip_indices, terminal_clip_idx, scan_idx

    def plan(self, state: Any) -> Dict[str, Any]:
        if state.env is None:
            raise RuntimeError("Environment not available.")
        if state.rgb_image is None:
            raise RuntimeError("No RGB image available.")
        if state.routing is None or len(state.routing) < 2:
            raise RuntimeError("Routing must contain at least 2 clips for next peg routing.")
        if state.clips is None:
            raise RuntimeError("Clip data not available.")

        routing = [int(v) for v in state.routing]
        route_idx = getattr(state, "next_route_routing_index", None)
        if route_idx is None:
            next_clip_id = getattr(state, "first_route_next_clip_id", None)
            route_idx = routing.index(int(next_clip_id)) if next_clip_id is not None else 2
        route_idx = int(route_idx)
        if route_idx <= 0 or route_idx >= len(routing):
            raise RuntimeError(f"Invalid next_route_routing_index={route_idx} for routing length {len(routing)}.")

        prev_clip_idx = routing[route_idx - 1]
        peg_clip_indices, terminal_clip_idx, route_after_idx = self._peg_chain_from_route(state, route_idx)
        if not peg_clip_indices:
            raise RuntimeError(f"Next routing clip {routing[route_idx]} is not a peg.")
        curr_clip_idx = peg_clip_indices[0]

        arm = getattr(state, "descend_second_arm", None) or getattr(state, "first_route_secondary_arm", None)
        if arm not in ("left", "right"):
            arm = "right" if getattr(state, "current_primary_arm", "left") == "left" else "left"
        other_arm = "right" if arm == "left" else "left"

        current_poses = getattr(state, "current_arm_poses_world", None) or {}
        if arm not in current_poses:
            raise RuntimeError(f"No current pose stored for cable-end arm '{arm}'. Run execute_first_route first.")
        start_pos = np.asarray(current_poses[arm]["position"], dtype=float).reshape(3)
        start_rot = np.asarray(current_poses[arm]["rotation"], dtype=float).reshape(3, 3)

        plane = get_routing_plane(state.config, clip_id=curr_clip_idx)
        route_height = float(
            getattr(
                state,
                "first_route_route_height_m",
                float(state.config.routing_height_above_plane_m),
            )
        )
        radius = float(getattr(state.config, "peg_route_clearance_radius_m", 0.035))
        samples = int(getattr(state.config, "peg_route_arc_samples", 10))
        cross_y = float(getattr(state.config, "peg_route_workspace_cross_y_m", 0.2))
        collinear_tol = float(getattr(state.config, "peg_route_collinear_tolerance_m", 0.02))
        height_tol = float(getattr(state.config, "peg_route_height_tolerance_m", 0.015))
        lateral_tol = float(getattr(state.config, "peg_route_lateral_tolerance_m", 0.01))

        prev_world = point_at_plane_height(_clip_center_world(state, prev_clip_idx, arm), plane, route_height)
        peg_centers_world = [
            point_at_plane_height(_clip_center_world(state, clip_idx, arm), plane, route_height)
            for clip_idx in peg_clip_indices
        ]
        curr_world = peg_centers_world[0]
        if terminal_clip_idx is not None:
            terminal_world = point_at_plane_height(_clip_center_world(state, terminal_clip_idx, arm), plane, route_height)
        elif len(peg_centers_world) >= 2:
            terminal_world = peg_centers_world[-1] + _normalize(peg_centers_world[-1] - peg_centers_world[-2]) * max(radius * 2.0, 0.08)
        else:
            terminal_world = peg_centers_world[-1] + _normalize(peg_centers_world[-1] - prev_world) * max(radius * 2.0, 0.08)
        start_world = point_at_plane_height(start_pos, plane, route_height)

        reachable = all(_arm_reaches_y(arm, float(center[1]), cross_y) for center in peg_centers_world)
        needs_handover = not reachable

        axis_x = _normalize(plane.v_axis)
        axis_y = _normalize(plane.u_axis)
        plane_origin = np.asarray(plane.origin, dtype=float).reshape(3)

        start_2d = _plane_coords(start_world, plane_origin, axis_x, axis_y)
        prev_clip_2d = _plane_coords(prev_world, plane_origin, axis_x, axis_y)
        terminal_2d = _plane_coords(terminal_world, plane_origin, axis_x, axis_y)
        peg_centers_2d = [_plane_coords(center, plane_origin, axis_x, axis_y) for center in peg_centers_world]
        preferred_sides_2d: List[np.ndarray] = []
        side_names: List[str] = []
        side_reasons: List[str] = []
        side_debug: List[Dict[str, Any]] = []
        previous_side_2d: Optional[np.ndarray] = None
        for idx, center_2d in enumerate(peg_centers_2d):
            prev_2d = prev_clip_2d if idx == 0 else peg_centers_2d[idx - 1]
            prev_rule_clip_idx = prev_clip_idx if idx == 0 else peg_clip_indices[idx - 1]
            successor_route_idx = int(route_idx) + idx + 1
            successor_is_real_peg = False
            successor_clip_idx = None
            if successor_route_idx < len(routing):
                successor_clip_idx = int(routing[successor_route_idx])
                successor_is_real_peg = int(state.clips[successor_clip_idx].clip_type) == CLIP_TYPE_PEG
                successor_world = point_at_plane_height(
                    _clip_center_world(state, successor_clip_idx, arm),
                    plane,
                    route_height,
                )
            else:
                successor_world = terminal_world
            successor_2d = _plane_coords(successor_world, plane_origin, axis_x, axis_y)
            preferred_side_2d, side_reason = _preferred_side_for_peg(
                prev_2d,
                center_2d,
                successor_2d,
                previous_side_2d,
                successor_is_real_peg,
                collinear_tol,
                height_tol,
                lateral_tol,
            )
            preferred_sides_2d.append(preferred_side_2d)
            side_names.append(_side_label(preferred_side_2d))
            side_reasons.append(side_reason)
            side_debug.append(
                {
                    "prev_clip": str(state.clips[int(prev_rule_clip_idx)].clip_id),
                    "curr_clip": str(state.clips[int(peg_clip_indices[idx])].clip_id),
                    "next_clip": (
                        str(state.clips[int(successor_clip_idx)].clip_id)
                        if successor_clip_idx is not None
                        else "terminal"
                    ),
                    "prev_2d": prev_2d.tolist(),
                    "curr_2d": center_2d.tolist(),
                    "next_2d": successor_2d.tolist(),
                    "side": side_names[-1],
                    "side_vector_2d": preferred_side_2d.tolist(),
                    "reason": side_reason,
                }
            )
            previous_side_2d = preferred_side_2d

        start_rel = start_2d - peg_centers_2d[0]
        start_options = [
            {"entry": peg_centers_2d[0] + rel}
            for rel in _tangent_points_from_external(start_rel, radius)
        ]

        bridge_options: List[List[Dict[str, np.ndarray]]] = []
        for idx in range(len(peg_centers_2d) - 1):
            bridge_options.append(
                [
                    {"exit": exit_point, "entry": entry_point}
                    for exit_point, entry_point in _common_tangents_equal_radius(
                        peg_centers_2d[idx],
                        peg_centers_2d[idx + 1],
                        radius,
                    )
                ]
            )

        terminal_rel = terminal_2d - peg_centers_2d[-1]
        terminal_options = [
            {"exit": peg_centers_2d[-1] + rel}
            for rel in _tangent_points_from_external(terminal_rel, radius)
        ]

        option_groups = [start_options] + bridge_options + [terminal_options]
        best = None
        for combo in product(*option_groups):
            entries: List[np.ndarray] = [np.asarray(combo[0]["entry"], dtype=float)]
            exits: List[np.ndarray] = [np.zeros(2, dtype=float) for _ in peg_centers_2d]
            for bridge_idx in range(len(bridge_options)):
                bridge = combo[1 + bridge_idx]
                exits[bridge_idx] = np.asarray(bridge["exit"], dtype=float)
                entries.append(np.asarray(bridge["entry"], dtype=float))
            exits[-1] = np.asarray(combo[-1]["exit"], dtype=float)

            total_score = 0.0
            route_positions_2d: List[np.ndarray] = []
            arc_directions: List[str] = []
            arc_scores: List[float] = []
            side_scores: List[float] = []
            for peg_i, center_2d in enumerate(peg_centers_2d):
                prev_point = start_2d if peg_i == 0 else exits[peg_i - 1]
                next_point = terminal_2d if peg_i == len(peg_centers_2d) - 1 else entries[peg_i + 1]
                entry = entries[peg_i]
                exit_point = exits[peg_i]
                incoming = entry - prev_point
                outgoing = next_point - exit_point
                entry_rel = entry - center_2d
                exit_rel = exit_point - center_2d
                angles, direction, arc_score = _arc_angles(
                    _angle_of(entry_rel),
                    _angle_of(exit_rel),
                    incoming,
                    outgoing,
                    samples,
                    preferred_sides_2d[peg_i],
                )
                preferred_side = _normalize(preferred_sides_2d[peg_i])
                side_score = max(
                    float(
                        np.dot(
                            np.array([np.cos(float(angle)), np.sin(float(angle))], dtype=float),
                            preferred_side,
                        )
                    )
                    for angle in angles
                )
                side_scores.append(float(side_score))
                if side_score < -0.02:
                    total_score -= 10.0
                total_score += arc_score + 5.0 * side_score
                arc_directions.append(direction)
                arc_scores.append(float(arc_score))

                arc_points = [
                    center_2d + radius * np.array([np.cos(a), np.sin(a)], dtype=float)
                    for a in angles
                ]
                if not route_positions_2d:
                    route_positions_2d.append(arc_points[0])
                elif float(np.linalg.norm(route_positions_2d[-1] - arc_points[0])) > 1e-6:
                    route_positions_2d.append(arc_points[0])
                route_positions_2d.extend(arc_points[1:])

            candidate = {
                "score": float(total_score),
                "route_positions_2d": route_positions_2d,
                "arc_directions": arc_directions,
                "arc_scores": arc_scores,
                "side_names": side_names,
                "side_scores": side_scores,
                "side_violation_count": sum(1 for score in side_scores if score < -0.02),
                "min_side_score": min(side_scores) if side_scores else 0.0,
            }
            candidate["selection_key"] = (
                -int(candidate["side_violation_count"]),
                float(candidate["min_side_score"]),
                float(candidate["score"]),
            )
            if best is None or candidate["selection_key"] > best["selection_key"]:
                best = candidate

        if best is None:
            raise RuntimeError("Failed to build peg-chain route candidates.")

        route_positions = [
            _from_plane_coords(point_2d, plane_origin, axis_x, axis_y)
            for point_2d in best["route_positions_2d"]
        ]

        away_from_prev = start_world - prev_world
        away_from_prev = away_from_prev - float(np.dot(away_from_prev, plane.normal)) * plane.normal
        if float(np.linalg.norm(away_from_prev)) < 1e-6 and route_positions:
            away_from_prev = route_positions[0] - start_world
        tool_y = start_rot[:, 1]
        pull_side_sign = 1.0 if float(np.dot(tool_y, away_from_prev)) >= 0.0 else -1.0

        poses: List[Dict[str, np.ndarray]] = []
        all_points = [start_world] + route_positions
        for idx, pos in enumerate(route_positions):
            prev = all_points[idx]
            if idx + 1 < len(route_positions):
                nxt = route_positions[idx + 1]
                pull = nxt - prev
            else:
                pull = terminal_world - pos
            if float(np.linalg.norm(pull)) < 1e-6:
                pull = terminal_world - pos
            poses.append(
                {
                    "position": np.asarray(pos, dtype=float).reshape(3),
                    "rotation": _rotation_from_pull_direction(plane.normal, pull, pull_side_sign),
                }
            )

        min_other_dist = None
        other_arm_should_move_aside = False
        if other_arm in current_poses:
            other_pos = np.asarray(current_poses[other_arm]["position"], dtype=float).reshape(3)
            dists = [float(np.linalg.norm(p["position"] - other_pos)) for p in poses]
            if dists:
                min_other_dist = min(dists)
                other_arm_should_move_aside = min_other_dist < float(
                    getattr(state.config, "peg_route_min_other_arm_distance_m", 0.1)
                )

        return {
            "arm": arm,
            "other_arm": other_arm,
            "route_idx": route_idx,
            "route_after_idx": route_after_idx,
            "prev_clip_idx": prev_clip_idx,
            "curr_clip_idx": curr_clip_idx,
            "next_clip_idx": terminal_clip_idx,
            "peg_clip_indices": peg_clip_indices,
            "prev_clip_label": str(state.clips[int(prev_clip_idx)].clip_id),
            "curr_clip_label": str(state.clips[int(curr_clip_idx)].clip_id),
            "peg_clip_labels": [str(state.clips[int(clip_idx)].clip_id) for clip_idx in peg_clip_indices],
            "terminal_clip_label": (
                str(state.clips[int(terminal_clip_idx)].clip_id)
                if terminal_clip_idx is not None
                else None
            ),
            "terminal_clip_idx": terminal_clip_idx,
            "start_position": start_world,
            "prev_clip_world": prev_world,
            "curr_clip_world": curr_world,
            "peg_centers_world": peg_centers_world,
            "next_clip_world": terminal_world,
            "poses": poses,
            "waypoints_world": [p["position"] for p in poses],
            "side": ",".join(best["side_names"]),
            "side_sign": float(np.sign(preferred_sides_2d[0][0])) if preferred_sides_2d else 0.0,
            "side_vectors_2d": [side.tolist() for side in preferred_sides_2d],
            "side_reasons": side_reasons,
            "side_debug": side_debug,
            "arc_direction": ",".join(best["arc_directions"]),
            "arc_direction_score": float(sum(best["arc_scores"])),
            "arc_directions": best["arc_directions"],
            "arc_direction_scores": best["arc_scores"],
            "side_scores": best["side_scores"],
            "side_violation_count": best["side_violation_count"],
            "min_side_score": best["min_side_score"],
            "pull_side_sign": pull_side_sign,
            "clearance_radius_m": radius,
            "continuation_world": [route_positions[-1], terminal_world] if route_positions else [],
            "reachable": reachable,
            "needs_handover": needs_handover,
            "other_arm_should_move_aside": other_arm_should_move_aside,
            "min_other_arm_distance_m": min_other_dist,
        }
