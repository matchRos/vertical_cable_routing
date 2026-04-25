from __future__ import annotations

import math
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from cable_perception.image_utils import find_nearest_white_pixel


def path_quality_metrics(path_in_pixels: Any) -> Tuple[int, float]:
    if path_in_pixels is None:
        return 0, 0.0
    pts = np.asarray(path_in_pixels, dtype=float)
    if pts.size == 0:
        return 0, 0.0
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    n_pts = int(pts.shape[0])
    if n_pts < 2:
        return n_pts, 0.0
    start = pts[0].reshape(-1)[:2]
    end = pts[-1].reshape(-1)[:2]
    return n_pts, float(np.linalg.norm(end - start))


def path_meets_quality(
    path_in_pixels: Any,
    min_path_points: int,
    min_end_to_start_px: float,
) -> bool:
    n_pts, end_dist = path_quality_metrics(path_in_pixels)
    return n_pts >= int(min_path_points) and end_dist >= float(min_end_to_start_px)


def snap_to_bright_pixel(image_rgb: np.ndarray, pt, radius: int = 5):
    x = int(round(float(pt[0])))
    y = int(round(float(pt[1])))
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    r = int(radius)
    y0 = int(max(0, y - r))
    y1 = int(min(gray.shape[0], y + r + 1))
    x0 = int(max(0, x - r))
    x1 = int(min(gray.shape[1], x + r + 1))

    patch = gray[y0:y1, x0:x1]
    ys, xs = np.where(patch > 150)
    if len(xs) == 0:
        return pt

    d2 = (xs + x0 - x) ** 2 + (ys + y0 - y) ** 2
    idx = int(np.argmin(d2))
    return (int(xs[idx] + x0), int(ys[idx] + y0))


def build_three_start_points_from_start_and_direction(
    image_rgb: np.ndarray,
    start_xy,
    direction_xy,
    step_px: float = 10,
):
    start_xy = snap_to_bright_pixel(image_rgb, start_xy)
    direction_xy = snap_to_bright_pixel(image_rgb, direction_xy)

    start = np.array(start_xy, dtype=float)
    direction_point = np.array(direction_xy, dtype=float)
    direction = direction_point - start
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([1.0, 0.0], dtype=float)
        norm = 1.0
    direction /= norm

    p0_xy = start
    p1_xy = start + direction * step_px
    p2_xy = start + direction * (2 * step_px)

    p0_xy = snap_to_bright_pixel(image_rgb, (int(round(p0_xy[0])), int(round(p0_xy[1]))))
    p1_xy = snap_to_bright_pixel(image_rgb, (int(round(p1_xy[0])), int(round(p1_xy[1]))), radius=7)
    p2_xy = snap_to_bright_pixel(image_rgb, (int(round(p2_xy[0])), int(round(p2_xy[1]))), radius=7)

    return [
        (p0_xy[1], p0_xy[0]),
        (p1_xy[1], p1_xy[0]),
        (p2_xy[1], p2_xy[0]),
    ]


def _pixels_on_euclidean_ring(
    cx: int,
    cy: int,
    radius: float,
    width: int,
    height: int,
) -> List[Tuple[int, int]]:
    rlo, rhi = float(radius) - 0.5, float(radius) + 0.5
    ri = int(math.ceil(float(radius) + 2))
    out: List[Tuple[int, int]] = []
    for dy in range(-ri, ri + 1):
        for dx in range(-ri, ri + 1):
            dist = math.hypot(dx, dy)
            if rlo <= dist <= rhi:
                x, y = cx + dx, cy + dy
                if 0 <= x < width and 0 <= y < height:
                    out.append((x, y))
    return out


def pick_whitest_pixel_on_ring(
    image_rgb: np.ndarray,
    cx: int,
    cy: int,
    radius: float,
) -> Tuple[int, int]:
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    pts = _pixels_on_euclidean_ring(cx, cy, radius, w, h)

    if not pts:
        best_xy: Optional[Tuple[int, int]] = None
        best_g = -1
        for k in range(720):
            ang = 2 * math.pi * k / 720.0
            x = int(round(cx + radius * math.cos(ang)))
            y = int(round(cy + radius * math.sin(ang)))
            if 0 <= x < w and 0 <= y < h:
                gray_val = int(gray[y, x])
                if gray_val > best_g or (gray_val == best_g and best_xy is not None and (x, y) < best_xy):
                    best_g = gray_val
                    best_xy = (x, y)
        if best_xy is not None:
            return best_xy
        return (max(0, min(w - 1, cx)), max(0, min(h - 1, cy)))

    return min(pts, key=lambda p: (-int(gray[p[1], p[0]]), p[0], p[1]))


def nearest_bright_pixel_global(
    image_rgb: np.ndarray,
    anchor_xy: Tuple[int, int],
    threshold: int = 150,
) -> Optional[Tuple[int, int]]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(gray > threshold)
    if len(xs) == 0:
        return None

    pts = np.stack([xs, ys], axis=1).astype(float)
    anchor = np.asarray(anchor_xy, dtype=float).reshape(1, 2)
    d2 = np.sum((pts - anchor) ** 2, axis=1)
    best = pts[int(np.argmin(d2))]
    return (int(best[0]), int(best[1]))


def run_white_rings_k_retry(
    tracer: Any,
    image_rgb: np.ndarray,
    anchor_point: Tuple[int, int],
    step: float,
    k_candidates: Tuple[float, ...],
    min_path_points: int,
    min_end_to_start_px: float,
    end_points: Optional[List[Tuple[int, int]]],
    viz: bool,
) -> Tuple[np.ndarray, Any, List[Tuple[int, int]], Dict[str, Any]]:
    cx, cy = int(anchor_point[0]), int(anchor_point[1])
    step_f = float(step)
    last_n = 0
    last_d = 0.0
    last_k: Optional[float] = None

    for k in k_candidates:
        kf = float(k)
        radii = [kf * step_f, (1.0 + kf) * step_f, (2.0 + kf) * step_f]
        pts_xy: List[Tuple[int, int]] = []
        for rad in radii:
            pts_xy.append(pick_whitest_pixel_on_ring(image_rgb, cx, cy, float(rad)))
        tracer_start_points = [(int(xy[1]), int(xy[0])) for xy in pts_xy]
        trace_ring_debug: Dict[str, Any] = {
            "anchor_xy": (cx, cy),
            "step_px": step_f,
            "white_ring_k": kf,
            "ring_radii_px": radii,
            "ring_points_xy": pts_xy,
        }
        print(
            "auto_white_rings_from_clip try k=",
            kf,
            "tracer points (y,x):",
            tracer_start_points,
            "anchor xy:",
            (cx, cy),
            "step_px:",
            step_f,
            "radii_px:",
            radii,
        )
        try:
            result = tracer.trace(
                img=image_rgb,
                start_points=tracer_start_points,
                end_points=end_points,
                viz=viz,
            )
        except Exception as exc:
            msg = str(exc)
            if "Not enough starting points" in msg:
                print(f"white_rings k={kf}: {msg} — try next k")
                continue
            raise

        if result is None:
            print(f"white_rings k={kf}: trace returned None — try next k")
            continue

        path, status = result
        n_pts, end_dist = path_quality_metrics(path)
        last_n, last_d, last_k = n_pts, end_dist, kf
        trace_ring_debug["trace_quality"] = {
            "n_points": n_pts,
            "end_to_start_px": end_dist,
        }
        ok = path_meets_quality(path, min_path_points, min_end_to_start_px)
        print(
            f"white_rings k={kf}: quality n={n_pts} end_dist={end_dist:.1f}px "
            f"(min n={min_path_points}, min dist={min_end_to_start_px}) "
            f"-> {'OK' if ok else 'FAIL'}"
        )
        if ok:
            return path, status, tracer_start_points, trace_ring_debug

    raise RuntimeError(
        "White-ring cable trace did not meet quality after trying k in "
        f"{list(k_candidates)} (last k={last_k}, n={last_n}, end_dist={last_d:.1f}px)."
    )


class TracingService:
    """
    Standalone helper for image acquisition, trace execution, and visualization.
    """

    def get_image_from_camera(self, camera: Any) -> Optional[np.ndarray]:
        if camera is None:
            return None

        for method_name in ("get_rgb", "get_rgb_image", "get_image", "get_frame", "read"):
            if hasattr(camera, method_name):
                try:
                    result = getattr(camera, method_name)()
                    if isinstance(result, np.ndarray):
                        return self._ensure_rgb_uint8(result)
                except Exception:
                    pass
        return None

    def load_image_from_disk(self, image_path: str) -> Optional[np.ndarray]:
        if not image_path:
            return None
        path = Path(image_path)
        if not path.exists():
            return None
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def acquire_image(
        self,
        camera: Any = None,
        fallback_image_path: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], str]:
        image = self.get_image_from_camera(camera)
        if image is not None:
            return image, "camera"
        image = self.load_image_from_disk(fallback_image_path or "")
        if image is not None:
            return image, "disk"
        return None, "none"

    def run_trace(
        self,
        tracer: Any,
        image_rgb: np.ndarray,
        start_points: List[Tuple[int, int]],
        end_points: Optional[List[Tuple[int, int]]] = None,
        viz: bool = False,
        start_mode: str = "auto_from_config",
        anchor_point: Optional[Tuple[int, int]] = None,
        clip_points: Optional[List[Tuple[int, int]]] = None,
        preferred_direction_xy: Optional[Tuple[float, float]] = None,
        max_start_dist_px: float = 260.0,
        min_route_dot: float = -0.15,
        outward_min_delta_px: float = 8.0,
        seed_order_descending_from_anchor: bool = True,
        clip_a_p1_offset_px: float = 20.0,
        clip_a_p2_offset_px: float = 40.0,
        trace_white_ring_step_px: float = 20.0,
        trace_min_path_points: int = 150,
        trace_min_end_to_start_px: float = 100.0,
        trace_white_ring_k_candidates: Tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0),
    ) -> Dict[str, Any]:
        if tracer is None:
            raise RuntimeError("Tracer object is not available.")

        tracer_start_points = start_points
        trace_ring_debug: Optional[Dict[str, Any]] = None

        def _build_anchor_white_candidates() -> List[List[Tuple[int, int]]]:
            if anchor_point is None:
                return []
            clip_dict = {"x": int(anchor_point[0]), "y": int(anchor_point[1])}
            try:
                valid_points = find_nearest_white_pixel(
                    image_rgb,
                    clip_dict,
                    num_options=20,
                    display=False,
                )
            except Exception:
                return []

            if clip_points:
                filtered_points = []
                for p in valid_points:
                    ok = True
                    for clip_pt in clip_points:
                        if (
                            anchor_point is not None
                            and np.linalg.norm(
                                np.asarray(clip_pt, dtype=float)
                                - np.asarray(anchor_point, dtype=float)
                            )
                            < 1.0
                        ):
                            continue
                        if float(np.linalg.norm(np.asarray(p) - np.asarray(clip_pt))) < 20.0:
                            ok = False
                            break
                    if ok:
                        filtered_points.append(p)
            else:
                filtered_points = list(valid_points)

            filtered_points.sort(
                key=lambda p: float(
                    np.linalg.norm(np.asarray(p, dtype=float) - np.asarray(anchor_point, dtype=float))
                )
            )

            candidates: List[List[Tuple[int, int]]] = []
            for p in filtered_points[:6]:
                direction_xy = (
                    int(round(2 * p[0] - anchor_point[0])),
                    int(round(2 * p[1] - anchor_point[1])),
                )
                cand = build_three_start_points_from_start_and_direction(
                    image_rgb,
                    p,
                    direction_xy,
                    step_px=20,
                )
                candidates.append(cand)
                candidates.append([cand[0], cand[-1]])
                candidates.append([cand[-1], cand[0]])
            return candidates

        def _rank_and_filter_candidates(
            candidates: List[List[Tuple[int, int]]],
        ) -> List[List[Tuple[int, int]]]:
            if not candidates:
                return candidates

            pref = None
            if preferred_direction_xy is not None:
                pref = np.asarray(preferred_direction_xy, dtype=float).reshape(2)
                norm = float(np.linalg.norm(pref))
                pref = pref / norm if norm > 1e-6 else None

            scored = []
            for cand in candidates:
                if cand is None or len(cand) < 2:
                    continue

                if anchor_point is not None and len(cand) >= 3:
                    pts_xy = []
                    for pt in cand:
                        arr = np.asarray(pt, dtype=float).reshape(-1)[:2]
                        pts_xy.append(np.array([arr[1], arr[0]], dtype=float))
                    dists = [
                        float(np.linalg.norm(pxy - np.asarray(anchor_point, dtype=float).reshape(2)))
                        for pxy in pts_xy
                    ]
                    order = np.argsort(dists)
                    if seed_order_descending_from_anchor:
                        order = order[::-1]
                    cand = [cand[int(i)] for i in order.tolist()]

                p0 = np.asarray(cand[0], dtype=float).reshape(-1)[:2]
                p1 = np.asarray(cand[1], dtype=float).reshape(-1)[:2]
                p0_xy = np.array([p0[1], p0[0]], dtype=float)
                p1_xy = np.array([p1[1], p1[0]], dtype=float)

                d_anchor = 0.0
                if anchor_point is not None:
                    anchor_arr = np.asarray(anchor_point, dtype=float)
                    d0 = float(np.linalg.norm(p0_xy - anchor_arr))
                    d1 = float(np.linalg.norm(p1_xy - anchor_arr))
                    d_anchor = d0
                    if d_anchor > max_start_dist_px:
                        continue
                    delta = d1 - d0
                    if seed_order_descending_from_anchor:
                        if (-delta) < outward_min_delta_px:
                            continue
                    else:
                        if delta < outward_min_delta_px:
                            continue

                dir_xy = p1_xy - p0_xy
                dir_n = float(np.linalg.norm(dir_xy))
                dot = 0.0
                if pref is not None and dir_n > 1e-6:
                    dot = float(np.dot(dir_xy / dir_n, pref))
                    if dot < min_route_dot:
                        continue

                score = d_anchor + 80.0 * (1.0 - dot)
                scored.append((score, cand))

            if not scored:
                return candidates
            scored.sort(key=lambda x: x[0])
            return [cand for _, cand in scored]

        try:
            if start_mode == "auto_white_rings_from_clip":
                if anchor_point is None:
                    raise RuntimeError(
                        "trace_start_mode=auto_white_rings_from_clip requires anchor_point "
                        "(first routing clip)."
                    )
                path, status, tracer_start_points, trace_ring_debug = run_white_rings_k_retry(
                    tracer=tracer,
                    image_rgb=image_rgb,
                    anchor_point=(int(anchor_point[0]), int(anchor_point[1])),
                    step=float(trace_white_ring_step_px),
                    k_candidates=tuple(float(x) for x in trace_white_ring_k_candidates),
                    min_path_points=int(trace_min_path_points),
                    min_end_to_start_px=float(trace_min_end_to_start_px),
                    end_points=end_points,
                    viz=viz,
                )
            else:
                if start_mode == "auto_from_clip_a":
                    clip_dict = {"x": int(anchor_point[0]), "y": int(anchor_point[1])} if anchor_point is not None else None
                    nearest = (
                        nearest_bright_pixel_global(image_rgb, anchor_point, threshold=150)
                        if anchor_point is not None
                        else None
                    )
                    direction = None
                    valid_points = []
                    if clip_dict is not None:
                        try:
                            valid_points = find_nearest_white_pixel(
                                image_rgb,
                                clip_dict,
                                num_options=25,
                                display=False,
                            )
                        except Exception:
                            valid_points = []
                    pref = None
                    if preferred_direction_xy is not None:
                        pref = np.asarray(preferred_direction_xy, dtype=float).reshape(2)
                        n_pref = float(np.linalg.norm(pref))
                        pref = pref / n_pref if n_pref > 1e-6 else None
                    best_score = float("inf")
                    best_point = None
                    for p in valid_points:
                        vec = np.asarray(p, dtype=float) - np.asarray(anchor_point, dtype=float)
                        dist = float(np.linalg.norm(vec))
                        if dist < 3.0:
                            continue
                        vec_norm = vec / (dist + 1e-8)
                        dot_penalty = 0.0
                        if pref is not None:
                            dot_penalty = 40.0 * (1.0 - float(np.dot(vec_norm, pref)))
                        score = abs(dist - 20.0) + dot_penalty
                        if score < best_score:
                            best_score = score
                            best_point = p
                    if best_point is not None:
                        direction = np.asarray(best_point, dtype=float) - np.asarray(anchor_point, dtype=float)
                    elif nearest is not None:
                        direction = np.asarray(nearest, dtype=float) - np.asarray(anchor_point, dtype=float)
                    elif pref is not None:
                        direction = pref.copy()
                    else:
                        direction = np.array([1.0, 0.0], dtype=float)

                    norm = float(np.linalg.norm(direction))
                    if norm < 1e-6:
                        direction = np.array([1.0, 0.0], dtype=float)
                        norm = 1.0
                    direction /= norm

                    p0_xy = (int(anchor_point[0]), int(anchor_point[1]))
                    p1_xy = (
                        int(round(anchor_point[0] + direction[0] * float(clip_a_p1_offset_px))),
                        int(round(anchor_point[1] + direction[1] * float(clip_a_p1_offset_px))),
                    )
                    p2_xy = (
                        int(round(anchor_point[0] + direction[0] * float(clip_a_p2_offset_px))),
                        int(round(anchor_point[1] + direction[1] * float(clip_a_p2_offset_px))),
                    )
                    p0_xy = snap_to_bright_pixel(image_rgb, p0_xy, radius=5)
                    p1_xy = snap_to_bright_pixel(image_rgb, p1_xy, radius=7)
                    p2_xy = snap_to_bright_pixel(image_rgb, p2_xy, radius=7)
                    tracer_start_points = [
                        (int(p0_xy[1]), int(p0_xy[0])),
                        (int(p1_xy[1]), int(p1_xy[0])),
                        (int(p2_xy[1]), int(p2_xy[0])),
                    ]
                else:
                    cfg_pts = [tuple(np.asarray(p).reshape(-1)[:2].astype(int)) for p in start_points]
                    if len(cfg_pts) >= 2:
                        tracer_start_points = build_three_start_points_from_start_and_direction(
                            image_rgb,
                            cfg_pts[0],
                            cfg_pts[1],
                            step_px=20,
                        )
                    elif len(cfg_pts) == 1:
                        p0 = cfg_pts[0]
                        p1 = (int(p0[0] + 20), int(p0[1]))
                        tracer_start_points = build_three_start_points_from_start_and_direction(
                            image_rgb,
                            p0,
                            p1,
                            step_px=20,
                        )
                    else:
                        raise RuntimeError(
                            "trace_start_points is empty. Provide at least one point in config."
                        )

                candidate_pool = [tracer_start_points]
                candidate_pool.extend(_build_anchor_white_candidates())
                candidate_pool = _rank_and_filter_candidates(candidate_pool)
                print(f"trace candidate pool size: {len(candidate_pool)} (mode={start_mode})")

                result = None
                last_exc: Optional[Exception] = None
                for candidate_idx, candidate in enumerate(candidate_pool):
                    tracer_start_points = candidate
                    try:
                        result = tracer.trace(
                            img=image_rgb,
                            start_points=tracer_start_points,
                            end_points=end_points,
                            viz=viz,
                        )
                        if result is not None:
                            if candidate_idx > 0:
                                print(
                                    "trace succeeded with fallback candidate "
                                    f"#{candidate_idx}: {tracer_start_points}"
                                )
                            break
                    except Exception as exc:
                        if "Not enough starting points" in str(exc):
                            last_exc = exc
                            continue
                        raise

                if result is None and last_exc is not None:
                    raise last_exc
                if result is None:
                    raise RuntimeError(
                        "Tracing failed. The start point is likely not on the cable or "
                        f"the analytic tracer could not initialize from start_points={start_points}."
                    )

                path, status = result
                n_pts, end_dist = path_quality_metrics(path)
                if not path_meets_quality(
                    path,
                    int(trace_min_path_points),
                    float(trace_min_end_to_start_px),
                ):
                    raise RuntimeError(
                        "Trace quality check failed: "
                        f"points={n_pts} (min {int(trace_min_path_points)}), "
                        f"end-start distance={end_dist:.1f}px "
                        f"(min {float(trace_min_end_to_start_px)})."
                    )
        except Exception as exc:
            print("\n=== TRACE ERROR ===")
            print(f"type: {type(exc).__name__}")
            print(f"message: {exc}")
            print(f"start_points(type/raw): {type(start_points)} -> {start_points}")
            print(
                f"tracer_start_points(type/used): {type(tracer_start_points)} -> {tracer_start_points}"
            )
            print(f"end_points type: {type(end_points)}")
            print(f"end_points value: {end_points}")
            traceback.print_exc()
            raise

        return {
            "path_in_pixels": path,
            "trace_status": status,
            "tracer_start_points_used": tracer_start_points,
            "tracer_start_point_count": (
                len(tracer_start_points) if tracer_start_points is not None else 0
            ),
            "trace_ring_debug": trace_ring_debug,
        }

    def create_trace_overlay(
        self,
        image_rgb: np.ndarray,
        start_points: List[Tuple[int, int]],
        end_points: Optional[List[Tuple[int, int]]] = None,
        path_in_pixels: Optional[np.ndarray] = None,
        tracer_start_points_used: Optional[List[Tuple[int, int]]] = None,
        configured_clip_positions: Optional[List[Tuple[str, int, int]]] = None,
        white_rings_debug: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        overlay = image_rgb.copy()

        for idx, pt in enumerate(start_points):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(overlay, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(overlay, f"S{idx}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        if end_points is not None:
            for idx, pt in enumerate(end_points):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(overlay, (x, y), 8, (255, 0, 0), -1)
                cv2.putText(overlay, f"E{idx}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

        if configured_clip_positions is not None:
            for clip_id, x, y in configured_clip_positions:
                xi, yi = int(x), int(y)
                if xi < 0 or yi < 0 or yi >= overlay.shape[0] or xi >= overlay.shape[1]:
                    continue
                cv2.circle(overlay, (xi, yi), 10, (0, 255, 255), 2)
                cv2.drawMarker(overlay, (xi, yi), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
                cv2.putText(overlay, f"C{clip_id}", (xi + 10, yi + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(overlay, f"C{clip_id}", (xi + 10, yi + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        if white_rings_debug is not None:
            ax, ay = white_rings_debug["anchor_xy"]
            step = float(white_rings_debug["step_px"])
            for k in (1, 2, 3):
                cv2.circle(overlay, (int(ax), int(ay)), int(round(k * step)), (0, 220, 255), 2)
            cv2.drawMarker(overlay, (int(ax), int(ay)), (0, 220, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
            cv2.putText(overlay, "anchor", (int(ax) + 12, int(ay) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)

        if path_in_pixels is not None and len(path_in_pixels) > 1:
            pts = np.asarray(path_in_pixels).astype(np.int32)
            for i in range(len(pts) - 1):
                cv2.line(overlay, tuple(pts[i]), tuple(pts[i + 1]), (255, 255, 0), 2)
            cv2.circle(overlay, tuple(pts[0]), 6, (255, 255, 255), -1)
            cv2.circle(overlay, tuple(pts[-1]), 6, (255, 255, 255), -1)

        if tracer_start_points_used is not None and len(tracer_start_points_used) >= 2:
            pxy_list = []
            seed_colors = [(255, 0, 255), (255, 165, 0), (0, 200, 255)]
            for idx, pt in enumerate(tracer_start_points_used[:3]):
                arr = np.asarray(pt).reshape(-1)
                if arr.size < 2:
                    continue
                y = int(arr[0])
                x = int(arr[1])
                if x < 0 or y < 0 or y >= overlay.shape[0] or x >= overlay.shape[1]:
                    continue
                pxy_list.append((x, y))
                color = seed_colors[idx] if idx < len(seed_colors) else (200, 200, 200)
                label = f"P{idx}"
                cv2.circle(overlay, (x, y), 16, color, 3)
                cv2.circle(overlay, (x, y), 5, color, -1)
                cv2.circle(overlay, (x, y), 22, (0, 0, 0), 1)
                cv2.putText(overlay, label, (x + 18, y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(overlay, label, (x + 18, y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

            if len(pxy_list) >= 2:
                cv2.arrowedLine(overlay, pxy_list[0], pxy_list[1], (255, 255, 255), 3, tipLength=0.22)
            if len(pxy_list) >= 3:
                cv2.arrowedLine(overlay, pxy_list[1], pxy_list[2], (220, 220, 255), 2, tipLength=0.2)

        return overlay

    def create_no_trace_overlay(
        self,
        image_rgb: np.ndarray,
        start_points: List[Tuple[int, int]],
        end_points: Optional[List[Tuple[int, int]]] = None,
        message: str = "Tracer unavailable",
        configured_clip_positions: Optional[List[Tuple[str, int, int]]] = None,
    ) -> np.ndarray:
        overlay = self.create_trace_overlay(
            image_rgb=image_rgb,
            start_points=start_points,
            end_points=end_points,
            path_in_pixels=None,
            configured_clip_positions=configured_clip_positions,
        )
        cv2.putText(overlay, message, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 100), 2, cv2.LINE_AA)
        return overlay

    def _ensure_rgb_uint8(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image
