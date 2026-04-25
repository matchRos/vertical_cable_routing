from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from cable_core.board_models import DebugBoard, DebugClip


class BoardService:
    """
    Utility methods for preparing routing overlays and clip-related debug output.
    """

    def get_clips(self, board: DebugBoard) -> List[DebugClip]:
        return board.get_clips()

    def create_base_board_image(
        self,
        width: int = 1500,
        height: int = 800,
        background_color: Tuple[int, int, int] = (30, 30, 30),
    ) -> np.ndarray:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :] = background_color
        return image

    def extract_clip_centers(self, clips: List[DebugClip]) -> List[Tuple[int, int]]:
        return [(clip.x, clip.y) for clip in clips]

    def draw_clip_centers(self, image: np.ndarray, clips: List[DebugClip]) -> np.ndarray:
        out = image.copy()
        for idx, clip in enumerate(clips):
            x, y = clip.x, clip.y
            cv2.circle(out, (x, y), 10, (0, 255, 0), 2)
            label = f"{idx}:{clip.clip_id}"
            cv2.putText(
                out,
                label,
                (x + 12, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return out

    def draw_routing_path(
        self,
        image: np.ndarray,
        clips: List[DebugClip],
        routing: List[int],
    ) -> np.ndarray:
        out = image.copy()
        if routing is None or len(routing) < 2:
            return out
        for idx in range(len(routing) - 1):
            idx_a = routing[idx]
            idx_b = routing[idx + 1]
            if idx_a >= len(clips) or idx_b >= len(clips):
                continue
            pt1 = (clips[idx_a].x, clips[idx_a].y)
            pt2 = (clips[idx_b].x, clips[idx_b].y)
            cv2.arrowedLine(out, pt1, pt2, (255, 0, 0), 3, tipLength=0.04)
        return out

    def estimate_crossing_fixture_ids(self, routing: List[int]) -> List[int]:
        return []

    def prepare_routing_debug_data(
        self,
        board: DebugBoard,
        routing: List[int],
        image_width: int = 1500,
        image_height: int = 800,
    ) -> Dict[str, Any]:
        clips = self.get_clips(board)
        clip_centers = self.extract_clip_centers(clips)
        overlay = self.create_base_board_image(width=image_width, height=image_height)
        overlay = self.draw_clip_centers(overlay, clips)
        overlay = self.draw_routing_path(overlay, clips, routing)
        crossing_fixture_id_list = self.estimate_crossing_fixture_ids(routing)
        return {
            "clips": clips,
            "clip_centers": clip_centers,
            "routing_overlay": overlay,
            "crossing_fixture_id_list": crossing_fixture_id_list,
            "num_clips": len(clips),
            "clip_ids": board.get_clip_ids(),
        }
