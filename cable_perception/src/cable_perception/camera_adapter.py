from __future__ import annotations

from typing import Any, Optional


def create_camera_subscriber(
    topic_rgb: Optional[str] = None,
    topic_depth: Optional[str] = None,
    topic_camera_info: Optional[str] = None,
    require_depth: bool = False,
    wait_timeout_sec: float = 5.0,
) -> Any:
    from cable_perception.zed_camera import ZedCameraSubscriber

    return ZedCameraSubscriber(
        topic_rgb=topic_rgb or "/zedm/zed_node/left/image_rect_color",
        topic_depth=topic_depth or "/zedm/zed_node/depth/depth_registered",
        topic_camera_info=topic_camera_info,
        require_depth=require_depth,
        wait_timeout_sec=wait_timeout_sec,
    )
