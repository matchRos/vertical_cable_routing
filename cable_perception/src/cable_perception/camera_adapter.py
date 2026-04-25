from __future__ import annotations

from typing import Any


def create_camera_subscriber() -> Any:
    from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber

    return ZedCameraSubscriber()
