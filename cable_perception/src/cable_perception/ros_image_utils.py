from __future__ import annotations

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError


def image_msg_to_numpy(
    msg,
    empty_value=None,
    output_resolution=None,
    max_depth=None,
    use_bridge=False,
) -> np.ndarray:
    if output_resolution is None:
        output_resolution = (msg.width, msg.height)

    is_rgb = "8" in msg.encoding.lower()
    is_depth16 = "16" in msg.encoding.lower()
    is_depth32 = "32" in msg.encoding.lower()

    if not use_bridge:
        if is_rgb:
            data = (
                np.frombuffer(msg.data, dtype=np.uint8)
                .reshape(msg.height, msg.width, -1)[:, :, :3]
                .copy()
            )
        elif is_depth16:
            data = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).copy() / -1000
            data = np.array(data.astype(np.float32))
        elif is_depth32:
            data = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
        else:
            raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    else:
        bridge = CvBridge()
        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            raise RuntimeError(f"Failed to convert ROS image: {exc}") from exc
        if is_rgb:
            data = cv_img[:, :, :3]
        elif is_depth16 or is_depth32:
            if max_depth is None:
                raise ValueError("max_depth is required when use_bridge=True for depth images")
            data = np.clip(cv_img, 0, max_depth) / max_depth
            data = np.uint8(data * 255)
        else:
            raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    if empty_value is not None:
        mask = np.isclose(abs(data), empty_value)
    else:
        mask = np.isnan(data)

    if np.any(mask):
        fill_value = np.percentile(data[~mask], 99)
        data[mask] = fill_value

    if output_resolution != (msg.width, msg.height):
        data = cv2.resize(
            data,
            dsize=(output_resolution[0], output_resolution[1]),
            interpolation=cv2.INTER_AREA,
        )

    return data
