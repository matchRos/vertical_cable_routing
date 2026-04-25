from __future__ import annotations

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image

from cable_perception.ros_image_utils import image_msg_to_numpy


class SimpleCameraIntrinsics:
    def __init__(self, fx, fy, cx, cy, width, height, frame="zed"):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = int(width)
        self.height = int(height)
        self.frame = frame
        self._K = np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )


class ZedCameraSubscriber:
    def __init__(
        self,
        topic_depth="/zedm/zed_node/depth/depth_registered",
        topic_rgb="/zedm/zed_node/left/image_rect_color",
        topic_camera_info=None,
        require_depth=False,
        wait_timeout_sec=5.0,
        display=False,
    ):
        self._topic_name = topic_depth
        self._rgb_name = topic_rgb
        self._camera_info_name = topic_camera_info or self._derive_camera_info_topic(topic_rgb)
        self.require_depth = require_depth
        self.wait_timeout_sec = float(wait_timeout_sec)

        self.intrinsic = None
        self.w = 0
        self.h = 0
        try:
            camera_info = rospy.wait_for_message(
                self._camera_info_name, CameraInfo, timeout=min(2.0, self.wait_timeout_sec)
            )
            self.intrinsic = SimpleCameraIntrinsics(
                fx=camera_info.K[0],
                fy=camera_info.K[4],
                cx=camera_info.K[2],
                cy=camera_info.K[5],
                width=camera_info.width,
                height=camera_info.height,
                frame="zed",
            )
            self.w = camera_info.width
            self.h = camera_info.height
        except Exception as exc:
            rospy.logwarn(
                "CameraInfo topic '%s' not ready: %s. Continuing with RGB-only camera.",
                self._camera_info_name,
                exc,
            )
        self.img_shape = (self.h, self.w)
        self.far_clip = 1
        self.near_clip = 0.1
        self.dis_noise = 0.0
        self.display = display
        self.zed_init = False
        self.init_success = False

        self.rgb_image = None
        self.depth_image = None

        self._check_rgb_ready()
        if self.rgb_image is not None and (self.w <= 0 or self.h <= 0):
            self.h, self.w = self.rgb_image.shape[:2]
            self.img_shape = (self.h, self.w)

        if require_depth:
            self._check_depth_ready()

        if topic_depth:
            self._depth_subscriber = rospy.Subscriber(
                self._topic_name, Image, self.depth_callback, queue_size=2
            )
        else:
            self._depth_subscriber = None
        self._image_subscriber = rospy.Subscriber(
            self._rgb_name, Image, self.rgb_callback, queue_size=2
        )

    @staticmethod
    def _derive_camera_info_topic(topic_rgb):
        suffixes = ("/image_rect_color", "/image_raw", "/image_color")
        for suffix in suffixes:
            if topic_rgb.endswith(suffix):
                return topic_rgb[: -len(suffix)] + "/camera_info"
        return "/zedm/zed_node/left/camera_info"

    def _check_rgb_ready(self):
        try:
            msg = rospy.wait_for_message(
                self._rgb_name, Image, timeout=self.wait_timeout_sec
            )
            self.start_time = rospy.get_time()
            self.rgb_image = image_msg_to_numpy(msg)
        except Exception as exc:
            raise RuntimeError(
                f"RGB image topic '{self._rgb_name}' not ready after "
                f"{self.wait_timeout_sec:.1f}s"
            ) from exc
        return self.rgb_image

    def _check_depth_ready(self):
        try:
            msg = rospy.wait_for_message(
                self._topic_name, Image, timeout=self.wait_timeout_sec
            )
            self.zed_init = True
            self.depth_image = image_msg_to_numpy(msg)
            self.start_time = rospy.get_time()
        except Exception as exc:
            raise RuntimeError(
                f"Depth image topic '{self._topic_name}' not ready after "
                f"{self.wait_timeout_sec:.1f}s"
            ) from exc
        return self.depth_image

    def rgb_callback(self, msg):
        try:
            self.rgb_image = image_msg_to_numpy(msg)
        except Exception:
            return

    def depth_callback(self, msg):
        try:
            self.depth_image = image_msg_to_numpy(msg)
        except Exception:
            return

    def get_frames(self):
        return self.rgb_image, self.depth_image

    def get_rgb(self):
        return self.rgb_image

    def get_depth(self):
        return self.depth_image

    def process_depth_image(self, depth_image):
        depth_image = np.clip(depth_image, self.near_clip, self.far_clip)
        return self.normalize_depth_image(depth_image)

    def normalize_depth_image(self, depth_image):
        return (depth_image - self.near_clip) / (self.far_clip - self.near_clip)

    def crop_depth_image(self, depth_image):
        return depth_image


if __name__ == "__main__":
    rospy.init_node("zed_pub")
    zed_cam = ZedCameraSubscriber(display=True)
    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        if (
            zed_cam.display
            and zed_cam.depth_image is not None
            and zed_cam.rgb_image is not None
        ):
            cv2.imshow(
                "Depth Image",
                np.expand_dims(zed_cam.depth_image, 0).transpose(1, 2, 0),
            )
            cv2.imshow("RGB Image", zed_cam.rgb_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        rate.sleep()
