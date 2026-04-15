#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import yaml
import rospy
import numpy as np
import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class CameraRobot2DCalibrator:
    def __init__(self):
        rospy.init_node("camera_robot_2d_calibrator")

        # Parameters
        self.image_topic = rospy.get_param(
            "~image_topic", "/zedm/zed_node/left/image_rect_color"
        )
        self.base_frame = rospy.get_param("~base_frame", "yumi_base_link")
        self.tcp_frame = rospy.get_param("~tcp_frame", "yumi_tcp_r")
        self.output_yaml = rospy.get_param(
            "~output_yaml", "camera_robot_2d_calibration.yaml"
        )
        self.min_area = rospy.get_param("~min_area", 20)
        self.use_hsv_defaults = rospy.get_param("~use_hsv_defaults", True)

        # HSV defaults for yellow
        self.h_low = rospy.get_param("~h_low", 20)
        self.s_low = rospy.get_param("~s_low", 80)
        self.v_low = rospy.get_param("~v_low", 80)
        self.h_high = rospy.get_param("~h_high", 40)
        self.s_high = rospy.get_param("~s_high", 255)
        self.v_high = rospy.get_param("~v_high", 255)

        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()

        self.latest_image = None
        self.latest_stamp = None

        self.roi_defined = False
        self.roi = None  # (x1, y1, x2, y2)
        self.dragging = False
        self.pt1 = None
        self.pt2 = None

        self.detected_center = None  # (u, v) in full image coordinates
        self.detected_mask = None

        self.pixel_points = []
        self.robot_points = []

        self.affine_matrix = None  # 2x3
        self.homography_matrix = None  # 3x3

        self.window_name = "Camera-Robot 2D Calibration"

        self.sub = rospy.Subscriber(
            self.image_topic, Image, self.image_cb, queue_size=1
        )

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_cb)

        rospy.sleep(1.0)

    def image_cb(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_stamp = msg.header.stamp
        except Exception as e:
            rospy.logwarn("CV bridge conversion failed: %s", str(e))

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.pt1 = (x, y)
            self.pt2 = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.pt2 = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.pt2 = (x, y)
            self.set_roi_from_points(self.pt1, self.pt2)

    def set_roi_from_points(self, p1, p2):
        if p1 is None or p2 is None:
            return

        x1 = min(p1[0], p2[0])
        y1 = min(p1[1], p2[1])
        x2 = max(p1[0], p2[0])
        y2 = max(p1[1], p2[1])

        if x2 - x1 < 5 or y2 - y1 < 5:
            rospy.logwarn("ROI too small, ignored.")
            return

        self.roi = (x1, y1, x2, y2)
        self.roi_defined = True
        rospy.loginfo("ROI set to: (%d, %d) - (%d, %d)", x1, y1, x2, y2)

    def get_tcp_yz(self):
        try:
            self.tf_listener.waitForTransform(
                self.base_frame, self.tcp_frame, rospy.Time(0), rospy.Duration(0.5)
            )
            trans, rot = self.tf_listener.lookupTransform(
                self.base_frame, self.tcp_frame, rospy.Time(0)
            )
            # trans = (x, y, z)
            return np.array([trans[1], trans[2]], dtype=np.float64)
        except Exception as e:
            rospy.logwarn("TF lookup failed: %s", str(e))
            return None

    def detect_marker(self, image):
        self.detected_center = None
        self.detected_mask = None

        if not self.roi_defined or self.roi is None:
            return image

        x1, y1, x2, y2 = self.roi
        roi_img = image[y1:y2, x1:x2].copy()

        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_low, self.s_low, self.v_low], dtype=np.uint8)
        upper = np.array([self.h_high, self.s_high, self.v_high], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        self.detected_mask = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < self.min_area:
            return image

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return image

        cx_roi = int(M["m10"] / M["m00"])
        cy_roi = int(M["m01"] / M["m00"])

        cx = x1 + cx_roi
        cy = y1 + cy_roi

        self.detected_center = (cx, cy)

        # draw contour and center in full image
        contour_shifted = largest + np.array([[[x1, y1]]], dtype=np.int32)
        cv2.drawContours(image, [contour_shifted], -1, (0, 255, 255), 2)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            image,
            f"Marker: ({cx}, {cy})",
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        return image

    def save_sample(self):
        if self.detected_center is None:
            rospy.logwarn("No marker detected. Sample not saved.")
            return

        yz = self.get_tcp_yz()
        if yz is None:
            rospy.logwarn("No TCP transform available. Sample not saved.")
            return

        u, v = self.detected_center
        self.pixel_points.append([float(u), float(v)])
        self.robot_points.append([float(yz[0]), float(yz[1])])

        rospy.loginfo(
            "Saved sample #%d | pixel=(%.1f, %.1f) -> robot_yz=(%.5f, %.5f)",
            len(self.pixel_points),
            float(u),
            float(v),
            float(yz[0]),
            float(yz[1]),
        )

    def fit_affine(self):
        if len(self.pixel_points) < 3:
            rospy.logwarn("Need at least 3 points for affine fit.")
            return None

        src = np.array(self.pixel_points, dtype=np.float64)
        dst = np.array(self.robot_points, dtype=np.float64)

        # Solve:
        # [y z] = [u v 1] * A^T
        X = np.hstack([src, np.ones((src.shape[0], 1))])  # Nx3
        Y = dst  # Nx2

        A, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        # A is 3x2, convert to OpenCV-like 2x3
        affine = A.T  # 2x3

        self.affine_matrix = affine
        return affine

    def apply_affine(self, uv):
        if self.affine_matrix is None:
            return None
        u, v = uv
        vec = np.array([u, v, 1.0], dtype=np.float64)
        yz = self.affine_matrix.dot(vec)
        return yz

    def fit_homography(self):
        if len(self.pixel_points) < 4:
            rospy.logwarn("Need at least 4 points for homography.")
            return None

        src = np.array(self.pixel_points, dtype=np.float32).reshape(-1, 1, 2)
        dst = np.array(self.robot_points, dtype=np.float32).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src, dst, method=0)
        if H is None:
            rospy.logwarn("Homography fit failed.")
            return None

        self.homography_matrix = H
        return H

    def apply_homography(self, uv):
        if self.homography_matrix is None:
            return None

        pt = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float32)
        yz = cv2.perspectiveTransform(pt, self.homography_matrix)[0, 0]
        return yz

    def compute_errors(self):
        if len(self.pixel_points) == 0:
            return

        src = np.array(self.pixel_points, dtype=np.float64)
        dst = np.array(self.robot_points, dtype=np.float64)

        if self.affine_matrix is not None:
            pred_aff = np.array([self.apply_affine(p) for p in src], dtype=np.float64)
            err_aff = np.linalg.norm(pred_aff - dst, axis=1)
            rospy.loginfo(
                "Affine mean error: %.6f m | max error: %.6f m",
                float(np.mean(err_aff)),
                float(np.max(err_aff)),
            )

        if self.homography_matrix is not None:
            pred_h = np.array([self.apply_homography(p) for p in src], dtype=np.float64)
            err_h = np.linalg.norm(pred_h - dst, axis=1)
            rospy.loginfo(
                "Homography mean error: %.6f m | max error: %.6f m",
                float(np.mean(err_h)),
                float(np.max(err_h)),
            )

    def save_yaml(self):
        data = {
            "base_frame": self.base_frame,
            "tcp_frame": self.tcp_frame,
            "image_topic": self.image_topic,
            "roi": list(self.roi) if self.roi is not None else None,
            "hsv_thresholds": {
                "h_low": int(self.h_low),
                "s_low": int(self.s_low),
                "v_low": int(self.v_low),
                "h_high": int(self.h_high),
                "s_high": int(self.s_high),
                "v_high": int(self.v_high),
            },
            "pixel_points": self.pixel_points,
            "robot_yz_points": self.robot_points,
            "affine_matrix_2x3": (
                self.affine_matrix.tolist() if self.affine_matrix is not None else None
            ),
            "homography_matrix_3x3": (
                self.homography_matrix.tolist()
                if self.homography_matrix is not None
                else None
            ),
        }

        with open(self.output_yaml, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        rospy.loginfo("Calibration saved to: %s", os.path.abspath(self.output_yaml))

    def clear_samples(self):
        self.pixel_points = []
        self.robot_points = []
        self.affine_matrix = None
        self.homography_matrix = None
        rospy.loginfo("All samples and fitted models cleared.")

    def draw_overlay(self, image):
        vis = image.copy()

        # Draw ROI
        if self.roi_defined and self.roi is not None:
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Temporary ROI while dragging
        if self.dragging and self.pt1 is not None and self.pt2 is not None:
            cv2.rectangle(vis, self.pt1, self.pt2, (255, 255, 0), 1)

        # Detect marker
        vis = self.detect_marker(vis)

        # Display current TCP y,z
        yz = self.get_tcp_yz()
        if yz is not None:
            text = f"TCP y,z = ({yz[0]:.4f}, {yz[1]:.4f}) m"
            cv2.putText(
                vis,
                text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Display sample count
        cv2.putText(
            vis,
            f"Samples: {len(self.pixel_points)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Help text
        help_lines = [
            "Keys:",
            "r = reset ROI",
            "s = save sample",
            "f = fit affine + homography",
            "w = write YAML",
            "c = clear samples",
            "q = quit",
        ]

        y0 = 90
        for i, line in enumerate(help_lines):
            cv2.putText(
                vis,
                line,
                (20, y0 + 25 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return vis

    def run(self):
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            if self.latest_image is None:
                rate.sleep()
                continue

            vis = self.draw_overlay(self.latest_image)

            cv2.imshow(self.window_name, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("r"):
                self.roi_defined = False
                self.roi = None
                self.pt1 = None
                self.pt2 = None
                rospy.loginfo("ROI reset. Draw a new ROI with mouse.")

            elif key == ord("s"):
                self.save_sample()

            elif key == ord("f"):
                aff = self.fit_affine()
                if aff is not None:
                    rospy.loginfo("Affine matrix:\n%s", str(aff))

                H = self.fit_homography()
                if H is not None:
                    rospy.loginfo("Homography matrix:\n%s", str(H))

                self.compute_errors()

            elif key == ord("w"):
                self.save_yaml()

            elif key == ord("c"):
                self.clear_samples()

            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        node = CameraRobot2DCalibrator()
        node.run()
    except rospy.ROSInterruptException:
        pass
