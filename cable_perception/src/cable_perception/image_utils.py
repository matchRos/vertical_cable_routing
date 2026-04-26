from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch

import cv2
import matplotlib.pyplot as plt
import numpy as np


def center_pixels_on_cable(
    image: np.ndarray,
    pixels: Iterable[Sequence[int]],
    num_options: int = 10,
    display: bool = False,
) -> np.ndarray:
    image_mask = image[:, :, 0] > 100
    kernel = np.ones((2, 2), np.uint8)
    image_mask = cv2.erode(image_mask.astype(np.uint8), kernel, iterations=1)
    white_pixels = np.argwhere(image_mask)

    processed_pixels = []
    for pixel in pixels:
        distances = np.linalg.norm(white_pixels - pixel, axis=1)
        valid_indices = np.where(distances >= 100)[0]
        if len(valid_indices) > 0:
            sorted_indices = np.argsort(distances[valid_indices])
            selected_pixels = white_pixels[valid_indices[sorted_indices[:num_options]]]
            processed_pixels.append(selected_pixels)

    if display:
        pixels = np.atleast_2d(list(pixels))
        plt.imshow(image_mask, cmap="gray")
        for pixel in pixels:
            plt.scatter(*pixel[::-1], c="r")
        for pixel_set in processed_pixels:
            for pt in pixel_set:
                plt.scatter(*pt[::-1], c="g")
        plt.show()

    return np.array(processed_pixels)


def find_nearest_white_pixel(
    image: np.ndarray,
    clip: dict,
    num_options: int = 10,
    display: bool = False,
) -> np.ndarray:
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image

    clip_pixel = np.array([[clip["y"], clip["x"]]])
    masked_image = cv2.bitwise_and(image_gray, image_gray)
    centered_pixels = center_pixels_on_cable(
        masked_image[..., None],
        clip_pixel,
        num_options=num_options,
        display=display,
    )
    nearest_pixels = centered_pixels[0]

    if display:
        vis = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        cv2.circle(vis, (clip["x"], clip["y"]), 5, (0, 0, 255), -1)
        for pixel in nearest_pixels:
            cv2.circle(vis, (pixel[1], pixel[0]), 5, (255, 255, 0), -1)
        cv2.imshow("Image with Nearest Cable Pixels", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return np.array([(int(pixel[1]), int(pixel[0])) for pixel in nearest_pixels])

def is_cable_in_gripper(
        image: np.ndarray,
        gripper_bbox: Tuple[int, int, int, int],
        clip: dict
) -> bool:
    # gripper_bbox is (x_min, y_min, x_max, y_max)
    bounded_image = image[gripper_bbox[0]:gripper_bbox[2], gripper_bbox[1]:gripper_bbox[3]]
    nearest_cable = find_nearest_white_pixel(
        bounded_image,
        clip,
        num_options=1)
    return len(nearest_cable) > 0

def is_cable_too_lax_nn(
        image: np.ndarray,
        model_type,
        weights
 ) -> bool:
    # This assumes we've retrained a model like ResNet or YOLO for cable laxness classification using PyTorch
    model = model_type(weights = weights)
    model.eval()
    image_transformed = model.preprocess(image)
    with torch.no_grad():
        # This inferences the model and gets the predicted class (0 for not too lax, 1 for too lax)
        output = torch.argmax(model(image_transformed.unsqueeze(0)), dim = 1) 
    return output.item() == 1

def is_cable_too_lax_contours(
        image: np.ndarray,
        cable_bbox: Tuple[int, int, int, int],
        area_threshold: float = 500.0
) -> bool:
    # cable_bbox is (x_min, y_min, x_max, y_max)
    # area_threshold is the threshold to avoid counting pegs (which have a large area) as cables

    bounded_image = image[cable_bbox[0]:cable_bbox[2], cable_bbox[1]:cable_bbox[3]] # Crops image to a single peg-to-peg segment
    gray = cv2.cvtColor(bounded_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    useful_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < area_threshold]
    if len(useful_contours) == 0:
        return False
    for cnt in useful_contours:
        vx, vy, x0, y0 = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        slope = vy / vx
        avg_dist = 0
        k = 0
        for point in cnt:
            distance = abs(slope * (point[0][0] - x0) - (point[0][1] - y0))
            avg_dist += distance
            k += 1
        if k > 0:
            avg_dist /= k
        if avg_dist > 2:  # Threshold for deviation from straight line where we guess it's drooping
            return True
    return False
    