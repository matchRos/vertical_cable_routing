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
) -> List[Tuple[int, int]]:
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

    return [(int(pixel[1]), int(pixel[0])) for pixel in nearest_pixels]

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

def is_cable_too_lax(
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
    