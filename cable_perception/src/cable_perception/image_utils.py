from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


def center_pixels_on_cable(
    image: np.ndarray,
    pixels: Iterable[Sequence[int]],
    num_options: int = 10,
    display: bool = False,
    min_distance_px: float = 0.0,
    max_distance_px: float | None = None,
) -> np.ndarray:
    image_mask = image[:, :, 0] > 100
    kernel = np.ones((2, 2), np.uint8)
    image_mask = cv2.erode(image_mask.astype(np.uint8), kernel, iterations=1)
    white_pixels = np.argwhere(image_mask)

    processed_pixels = []
    for pixel in pixels:
        if len(white_pixels) == 0:
            continue
        pixel_arr = np.asarray(pixel, dtype=float).reshape(1, 2)
        delta = white_pixels.astype(float) - pixel_arr
        distances_sq = np.einsum("ij,ij->i", delta, delta)
        valid = np.ones(len(white_pixels), dtype=bool)
        if min_distance_px > 0.0:
            valid &= distances_sq >= float(min_distance_px) ** 2
        if max_distance_px is not None:
            valid &= distances_sq <= float(max_distance_px) ** 2
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) == 0:
            if max_distance_px is not None:
                continue
            valid_indices = np.arange(len(white_pixels))
        sorted_indices = np.argsort(distances_sq[valid_indices])
        selected_pixels = white_pixels[valid_indices[sorted_indices[:num_options]]]
        processed_pixels.append(selected_pixels)

    if display:
        import matplotlib.pyplot as plt

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
    min_distance_px: float = 0.0,
    max_distance_px: float | None = None,
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
        min_distance_px=min_distance_px,
        max_distance_px=max_distance_px,
    )
    if len(centered_pixels) == 0:
        return []
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
