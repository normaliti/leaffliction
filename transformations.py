"""Image transformations.

Every transformation takes an OpenCV image (numpy array, BGR)
and returns a new OpenCV image of the same size.
"""

import cv2
import numpy as np


def flip(image):
    return cv2.flip(image, 1)


def rotate(image, angle=30):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return _warp_affine(image, matrix)


def shear(image, factor=0.2):
    height = image.shape[0]
    matrix = np.float32([
        [1, factor, -factor * height / 2],
        [0, 1, 0],
    ])
    return _warp_affine(image, matrix)


def skew(image, strength=0.2):
    height, width = image.shape[:2]
    shift = width * strength
    corners = np.float32([
        [0, 0], [width, 0], [width, height], [0, height],
    ])
    skewed_corners = np.float32([
        [shift, 0], [width - shift, 0], [width, height], [0, height],
    ])
    matrix = cv2.getPerspectiveTransform(corners, skewed_corners)
    return cv2.warpPerspective(
        image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT
    )


def crop(image, keep_ratio=0.8):
    height, width = image.shape[:2]
    crop_height = int(height * keep_ratio)
    crop_width = int(width * keep_ratio)
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    cropped = image[top:top + crop_height, left:left + crop_width]
    return cv2.resize(cropped, (width, height))


def distortion(image, amplitude=4, wavelength=120):
    height, width = image.shape[:2]
    x_grid, y_grid = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    x_wave = amplitude * np.sin(2 * np.pi * y_grid / wavelength)
    y_wave = amplitude * np.sin(2 * np.pi * x_grid / wavelength)
    return cv2.remap(
        image, x_grid + x_wave, y_grid + y_wave,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )


def _warp_affine(image, matrix):
    height, width = image.shape[:2]
    return cv2.warpAffine(
        image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT
    )


TRANSFORMATIONS = {
    "Flip": flip,
    "Rotate": rotate,
    "Skew": skew,
    "Shear": shear,
    "Crop": crop,
    "Distortion": distortion,
}
