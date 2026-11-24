import cv2
import numpy as np


def apply_gaussian_blur(image):
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    return blurred_image


""" def create_mask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv_image[:, :, 1]
    _, mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return mask """


def create_mask(image):
    height, width = image.shape[:2]

    grabcut_mask = np.zeros((height, width), dtype=np.uint8)

    background_model = np.zeros((1, 65), dtype=np.float64)

    foreground_model = np.zeros((1, 65), dtype=np.float64)

    margin = 5

    rectangle = (
        margin,
        margin,
        width - margin * 2,
        height - margin * 2,
    )

    cv2.grabCut(
        image,
        grabcut_mask,
        rectangle,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT,
    )

    mask = np.where(
        (grabcut_mask == cv2.GC_FGD)
        | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype("uint8")

    return mask


def find_roi_objects(image, mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return image.copy(), None

    leaf_contour = max(
        contours,
        key=cv2.contourArea,
    )

    roi_image = image.copy()

    cv2.drawContours(
        roi_image,
        [leaf_contour],
        -1,
        (0, 255, 0),
        2,
    )

    x, y, width, height = cv2.boundingRect(
        leaf_contour
    )

    cv2.rectangle(
        roi_image,
        (x, y),
        (x + width, y + height),
        (255, 0, 0),
        2,
    )

    return roi_image, leaf_contour


def analyze_object(image, leaf_contour):
    analyzed_image = image.copy()

    if leaf_contour is None:
        return analyzed_image, {}

    area = cv2.contourArea(leaf_contour)

    perimeter = cv2.arcLength(
        leaf_contour,
        True,
    )

    x, y, width, height = cv2.boundingRect(
        leaf_contour
    )

    moments = cv2.moments(leaf_contour)

    if moments["m00"] != 0:
        center_x = int(
            moments["m10"] / moments["m00"]
        )
        center_y = int(
            moments["m01"] / moments["m00"]
        )
    else:
        center_x = 0
        center_y = 0

    aspect_ratio = 0

    if height != 0:
        aspect_ratio = width / height

    circularity = 0

    if perimeter != 0:
        circularity = (
            4 * np.pi * area
        ) / (perimeter * perimeter)

    cv2.drawContours(
        analyzed_image,
        [leaf_contour],
        -1,
        (255, 0, 255),
        2,
    )

    cv2.circle(
        analyzed_image,
        (center_x, center_y),
        5,
        (0, 255, 255),
        -1,
    )

    cv2.line(
        analyzed_image,
        (x, center_y),
        (x + width, center_y),
        (255, 0, 0),
        2,
    )

    cv2.line(
        analyzed_image,
        (center_x, y),
        (center_x, y + height),
        (255, 0, 0),
        2,
    )

    features = {
        "area": area,
        "perimeter": perimeter,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "center_x": center_x,
        "center_y": center_y,
    }

    return analyzed_image, features


def find_pseudolandmarks(
    image,
    leaf_contour,
    number_of_points=20,
):
    landmarks_image = image.copy()
    landmarks = []

    if leaf_contour is None:
        return landmarks_image, landmarks

    contour_points = leaf_contour[:, 0, :]

    point_count = min(
        number_of_points,
        len(contour_points),
    )

    indexes = np.linspace(
        0,
        len(contour_points) - 1,
        point_count,
        dtype=int,
    )

    for index in indexes:
        x = int(contour_points[index][0])
        y = int(contour_points[index][1])

        landmarks.append((x, y))

        cv2.circle(
            landmarks_image,
            (x, y),
            4,
            (255, 0, 255),
            -1,
        )

    return landmarks_image, landmarks


def calculate_color_histograms(image, mask):
    hsv_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    lab_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    channels = {
        "blue": (image, 0),
        "green": (image, 1),
        "red": (image, 2),
        "hue": (hsv_image, 0),
        "saturation": (hsv_image, 1),
        "value": (hsv_image, 2),
        "lightness": (lab_image, 0),
        "green-magenta": (lab_image, 1),
        "blue-yellow": (lab_image, 2),
    }

    histograms = {}

    for channel_name in channels:
        channel_image = channels[channel_name][0]
        channel_index = channels[channel_name][1]

        histogram = cv2.calcHist(
            [channel_image],
            [channel_index],
            mask,
            [256],
            [0, 256],
        )

        total = histogram.sum()

        if total != 0:
            histogram = histogram / total

        histograms[channel_name] = histogram.flatten()

    return histograms


def create_damage_mask(image, leaf_mask):
    hsv_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    yellow_mask = cv2.inRange(
        hsv_image,
        (12, 45, 35),
        (38, 255, 255),
    )

    brown_mask = cv2.inRange(
        hsv_image,
        (0, 45, 20),
        (22, 255, 210),
    )

    red_brown_mask = cv2.inRange(
        hsv_image,
        (165, 45, 20),
        (179, 255, 220),
    )

    dark_mask = cv2.inRange(
        hsv_image,
        (0, 0, 0),
        (179, 255, 65),
    )

    damage_mask = cv2.bitwise_or(
        yellow_mask,
        brown_mask,
    )

    damage_mask = cv2.bitwise_or(
        damage_mask,
        red_brown_mask,
    )

    damage_mask = cv2.bitwise_or(
        damage_mask,
        dark_mask,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    inner_leaf_mask = cv2.erode(
        leaf_mask,
        kernel,
        iterations=1,
    )

    damage_mask = cv2.bitwise_and(
        damage_mask,
        inner_leaf_mask,
    )

    damage_mask = cv2.morphologyEx(
        damage_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    return damage_mask


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


AUGMENTATIONS = {
    "Flip": flip,
    "Rotate": rotate,
    "Skew": skew,
    "Shear": shear,
    "Crop": crop,
    "Distortion": distortion,
}
