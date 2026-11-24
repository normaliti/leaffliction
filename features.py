import cv2
import numpy as np


def extract_shape_features(image, leaf_contour):
    if leaf_contour is None:
        return None

    image_height, image_width = image.shape[:2]
    image_area = image_height * image_width

    area = cv2.contourArea(leaf_contour)

    perimeter = cv2.arcLength(
        leaf_contour,
        True,
    )

    x, y, width, height = cv2.boundingRect(
        leaf_contour
    )

    rectangle_area = width * height

    convex_hull = cv2.convexHull(leaf_contour)
    convex_hull_area = cv2.contourArea(convex_hull)

    area_ratio = 0
    perimeter_normalized = 0
    aspect_ratio = 0
    extent = 0
    solidity = 0
    circularity = 0
    eccentricity = 0

    if image_area != 0:
        area_ratio = area / image_area

    if area != 0:
        perimeter_normalized = (
            perimeter / np.sqrt(area)
        )

    if height != 0:
        aspect_ratio = width / height

    if rectangle_area != 0:
        extent = area / rectangle_area

    if convex_hull_area != 0:
        solidity = area / convex_hull_area

    if perimeter != 0:
        circularity = (
            4 * np.pi * area
        ) / (perimeter * perimeter)

    if len(leaf_contour) >= 5:
        ellipse = cv2.fitEllipse(leaf_contour)
        axes = ellipse[1]

        major_axis = max(axes)
        minor_axis = min(axes)

        if major_axis != 0:
            eccentricity = np.sqrt(
                1
                - (minor_axis / major_axis) ** 2
            )

    features = {
        "area_ratio": area_ratio,
        "perimeter_normalized": perimeter_normalized,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "circularity": circularity,
        "eccentricity": eccentricity,
    }

    moments = cv2.moments(leaf_contour)
    hu_moments = cv2.HuMoments(moments).flatten()

    for index in range(len(hu_moments)):
        value = hu_moments[index]

        if value != 0:
            value = (
                -np.sign(value)
                * np.log10(abs(value))
            )

        feature_name = "hu_" + str(index + 1)
        features[feature_name] = value

    return features


def extract_color_features(
    image,
    mask,
    number_of_bins=16,
):
    if mask is None:
        return None

    if cv2.countNonZero(mask) == 0:
        return None

    hsv_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    lab_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    channels = {
        "hue": (hsv_image, 0, 180),
        "saturation": (hsv_image, 1, 256),
        "value": (hsv_image, 2, 256),
        "lightness": (lab_image, 0, 256),
        "green_magenta": (lab_image, 1, 256),
        "blue_yellow": (lab_image, 2, 256),
    }

    features = {}

    for channel_name in channels:
        channel_image = channels[channel_name][0]
        channel_index = channels[channel_name][1]
        maximum_value = channels[channel_name][2]

        channel = channel_image[:, :, channel_index]

        mean, standard_deviation = cv2.meanStdDev(
            channel,
            mask=mask,
        )

        features[channel_name + "_mean"] = float(
            mean[0][0]
        )

        features[channel_name + "_std"] = float(
            standard_deviation[0][0]
        )

        histogram = cv2.calcHist(
            [channel_image],
            [channel_index],
            mask,
            [number_of_bins],
            [0, maximum_value],
        )

        total = histogram.sum()

        if total != 0:
            histogram = histogram / total

        histogram = histogram.flatten()

        for index in range(number_of_bins):
            feature_name = (
                channel_name
                + "_hist_"
                + str(index)
            )

            features[feature_name] = float(
                histogram[index]
            )

    return features


def extract_damage_features(
    damage_mask,
    leaf_mask,
):
    leaf_area = cv2.countNonZero(leaf_mask)

    if leaf_area == 0:
        return None

    damage_area = cv2.countNonZero(
        damage_mask
    )

    contours, _ = cv2.findContours(
        damage_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    minimum_spot_area = max(
        5,
        leaf_area * 0.0005,
    )

    spot_areas = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= minimum_spot_area:
            spot_areas.append(area)

    spots_count = len(spot_areas)
    average_spot_area_ratio = 0
    largest_spot_area_ratio = 0

    if spots_count != 0:
        average_spot_area = (
            sum(spot_areas) / spots_count
        )

        largest_spot_area = max(spot_areas)

        average_spot_area_ratio = (
            average_spot_area / leaf_area
        )

        largest_spot_area_ratio = (
            largest_spot_area / leaf_area
        )

    features = {
        "damage_ratio": (
            damage_area / leaf_area
        ),
        "spots_count": spots_count,
        "average_spot_area_ratio":
            average_spot_area_ratio,
        "largest_spot_area_ratio":
            largest_spot_area_ratio,
    }

    return features


def extract_texture_features(image, mask):
    if mask is None:
        return None

    leaf_area = cv2.countNonZero(mask)

    if leaf_area == 0:
        return None

    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    edges = cv2.Canny(
        gray_image,
        100,
        200,
    )

    leaf_edges = cv2.bitwise_and(
        edges,
        mask,
    )

    edge_density = (
        cv2.countNonZero(leaf_edges)
        / leaf_area
    )

    laplacian = cv2.Laplacian(
        gray_image,
        cv2.CV_64F,
    )

    leaf_laplacian = laplacian[mask > 0]

    laplacian_variance = float(
        leaf_laplacian.var()
    )

    histogram = cv2.calcHist(
        [gray_image],
        [0],
        mask,
        [256],
        [0, 256],
    ).flatten()

    probabilities = histogram / leaf_area
    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = float(
        -np.sum(
            probabilities
            * np.log2(probabilities)
        )
    )

    features = {
        "edge_density": edge_density,
        "laplacian_variance":
            laplacian_variance,
        "entropy": entropy,
    }

    return features


def build_feature_vector(
    shape_features,
    color_features,
    damage_features,
    texture_features,
):
    feature_groups = [
        shape_features,
        color_features,
        damage_features,
        texture_features,
    ]

    for feature_group in feature_groups:
        if feature_group is None:
            return None

    all_features = {}

    for feature_group in feature_groups:
        all_features.update(feature_group)

    return all_features
