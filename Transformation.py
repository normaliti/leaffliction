#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from image_io import list_images, save_image
from transforms import (
    apply_gaussian_blur,
    create_mask,
    find_roi_objects,
    analyze_object,
    find_pseudolandmarks,
    calculate_color_histograms,
)

TRANSFORMATION_OPTIONS = {
    "blur": ("Gaussian blur", "GaussianBlur"),
    "mask": ("Mask", "Mask"),
    "roi": ("ROI objects", "ROI"),
    "analyze": ("Analyze object", "Analyze"),
    "landmarks": ("Pseudolandmarks", "Pseudolandmarks"),
}

ALL_TRANSFORMATIONS = [
    "blur",
    "mask",
    "roi",
    "analyze",
    "landmarks",
    "histogram",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply transformations to leaf images"
    )
    parser.add_argument(
        "image", nargs="?", type=Path, help="path to one image"
    )
    parser.add_argument("-src", type=Path, help="source directory")
    parser.add_argument("-dst", type=Path, help="destination directory")
    parser.add_argument(
        "-blur", action="store_true", help="save Gaussian blur"
    )
    parser.add_argument("-mask", action="store_true", help="save leaf mask")
    parser.add_argument("-roi", action="store_true", help="save ROI objects")
    parser.add_argument(
        "-analyze", action="store_true", help="save object analysis"
    )
    parser.add_argument(
        "-landmarks", action="store_true", help="save pseudolandmarks"
    )
    parser.add_argument(
        "-histogram", action="store_true", help="save color histogram"
    )
    parser.add_argument(
        "-all", action="store_true", help="save all transformations"
    )
    return parser.parse_args()


def load_image(image_path):
    if not image_path.exists():
        print("Error: image does not exist")
        return
    if not image_path.is_file():
        print("Error: given path is not a file")
        return
    image = cv2.imread(str(image_path))

    if image is None:
        print("Error: could not load a file")
        return None
    return image


def create_transformations(image):
    blurred_image = apply_gaussian_blur(image)
    mask = create_mask(blurred_image)
    roi_image, leaf_contour = find_roi_objects(image, mask)
    analyzed_image, _ = analyze_object(image, leaf_contour)
    landmarks_image, _ = find_pseudolandmarks(image, leaf_contour)
    histograms = calculate_color_histograms(image, mask)

    images = {
        "Original": image,
        "Gaussian blur": blurred_image,
        "Mask": mask,
        "ROI objects": roi_image,
        "Analyze object": analyzed_image,
        "Pseudolandmarks": landmarks_image,
    }

    return images, histograms


def show_transformations(images, histograms):
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for axis, (title, image) in zip(axes, images.items()):
        if len(image.shape) == 2:
            axis.imshow(image, cmap="gray")
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axis.imshow(rgb_image)

        axis.set_title(title)
        axis.axis("off")

    colors = {
        "blue": "blue",
        "green": "green",
        "red": "red",
        "hue": "purple",
        "saturation": "cyan",
        "value": "orange",
        "lightness": "gray",
        "green-magenta": "magenta",
        "blue-yellow": "gold",
    }

    histogram_axis = axes[len(images)]

    for channel_name, histogram in histograms.items():
        histogram_axis.plot(
            histogram,
            color=colors[channel_name],
            label=channel_name,
        )

    histogram_axis.set_title("Color histogram")
    histogram_axis.set_xlabel("Pixel intensity")
    histogram_axis.set_ylabel("Proportion")
    histogram_axis.legend(fontsize=7, ncol=2)

    for axis in axes[len(images) + 1:]:
        axis.axis("off")

    figure.tight_layout()
    plt.show()


def get_selected_transformations(arguments):
    if arguments.all:
        return list(ALL_TRANSFORMATIONS)

    selected = []

    for name in ALL_TRANSFORMATIONS:
        if getattr(arguments, name):
            selected.append(name)

    return selected


def save_histogram(histograms, output_path):
    colors = {
        "blue": "blue",
        "green": "green",
        "red": "red",
        "hue": "purple",
        "saturation": "cyan",
        "value": "orange",
        "lightness": "gray",
        "green-magenta": "magenta",
        "blue-yellow": "gold",
    }

    figure, axis = plt.subplots(figsize=(8, 5))

    for channel_name, histogram in histograms.items():
        color = colors[channel_name]
        axis.plot(histogram, color=color, label=channel_name)

    axis.set_title("Color histogram")
    axis.set_xlabel("Pixel intensity")
    axis.set_ylabel("Proportion")
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_transformations(
    source_directory,
    destination_directory,
    selected_transformations,
):
    if not source_directory.is_dir():
        print("Error: source directory does not exist")
        return

    destination_directory.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(source_directory)

    if not image_paths:
        print("Error: no images found")
        return

    saved_count = 0

    for image_path in image_paths:
        image = load_image(image_path)

        if image is None:
            continue

        images, histograms = create_transformations(image)

        for name in selected_transformations:
            if name == "histogram":
                output_name = image_path.stem + "_ColorHistogram.png"
                output_path = destination_directory / output_name
                save_histogram(histograms, output_path)
            else:
                title, suffix = TRANSFORMATION_OPTIONS[name]
                extension = ".png" if name == "mask" else image_path.suffix
                output_name = image_path.stem + "_" + suffix + extension
                output_path = destination_directory / output_name
                save_image(images[title], output_path)

            saved_count += 1

    print("Saved transformations:", saved_count)


def main():
    arguments = parse_args()

    if arguments.image is not None:
        image = load_image(arguments.image)

        if image is None:
            return

        images, histograms = create_transformations(image)
        show_transformations(images, histograms)
        return

    if arguments.src is not None and arguments.dst is not None:
        selected = get_selected_transformations(arguments)

        if not selected:
            print("Error: choose at least one transformation or -all")
            return

        save_transformations(arguments.src, arguments.dst, selected)
        return

    print("Error: provide an image or use -src and -dst")


if __name__ == "__main__":
    main()
