#!/usr/bin/env python3
"""Data augmentation.

Command line:
    ./Augmentation.py "images/Apple_rust/image (1).JPG"
    ./Augmentation.py images --output augmented_directory

From code:
    from Augmentation import augment_single_image, augment_dataset

    augment_single_image(Path("leaf.JPG"), show=False)

    counts = augment_dataset(Path("images"), Path("augmented_directory"))
    # {"Apple_rust": (275, 1365), ...}
"""

import argparse
import shutil
import sys
from pathlib import Path

from image_io import (
    find_image_directories,
    list_images,
    load_image,
    path_with_suffix,
    save_image,
    show_images,
)
from transforms import AUGMENTATIONS

AUGMENTATION_PRIORITY = [
    "Flip",
    "Rotate",
    "Skew",
    "Shear",
    "Crop",
    "Distortion",
]


def augment_single_image(image_path, show=True):
    original = load_image(image_path)
    augmented_by_name = {
        name: AUGMENTATIONS[name](original)
        for name in AUGMENTATION_PRIORITY
    }
    for name, augmented in augmented_by_name.items():
        save_image(augmented, path_with_suffix(image_path, name))
    if show:
        show_images(
            {"Original": original, **augmented_by_name}, image_path.name
        )


def plan_augmentations(image_paths, missing_count):
    """Return {image path: [augmentation names]}.

    Images are augmented in rounds. In round R, image I gets
    augmentation number (I + R) of the priority list, wrapping around.
    """
    if missing_count <= 0:
        return {}
    full_rounds, extra_images = divmod(missing_count, len(image_paths))
    plan = {}
    for image_index, image_path in enumerate(image_paths):
        rounds_count = full_rounds + (1 if image_index < extra_images else 0)
        plan[image_path] = augmentations_starting_at(image_index, rounds_count)
    return plan


def augmentations_starting_at(start_index, count):
    count = min(count, len(AUGMENTATION_PRIORITY))
    return [
        AUGMENTATION_PRIORITY[(start_index + offset)
                              % len(AUGMENTATION_PRIORITY)]
        for offset in range(count)
    ]


def balance_class(class_directory, target_count):
    original_paths = list_images(class_directory)
    missing_count = target_count - len(original_paths)
    plan = plan_augmentations(original_paths, missing_count)
    for image_path, augmentation_names in plan.items():
        image = load_image(image_path)
        for name in augmentation_names:
            augmented = AUGMENTATIONS[name](image)
            save_image(augmented, path_with_suffix(image_path, name))
    added_count = sum(len(names) for names in plan.values())
    return len(original_paths), added_count


def augment_dataset(source_directory, output_directory):
    """Return {class name: (original count, added count)}."""
    if output_directory.exists():
        raise ValueError(f"'{output_directory}' already exists")
    shutil.copytree(source_directory, output_directory)

    class_directories = find_image_directories(output_directory)
    if not class_directories:
        raise ValueError(f"no images found in '{source_directory}'")
    target_count = max(
        len(list_images(directory)) for directory in class_directories
    )

    return {
        directory.name: balance_class(directory, target_count)
        for directory in class_directories
    }


def print_counts(counts_by_class):
    for class_name, (original_count, added_count) in counts_by_class.items():
        total_count = original_count + added_count
        print(f"{class_name}: "
              f"{original_count} + {added_count} = {total_count}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment one image or balance a whole dataset."
    )
    parser.add_argument("path", type=Path, help="an image or a dataset")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("augmented_directory"),
        help="where to write the balanced dataset (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    arguments = parse_args()
    try:
        if arguments.path.is_dir():
            counts_by_class = augment_dataset(arguments.path, arguments.output)
            print_counts(counts_by_class)
        elif arguments.path.is_file():
            augment_single_image(arguments.path)
        else:
            raise ValueError(f"'{arguments.path}' does not exist")
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
