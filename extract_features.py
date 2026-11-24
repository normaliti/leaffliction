import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd

from features import (
    build_feature_vector,
    extract_color_features,
    extract_damage_features,
    extract_shape_features,
    extract_texture_features,
)
from transforms import (
    apply_gaussian_blur,
    create_damage_mask,
    create_mask,
    find_roi_objects,
)

IMAGES_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def extract_image_features(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not load image:", image_path)
        return None

    blurred_image = apply_gaussian_blur(image)
    leaf_mask = create_mask(blurred_image)
    _, leaf_contour = find_roi_objects(image, leaf_mask)

    if leaf_contour is None:
        print("Could not find leaf:", image_path)
        return None

    damage_mask = create_damage_mask(image, leaf_mask)
    shape_features = extract_shape_features(image, leaf_contour)
    color_features = extract_color_features(image, leaf_mask)
    damage_features = extract_damage_features(damage_mask, leaf_mask)
    texture_features = extract_texture_features(image, leaf_mask)

    feature_vector = build_feature_vector(
        shape_features, color_features, damage_features, texture_features
    )

    return feature_vector


def build_feature_table(image_paths, labels):
    if len(image_paths) != len(labels):
        raise ValueError("paths and labels have different lengths")

    feature_rows = []
    valid_labels = []
    valid_paths = []
    image_data = zip(image_paths, labels)

    for index, (image_path, label) in enumerate(image_data, start=1):
        features = extract_image_features(image_path)

        if features is None:
            continue

        feature_rows.append(features)
        valid_labels.append(label)
        valid_paths.append(image_path)

        if index % 100 == 0:
            print("Extracted features:", index)

    if not feature_rows:
        raise ValueError("could not extract any features")

    feature_table = pd.DataFrame(feature_rows)

    return feature_table, valid_labels, valid_paths


def extract_dataset_features(dataset_directory, output_path, limit=None):
    processed_count = 0
    skipped_count = 0
    writer = None

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        for class_directory in sorted(dataset_directory.iterdir()):
            if not class_directory.is_dir():
                continue

            label = class_directory.name

            for image_path in sorted(class_directory.rglob("*")):
                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in IMAGES_EXT:
                    continue

                if limit is not None and processed_count >= limit:
                    break

                features = extract_image_features(image_path)

                if features is None:
                    skipped_count += 1
                    continue

                if writer is None:
                    field_names = ["image_path", "label"]
                    field_names.extend(features.keys())
                    writer = csv.DictWriter(csv_file, fieldnames=field_names)
                    writer.writeheader()

                row = {"image_path": str(image_path), "label": label}
                row.update(features)
                writer.writerow(row)
                processed_count += 1

                if processed_count % 100 == 0:
                    print("Processed:", processed_count)

            if limit is not None and processed_count >= limit:
                break

    print("Finished.")
    print("Processed:", processed_count)
    print("Skipped:", skipped_count)
    print("Saved to:", output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract features from leaf images."
    )
    parser.add_argument("directory", help="Path to the dataset directory.")
    parser.add_argument("output", help="Path to the output CSV file.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N images.",
    )

    return parser.parse_args()


def main():
    arguments = parse_args()
    dataset_directory = Path(arguments.directory)
    output_path = Path(arguments.output)

    if not dataset_directory.exists():
        print("Dataset directory does not exist.")
        return

    if not dataset_directory.is_dir():
        print("Dataset path is not a directory.")
        return

    extract_dataset_features(dataset_directory, output_path, arguments.limit)


if __name__ == "__main__":
    main()
