from shutil import copy2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from image_io import list_images


def load_data(csv_path):
    data = pd.read_csv(csv_path)
    image_paths = data["image_path"]
    labels = data["label"]
    features = data.drop(columns=["image_path", "label"])

    return features, labels, image_paths


def collect_dataset_images(dataset_directory):
    if not dataset_directory.is_dir():
        raise ValueError("dataset directory does not exist")

    image_paths = []
    labels = []

    for class_directory in sorted(dataset_directory.iterdir()):
        if not class_directory.is_dir():
            continue

        for image_path in list_images(class_directory):
            image_paths.append(image_path)
            labels.append(class_directory.name)

    if not image_paths:
        raise ValueError("no images found in dataset")

    return image_paths, labels


def split_image_paths(image_paths, labels, validation_size=0.2):
    return train_test_split(
        image_paths, labels, test_size=validation_size,
        random_state=42, stratify=labels,
    )


def split_data(features, labels, image_paths):
    return train_test_split(
        features, labels, image_paths, test_size=0.2,
        random_state=42, stratify=labels,
    )


def normalize_data(features_train, features_test):
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    return features_train_scaled, features_test_scaled, scaler


def remove_constant_features(features_train, features_test):
    columns_to_keep = []

    for column_name in features_train.columns:
        unique_values_count = features_train[column_name].nunique()

        if unique_values_count > 1:
            columns_to_keep.append(column_name)

    features_train = features_train[columns_to_keep]
    features_test = features_test[columns_to_keep]

    return features_train, features_test, columns_to_keep


def create_subset_directory(image_paths, labels, output_directory):
    if len(image_paths) != len(labels):
        raise ValueError("paths and labels have different lengths")

    if output_directory.exists():
        raise ValueError("output directory already exists")

    for image_path, label in zip(image_paths, labels):
        class_directory = output_directory / label
        class_directory.mkdir(parents=True, exist_ok=True)
        output_path = class_directory / image_path.name
        copy2(image_path, output_path)

    return len(image_paths)


def main():
    features, labels, image_paths = load_data("features.csv")

    (
        features_train,
        features_test,
        labels_train,
        labels_test,
        paths_train,
        paths_test,
    ) = split_data(features, labels, image_paths)

    (
        features_train,
        features_test,
        columns_to_keep,
    ) = remove_constant_features(features_train, features_test)

    (
        features_train_scaled,
        features_test_scaled,
        scaler,
    ) = normalize_data(features_train, features_test)

    print("Training images:", len(labels_train))
    print("Test images:", len(labels_test))
    print("Features before:", features.shape[1])
    print("Features after:", len(columns_to_keep))
    print("Training data shape:", features_train_scaled.shape)
    print("Test data shape:", features_test_scaled.shape)


if __name__ == "__main__":
    main()
