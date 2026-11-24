#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.svm import SVC

from Augmentation import augment_dataset, print_counts
from evaluate import save_evaluation_results, save_validation_data
from extract_features import build_feature_table
from prepare_data import (
    collect_dataset_images,
    create_subset_directory,
    normalize_data,
    remove_constant_features,
    split_image_paths,
)


def prepare_training_images(
    dataset_directory, output_directory, validation_size=0.2
):
    image_paths, labels = collect_dataset_images(dataset_directory)

    (
        training_paths,
        validation_paths,
        training_labels,
        validation_labels,
    ) = split_image_paths(image_paths, labels, validation_size)

    output_directory.mkdir(parents=True, exist_ok=True)
    augmented_directory = output_directory / "augmented_training"

    if augmented_directory.exists():
        raise ValueError("augmented training directory already exists")

    with TemporaryDirectory() as temporary_directory:
        training_directory = Path(temporary_directory) / "training"
        create_subset_directory(
            training_paths, training_labels, training_directory
        )
        counts_by_class = augment_dataset(
            training_directory, augmented_directory
        )

    return (
        augmented_directory, validation_paths,
        validation_labels, counts_by_class,
    )


def prepare_feature_tables(
    augmented_directory, validation_paths, validation_labels
):
    training_paths, training_labels = collect_dataset_images(
        augmented_directory
    )

    training_data = build_feature_table(training_paths, training_labels)
    training_features, training_labels, training_paths = training_data

    validation_data = build_feature_table(validation_paths, validation_labels)
    validation_features, validation_labels, validation_paths = validation_data

    return (
        training_features,
        validation_features,
        training_labels,
        validation_labels,
        training_paths,
        validation_paths,
    )


def train_model(features_train, labels_train):
    model = SVC(kernel="rbf", class_weight="balanced")
    model.fit(features_train, labels_train)

    return model


def evaluate_model(model, features_validation, labels_validation):
    predictions = model.predict(features_validation)
    accuracy = accuracy_score(labels_validation, predictions)

    print("Validation accuracy:", accuracy)
    print()
    print(classification_report(labels_validation, predictions))

    return predictions


def save_training(model, scaler, columns_to_keep, output_path):
    training_data = {
        "model": model,
        "scaler": scaler,
        "columns_to_keep": columns_to_keep,
    }

    joblib.dump(training_data, output_path)
    print("Model saved to:", output_path)


def show_confusion_matrix(labels_validation, predictions):
    ConfusionMatrixDisplay.from_predictions(
        labels_validation, predictions, xticks_rotation=45, cmap="Blues"
    )
    plt.title("Validation confusion matrix")
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a leaf disease classifier"
    )
    parser.add_argument("directory", type=Path, help="dataset directory")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("training_output"),
        help="training output directory",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="validation proportion",
    )

    return parser.parse_args()


def main():
    arguments = parse_args()

    try:
        prepared_images = prepare_training_images(
            arguments.directory, arguments.output,
            arguments.validation_size,
        )
        (
            augmented_directory,
            validation_paths,
            validation_labels,
            counts_by_class,
        ) = prepared_images

        print()
        print("Training balance after augmentation:")
        print_counts(counts_by_class)

        prepared_features = prepare_feature_tables(
            augmented_directory, validation_paths, validation_labels
        )
        (
            training_features,
            validation_features,
            training_labels,
            validation_labels,
            _,
            validation_paths,
        ) = prepared_features

        if len(validation_labels) < 100:
            print("Warning: validation contains fewer than 100 images")

        validation_data_path = arguments.output / "validation_features.joblib"
        save_validation_data(
            validation_features,
            validation_labels,
            validation_paths,
            validation_data_path,
        )

        (
            training_features,
            validation_features,
            columns_to_keep,
        ) = remove_constant_features(training_features, validation_features)

        (
            training_features,
            validation_features,
            scaler,
        ) = normalize_data(training_features, validation_features)

        print()
        print("Training images:", len(training_labels))
        print("Validation images:", len(validation_labels))
        print("Features:", len(columns_to_keep))
        print("Training model...")

        model = train_model(training_features, training_labels)
        predictions = evaluate_model(
            model, validation_features, validation_labels
        )

        model_path = arguments.output / "leaf_model.joblib"
        save_training(model, scaler, columns_to_keep, model_path)
        save_evaluation_results(
            validation_labels,
            predictions,
            validation_paths,
            arguments.output,
        )
        show_confusion_matrix(validation_labels, predictions)

    except (OSError, ValueError) as error:
        print("Error:", error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
