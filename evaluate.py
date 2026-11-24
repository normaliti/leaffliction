#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)


def save_validation_data(features, labels, paths, output_path):
    validation_data = {
        "features": features,
        "labels": list(labels),
        "paths": [str(path) for path in paths],
    }
    joblib.dump(validation_data, output_path)
    print("Validation data saved to:", output_path)


def load_evaluation_data(output_directory):
    model_path = output_directory / "leaf_model.joblib"
    validation_path = output_directory / "validation_features.joblib"
    saved_data = joblib.load(model_path)
    validation_data = joblib.load(validation_path)
    return saved_data, validation_data


def prepare_validation_features(saved_data, validation_data):
    features = validation_data["features"]
    columns_to_keep = saved_data["columns_to_keep"]
    features = features[columns_to_keep]
    scaler = saved_data["scaler"]
    return scaler.transform(features)


def evaluate_saved_model(saved_data, validation_data):
    features = prepare_validation_features(saved_data, validation_data)
    labels = validation_data["labels"]
    predictions = saved_data["model"].predict(features)
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions)
    return predictions, accuracy, report


def create_confusion_matrix(labels, predictions):
    figure, axis = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        labels, predictions, xticks_rotation=45, cmap="Blues", ax=axis
    )
    axis.set_title("Validation confusion matrix")
    figure.tight_layout()
    return figure


def save_evaluation_results(labels, predictions, paths, output_directory):
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions)
    report_content = (
        "Validation images: " + str(len(labels)) + "\n"
        "Validation accuracy: " + str(accuracy) + "\n\n" + report
    )
    report_path = output_directory / "validation_report.txt"
    report_path.write_text(report_content, encoding="utf-8")

    results = pd.DataFrame({
        "image_path": [str(path) for path in paths],
        "label": labels,
        "prediction": predictions,
    })
    results["correct"] = results["label"] == results["prediction"]
    results_path = output_directory / "validation_predictions.csv"
    results.to_csv(results_path, index=False)

    figure = create_confusion_matrix(labels, predictions)
    figure.savefig(
        output_directory / "confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the saved model on the validation set"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("training_output"),
        help="directory with the model and validation features",
    )
    return parser.parse_args()


def main():
    arguments = parse_args()

    try:
        saved_data, validation_data = load_evaluation_data(arguments.output)
        predictions, accuracy, report = evaluate_saved_model(
            saved_data, validation_data
        )
        labels = validation_data["labels"]
        print("Validation images:", len(labels))
        print("Validation accuracy:", accuracy)
        print()
        print(report)
        create_confusion_matrix(labels, predictions)
        plt.show()
    except (OSError, ValueError, KeyError) as error:
        print("Error:", error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
