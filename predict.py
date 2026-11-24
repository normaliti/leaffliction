#!/usr/bin/env python3

import argparse
from pathlib import Path

import joblib
import pandas as pd

from extract_features import extract_image_features
from image_io import load_image, show_images
from transforms import apply_gaussian_blur, create_mask


def load_saved_data(model_path):
    training_data = joblib.load(model_path)
    return training_data


def prepare_image_features(image_path, saved_data):
    features = extract_image_features(image_path)

    if features is None:
        return None

    features_table = pd.DataFrame([features])
    columns_to_keep = saved_data["columns_to_keep"]
    features_table = features_table[columns_to_keep]
    scaler = saved_data["scaler"]
    scaled_features = scaler.transform(features_table)

    return scaled_features


def predict_class(scaled_features, model):
    predictions = model.predict(scaled_features)
    return predictions[0]


def show_prediction(image_path, predicted_class):
    original_image = load_image(image_path)
    blurred_image = apply_gaussian_blur(original_image)
    mask = create_mask(blurred_image)
    transformed_image = original_image.copy()
    transformed_image[mask == 0] = 255

    images = {
        "Original": original_image,
        "Transformed": transformed_image,
    }

    show_images(
        images, window_title="Class predicted: " + predicted_class
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict a leaf disease"
    )
    parser.add_argument(
        "image_path", type=Path, help="path to a leaf image"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        default=Path("training_output/leaf_model.joblib"),
        help="path to the trained model",
    )

    return parser.parse_args()


def main():
    arguments = parse_args()
    saved_data = load_saved_data(arguments.model)
    scaled_features = prepare_image_features(
        arguments.image_path, saved_data
    )

    if scaled_features is None:
        print("Could not prepare image features.")
        return

    model = saved_data["model"]
    predicted_class = predict_class(scaled_features, model)
    print("Class predicted:", predicted_class)
    show_prediction(arguments.image_path, predicted_class)


if __name__ == "__main__":
    main()
