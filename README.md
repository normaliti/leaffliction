# Leaffliction

Leaffliction is a classical computer vision and machine learning project for
classifying diseases in apple and grape leaves.

The project analyzes the dataset distribution, balances classes with image
augmentation, visualizes image transformations, extracts handcrafted features
and trains an RBF Support Vector Classifier.

The classifier recognizes eight classes:

- `Apple_Black_rot`
- `Apple_healthy`
- `Apple_rust`
- `Apple_scab`
- `Grape_Black_rot`
- `Grape_Esca`
- `Grape_healthy`
- `Grape_spot`

## Requirements

- Python 3
- `pip`
- `unzip`
- `zip`

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make the main programs executable if necessary:

```bash
chmod +x Distribution.py Augmentation.py Transformation.py train.py predict.py \
    evaluate.py
```

## Prepare the data

The datasets are not stored in the Git repository.

Download the training dataset from the location provided for the project and
place its archive in the project root:

```text
leaves.zip
```

Extract the training dataset:

```bash
unzip leaves.zip
```

The extracted training directory should look like this:

```text
images/
├── Apple_Black_rot/
├── Apple_healthy/
├── Apple_rust/
├── Apple_scab/
├── Grape_Black_rot/
├── Grape_Esca/
├── Grape_healthy/
└── Grape_spot/
```

## Part 1: Dataset distribution

Display the number and proportion of images in every class:

```bash
./Distribution.py images
```

The program prints the class counts and displays a pie chart and a bar chart
for each plant.

## Part 2: Data augmentation

### One image

The single-image mode creates six transformed images next to the input image.
To keep the original dataset unchanged, demonstrate it on a copy:

```bash
DEMO_DIR=/tmp/leaffliction_augmentation_demo
mkdir -p "$DEMO_DIR"
cp "images/Apple_healthy/image (1).JPG" "$DEMO_DIR"
./Augmentation.py "$DEMO_DIR/image (1).JPG"
ls "$DEMO_DIR"
```

The following transformations are created:

```text
Flip
Rotate
Skew
Shear
Crop
Distortion
```

### Balance a dataset

Create a balanced copy of the complete dataset:

```bash
./Augmentation.py images -o augmented_directory
```

The source directory is not modified. The output directory must not already
exist.

Verify the balance visually:

```bash
./Distribution.py augmented_directory
```

All bars should have the same height and all parts of each pie chart should be
equal.

## Part 3: Image transformation

Display all transformations for one image:

```bash
./Transformation.py "images/Apple_healthy/image (1).JPG"
```

The window contains the original image, Gaussian blur, leaf mask, ROI,
analyzed object, pseudolandmarks and color histograms.

Apply all transformations to a directory:

```bash
./Transformation.py -src images/Apple_healthy \
    -dst transformation_output -all
```

Individual transformations can also be selected and combined:

```bash
./Transformation.py -src images/Apple_healthy \
    -dst transformation_output -mask -histogram
```

Available options are:

```text
-blur
-mask
-roi
-analyze
-landmarks
-histogram
-all
```

## Part 4: Training

> **Important:** Feature extraction from the complete dataset takes a long
> time. During the defense, avoid running the full training process unless the
> evaluator explicitly requests it. Extract the prepared
> `training_output.zip` and continue with evaluation and prediction instead.

Train the classifier:

```bash
./train.py images
```

The training program:

1. Creates a stratified 80/20 training and validation split.
2. Augments only the training part.
3. Balances all training classes.
4. Extracts shape, color, damage and texture features.
5. Removes constant features.
6. Standardizes the feature values.
7. Trains an RBF Support Vector Classifier.
8. Evaluates it on the untouched validation set.
9. Displays the classification report and confusion matrix.
10. Saves the model, validation data, report and confusion matrix.

The default output is:

```text
training_output/
├── augmented_training/
├── confusion_matrix.png
├── leaf_model.joblib
├── validation_features.joblib
├── validation_predictions.csv
└── validation_report.txt
```

The model file contains the trained `SVC`, the fitted scaler and the ordered
list of feature columns.

The latest local reference run obtained a validation accuracy of `0.9439` on
1,445 validation images.

## Use the prepared training output

After a clean clone, place `training_output.zip` in the project root. Verify
its SHA-1 before extracting it:

```bash
shasum -c signature.txt
```

The command must print `training_output.zip: OK`. Then extract the archive:

```bash
unzip training_output.zip
```

## Reproduce the validation result

Recalculate the validation accuracy with the saved model:

```bash
./evaluate.py
```

The program loads the saved validation features, applies the saved scaler and
classifier, prints the accuracy and classification report, and displays the
confusion matrix. It does not retrain the model.

## Verify the training balance

The training output contains 1,312 images in each of the eight classes. Display
the counts and charts with the commands below.

```bash
./Distribution.py training_output/augmented_training
```

## Prediction

Download and extract the test dataset provided for the evaluation. Run the
prediction program on each test image:

```bash
./predict.py path/to/test_image.JPG
```

The default model path is:

```text
training_output/leaf_model.joblib
```

Use a different model if necessary:

```bash
./predict.py image.JPG --model path/to/leaf_model.joblib
```

The program prints the predicted class and displays the original and
transformed images. The prediction uses the image content, not its filename.

## Style check

Install and run `flake8`:

```bash
pip install flake8
flake8 . --exclude=.venv
```

## Defense sequence

1. Clone the Git repository into an empty directory.
2. Install the dependencies.
3. Download and extract the supplied datasets.
4. Compare the SHA-1 of `training_output.zip` with `signature.txt`.
5. Extract `training_output.zip`.
6. Run `flake8`.
7. Run `Distribution.py` on the original dataset.
8. Demonstrate the six augmentation operations on one copied image.
9. Run `Distribution.py` on the balanced dataset.
10. Demonstrate the image transformations.
11. Run `evaluate.py` and explain the SVC model and validation result.
12. Run `predict.py` on the hidden evaluation images.

## Files excluded from Git

The repository does not include datasets, generated images, model files,
archives, virtual environments or local editor settings. These files are
excluded by `.gitignore`.
