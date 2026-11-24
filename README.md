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
chmod +x Distribution.py Augmentation.py Transformation.py train.py predict.py
```

## Download the data

The datasets are not stored in the Git repository.

Download the training dataset:

```bash
curl -L https://cdn.intra.42.fr/document/document/17547/leaves.zip \
    -o leaves.zip
unzip leaves.zip
```

Download the evaluation images:

```bash
curl -L https://cdn.intra.42.fr/document/document/17546/test_images.zip \
    -o test_images.zip
unzip test_images.zip
```

The extracted directories should look like this:

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

test_images/
├── Unit_test1/
└── Unit_test2/
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
10. Saves the trained model.

The default output is:

```text
training_output/
├── augmented_training/
└── leaf_model.joblib
```

The model file contains the trained `SVC`, the fitted scaler and the ordered
list of feature columns.

A different output directory and validation proportion can be specified:

```bash
./train.py images -o another_training_output --validation-size 0.2
```

The latest local reference run obtained a validation accuracy of `0.94` on
1,445 validation images.

## Verify the training balance

The training output contains 1,312 images in each of the eight classes. Display
the counts and charts with:

```bash
./Distribution.py training_output/augmented_training
```

## Package the trained artifacts

The trained model and augmented training images must be provided as an external
archive. They must not be committed to Git.

Create the archive after training:

```bash
zip -r training_output.zip training_output
```

Create its SHA-1 signature on macOS:

```bash
shasum training_output.zip > signature.txt
```

On Linux, use:

```bash
sha1sum training_output.zip > signature.txt
```

Verify the signature:

```bash
cat signature.txt
shasum training_output.zip
```

Commit `signature.txt` together with the source files. Keep
`training_output.zip` outside Git but available locally for the defense.

To restore the trained artifacts after a clean clone:

```bash
unzip training_output.zip
```

## Prediction

Predict one image with the default model:

```bash
./predict.py test_images/Unit_test1/Apple_healthy1.JPG
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

## Evaluation images

Apple examples are stored in `Unit_test1` and grape examples in `Unit_test2`.

```bash
./predict.py test_images/Unit_test1/Apple_Black_rot1.JPG
./predict.py test_images/Unit_test1/Apple_healthy1.JPG
./predict.py test_images/Unit_test1/Apple_healthy2.JPG
./predict.py test_images/Unit_test1/Apple_rust.JPG
./predict.py test_images/Unit_test1/Apple_scab.JPG

./predict.py test_images/Unit_test2/Grape_Black_rot1.JPG
./predict.py test_images/Unit_test2/Grape_Black_rot2.JPG
./predict.py test_images/Unit_test2/Grape_Esca.JPG
./predict.py test_images/Unit_test2/Grape_healthy.JPG
./predict.py test_images/Unit_test2/Grape_spot.JPG
```

The latest trained model classified all 10 supplied evaluation images
correctly.

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
4. Compare `signature.txt` with `training_output.zip`.
5. Extract `training_output.zip`.
6. Run `flake8`.
7. Run `Distribution.py` on the original dataset.
8. Demonstrate the six augmentation operations on one copied image.
9. Run `Distribution.py` on the balanced dataset.
10. Demonstrate the image transformations.
11. Show the validation results and explain the SVC model.
12. Run `predict.py` on the hidden evaluation images.

## Files excluded from Git

The repository does not include datasets, generated images, model files,
archives, virtual environments or local editor settings. These files are
excluded by `.gitignore`.
