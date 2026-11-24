"""Loading, saving, finding and displaying OpenCV images."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"cannot read image '{path}'")
    return image


def save_image(image, path):
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"cannot write image '{path}'")


def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_images(directory):
    return sorted(path for path in Path(directory).iterdir() if is_image(path))


def find_image_directories(root):
    directories = [root, *Path(root).rglob("*")]
    return [
        directory for directory in sorted(directories)
        if directory.is_dir() and list_images(directory)
    ]


def path_with_suffix(image_path, suffix):
    image_path = Path(image_path)
    new_name = f"{image_path.stem}_{suffix}{image_path.suffix}"
    return image_path.with_name(new_name)


def show_images(images_by_title, window_title=""):
    image_count = len(images_by_title)
    column_count = min(4, image_count)
    row_count = (image_count + column_count - 1) // column_count

    figure_size = (column_count * 4, row_count * 3.5)
    figure, _ = plt.subplots(
        row_count, column_count, figsize=figure_size
    )
    axes = figure.axes

    for axis, (title, image) in zip(axes, images_by_title.items()):
        if len(image.shape) == 2:
            axis.imshow(image, cmap="gray")
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axis.imshow(rgb_image)

        axis.set_title(title)
        axis.axis("off")

    for axis in axes[image_count:]:
        axis.axis("off")

    figure.suptitle(window_title)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
