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
    figure, axes = plt.subplots(1, len(images_by_title), figsize=(20, 4))
    for axis, (title, image) in zip(axes, images_by_title.items()):
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(window_title)
    figure.tight_layout()
    plt.show()
