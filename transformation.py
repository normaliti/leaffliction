import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from transforms import apply_gaussian_blur

def parse_args():
    parser = argparse.ArgumentParser(description="Apply transformation to an image")
    parser.add_argument("image", help="Path to an image")
    return parser.parse_args()


def load_image(image_path):
    if not image_path.exists():
        print("Error: image does not exist")
        return
    if not image_path.is_file():
        print("Error: given path is not a file")
        return 
    image = cv2.imread(str(image_path))

    if image is None:
        print("Error: could not loaf a file")
        return None
    return image

def show_image(image, title):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()



def main():
    arguments = parse_args()
    image_path = Path(arguments.image)

    image = load_image(image_path)
    if image is None:
        return

    height = image.shape[0]
    width = image.shape[1]

    print("Image loaded successfully.")
    print("Width:", width)
    print("Height:", height)
    show_image(image, "original image")

    blurred_image = apply_gaussian_blur(image)
    show_image(blurred_image, "Gaussian blur")

if __name__ == "__main__":
    main()