import argparse
from pathlib import Path

import matplotlib.pyplot as plt

IMAGES_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def count_images(directory):
    count = 0
    for path in directory.rglob("*"):
        if path.is_file():
            if path.suffix.lower() in IMAGES_EXT:
                count += 1
    return count

def make_plant_dict(root_dir):
    plant_dict = {}
    for class_dir in sorted(root_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        folder_name = class_dir.name
        if "_" not in folder_name:
            continue
        plant_name = folder_name.split("_", 1)[0]
        image_count = count_images(class_dir)

        if plant_name not in plant_dict:
            plant_dict[plant_name] = {}
        plant_dict[plant_name][folder_name] = image_count
    return plant_dict

def print_plant_dict(plant_dict):
    for plant_name in plant_dict:
        classes = plant_dict[plant_name]
        total_class_imgs = 0
        for image_count in classes.values():
            total_class_imgs += image_count
        print()
        print(plant_name + ": " + str(total_class_imgs))

        for class_name in classes:
            image_count = classes[class_name]
            print(" " + class_name + ": " + str(image_count))


def plot_bar_chart(plant_dict):
    for plant_name in plant_dict:
        classes = plant_dict[plant_name]

        class_names = list(classes.keys())
        image_count = list(classes.values())

        plt.figure(figsize=(10, 6))

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        plt.bar(class_names, image_count, color=colors)

        plt.title(plant_name + " image distribution")
        plt.xlabel("Classes")
        plt.ylabel("Number of images")

        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

    #plt.show()


def plot_pie_chart(plant_dict):
    for plant_name in plant_dict:
        classes = plant_dict[plant_name]
        class_names = list(classes.keys())
        image_count = list(classes.values())

        plt.figure(figsize=(8, 6))

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        plt.pie(
            image_count,
            labels=class_names,
            colors = colors,
            autopct="%1.1f%%",
            startangle=90,
        )

        plt.title(plant_name + " class distribution")
        plt.axis("equal")
        plt.tight_layout()
    
    #plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze leaf image distribution")
    parser.add_argument("directory", help="Path to the dataset directory")

    return parser.parse_args()


def main():
    arguments = parse_args()
    root_dir = Path(arguments.directory)
    if not root_dir.exists():
        print("Error: directory does not exist")
        return

    if not root_dir.is_dir():
        print("Error: the given path is not a directory")
        return
    
    plant_dict = make_plant_dict(root_dir)
    
    if not plant_dict:
        print("Error: no plant directories found")
        return

    print_plant_dict(plant_dict)
    plot_bar_chart(plant_dict)
    plot_pie_chart(plant_dict)

    plt.show()

if __name__ == "__main__":
    main()

    





