"""
Convert filename-encoded JPG dataset to the XYDataset format used in train.py.

Expected source filenames:
    <x>_<y>_<anything>.jpg
Example:
    216_13_001.jpg  -> x=216, y=13

Output structure (mirrors labelme2Dataset.py):
    datasets/<DATASET_NAME>_processed/
        train/
            images/*.jpg
            *.json (with {"x": x, "y": y})
        valid/
            images/*.jpg
            *.json
        train.txt   (absolute paths to train images)
        valid.txt   (absolute paths to valid images)
"""

import os
import json
import shutil
import random
from pathlib import Path
from typing import Optional, Tuple

# ----------- Configuration -----------
# Source folder containing the JPGs. You can set an absolute path if needed.
DATASET_NAME = "newCapture"  # used for output folder naming only
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_FOLDER = os.path.join(SCRIPT_DIR, "datasets", DATASET_NAME)

# Train/Valid split ratio
TRAIN_RATIO = 0.8

# Output parent folder that will contain train/valid and txt files
OUTPUT_PARENT_FOLDER = os.path.join(SCRIPT_DIR, "datasets", f"{DATASET_NAME}_processed")
TRAIN_DEST_FOLDER = os.path.join(OUTPUT_PARENT_FOLDER, "train")
VALID_DEST_FOLDER = os.path.join(OUTPUT_PARENT_FOLDER, "valid")


# ----------- Helpers -----------
def parse_xy_from_filename(filename: str) -> Optional[Tuple[float, float]]:
    """
    Parse x, y from filenames shaped like: <x>_<y>_<anything>.jpg
    Returns (x, y) as floats, or None if pattern does not match.
    """
    stem = Path(filename).stem  # drop extension
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        x = float(parts[0])
        y = float(parts[1])
        return x, y
    except ValueError:
        return None


def create_xy_annotation(x: float, y: float, annotation_path: str):
    os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
    with open(annotation_path, "w") as f:
        json.dump({"x": x, "y": y}, f, indent=4)


def process_file(image_filename: str, src_folder: str, dest_folder: str, dest_images_folder: str, records_list: list) -> bool:
    try:
        xy = parse_xy_from_filename(image_filename)
        if xy is None:
            print(f"Skip (bad name): {image_filename}")
            return False
        x, y = xy

        src_image_path = os.path.join(src_folder, image_filename)
        if not os.path.exists(src_image_path):
            print(f"Warning: Image not found: {src_image_path}")
            return False

        dest_image_path = os.path.join(dest_images_folder, image_filename)
        dest_json_path = dest_image_path.replace(".jpg", ".json")

        os.makedirs(dest_images_folder, exist_ok=True)
        os.makedirs(dest_folder, exist_ok=True)

        shutil.copy2(src_image_path, dest_image_path)
        create_xy_annotation(x, y, dest_json_path)
        records_list.append(dest_image_path)
        return True
    except Exception as e:
        print(f"Error processing {image_filename}: {e}")
        return False


def main():
    print(f"Converting filename-encoded dataset from '{SRC_FOLDER}'...")
    if not os.path.exists(SRC_FOLDER):
        print(f"Error: Source folder '{SRC_FOLDER}' does not exist!")
        return False

    image_files = [f for f in os.listdir(SRC_FOLDER) if f.lower().endswith(".jpg")]
    if not image_files:
        print(f"Error: No JPG files found in '{SRC_FOLDER}'")
        return False

    random.shuffle(image_files)
    split_idx = int(len(image_files) * TRAIN_RATIO)
    train_files = image_files[:split_idx]
    valid_files = image_files[split_idx:]

    train_records = []
    valid_records = []

    train_images_folder = os.path.join(TRAIN_DEST_FOLDER, "images")
    valid_images_folder = os.path.join(VALID_DEST_FOLDER, "images")

    print(f"Train/Valid split: {len(train_files)} / {len(valid_files)}")
    print("\nProcessing training files...")
    for i, filename in enumerate(train_files):
        if process_file(filename, SRC_FOLDER, TRAIN_DEST_FOLDER, train_images_folder, train_records):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(train_files)} training files")

    print("Processing validation files...")
    for i, filename in enumerate(valid_files):
        if process_file(filename, SRC_FOLDER, VALID_DEST_FOLDER, valid_images_folder, valid_records):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(valid_files)} validation files")

    os.makedirs(OUTPUT_PARENT_FOLDER, exist_ok=True)
    train_txt_path = os.path.join(OUTPUT_PARENT_FOLDER, "train.txt")
    valid_txt_path = os.path.join(OUTPUT_PARENT_FOLDER, "valid.txt")

    with open(train_txt_path, "w") as f:
        for record in train_records:
            f.write(record + "\n")

    with open(valid_txt_path, "w") as f:
        for record in valid_records:
            f.write(record + "\n")

    print("\nDone!")
    print(f"Training images: {len(train_records)} -> {TRAIN_DEST_FOLDER}")
    print(f"Validation images: {len(valid_records)} -> {VALID_DEST_FOLDER}")
    print(f"Train list: {train_txt_path}")
    print(f"Valid list: {valid_txt_path}")
    return True


if __name__ == "__main__":
    ok = main()
    if not ok:
        exit(1)

