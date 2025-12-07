"""
Convert LabelMe annotated dataset to XY coordinate dataset for training.

This script processes the baseline dataset (images + LabelMe JSON annotations)
and converts it into a format compatible with the XYDataset in DataLoader.py.

Expected structure:
- datasets/baseline/
  ├── img1.png
  ├── img1.json
  ├── img2.png
  ├── img2.json
  └── ...

Output structure:
- datasets/baseline_processed/
  ├── train/
  │   ├── images/
  │   │   ├── img1.png
  │   │   ├── img2.png
  │   │   └── ...
  │   ├── img1.json (with x, y coordinates)
  │   ├── img2.json
  │   └── ...
  ├── valid/
  │   ├── images/
  │   │   ├── img3.png
  │   │   └── ...
  │   ├── img3.json
  │   └── ...
  ├── train.txt (paths to training images)
  └── valid.txt (paths to validation images)
"""

import os
import json
import shutil
import random
from pathlib import Path

# Configuration
FILE_EXT = '.png'
DATASET = 'baseline'

# Use the directory where this script is located as the base
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_FOLDER = os.path.join(SCRIPT_DIR, 'datasets', DATASET)

# Output parent folder that will contain train, valid folders and txt files
OUTPUT_PARENT_FOLDER = os.path.join(SCRIPT_DIR, 'datasets', f'{DATASET}_processed')
TRAIN_DEST_FOLDER = os.path.join(OUTPUT_PARENT_FOLDER, 'train')
VALID_DEST_FOLDER = os.path.join(OUTPUT_PARENT_FOLDER, 'valid')

# Train/Valid split ratio
TRAIN_RATIO = 0.8

def create_xy_annotation(x, y, annotation_path):
    """
    Create a JSON annotation file with x, y coordinates.
    
    Args:
        x (float): X coordinate
        y (float): Y coordinate
        annotation_path (str): Path where to save the annotation JSON
    """
    annotation_data = {
        "x": x,
        "y": y
    }
    os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
    with open(annotation_path, 'w') as f:
        json.dump(annotation_data, f, indent=4)


def process_image_annotation(image_filename, src_folder, dest_folder, dest_images_folder, records_list):
    """
    Process a single image-annotation pair and copy to destination.
    
    Args:
        image_filename (str): Name of the image file (e.g., 'img1.png')
        src_folder (str): Source folder path
        dest_folder (str): Destination folder path
        dest_images_folder (str): Destination images subfolder path
        records_list (list): List to append the final image path to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Construct paths
        src_image_path = os.path.join(src_folder, image_filename)
        dest_image_path = os.path.join(dest_images_folder, image_filename)
        
        json_filename = image_filename.replace(FILE_EXT, '.json')
        src_json_path = os.path.join(src_folder, json_filename)
        dest_json_path = os.path.join(dest_images_folder, json_filename)
        
        # Check if files exist
        if not os.path.exists(src_image_path):
            print(f"Warning: Image not found: {src_image_path}")
            return False
            
        if not os.path.exists(src_json_path):
            print(f"Warning: Annotation not found: {src_json_path}")
            return False
        
        # Read annotation
        with open(src_json_path, 'r') as f:
            ann_data = json.load(f)
        
        # Extract coordinates from LabelMe format
        if 'shapes' not in ann_data or len(ann_data['shapes']) == 0:
            print(f"Warning: No shapes found in {src_json_path}")
            return False
        
        # Get the first shape's first point (LabelMe format)
        shape = ann_data['shapes'][0]
        if 'points' not in shape or len(shape['points']) == 0:
            print(f"Warning: No points found in shape in {src_json_path}")
            return False
        
        point = shape['points'][0]
        x, y = point[0], point[1]
        
        # Create destination directories
        os.makedirs(dest_images_folder, exist_ok=True)
        os.makedirs(dest_folder, exist_ok=True)
        
        # Copy image
        shutil.copy2(src_image_path, dest_image_path)
        
        # Create annotation file with x, y coordinates
        create_xy_annotation(x, y, dest_json_path)
        
        # Record the image path
        records_list.append(dest_image_path)
        
        return True
        
    except Exception as e:
        print(f"Error processing {image_filename}: {str(e)}")
        return False


def main():
    """Main conversion function."""
    
    print(f"Starting dataset conversion from '{SRC_FOLDER}'...")
    
    # Check if source folder exists
    if not os.path.exists(SRC_FOLDER):
        print(f"Error: Source folder '{SRC_FOLDER}' does not exist!")
        return False
    
    # Get all image files
    image_files = [f for f in os.listdir(SRC_FOLDER) if f.endswith(FILE_EXT)]
    
    if not image_files:
        print(f"Error: No image files found in '{SRC_FOLDER}'")
        return False
    
    print(f"Found {len(image_files)} image files")
    
    # Shuffle and split into train/valid
    random.shuffle(image_files)
    split_idx = int(len(image_files) * TRAIN_RATIO)
    train_files = image_files[:split_idx]
    valid_files = image_files[split_idx:]
    
    print(f"Split into {len(train_files)} training and {len(valid_files)} validation images")
    
    # Process files
    train_records = []
    valid_records = []
    
    # Create destination folder structures
    train_images_folder = os.path.join(TRAIN_DEST_FOLDER, 'images')
    valid_images_folder = os.path.join(VALID_DEST_FOLDER, 'images')
    
    print("\nProcessing training files...")
    for i, filename in enumerate(train_files):
        if process_image_annotation(filename, SRC_FOLDER, TRAIN_DEST_FOLDER, 
                                    train_images_folder, train_records):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(train_files)} training files")
    
    print(f"Successfully processed {len(train_records)} training files")
    
    print("\nProcessing validation files...")
    for i, filename in enumerate(valid_files):
        if process_image_annotation(filename, SRC_FOLDER, VALID_DEST_FOLDER,
                                    valid_images_folder, valid_records):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(valid_files)} validation files")
    
    print(f"Successfully processed {len(valid_records)} validation files")
    
    # Create train.txt and valid.txt in the OUTPUT_PARENT_FOLDER
    train_txt_path = os.path.join(OUTPUT_PARENT_FOLDER, 'train.txt')
    valid_txt_path = os.path.join(OUTPUT_PARENT_FOLDER, 'valid.txt')
    
    os.makedirs(OUTPUT_PARENT_FOLDER, exist_ok=True)
    
    with open(train_txt_path, 'w') as f:
        for record in train_records:
            f.write(record + '\n')
    
    with open(valid_txt_path, 'w') as f:
        for record in valid_records:
            f.write(record + '\n')
    
    print(f"\nDataset conversion complete!")
    print(f"Training images: {len(train_records)} (saved to {TRAIN_DEST_FOLDER})")
    print(f"Validation images: {len(valid_records)} (saved to {VALID_DEST_FOLDER})")
    print(f"Train list saved to: {train_txt_path}")
    print(f"Valid list saved to: {valid_txt_path}")
    
    return True


if __name__ == '__main__':
    success = main()
    if not success:
        print("\nDataset conversion failed!")
        exit(1)
