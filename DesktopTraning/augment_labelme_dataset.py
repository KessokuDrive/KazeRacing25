"""
Augment LabelMe dataset by adding horizontally flipped versions of half the images.
This makes the dataset more robust by doubling the data with flipped images where x-coordinates are adjusted.

Expected input structure (LabelMe format):
- datasets/<DATASET_NAME>/
  ├── img1.png (or .jpg)
  ├── img1.json
  ├── img2.png
  ├── img2.json
  └── ...

Output structure:
- datasets/<DATASET_NAME>_augmented/
  ├── img1.png (original)
  ├── img1.json (original)
  ├── img1_flipped.png (flipped version)
  ├── img1_flipped.json (with flipped x coordinate)
  └── ...

This output can then be processed by labelme2Dataset.py.
"""

import os
import json
import shutil
import random
from PIL import Image
import numpy as np

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_NAME = 'dataset0254'  # Change this to your source dataset name
SRC_FOLDER = os.path.join(SCRIPT_DIR, 'datasets', DATASET_NAME)
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, 'datasets', f'{DATASET_NAME}_augmented')

# Supported image extensions
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']

# Percentage of images to flip (0.5 = 50% = half)
FLIP_RATIO = 0.5


def get_image_files(folder):
    """Get all image files from the folder."""
    image_files = []
    for filename in os.listdir(folder):
        _, ext = os.path.splitext(filename)
        if ext in IMAGE_EXTENSIONS:
            image_files.append(filename)
    return sorted(image_files)


def flip_image(image_path, output_path):
    """Horizontally flip an image and save it."""
    try:
        img = Image.open(image_path)
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_img.save(output_path)
        return True
    except Exception as e:
        print(f"Error flipping image {image_path}: {e}")
        return False


def flip_labelme_json(json_path, output_path):
    """
    Flip LabelMe JSON annotation by adjusting x-coordinates.
    x_new = imageWidth - x_old
    """
    try:
        with open(json_path, 'r') as f:
            ann_data = json.load(f)
        
        # Get image width from annotation
        if 'imageWidth' not in ann_data:
            print(f"Warning: imageWidth not found in {json_path}, skipping flip")
            return False
        
        image_width = ann_data['imageWidth']
        
        # Flip x-coordinates for all points in all shapes
        if 'shapes' in ann_data:
            for shape in ann_data['shapes']:
                if 'points' in shape:
                    for point in shape['points']:
                        if len(point) >= 1:
                            # Flip x coordinate: x_new = width - x_old
                            point[0] = image_width - point[0]
        
        # Save flipped annotation
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(ann_data, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error flipping JSON {json_path}: {e}")
        return False


def process_dataset():
    """Main function to augment the dataset."""
    print(f"Augmenting dataset from '{SRC_FOLDER}'...")
    
    # Check if source folder exists
    if not os.path.exists(SRC_FOLDER):
        print(f"Error: Source folder '{SRC_FOLDER}' does not exist!")
        return False
    
    # Get all image files
    image_files = get_image_files(SRC_FOLDER)
    if not image_files:
        print(f"Error: No image files found in '{SRC_FOLDER}'")
        return False
    
    print(f"Found {len(image_files)} image files")
    
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Determine which images to flip (randomly select half)
    num_to_flip = int(len(image_files) * FLIP_RATIO)
    files_to_flip = set(random.sample(image_files, num_to_flip))
    
    print(f"Will flip {num_to_flip} images ({FLIP_RATIO*100:.0f}% of dataset)")
    
    # Process each image
    copied_count = 0
    flipped_count = 0
    skipped_count = 0
    
    for filename in image_files:
        # Get file extension and base name
        base_name, ext = os.path.splitext(filename)
        json_filename = base_name + '.json'
        
        src_image_path = os.path.join(SRC_FOLDER, filename)
        src_json_path = os.path.join(SRC_FOLDER, json_filename)
        
        # Check if JSON exists
        if not os.path.exists(src_json_path):
            print(f"Warning: JSON not found for {filename}, skipping")
            skipped_count += 1
            continue
        
        # Always copy original files
        dest_image_path = os.path.join(OUTPUT_FOLDER, filename)
        dest_json_path = os.path.join(OUTPUT_FOLDER, json_filename)
        
        try:
            shutil.copy2(src_image_path, dest_image_path)
            shutil.copy2(src_json_path, dest_json_path)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {filename}: {e}")
            skipped_count += 1
            continue
        
        # If this image should be flipped, create flipped version
        if filename in files_to_flip:
            flipped_image_name = f"{base_name}_flipped{ext}"
            flipped_json_name = f"{base_name}_flipped.json"
            
            flipped_image_path = os.path.join(OUTPUT_FOLDER, flipped_image_name)
            flipped_json_path = os.path.join(OUTPUT_FOLDER, flipped_json_name)
            
            # Flip image
            if flip_image(src_image_path, flipped_image_path):
                # Flip JSON annotation
                if flip_labelme_json(src_json_path, flipped_json_path):
                    flipped_count += 1
                    if flipped_count % 10 == 0:
                        print(f"  Processed {flipped_count}/{num_to_flip} flipped images")
                else:
                    # If JSON flip failed, remove the flipped image
                    if os.path.exists(flipped_image_path):
                        os.remove(flipped_image_path)
    
    print(f"\nAugmentation complete!")
    print(f"  Original images copied: {copied_count}")
    print(f"  Flipped images created: {flipped_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total output images: {copied_count + flipped_count}")
    print(f"  Output folder: {OUTPUT_FOLDER}")
    print(f"\nYou can now process '{OUTPUT_FOLDER}' with labelme2Dataset.py")
    
    return True


if __name__ == '__main__':
    success = process_dataset()
    if not success:
        exit(1)
