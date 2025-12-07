import os
import json

# Specify the root folder path containing the JSON files and subfolders
root_folder_path = 'datasets'  # Replace with the actual root folder path

# Iterate over all files in the root folder and its subfolders
for dirpath, dirnames, filenames in os.walk(root_folder_path):
    for filename in filenames:
        if filename.endswith('.json'):
            file_path = os.path.join(dirpath, filename)
            
            # Load the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            # Remove the 'imageData' key if it exists
            if 'imageData' in data:
                del data['imageData']
            
            # Save the modified JSON back to the file
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)  # indent=4 for pretty-printing, optional

print("Processing complete. 'imageData' removed from all JSON files in the folder and subfolders.")