
import numpy as np
import cv2
import json
from pathlib import Path  # Used for cleaner file path handling

# Open the JSON file
with open('/home/rakib/data/MIDOGpp-main/databases/MIDOG++.json') as json_file:
    labels = json.load(json_file)

# It's better to group annotations by image_id first for performance
print("Preprocessing annotations...")
annotations_by_image_id = {}
for anno in labels['annotations']:
    image_id = anno['image_id']
    if image_id not in annotations_by_image_id:
        annotations_by_image_id[image_id] = []
    annotations_by_image_id[image_id].append(anno)
print("Preprocessing complete.")

# Iterate directly over the images list - this is more robust
for image_info in labels['images']:
    # --- Get image details ---
    image_id = image_info['id']
    file_name = image_info['file_name']
    height = image_info['height']
    width = image_info['width']

    # --- Create a blank mask ---
    mask = np.zeros((height, width), dtype=np.uint8)

    # --- Find and draw all annotations for this image ---
    # Use the preprocessed dictionary for a fast lookup
    if image_id in annotations_by_image_id:
        for annotation in annotations_by_image_id[image_id]:
            x1, y1, x2, y2 = annotation['bbox']
            category_id = annotation['category_id']

            # Create polygon from bounding box
            polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

            # Draw the filled polygon on the mask
            cv2.fillPoly(mask, [polygon], color=category_id)

    # --- Save the final mask file ONLY ONCE per image ---
    output_path = Path('/home/rakib/data/MIDOGpp-main/masks/')
    # home/rakib/data/MIDOGpp-main/224/data/img/
    output_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    cv2.imwrite(str(output_path / file_name), mask)

print("Completed")