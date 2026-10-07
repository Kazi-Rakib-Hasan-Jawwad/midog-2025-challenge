import os
import json
from PIL import Image
from pathlib import Path
import numpy as np

# --- Configuration ---
image_input_dir = "/home/rakib/data/MIDOGpp-main/images"
mask_input_dir = "/home/rakib/data/MIDOGpp-main/masks"
json_path = "/home/rakib/data/MIDOGpp-main/databases/MIDOG++.json"

# These are the output directories for the CLEANED, BALANCED patches
output_dir_img = "/home/rakib/data/MIDOGpp-main/256_balanced/img/"
output_dir_mask = "/home/rakib/data/MIDOGpp-main/256_balanced/mask/"
patch_size = 256

# --- Setup ---
Path(output_dir_img).mkdir(parents=True, exist_ok=True)
Path(output_dir_mask).mkdir(parents=True, exist_ok=True)

print("Loading and preprocessing JSON annotations...")
with open(json_path) as f:
    data = json.load(f)

# Create a mapping from image filename to image_id for easy lookup
filename_to_id = {img['file_name']: img['id'] for img in data['images']}

# Group annotations by image_id for efficient processing
annotations_by_image_id = {}
for anno in data['annotations']:
    img_id = anno['image_id']
    if img_id not in annotations_by_image_id:
        annotations_by_image_id[img_id] = []
    annotations_by_image_id[img_id].append(anno)

print("Setup complete. Starting annotation-centric cropping...")

# --- Reporting Setup ---
# Initialize counters for the class report
class_1_only_count = 0
class_2_only_count = 0
both_classes_count = 0

# --- Main Script Execution ---
image_files = os.listdir(image_input_dir)
total_patches_saved = 0

for image_filename in image_files:
    if image_filename not in filename_to_id:
        print(f"Skipping {image_filename}: Not found in JSON file.")
        continue

    image_id = filename_to_id[image_filename]
    if image_id not in annotations_by_image_id:
        print(f"Skipping {image_filename}: No annotations for this image.")
        continue

    try:
        print(f"\n--- Processing: {image_filename} (Image ID: {image_id}) ---")
        image_path = os.path.join(image_input_dir, image_filename)
        mask_path = os.path.join(mask_input_dir, image_filename)

        full_img = Image.open(image_path)
        full_mask = Image.open(mask_path)
        img_w, img_h = full_img.size

        patches_from_this_slide = 0
        # Iterate through every annotation for this specific image
        for anno in annotations_by_image_id[image_id]:
            bbox = anno['bbox']
            cat_id = anno['category_id']
            anno_id = anno['id']

            # Bounding box coordinates
            x1, y1, w, h = bbox

            # Calculate the center of the bounding box
            center_x = x1 + w / 2
            center_y = y1 + h / 2

            # Calculate the crop box coordinates for a 256x256 patch
            left = int(center_x - (patch_size / 2))
            up = int(center_y - (patch_size / 2))
            right = left + patch_size
            down = up + patch_size

            # --- Edge Case Handling ---
            # Ensure the crop box does not go outside the image boundaries
            if left < 0:
                left = 0
                right = patch_size
            if up < 0:
                up = 0
                down = patch_size
            if right > img_w:
                right = img_w
                left = img_w - patch_size
            if down > img_h:
                down = img_h
                up = img_h - patch_size

            # Crop the image and mask
            image_patch = full_img.crop((left, up, right, down))
            mask_patch = full_mask.crop((left, up, right, down))

            # --- Reporting Logic ---
            # Analyze the content of the mask patch for reporting
            mask_array = np.array(mask_patch)
            has_class_1 = np.any(mask_array == 1)
            has_class_2 = np.any(mask_array == 2)

            if has_class_1 and has_class_2:
                both_classes_count += 1
            elif has_class_1:
                class_1_only_count += 1
            elif has_class_2:
                class_2_only_count += 1
            # -------------------------

            # Save the patch pair with a descriptive name
            file_stem = Path(image_path).stem
            dst_filename = f"{file_stem}_anno_{anno_id}_cat_{cat_id}.png"

            image_patch.save(os.path.join(output_dir_img, dst_filename))
            mask_patch.save(os.path.join(output_dir_mask, dst_filename))

            patches_from_this_slide += 1
            total_patches_saved += 1

        print(f"Saved {patches_from_this_slide} patches based on {len(annotations_by_image_id[image_id])} annotations.")

    except Exception as e:
        print(f"CRITICAL ERROR processing {image_filename}: {e}")
        continue

print(f"\n--- EXECUTION COMPLETE ---")
print(f"Total patches saved across all slides: {total_patches_saved}")
print(f"Balanced dataset created in: /home/rakib/data/MIDOGpp-main/256_balanced/")

# --- Final Class Report ---
print("\n--- CLASS DISTRIBUTION REPORT ---")
print(f"Patches with ONLY Class 1: {class_1_only_count}")
print(f"Patches with ONLY Class 2: {class_2_only_count}")
print(f"Patches with BOTH Class 1 and 2: {both_classes_count}")
print("---------------------------------")
