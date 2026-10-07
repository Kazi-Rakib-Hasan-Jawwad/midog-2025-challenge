import os
from pathlib import Path
import numpy as np
from PIL import Image

# --- Configuration ---
base_input_dir = "/home/rakib/data/MIDOGpp-main/224/data/"
image_input_dir = os.path.join(base_input_dir, "img")
mask_input_dir = os.path.join(base_input_dir, "mask")

# Define a base directory for all sorted output
base_output_dir = "/home/rakib/data/MIDOGpp-main/224/sorted_data/"

files = os.listdir(mask_input_dir)

for i in files:
    try:
        img_path = os.path.join(image_input_dir, i)
        mask_path = os.path.join(mask_input_dir, i)

        # Use 'with' to ensure files are properly closed
        with Image.open(img_path) as img, Image.open(mask_path) as mask:
            mask_np = np.array(mask)
            dst = i # The filename is already correct

            # This identifies all classes in the mask, excluding the background (0)
            unique_classes = set(np.unique(mask_np)) - {0}

            # Determine the target folder based on the unique classes found
            if not unique_classes:
                continue # Skip empty masks
            elif unique_classes == {1}:
                target_folder = "folder1"
            elif unique_classes == {2}:
                target_folder = "folder2"
            else:
                target_folder = "folder5"

            # --- Create Directories and Save Files ---
            # Construct the full output paths correctly
            output_img_dir = os.path.join(base_output_dir, target_folder, "img")
            output_mask_dir = os.path.join(base_output_dir, target_folder, "mask")

            # Ensure the output directories exist before saving
            Path(output_img_dir).mkdir(parents=True, exist_ok=True)
            Path(output_mask_dir).mkdir(parents=True, exist_ok=True)

            # Save the image and mask
            img.save(os.path.join(output_img_dir, dst))
            mask.save(os.path.join(output_mask_dir, dst))

    except FileNotFoundError:
        print(f"Warning: Could not find matching image for mask: {i}. Skipping.")
        continue
    except Exception as e:
        print(f"An error occurred with file {i}: {e}. Skipping.")
        continue

print("Sorting complete.")