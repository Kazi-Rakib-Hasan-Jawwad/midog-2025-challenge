import os
import shutil
from pathlib import Path

# --- 1. Define the directory paths ---
# Using pathlib for a modern and robust way to handle paths.
IMG_DIR = Path("/home/rakib/data/MIDOGpp-main/256/cropped/img/")
MASK_SRC_DIR = Path("/home/rakib/data/MIDOGpp-main/256/cropped/mask/")
MASK_DEST_DIR = Path("/home/rakib/data/MIDOGpp-main/256/cropped/mask_2/")

# --- 2. Create the destination directory if it doesn't exist ---
# The exist_ok=True argument prevents an error if the folder is already there.
print(f"Ensuring destination directory exists: {MASK_DEST_DIR}")
os.makedirs(MASK_DEST_DIR, exist_ok=True)

# --- 3. Get the list of all image files from Location A ---
try:
    image_files = os.listdir(IMG_DIR)
    print(f"Found {len(image_files)} images in {IMG_DIR}")
except FileNotFoundError:
    print(f"Error: The image directory was not found at {IMG_DIR}")
    # Exit the script if the main image folder doesn't exist.
    exit()

# --- 4. Initialize a list to track images with no matching mask ---
missing_masks_log = []
moved_count = 0

print("\nStarting to process files...")

# --- 5. Loop through each image filename ---
for filename in image_files:
    # Define the full path for the source and destination mask files
    source_mask_path = MASK_SRC_DIR / filename
    destination_mask_path = MASK_DEST_DIR / filename

    # Check if the corresponding mask exists in Location B
    if source_mask_path.exists():
        # If it exists, move it to Location C
        try:
            shutil.move(source_mask_path, destination_mask_path)
            moved_count += 1
            # Optional: uncomment the line below to see progress for every file
            # print(f"Moved: {filename}")
        except Exception as e:
            print(f"Error moving {filename}: {e}")
    else:
        # If it doesn't exist, add the image name to our log
        missing_masks_log.append(filename)

print("\n--- Processing Complete ---")
print(f"Successfully moved {moved_count} masks to {MASK_DEST_DIR}")

# --- 6. Report the results for missing masks ---
if missing_masks_log:
    print(f"\nFound {len(missing_masks_log)} images with no corresponding mask.")
    print("-----------------------------------------")
    # Print each filename that was missing a mask
    for missing_file in missing_masks_log:
        print(missing_file)
    print("-----------------------------------------")
else:
    print("\nSuccess! All images in Location A had a corresponding mask in Location B.")
