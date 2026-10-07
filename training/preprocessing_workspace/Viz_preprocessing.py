import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from PIL import Image

from stain_augmentation.randstainna import RandStainNA
from Dataset import MIDOG_Dataset


def visualize_randstainna(dm, yaml_path, num_samples=3):
    """Show 3x3 grid: original | after RandStainNA | after RandStainNA + datamodule transforms."""

    # 1. Prepare datasets without and with transforms
    clean_dataset = MIDOG_Dataset(dm.base_data_dir, dm.regular_folders + dm.minority_folders, transform=None)

    # 2. Instantiate RandStainNA (train mode so it accepts PIL.Image)
    rsna = RandStainNA(yaml_file=yaml_path, is_train=True, probability=1.0)

    # 3. Prepare figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    fig.suptitle("RandStainNA Visualization", fontsize=18)

    # Random sample indices
    indices = random.sample(range(len(clean_dataset)), num_samples)

    for row_idx, idx in enumerate(indices):
        img, label = clean_dataset[idx]  # img is a numpy array (H,W,C)

        # Ensure uint8 RGB
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        pil_img = Image.fromarray(img)

        # Column 1: Original
        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_title("Original")
        axes[row_idx, 0].axis("off")

        # Column 2: After RandStainNA only
        rsna_img_bgr = rsna(pil_img)  # returns BGR np.array
        rsna_img_rgb = rsna_img_bgr[:, :, ::-1]  # convert to RGB for plotting
        axes[row_idx, 1].imshow(rsna_img_rgb)
        axes[row_idx, 1].set_title("After RandStainNA")
        axes[row_idx, 1].axis("off")

        # Column 3: After RandStainNA + original datamodule train_transforms
        rsna_img_for_aug = Image.fromarray(rsna_img_rgb)
        rsna_img_np = np.array(rsna_img_for_aug)

        aug_result = dm.train_transforms(image=rsna_img_np)
        aug_img_tensor = aug_result["image"]

        # Denormalize for plotting
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        aug_img_vis = aug_img_tensor * std + mean
        aug_img_vis = torch.clamp(aug_img_vis, 0, 1).permute(1, 2, 0).numpy()

        axes[row_idx, 2].imshow(aug_img_vis)
        axes[row_idx, 2].set_title("RandStainNA + Train Augs")
        axes[row_idx, 2].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    from Datamodule_v2 import MIDOG_DataModule

    config = {
        "base_data_dir": "/home/rakib/data/MIDOGpp-main/224/sorted_data/",
        "regular_folders": ['folder1', 'folder2'],
        "minority_folders": ['folder5'],
        "batch_size": 1,
        "num_workers": 0,
        "yaml_path": "./stain_augmentation/output/random_images.yaml",
    }

    dm = MIDOG_DataModule(
        base_data_dir=config["base_data_dir"],
        regular_folders=config['regular_folders'],
        minority_folders=config["minority_folders"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        yaml_path=config["yaml_path"]
    )

    yaml_path = config["yaml_path"]  # <-- replace with your RandStainNA YAML file
    visualize_randstainna(dm, yaml_path, num_samples=4)
