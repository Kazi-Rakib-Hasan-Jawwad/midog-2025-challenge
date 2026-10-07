import os
from pathlib import Path
from typing import List, Optional, Callable

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch


class MIDOG_Dataset(Dataset):
    """
    Dataset class updated to use the Albumentations library for transformations.
    """

    def __init__(
            self,
            base_data_dir: str,
            sub_folders: List[str],
            transform: Optional[Callable] = None,  # This will now be an Albumentations transform
    ):
        """
        Args:
            base_data_dir (str): The base path to the sorted data.
            sub_folders (List[str]): A list of subfolders to include.
            transform (Callable, optional): An Albumentations transform pipeline.
        """
        self.transform = transform
        self.img_list = []
        self.mask_list = []

        print(f"Loading data from folders: {sub_folders}")
        for folder in sub_folders:
            img_dir = os.path.join(base_data_dir, folder, "img")
            mask_dir = os.path.join(base_data_dir, folder, "mask")

            if not os.path.isdir(img_dir):
                print(f"Warning: Image directory not found, skipping: {img_dir}")
                continue

            for filename in sorted(os.listdir(img_dir)):
                self.img_list.append(os.path.join(img_dir, filename))
                self.mask_list.append(os.path.join(mask_dir, filename))

        print(f"Found {len(self.img_list)} image-mask pairs.")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        mask_path = self.mask_list[idx]

        # Open image and mask as numpy arrays
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Apply Albumentations transforms
        if self.transform:
            # Albumentations takes a dictionary and returns a dictionary
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        return img, mask
