import os
from pathlib import Path
from typing import List, Optional, Callable

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch

DEFAULT_VALID_LABELS = (0, 1, 2)

class MIDOG_Dataset(Dataset):
    """
    Dataset class updated to use the Albumentations library for transformations.
    """

    def __init__(
            self,
            base_data_dir: str,
            img_size: tuple = (512, 512),
            valid_labels: list = DEFAULT_VALID_LABELS,
            transform: Optional[Callable] = None,  # This will now be an Albumentations transform
    ):
        """
        Args:
            base_data_dir (str): The base path to the sorted data.
            sub_folders (List[str]): A list of subfolders to include.
            transform (Callable, optional): An Albumentations transform pipeline.
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.valid_labels = valid_labels
        self.class_map = dict(zip(self.valid_labels, range(len(self.valid_labels))))
        self.transform = transform
        self.img_list = []
        self.mask_list = []

        print(f"Loading data from folders: {sub_folders}")
        
        img_dir = os.path.join(base_data_dir, "img")
        mask_dir = os.path.join(base_data_dir, "mask")
 
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
        img = np.array(Image.open(img_path).convert('RGB')).resize(self.img_size)
        mask = np.array(Image.open(mask_path).convert("L")).resize(self.img_size)

        # Apply Albumentations transforms
        if self.transform:
            # Albumentations takes a dictionary and returns a dictionary
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        return img, mask

