import os
import random
from typing import Any, Optional, List, Callable

import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, ConcatDataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.domain_adaptation import FDA
from PIL import Image

from Dataset import MIDOG_Dataset


# Main DataModule Class
class MIDOG_DataModule(LightningDataModule):
    name = "MIDOG_DataModule"

    def __init__(
            self,
            base_data_dir: str,
            regular_folders: List[str],
            minority_folders: List[str],
            minority_sampling_ratio: float = 0.5,
            val_split: float = 0.2,
            test_split: Optional[float] = None,
            batch_size: int = 32,
            num_workers: int = 8,
            seed: int = 42,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.base_data_dir = base_data_dir
        self.regular_folders = regular_folders
        self.minority_folders = minority_folders
        self.minority_sampling_ratio = minority_sampling_ratio
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.target_image_paths = self._get_all_image_paths(regular_folders + minority_folders)

        def read_fn(path):
            return np.array(Image.open(path).convert('RGB'))

        self.train_transforms = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.RandomRotate90(p=1.0),
            ], p=0.8),
            FDA(reference_images=self.target_image_paths, read_fn=read_fn, beta_limit=0.01, p=0.8),
            A.ElasticTransform(p=0.3, alpha=10, sigma=10, alpha_affine=5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.val_test_transforms = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def _get_all_image_paths(self, folders) -> List[str]:
        all_paths = []
        for folder in folders:
            img_dir = os.path.join(self.base_data_dir, folder, "img")
            if os.path.isdir(img_dir):
                for filename in os.listdir(img_dir):
                    all_paths.append(os.path.join(img_dir, filename))
        return all_paths

    def setup(self, stage: Optional[str] = None):
        # Create separate datasets for training (with augmentations)
        self.regular_train = MIDOG_Dataset(self.base_data_dir, self.regular_folders, transform=self.train_transforms)
        self.minority_train = MIDOG_Dataset(self.base_data_dir, self.minority_folders, transform=self.train_transforms)

        # Create separate datasets for validation/testing (no augmentations)
        regular_val_test = MIDOG_Dataset(self.base_data_dir, self.regular_folders, transform=self.val_test_transforms)
        minority_val_test = MIDOG_Dataset(self.base_data_dir, self.minority_folders, transform=self.val_test_transforms)

        full_val_test_dataset = ConcatDataset([regular_val_test, minority_val_test])

        dataset_size = len(full_val_test_dataset)
        val_len = int(self.val_split * dataset_size)

        if self.test_split is not None and self.test_split > 0:
            test_len = int(self.test_split * dataset_size)
            train_len = dataset_size - val_len - test_len
            _, self.val_dataset, self.test_dataset = random_split(
                full_val_test_dataset,
                lengths=[train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(self.seed)
            )
        else:
            train_len = dataset_size - val_len
            _, self.val_dataset = random_split(
                full_val_test_dataset,
                lengths=[train_len, val_len],
                generator=torch.Generator().manual_seed(self.seed)
            )
            self.test_dataset = None

    def train_dataloader(self) -> DataLoader:
        train_dataset = ConcatDataset([self.regular_train, self.minority_train])

        # Prevent division by zero if a dataset is empty
        len_regular = len(self.regular_train)
        len_minority = len(self.minority_train)

        if len_regular == 0 or len_minority == 0:
            return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

        regular_weight = (1.0 - self.minority_sampling_ratio) / len_regular
        minority_weight = self.minority_sampling_ratio / len_minority
        sample_weights = [regular_weight] * len_regular + [minority_weight] * len_minority

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("No test dataset configured.")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=True)
