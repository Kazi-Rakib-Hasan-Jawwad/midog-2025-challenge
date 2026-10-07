import os
from typing import Any, Optional, List

import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.domain_adaptation import FDA
from PIL import Image

from Dataset import MIDOG_Dataset


class MIDOG_DataModule(LightningDataModule):
    name = "MIDOG_DataModule"

    def __init__(
        self,
        data_dir: str,
        val_split: float = 0.2,
        test_split: Optional[float] = None,
        batch_size: int = 32,
        num_workers: int = 6,
        seed: int = 1234,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir.rstrip("/")
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # FDA reference images: all images under root/img
        self.target_image_paths = self._get_all_image_paths(self.data_dir)

        def read_fn(path):
            return np.array(Image.open(path).convert('RGB'))

        self.train_transforms = A.Compose([
            A.OneOf([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), A.RandomRotate90(p=1.0)], p=0.8),
            FDA(reference_images=self.target_image_paths, read_fn=read_fn, beta_limit=0.01, p=0.8),
            A.ElasticTransform(p=0.3, alpha=10, sigma=10, alpha_affine=5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.val_test_transforms = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _get_all_image_paths(self, root: str) -> List[str]:
        img_dir = os.path.join(root, "img")
        paths = []
        if os.path.isdir(img_dir):
            for fn in os.listdir(img_dir):
                paths.append(os.path.join(img_dir, fn))
        return paths

    def setup(self, stage: Optional[str] = None):
        # Two dataset instances (same files, different transforms)
        full_train = MIDOG_Dataset(self.data_dir, transform=self.train_transforms)
        full_eval  = MIDOG_Dataset(self.data_dir, transform=self.val_test_transforms)

        n = len(full_train)
        val_len = int(self.val_split * n)
        test_len = int(self.test_split * n) if self.test_split else 0
        train_len = n - val_len - test_len
        if train_len <= 0:
            raise ValueError("Train split is empty. Adjust val_split/test_split.")

        # Make a single set of indices to share between train/eval datasets
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=g).tolist()
        train_idx = perm[:train_len]
        val_idx   = perm[train_len:train_len + val_len]
        test_idx  = perm[train_len + val_len:] if test_len > 0 else []

        self.train_dataset = Subset(full_train, train_idx)
        self.val_dataset   = Subset(full_eval,  val_idx)
        self.test_dataset  = Subset(full_eval,  test_idx) if test_len > 0 else None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=True, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("No test dataset configured.")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, pin_memory=True)

