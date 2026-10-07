# Dataset.py
import os
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict

import numpy as np
from torch.utils.data import Dataset
from PIL import Image

DEFAULT_VALID_LABELS = (0, 1, 2)
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

class MIDOG_Dataset(Dataset):
    """
    Pairs images and masks from:
        base_data_dir/
            img/  *.{png,jpg,jpeg,tif,tiff,bmp}
            mask/ *.{png,jpg,jpeg,tif,tiff,bmp}  (same stem as image)
    Transform should be an Albumentations Compose with Resize/Normalize/ToTensorV2, etc.
    """

    def __init__(
        self,
        base_data_dir: str,
        img_size: Tuple[int, int] = (512, 512),
        valid_labels = DEFAULT_VALID_LABELS,
        transform: Optional[Callable] = None,
        ignore_index: int = 250,
    ):
        self.root = base_data_dir
        self.img_size = img_size
        self.valid_labels = tuple(valid_labels)
        self.class_map: Dict[int, int] = dict(zip(self.valid_labels, range(len(self.valid_labels))))
        self.transform = transform
        self.ignore_index = ignore_index

        img_dir = Path(base_data_dir) / "img"
        mask_dir = Path(base_data_dir) / "mask"

        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing image folder: {img_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing mask folder: {mask_dir}")

        # Index masks by stem for robust pairing
        mask_by_stem = {}
        for mp in mask_dir.iterdir():
            if mp.suffix.lower() in ALLOWED_EXTS and mp.is_file():
                mask_by_stem[mp.stem] = mp

        self.img_list = []
        self.mask_list = []
        for ip in sorted(img_dir.iterdir()):
            if ip.suffix.lower() not in ALLOWED_EXTS or not ip.is_file():
                continue
            stem = ip.stem
            mp = mask_by_stem.get(stem, None)
            if mp is None:
                # no matching mask → skip this image
                continue
            self.img_list.append(str(ip))
            self.mask_list.append(str(mp))

        if len(self.img_list) == 0:
            raise RuntimeError(f"No image-mask pairs found under {base_data_dir}/img and /mask")

        print(f"Found {len(self.img_list)} valid image–mask pairs at {base_data_dir}")

    def __len__(self):
        return len(self.img_list)

    def _remap_mask(self, mask_np: np.ndarray) -> np.ndarray:
        """
        Remap raw mask labels to contiguous [0..C-1]; anything else → ignore_index.
        """
        out = np.full_like(mask_np, fill_value=self.ignore_index, dtype=np.uint8)
        for raw_lbl, new_lbl in self.class_map.items():
            out[mask_np == raw_lbl] = np.uint8(new_lbl)
        return out

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        mask_path = self.mask_list[idx]

        # Read as PIL (RGB for image, L for mask), then to numpy (HxWxC and HxW)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = np.array(img)            # uint8, shape HxWx3
        mask = np.array(mask)          # uint8, shape HxW

        # Remap mask labels if necessary
        if self.class_map != {0:0, 1:1, 2:2}:
            mask = self._remap_mask(mask)

        if self.transform is not None:
            sample = self.transform(image=img, mask=mask)
            img, mask = sample["image"], sample["mask"]

        return img, mask

