import os
import random
from typing import Any, Optional, List, Callable
import argparse
import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, ConcatDataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.domain_adaptation import FDA
from PIL import Image
import matplotlib.pyplot as plt
from Dataset import MIDOG_Dataset

class MacenkoNormalizer:
    """
    Minimal Macenko stain normalization.
    Input/Output: uint8 RGB np.ndarray, shape (H, W, 3).
    """
    def __init__(self, Io=240, alpha=1, beta=0.15,
                 HERef=np.array([[0.5626, 0.2159],
                                 [0.7201, 0.8012],
                                 [0.4062, 0.5581]], dtype=np.float32),
                 maxCRef=np.array([1.9705, 1.0308], dtype=np.float32)):
        self.Io = float(Io)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.HERef = HERef.astype(np.float32)
        self.maxCRef = maxCRef.astype(np.float32)

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        # expects uint8 RGB
        h, w, _ = img_rgb.shape
        img = img_rgb.reshape((-1, 3)).astype(np.float32)

        # Optical Density (avoid log(0) with +1)
        OD = -np.log10((img + 1.0) / self.Io)

        # Remove transparent pixels
        ODhat = OD[~np.any(OD < self.beta, axis=1)]
        if ODhat.size == 0:
            return img_rgb  # fallback if no tissue

        # SVD
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

        # Project on the plane spanned by 2 largest eigenvectors
        That = ODhat @ eigvecs[:, 1:3]
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, self.alpha)
        maxPhi = np.percentile(phi, 100 - self.alpha)

        vMin = eigvecs[:, 1:3] @ np.array([[np.cos(minPhi)], [np.sin(minPhi)]], dtype=np.float32)
        vMax = eigvecs[:, 1:3] @ np.array([[np.cos(maxPhi)], [np.sin(maxPhi)]], dtype=np.float32)

        if vMin[0] > vMax[0]:
            HE = np.concatenate([vMin, vMax], axis=1)
        else:
            HE = np.concatenate([vMax, vMin], axis=1)

        Y = OD.T
        C, *_ = np.linalg.lstsq(HE, Y, rcond=None)
        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)], dtype=np.float32)
        C2 = C / (maxC / self.maxCRef)[:, None]

        Inorm = self.Io * np.exp(-self.HERef @ C2)
        Inorm = np.clip(Inorm.T.reshape(h, w, 3), 0, 255).astype(np.uint8)
        return Inorm

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
        self.macenko = MacenkoNormalizer()
        self.target_image_paths = self._get_all_image_paths(regular_folders + minority_folders)

        def read_fn(path):
            return np.array(Image.open(path).convert('RGB'))

        self.train_transforms = A.Compose([
            A.Lambda(image=lambda im, **k: self.macenko(im)),  # STAIN STEP (Macenko)
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.RandomRotate90(p=1.0),
            ], p=0.8),
            FDA(reference_images=self.target_image_paths, read_fn=read_fn, beta_limit=0.01, p=0.8),
            A.ElasticTransform(p=0.3, alpha=8, sigma=6, alpha_affine=5),
            #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.val_test_transforms = A.Compose([
            A.Lambda(image=lambda im, **k: self.macenko(im)),
            #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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

'''
# =================================================================================
# NEW: Visualization Function and Main Execution Block
# =================================================================================
def visualize_augmentations(dm, num_samples=4):
    """Loads samples and plots a detailed comparison of the FDA and full augmentation pipeline."""

    # --- Create a "clean" dataset without any augmentations ---
    clean_dataset = ConcatDataset([
        MIDOG_Dataset(dm.base_data_dir, dm.regular_folders, transform=None),
        MIDOG_Dataset(dm.base_data_dir, dm.minority_folders, transform=None)
    ])

    # --- Isolate the FDA transform from the main pipeline ---
    def read_fn(path):
        return np.array(Image.open(path).convert('RGB'))

    fda_transform = A.Compose([
        FDA(reference_images=dm.target_image_paths, read_fn=read_fn, beta_limit=0.2, p=1.0),  # p=1.0 to force it
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # --- Setup the plot ---
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    fig.suptitle("FDA (20% style transfer) Visualization", fontsize=20)

    # Get random indices for unique samples
    indices = random.sample(range(len(clean_dataset)), num_samples * 2)  # Get extra for style sources

    for i in range(num_samples):
        # --- 1. Get Original Target and Style Source ---
        target_idx = indices[i]
        style_idx = indices[i + num_samples]

        target_img, target_mask = clean_dataset[target_idx]
        style_img, _ = clean_dataset[style_idx]

        # --- 2. Apply FDA-Only Augmentation ---
        # We need to manually provide the style image to the FDA transform
        # The easiest way is to temporarily replace the reference images
        fda_transform.transforms[0].reference_images = [dm.target_image_paths[style_idx]]
        fda_result = fda_transform(image=target_img, mask=target_mask)
        fda_img_tensor = fda_result['image']

        # --- 3. Apply Full Augmentation Pipeline ---
        full_aug_result = dm.train_transforms(image=target_img, mask=target_mask)
        #full_aug_img_tensor = full_aug_result['image']
        #full_aug_mask_tensor = full_aug_result['mask']

        # --- 4. Denormalize images for plotting ---
        def denormalize(tensor):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = tensor * std + mean
            return torch.clamp(tensor, 0, 1).permute(1, 2, 0).numpy()

        def prepare_for_plotting(tensor):
            """Rescales a normalized tensor to [0,1] and permutes dimensions for plotting."""
            # Ensure tensor is on CPU and clone it to avoid modifying the original
            tensor = tensor.cpu().clone()
            # Permute from (C, H, W) to (H, W, C) for matplotlib
            tensor = tensor.permute(1, 2, 0)
            # Rescale from its normalized range to the [0, 1] range
            min_val = tensor.min()
            max_val = tensor.max()
            tensor = (tensor - min_val) / (max_val - min_val)
            return tensor.numpy()

        fda_img_vis = denormalize(fda_img_tensor)
        #full_aug_img_vis = denormalize(full_aug_img_tensor)
        #full_aug_mask_vis = full_aug_mask_tensor.cpu().numpy()

        # --- 5. Plotting ---
        axes[i, 0].imshow(style_img)
        axes[i, 0].set_title(f"Style Source #{style_idx}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(target_img)
        axes[i, 1].set_title(f"Original Target #{target_idx}")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(target_mask, cmap='gray')
        axes[i, 2].set_title(f"Original Mask #{target_idx}")
        axes[i, 2].axis('off')
        
        axes[i, 2].imshow(fda_img_vis)
        axes[i, 2].set_title("Target img after FDA-Only")
        axes[i, 2].axis('off')
        
        axes[i, 4].imshow(full_aug_img_vis)
        axes[i, 4].set_title("Target img after Full Augmentation")
        axes[i, 4].axis('off')

        axes[i, 5].imshow(full_aug_mask_vis)
        axes[i, 5].set_title("Target mask after Full Augmentation")
        axes[i, 5].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def cli_main():

    config = {
        "base_data_dir": "/home/rakib/data/MIDOGpp-main/224/sorted_data/",
        "regular_folders": ['folder1', 'folder2'],
        "minority_folders": ['folder5'],
        "minority_sampling_ratio": 0.6,
        "batch_size": 3,   # Batch size should be the number of samples I want
        "num_workers": 0,  # Use 0 workers for simple visualization to avoid multiprocessing issues
    }
    # --- Instantiate DataModule with your standard parameters ---
    dm = MIDOG_DataModule(
        base_data_dir=config["base_data_dir"],
        regular_folders=config['regular_folders'],
        minority_folders=config["minority_folders"],
        minority_sampling_ratio=config["minority_sampling_ratio"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )

    print(f"Loading {config['batch_size']} samples to visualize augmentations...")
    visualize_augmentations(dm, num_samples=config["batch_size"])


if __name__ == "__main__":
    # This block allows you to run this script directly to visualize data
    cli_main()
'''