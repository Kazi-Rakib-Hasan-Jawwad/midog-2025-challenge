import os
import torch
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

# --- User-Provided Code (with necessary imports) ---
# Assuming these custom modules are in the project directory
from network.DeepLabv3_plus import DeepLabv3Plus
from Datamodule import MIDOG_DataModule  # Assuming this file exists and contains the DataModule
from torch.nn import functional as F


# Helper function from user, updated to reflect num_classes=3 from training
def one_hot_label(labels: torch.Tensor, num_classes: int = 3, ignore_index: int = 250) -> torch.Tensor:
    """
    Converts a label tensor to a one-hot encoded tensor.
    Args:
        labels: Ground truth labels (B, H, W).
        num_classes: Number of classes.
        ignore_index: Index to ignore.
    Returns:
        One-hot encoded label tensor.
    """
    valid = (labels != ignore_index)
    safe = labels.clone()
    safe[~valid] = 0  # Clamp ignored to a valid class for one_hot
    oh = F.one_hot(safe, num_classes=num_classes).permute(0, 3, 1, 2).float()
    oh *= valid.unsqueeze(1)  # Zero-out ignored pixels across all classes
    return oh


# Placeholder for the custom loss function.
# This is required for the LightningModule to load correctly from the checkpoint.
def Class_Wise_TIL_Detection_FROC(out, label):
    # This function is kept for compatibility with the LightningModule definition.
    # We return dummy values as they are not used during inference.
    return 0.0, 0.0, 0.0


class SemSegment(LightningModule):
    """
    LightningModule for Semantic Segmentation.
    This definition MUST match the one in train.py for `load_from_checkpoint` to work correctly.
    """

    def __init__(
            self,
            lr: float = 0.01,
            num_classes: int = 3,  # Defaulting to 3 as per train.py
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
            after_training: bool = True,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        # This network initialization must match train.py
        self.net = DeepLabv3Plus()

    def forward(self, x):
        return self.net(x)

    # These steps are not executed during inference but are needed for the class definition
    def training_step(self, batch, batch_nb):
        img, mask = batch
        out = self(img.float())
        label = one_hot_label(mask.long(), num_classes=self.num_classes)
        loss, _, _ = Class_Wise_TIL_Detection_FROC(out, label)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        out = self(img.float())
        label = one_hot_label(mask.long(), num_classes=self.num_classes)
        loss, _, _ = Class_Wise_TIL_Detection_FROC(out, label)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        return [opt], [sch]


# --- Inference Logic ---

def run_inference(args):
    """
    Main function to run inference and calculate metrics.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    # MODIFIED: Correctly initialize the DataModule as per user's training script example.
    print("Initializing DataModule...")
    data_module = MIDOG_DataModule(
        base_data_dir=args.data_dir,
        regular_folders=['folder1', 'folder2'],  # Assuming these are constant for testing
        minority_folders=['folder5'],  # Assuming these are constant for testing
        test_split=0.2,  # Assuming this is the desired test split
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()
    print(f"Test dataset loaded with {len(test_loader.dataset)} samples.")

    # 3. Load Model from Checkpoint
    print(f"Loading model from checkpoint: {args.ckpt_path}")
    model = SemSegment.load_from_checkpoint(args.ckpt_path, map_location=device, strict=False)
    print("Model loaded successfully from checkpoint.")

    # 4. Freeze Model and Set to Eval Mode
    model.freeze()
    model.to(device)
    model.eval()
    print("Model frozen and set to evaluation mode.")

    # 5. Run Prediction Loop
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running Inference"):
            imgs, masks = batch
            imgs = imgs.float().to(device)
            masks = masks.long()  # Keep masks on CPU until needed

            # Get model predictions
            logits = model(imgs)

            if isinstance(logits, tuple):
                prediction_tensor = logits[0]
            else:
                prediction_tensor = logits

            preds = torch.argmax(prediction_tensor, dim=1)

            # MODIFIED: Convert tensors to a memory-efficient type (uint8) before storing.
            # This drastically reduces RAM usage for large datasets.
            all_preds.append(preds.view(-1).cpu().to(torch.uint8))
            all_labels.append(masks.view(-1).cpu().to(torch.uint8))

    # Concatenate all batch results
    print("Concatenating results...")
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # 6. Calculate and Display Metrics
    print("\n--- Evaluation Metrics ---")

    target_names = ['background', 'non-mitotic', 'mitotic']
    labels_of_interest = list(range(len(target_names)))

    print("Calculating precision, recall, and F1 scores...")
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average=None,
        labels=labels_of_interest,
        zero_division=0
    )

    avg_precision, avg_recall, avg_f1_score, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average='macro',
        zero_division=0
    )

    print(f"Mitotic F1 Score (Class 2): {f1_score[2]:.4f}")
    print(f"Non-Mitotic F1 Score (Class 1): {f1_score[1]:.4f}")
    print(f"Background F1 Score (Class 0): {f1_score[0]:.4f}")
    print("-" * 35)
    print(f"Average F1 Score (Macro): {avg_f1_score:.4f}")
    print(f"Overall Precision (Macro): {avg_precision:.4f}")
    print(f"Overall Recall (Macro): {avg_recall:.4f}")

    print("Calculating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds, labels=labels_of_interest)
    print("\nConfusion Matrix:")
    print(" " * 12 + " ".join([f"{name[:4]}." for name in target_names]) + " (Predicted)")
    print("-" * 45)
    for i, row in enumerate(cm):
        print(f"{target_names[i][:10]:<10s} | {' '.join(map(str, row))}")
    print("(Actual)")


def main():
    parser = ArgumentParser(description="Inference script for Mitosis Detection")

    # --- Arguments ---
    parser.add_argument("--ckpt_path", type=str, default="/home/rakib/PycharmProjects/MIDOG2025_MICCAI/aug_logs/default/server/v_4/checkpoints/midog-epoch=99-val_loss=0.0142.ckpt")
    parser.add_argument("--data_dir", type=str, default="/home/rakib/data/MIDOGpp-main/224/sorted_data/",
                        help="Directory containing the test data.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for the DataLoader.")

    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()
