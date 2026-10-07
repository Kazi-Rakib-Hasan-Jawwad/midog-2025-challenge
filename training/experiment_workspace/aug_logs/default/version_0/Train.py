from argparse import ArgumentParser
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch
import torchmetrics
import torch.nn as nn
from torchvision.utils import make_grid
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.nn import functional as F
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import Subset, DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from network.DeepLabv3_plus import DeepLabv3Plus
from sklearn.model_selection import KFold
from Datamodule import MIDOG_DataModule

#OUTPUT_ROOT = "/home/rakib/PycharmProjects/MIDOG2025_MICCAI/lightning_logs/V2_results"

from torch.utils.data import random_split

def make_loaders(dataset, batch_size, num_workers, val_ratio=0.2, seed=None):
    n = len(dataset)
    val_len = int(n * val_ratio)
    train_len = n - val_len
    g = torch.Generator()
    if seed is None:
        seed = torch.seed()  # different each call
    g.manual_seed(int(seed))
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader


class SemSegment(LightningModule):
    def __init__(
            self,
            lr: float = 0.01,
            num_classes: int = 2,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
            after_training: bool = True,
            **kwargs
    ):
        """Basic model for semantic segmentation. Uses UNet architecture by default.

        The default parameters in this model are for the KITTI dataset. Note, if you'd like to use this model as is,
        you will first need to download the KITTI dataset yourself. You can download the dataset `here.
        <http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015>`_

        Implemented by:

            - `Annika Brundyn <https://github.com/annikabrundyn>`_

        Args:
            num_layers: number of layers in each side of U-net (default 5)
            features_start: number of features in first layer (default 64)
            bilinear: whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.
            lr: learning (default 0.01)
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr
        self.after_training = True
        # self.net = UNetWithResnet50Encoder(n_classes = 2)
        self.net = DeepLabv3Plus()

        class_weights = torch.tensor([0.25, 0.9, 0.9], dtype=torch.float)
        self.register_buffer('class_weights', class_weights)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        # Use the correct F1 class name for your torchmetrics version
        self.train_f1 = torchmetrics.F1(num_classes=self.hparams.num_classes, average='macro')
        self.val_f1 = torchmetrics.F1(num_classes=self.hparams.num_classes, average='macro')

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss = self.criterion(out, mask.long())

        # --- Calculate and Log Metrics ---
        preds = torch.argmax(out, dim=1)
        self.train_f1.update(preds.flatten(), mask.flatten())

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_f1_macro', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss = self.criterion(out, mask.long())
        # --- Calculate and Log Metrics ---
        preds = torch.argmax(out, dim=1)

        self.val_f1.update(preds.flatten(), mask.flatten())

        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        self.log('val_f1_macro', self.val_f1, on_epoch=True, prog_bar=True)

        return loss

    def training_epoch_end(self, outputs):
        self.after_training = True

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.net.parameters(), lr=0.001, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        return [opt], [sch]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
        parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
        parser.add_argument(
            "--bilinear", action="store_true", default=False, help="whether to use bilinear interpolation or transposed"
        )

        return parser


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cli_main():
    seed_everything(1234, workers=True)
    # --- Data Parameters ---
    config = {
        "base_data_dir": "/home/rakib/data/sorted_data/",
        "regular_folders": ['folder1', 'folder2'],
        "minority_folders": ['folder5'],
        "minority_sampling_ratio": 0.25,
        "batch_size": 128,
        "num_workers": 12,
    }

    # --- Model Hyperparameters ---
    model_params = {
        "num_classes": 3,
        "class_weights": [0.2, 0.99, 0.99]
    }

    # --- Trainer Settings ---
    trainer_params = {
        "gpus": 1,
        "precision": 16,
        "max_epochs": 100,
    }

    # --- Instantiate DataModule ---
    dm = MIDOG_DataModule(
        base_data_dir=config["base_data_dir"],
        regular_folders=config["regular_folders"],
        minority_folders=config["minority_folders"],
        minority_sampling_ratio=config["minority_sampling_ratio"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # --- Initialize Model, Logger, and Callbacks ---
    model = SemSegment(**model_params)
    tb_logger = pl_loggers.TensorBoardLogger("aug_logs/")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_f1_macro",
        mode="max",
        filename="midog-{epoch:02d}-{val_f1_macro:.4f}"
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # --- Initialize Trainer ---
    trainer = Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        **trainer_params
    )

    print("Starting training with hardcoded parameters...")
    trainer.fit(model, datamodule=dm)
    print("Training complete.")

if __name__ == "__main__":
    cli_main()
    print("Run Success")
