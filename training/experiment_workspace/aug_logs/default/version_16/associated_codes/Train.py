from argparse import ArgumentParser
import os, math
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
from torch.nn import functional as F
from Loss_function_bce import Class_Wise_TIL_Detection_FROC
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from network.DeepLabv3_plus import DeepLabv3Plus
from Datamodule import MIDOG_DataModule
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def one_hot_label(labels: torch.Tensor, num_classes: int = 3, ignore_index: int = 250) -> torch.Tensor:
    # labels: (B,H,W), long
    valid = (labels != ignore_index)
    safe = labels.clone()
    safe[~valid] = 0                            # clamp ignored to a valid class for one_hot
    oh = F.one_hot(safe, num_classes=num_classes).permute(0, 3, 1, 2).float()
    oh *= valid.unsqueeze(1)                    # zero-out ignored pixels across all classes
    return oh

def make_loaders(dataset, batch_size, num_workers, val_ratio=0.2, seed=None):
    n = len(dataset)
    val_len = int(n * val_ratio)
    train_len = n - val_len
    g = torch.Generator()
    if seed is None:
        seed = torch.seed()  # different each call
    g.manual_seed(int(seed))
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=False, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, persistent_workers=True)
    return train_loader, val_loader

class SemSegment(LightningModule):
    def __init__(
            self,
            lr: float = 0.001,
            num_classes: int = 3,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
            after_training: bool = True,
            #effective_bs: int = 256,          # ← NEW: pass your target effective batch
            max_epochs: int = 100,            # ← NEW: for cosine warmup
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
        #self.effective_bs = effective_bs
        self.max_epochs = max_epochs
        self.after_training = True
        self.net = DeepLabv3Plus()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        label = one_hot_label(mask)
        loss, f1_score1, f1_score2 = Class_Wise_TIL_Detection_FROC(out, label)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_mitotic_f1', f1_score1, on_step=False, on_epoch=True)
        self.log('train_non_mitotic_f1', f1_score2, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        label = one_hot_label(mask)
        loss, f1_score1, f1_score2 = Class_Wise_TIL_Detection_FROC(out, label)

        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        self.log('val_mitotic_f1', f1_score1, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_non_mitotic_f1', f1_score2, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_training_epoch_end(self, outputs):
        self.after_training = True

    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.net.parameters(), lr=0.001, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)
        return [opt], [sch]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=0.001, help="adamw: learning rate")
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

    config = {
        "data_dir": "/home/rakib/data/512_seg_root/",
        "batch_size": 28,
        "num_workers": 6,
        "val_split": 0.2,
        "test_split": None,
    }
    
    # --- Model Hyperparameters ---
    model_params = {"num_classes": 3}

    # --- Trainer Settings ---
    trainer_params = {
        "gpus": 1, # "accelerator": "gpu",
        "precision": 16, #bf16-mixed",
        "max_epochs": 100,
    }


    # --- Data/Trainer settings ---
    dm = MIDOG_DataModule(
        data_dir=config["data_dir"],
        val_split=config["val_split"],
        test_split=config["test_split"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # --- Initialize Model, Logger, and Callbacks ---
    model = SemSegment(**model_params)
    
    tb_logger = pl_loggers.TensorBoardLogger("aug_logs/")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2, monitor="val_loss", mode="min", save_last=True, filename="midog-{epoch:02d}"
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # --- Initialize Trainer ---
    trainer = Trainer(
        logger=tb_logger,
        #reload_dataloader_every_n_epoch=10,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        accumulate_grad_batches=10, #{1: 4, 50: 6, 75: 5}
        **trainer_params
    )

    print("Starting training with hardcoded parameters...")
    trainer.fit(model, datamodule=dm)
    print("Training complete.")

if __name__ == "__main__":
    cli_main()
    print("Run Success")

