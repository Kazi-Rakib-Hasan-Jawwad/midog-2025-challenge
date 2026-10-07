from argparse import ArgumentParser
import os, math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"           # pinpoint failing op
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"     # richer CUDA stack
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
'''
class RefreshHardExamples(Callback):
    def __init__(self, every_n_epochs=3, miner_script="seg_hnm_miner.py"):
        super().__init__()
        self.every_n = every_n_epochs
        self.script = miner_script

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n == 0:
            pl_module.print(f"[HNM] Mining hard examples at epoch {epoch} …")
            # Your miner should copy/symlink mined FP/FN patches back into /img and /mask
            subprocess.run(["python", self.script], check=True)
            # Re-scan so new files are included
            trainer.datamodule.setup(stage=None)
            trainer.reset_train_dataloader(pl_module)
            pl_module.print("[HNM] Mining done. Dataloader refreshed.")
'''
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

def _freeze_backbone_bn(self):
    for module in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

class SemSegment(LightningModule):
    def __init__(
            self,
            lr: float = 0.0003,
            num_classes: int = 3,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
            after_training: bool = True,
            effective_bs: int = 256,          # ← NEW: pass your target effective batch
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
        self.effective_bs = effective_bs
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

    def training_epoch_end(self, outputs):
        self.after_training = True
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)
    
    def configure_optimizers(self):
        # --- safer √ scaling of LR by effective batch vs reference 16 ---
        ref_bs = 16.0
        scaled_lr = self.lr * math.sqrt(self.effective_bs / ref_bs)

        # Exclude norm/bias from weight decay
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad: 
                continue
            if any(k in n.lower() for k in ['bias', 'bn', 'norm']):
                no_decay.append(p)
            else:
                decay.append(p)

        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 1e-4},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=scaled_lr, betas=(0.9, 0.999), eps=1e-7
        )

        # Cosine with warmup across epochs (PL 1.5-safe)
        warmup_epochs = max(1, int(0.05 * self.max_epochs))
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            prog = (epoch - warmup_epochs) / max(1, (self.max_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
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
        "batch_size": 18, # ↓ from 36 (prevents OOM)
        "num_workers": 6,
        "val_split": 0.2,
        "test_split": None,
        "img_size": (512, 512),
    }
    
    # --- Target effective batch = 256 (ceil division if not divisible) ---
    target_eff_bs = 252
    micro_bs = config["batch_size"]
    accum = (target_eff_bs + micro_bs - 1) // micro_bs
    
    # --- Model Hyperparameters ---
    model_params = {"num_classes": 3}

    # --- Trainer Settings ---
    trainer_params = {
        "gpus": 1,
        "precision": 16,
        "max_epochs": 100,
    }


    # --- Data/Trainer settings ---
    dm = MIDOG_DataModule(
        data_dir=config["data_dir"],
        val_split=config["val_split"],
        test_split=config["test_split"],
        batch_size=micro_bs,
        num_workers=config["num_workers"],
        img_size=config["img_size"],
    )

    model = SemSegment(
        num_classes=3,
        lr=3e-4,                     # base LR for ref_bs=16 (scaled inside)
        effective_bs=accum * micro_bs,
        max_epochs=100,
    )
    model.net._freeze_backbone_bn()  # make sure BN stays frozen even if Trainer toggles modes
    
    tb_logger = pl_loggers.TensorBoardLogger("aug_logs/")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3, monitor="val_loss", mode="min", save_last=True, filename="midog-{epoch:02d}"
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # --- Initialize Trainer ---
    trainer = Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        amp_backend="native",
        accumulate_grad_batches=accum,   # keeps effective batch ~36
        gradient_clip_val=1.0,           # helps stability
        benchmark=True,                  # cudnn benchmark for speed
        deterministic=False,
        log_every_n_steps=25,
        num_sanity_val_steps=2,
        **trainer_params
    )

    print("Starting training with hardcoded parameters...")
    trainer.fit(model, datamodule=dm)
    print("Training complete.")

if __name__ == "__main__":
    cli_main()
    print("Run Success")

