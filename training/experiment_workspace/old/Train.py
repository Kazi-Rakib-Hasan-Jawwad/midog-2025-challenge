from argparse import ArgumentParser
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import torch
import torch.nn
from torchvision.utils import make_grid
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.nn import functional as F
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import Subset, DataLoader
from Loss_function import Class_Wise_TIL_Detection_FROC
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from network.DeepLabv3_plus import DeepLabv3Plus
from sklearn.model_selection import KFold


tb_loggers = pl_loggers.TensorBoardLogger("logs/")


def one_hot_label(
        labels: torch.Tensor,
        num_classes: int = 3,
        eps: float = 1e-6,
        ignore_index=250,
) -> torch.Tensor:
    device = labels.device
    dtype  = torch.int32
    shape = labels.shape
    one_hot = torch.zeros((shape[0], ignore_index + 1) + shape[1:], device=device, dtype=dtype)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
    ret = torch.split(one_hot, [num_classes, ignore_index + 1 - num_classes], dim=1)[0]

    return ret


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

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr
        self.after_training = True
        #self.net = UNetWithResnet50Encoder(n_classes = 2)


        self.net = DeepLabv3Plus()

        print('fdfasd')

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()

        label = one_hot_label(mask)
        out = self(img)
        '''
        ratios = torch.tensor([0.9863, 0.0063, 0.0074])  # 예시 비율
        weights = 1.0 / (ratios + 1e-6)
        weights = weights / weights.sum()
        '''
        loss_val, f1_score1, f1_score2 = Class_Wise_TIL_Detection_FROC(out, label)

        self.log('train_loss', loss_val, on_step=False, on_epoch=True)
        self.log('train_mitotic_f1', f1_score1, on_step=False, on_epoch=True)
        self.log('train_non_mitotic_f1', f1_score2, on_step=False, on_epoch=True)


        return loss_val

    def validation_step(self, batch, batch_idx):

        img, mask = batch
        img = img.float()
        mask = mask.long()
        label = one_hot_label(mask)
        out = self(img)
        pred = F.softmax(out)

        np_mask = mask.detach().cpu().numpy()
        np_mask = np.transpose(np_mask, (0, 1, 2))
        np_pred = pred.detach().cpu().numpy()
        np_pred = np.transpose(np_pred, (0, 2, 3, 1))
        pred_array = np.array(np_pred)
        argmax_pred = np_pred.argmax(3)

        ratios = torch.tensor([0.9863, 0.0063, 0.0074])  # 예시 비율
        weights = 1.0 / (ratios + 1e-6)
        weights = weights / weights.sum()

        loss_val, f1_score1, f1_score2 = Class_Wise_TIL_Detection_FROC(out, label)

        self.log('val_loss', loss_val, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_mitotic_f1', f1_score1, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_non_mitotic_f1', f1_score2, on_step=False, on_epoch=True, sync_dist=True)



        if (batch_idx == 0) and self.after_training:
            self.logger.experiment.add_image("val_input", make_grid(img, nrow=5))
            self.logger.experiment.add_image("val_label", make_grid(label[:, :3], nrow=5))
            self.logger.experiment.add_image("val_pred", make_grid(pred[:, :3], nrow=5))


        return loss_val

    def training_epoch_end(self, outputs):
        self.after_training = True

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=0.001)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
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
    from Datamodule import MIDOG_DataModule

    seed_everything(1234)

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = SemSegment.add_model_specific_args(parser)
    parser = MIDOG_DataModule.add_argparse_args(parser)

    args = parser.parse_args()
    args.gpus = 1
    args.batch_size = 20
    args.precision = 16
    args.max_epochs = 100
    args.num_workers = 12
    args.lr = 0.001
    args.callbacks = [ModelCheckpoint(save_top_k=1, save_last=False, monitor="val_loss"),
                      LearningRateMonitor()]

    # 데이터 준비
    dm = MIDOG_DataModule(args.data_dir).from_argparse_args(args)
    dm.prepare_data()
    dm.setup(stage='fit')

    full_trainset = dm.trainset  # trainset만 5-fold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_trainset)))):
        print(f"\n==== Fold {fold+1} ====")

        # Subset 생성
        train_subset = Subset(full_trainset, train_idx)
        val_subset = Subset(full_trainset, val_idx)

        # DataLoader 생성
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


        # 모델 초기화
        model = SemSegment(**vars(args))
        print("total parameter count:", count_parameters(model))

        trainer = Trainer.from_argparse_args(args)

        # 학습
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # fold 결과 저장
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            val_loss = val_loss.item()
            fold_results.append(val_loss)
            print(f"Fold {fold+1} val_loss: {val_loss:.4f}")
        else:
            print(f"Fold {fold+1}: val_loss 없음")

    # 평균 출력
    if len(fold_results) > 0:
        avg = sum(fold_results) / len(fold_results)
        print(f"\n=== 5-Fold CV 평균 val_loss: {avg:.4f} ===")
    else:
        print("val_loss를 기록하지 못함.")


if __name__ == "__main__":
    cli_main()
    print("Run Success")
    
