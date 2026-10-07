#/utils/model.py

from typing import Dict, List, Optional, Tuple, Union
import torchvision
import torch 
import logging 
from functools import partial
from torch import nn as nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from DeepLabv3_plus import DeepLabv3Plus

class SemSegment(nn.Module):
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

    def forward(self, x):
        return self.net(x)

def make_DLv3_model():
    # create model 
    model = DeepLabv3Plus()
        
    
    return model 




