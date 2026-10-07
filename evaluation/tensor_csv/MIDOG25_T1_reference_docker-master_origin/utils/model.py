from typing import Dict, List, Optional, Tuple, Union
import torchvision
import torch 
import logging 
from functools import partial

from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models._utils import  _ovewrite_value_param
from torchvision.models.resnet import ResNet50_Weights, ResNet
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _resnet_fpn_extractor

from torchvision.models.detection.fcos import FCOS
from torchvision.ops.feature_pyramid_network import LastLevelP6P7, LastLevelMaxPool




BACKBONES = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2"
    ]


WEIGHTS = [
    'IMAGENET1K_V1', 
    'IMAGENET1K_V2', 
    None
]



def load_resnet_backbone(
        backbone: str = 'resnet50',
        weights: str = 'IMAGENET1K_V1'
        ) -> Tuple[ResNet, Dict[str, List[float]]]:
    """Load a ResNet backbone with specified weights.

    Args:
        backbone (str, optional): ResNet architecture name. Defaults to 'resnet50'.
        weights (str, optional): Pretrained weights type. Defaults to 'IMAGENET1K_V1'.

    Raises:
        ValueError: If backbone is not in supported BACKBONES list
        ValueError: If weights is not in supported WEIGHTS list

    Returns:
        ResNet: Pretrained ResNet backbone model
    """
    if backbone not in BACKBONES:
        raise ValueError(f'Unsupported backbone: {backbone}.')

    if weights in WEIGHTS:
        backbone = resnet.__dict__[backbone](weights=weights)
    else:
        raise ValueError(f'Unsupported weights: {weights}.')

    return backbone



def make_fcos_model(
        num_classes: int = 2,
        backbone: str = 'resnet50',
        weights: str = 'IMAGENET1K_V2',
        extra_blocks: bool = False,
        returned_layers: List[int] = [1, 2, 3, 4],
        trainable_backbone_layers: int = 5,
        center_sampling_radius: float = 1.5,
        det_thresh: float = 0.2,
        patch_size: int = 512,
        detections_per_img: int = 300,
        topk_candidates: int = 1000,
        image_mean: List[float] = None,
        image_std: List[float] = None,
        **kwargs) -> FCOS:
    """Create an FCOS (Fully Convolutional One-Stage) object detection model.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 2.
        backbone (str, optional): Backbone architecture. Defaults to 'resnet50'.
        weights (str, optional): Pretrained weights type. Defaults to 'IMAGENET1K_V2'.
        extra_blocks (bool, optional): Whether to add P6/P7 FPN levels. Defaults to False.
        returned_layers (List[int], optional): FPN layers to use. Defaults to [1,2,3,4].
        trainable_backbone_layers (int, optional): Number of trainable backbone layers. Defaults to 5.
        center_sampling_radius (float, optional): Center sampling radius. Defaults to 1.5.
        det_thresh (float, optional): Detection confidence threshold. Defaults to 0.2.
        patch_size (int, optional): Input image size. Defaults to 512.
        detections_per_img (int, optional): Maximum detections per image. Defaults to 300.
        topk_candidates (int, optional): Number of top candidates to keep. Defaults to 1000.
        image_mean (List[float], optional): Image normalization mean. Defaults to None.
        image_std (List[float], optional): Image normalization std. Defaults to None.
        **kwargs: Additional arguments for FCOS model.

    Returns:
        FCOS: Configured FCOS model
    """
    
    # load backbone
    backbone = load_resnet_backbone(backbone=backbone, weights=weights)

    if extra_blocks:
        extra_blocks = LastLevelP6P7(256, 256)
    else:
        extra_blocks = LastLevelMaxPool()

    # load backbone with FPN
    backbone = _resnet_fpn_extractor(
        backbone=backbone,
        trainable_layers=trainable_backbone_layers,
        returned_layers=returned_layers,
        extra_blocks=extra_blocks
        )
    

    # if anchors are provided
    anchor_sizes = ((8,), (16,), (32,), (64,), (128,))  # equal to strides of multi-level feature map
    anchor_sizes = tuple([anchor_sizes[0]] + [anchor_sizes[i] for i in returned_layers])  # select specific feature maps
    aspect_ratios = ((1.0,),) * len(anchor_sizes)  # set only one anchor
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)


    # create model 
    model = FCOS(
        backbone,
        num_classes,
        anchor_generator=anchor_generator,
        min_size = patch_size,
        max_size = patch_size,
        image_mean = image_mean,
        image_std = image_std,
        center_sampling_radius = center_sampling_radius,
        score_thresh = det_thresh,
        nms_thresh = 0.6,
        detections_per_img = detections_per_img,
        topk_candidates = topk_candidates,
        **kwargs
        )
        
    
    return model 




