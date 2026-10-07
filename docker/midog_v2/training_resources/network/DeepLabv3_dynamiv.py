# network/DeepLabv3_plus.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.ASPP import ASPP_Bottleneck
from torchvision.models import resnet50

feature_extracting = True

def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1] * (num_blocks - 1)
    blocks = []
    for s in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=s, dilation=dilation))
        in_channels = block.expansion * channels
    return nn.Sequential(*blocks)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super().__init__()
        out_channels = self.expansion * channels
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if (stride != 1) or (in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.relu(out + self.downsample(x))
        return out

class DeepLabv3Plus(nn.Module):
    """Detection‑centric DeepLabv3+: 2‑ch logits + regional/local log‑variance heads."""
    def __init__(self, num_classes_det: int = 2, output_stride: int = 16):
        super().__init__()
        resnet = resnet50(pretrained=False)
        self.conv1  = nn.Sequential(*list(resnet.children())[:1])
        self.bn1    = nn.Sequential(*list(resnet.children())[1:2])
        self.relu   = nn.Sequential(*list(resnet.children())[2:3])
        self.maxpool= nn.Sequential(*list(resnet.children())[3:4])
        self.layer1 = nn.Sequential(*list(resnet.children())[4:5])   # 1/4
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6])   # 1/8
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7])   # 1/16
        # keep OS=16 backbone; for OS=8, set stride=1 in layer3 and increase dilation
        self.layer4 = make_layer(Bottleneck, in_channels=4 * 256, channels=512, num_blocks=3, stride=1, dilation=2)

        self.aspp_det = ASPP_Bottleneck(num_classes=256)  # 256 ch features

        self.low_level_conv = nn.Conv2d(512, 48, kernel_size=1)
        self.low_level_bn   = nn.BatchNorm2d(48)

        self.decoder_conv1_det = nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1)
        self.decoder_bn1 = nn.BatchNorm2d(256)
        self.decoder_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.decoder_bn2 = nn.BatchNorm2d(256)

        # heads
        self.final_conv_det = nn.Conv2d(256, num_classes_det, kernel_size=1)

        self.regional_uncertainty_head = nn.Sequential(
            nn.Conv2d(256 + 48, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes_det, kernel_size=1)
        )
        self.local_uncertainty_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes_det, kernel_size=1)
        )

    def forward(self, x):
        B, _, H, W = x.shape
        c = self.conv1(x); b = self.bn1(c); r = self.relu(b); m = self.maxpool(r)
        c1 = self.layer1(m)   # 1/4
        c2 = self.layer2(c1)  # 1/8
        c3 = self.layer3(c2)  # 1/16
        out = self.layer4(c3) # 1/16

        aspp_out = self.aspp_det(out)  # 1/16 → (B,256,.,.)
        low = self.low_level_bn(self.low_level_conv(c2))  # (B,48,H/8,W/8)
        aspp_up = F.interpolate(aspp_out, size=low.shape[2:], mode='bilinear', align_corners=False)
        feat_1_8 = torch.cat([aspp_up, low], dim=1)  # (B,304,H/8,W/8)

        # decoder @1/8
        x_det = F.relu(self.decoder_bn1(self.decoder_conv1_det(feat_1_8)))
        x_det = F.relu(self.decoder_bn2(self.decoder_conv2(x_det)))        # (B,256,H/8,W/8)

        # logits @full‑res
        x_full = F.interpolate(x_det, size=(H, W), mode='bilinear', align_corners=False)
        logits = self.final_conv_det(x_full)                                # (B,2,H,W)

        # regional/local log‑variance
        reg_logvar_1_8 = self.regional_uncertainty_head(feat_1_8)           # (B,2,H/8,W/8)
        reg_logvar = F.interpolate(reg_logvar_1_8, size=(H, W), mode='bilinear', align_corners=False)
        loc_logvar_1_8 = self.local_uncertainty_head(x_full)                    # (B,2,H,W)
        loc_logvar = F.interpolate(loc_logvar_1_8, size=(H, W), mode='bilinear', align_corners=False)


        return logits, reg_logvar, loc_logvar