"""
Backbone for stereo matching: timm MobileNetV2/V3 + FPN.
Outputs three-scale features [feat_full, feat_half, feat_quarter] with dynamic channels.
"""
from __future__ import annotations

from typing import List, Literal, Optional

import torch
import torch.nn as nn
from functools import partial

try:
    import timm
except ImportError:
    timm = None

from .submodules import BasicConv2d, BasicDeconv2d


def _convbn(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    pad: int = 1,
    dilation: int = 1,
) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_planes),
    )


class FPNLayer(nn.Module):
    """Single FPN layer: upsample low-res feature and fuse with high-res skip."""

    def __init__(self, chan_low: int, chan_high: int) -> None:
        super().__init__()
        self.deconv = BasicDeconv2d(
            chan_low,
            chan_high,
            kernel_size=4,
            stride=2,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=partial(nn.ReLU, inplace=True),
        )
        self.conv = BasicConv2d(
            chan_high * 2,
            chan_high,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=partial(nn.ReLU, inplace=True),
        )

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        low = self.deconv(low)
        feat = torch.cat([high, low], dim=1)
        return self.conv(feat)


class Backbone(nn.Module):
    """
    Encapsulates timm MobileNetV2 or MobileNetV3 with FPN.
    Outputs [feat_full, feat_half, feat_quarter] with channels set dynamically.
    """

    BackboneType = Literal["MobileNetv2", "MobileNetV2", "MobileNetV3", "mobilenetv3"]

    def __init__(
        self,
        backbone: str = "MobileNetV2",
        pretrained: bool = False,
        out_channels_half: Optional[int] = None,
        out_channels_quarter: Optional[int] = None,
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for Backbone. Install with: pip install timm")

        self._backbone_name = backbone
        use_v2 = "v2" in backbone.lower() or "V2" in backbone

        # 统一通过 timm 的 features_only 模型：只调用 forward() 获取多尺度特征，
        # 不依赖 conv_stem/bn1/act1/blocks 等内部属性，兼容不同 timm 版本（含 EfficientNetFeatures）
        if use_v2:
            self.feature_model = timm.create_model(
                "mobilenetv2_120d",
                pretrained=pretrained,
                features_only=True,
            )
            if hasattr(self.feature_model, "feature_info"):
                self._channels_from_blocks = self.feature_model.feature_info.channels()
            else:
                self._channels_from_blocks = [24, 32, 40, 112, 384]
        else:
            self.feature_model = timm.create_model(
                "mobilenetv3_large_100",
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3, 4),
            )
            self._channels_from_blocks = self.feature_model.feature_info.channels()

        channels = self._channels_from_blocks
        self.fpn_layer4 = FPNLayer(channels[4], channels[3])
        self.fpn_layer3 = FPNLayer(channels[3], channels[2])
        self.fpn_layer2 = FPNLayer(channels[2], channels[1])
        self.fpn_layer1 = FPNLayer(channels[1], channels[0])

        ch_half = out_channels_half if out_channels_half is not None else channels[0]
        ch_quarter = out_channels_quarter if out_channels_quarter is not None else channels[1]

        self.out_conv_half = BasicConv2d(
            channels[0],
            ch_half,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
        )
        self.out_conv_quarter = BasicConv2d(
            channels[1],
            ch_quarter,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
        )

        self.upsample_full = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            BasicConv2d(
                ch_half,
                ch_half,
                kernel_size=3,
                padding=1,
                norm_layer=nn.BatchNorm2d,
                act_layer=None,
            ),
        )

        self._ch_half = ch_half
        self._ch_quarter = ch_quarter
        self.output_channels: List[int] = [ch_half, ch_half, ch_quarter]

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            images: [B, 3, H, W]

        Returns:
            [feat_full, feat_half, feat_quarter]
            - feat_full:  [B, C_full, H, W]
            - feat_half:  [B, C_half, H/2, W/2]
            - feat_quarter: [B, C_quarter, H/4, W/4]
        """
        # 仅通过 forward 获取多尺度特征，兼容任意 timm features_only 实现（含 EfficientNetFeatures）
        features = self.feature_model(images)
        c1, c2, c3, c4, c5 = features[0], features[1], features[2], features[3], features[4]

        p4 = self.fpn_layer4(c5, c4)
        p3 = self.fpn_layer3(p4, c3)
        p2 = self.fpn_layer2(p3, c2)
        p1 = self.fpn_layer1(p2, c1)

        feat_half = self.out_conv_half(p1)
        feat_quarter = self.out_conv_quarter(p2)
        feat_full = self.upsample_full(feat_half)

        return [feat_full, feat_half, feat_quarter]
