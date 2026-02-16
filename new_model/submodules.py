"""
Base building blocks for stereo matching: conv layers, cost volumes, warping, 3D hourglass.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 2D Convolution blocks
# -----------------------------------------------------------------------------


class BasicConv2d(nn.Module):
    """Conv2d + optional BatchNorm + optional activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        norm_layer: Optional[type[nn.Module]] = None,
        act_layer: Optional[type[nn.Module]] = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                **kwargs,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if act_layer is not None:
            layers.append(act_layer())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BasicDeconv2d(nn.Module):
    """ConvTranspose2d + optional BatchNorm + optional activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        norm_layer: Optional[type[nn.Module]] = None,
        act_layer: Optional[type[nn.Module]] = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                **kwargs,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if act_layer is not None:
            layers.append(act_layer())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def convbn_3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    pad: int,
) -> nn.Sequential:
    """Conv3d + BatchNorm3d."""
    return nn.Sequential(
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        ),
        nn.BatchNorm3d(out_channels),
    )


def convbn1(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    pad: int,
    dilation: int,
) -> nn.Sequential:
    """Conv2d (with dilation support) + BatchNorm2d."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    )


class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


# -----------------------------------------------------------------------------
# Cost volumes
# -----------------------------------------------------------------------------


def _groupwise_correlation(
    fea1: torch.Tensor,
    fea2: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view(B, num_groups, channels_per_group, H, W).mean(dim=2)
    return cost


def build_gwc_volume(
    refimg_fea: torch.Tensor,
    targetimg_fea: torch.Tensor,
    maxdisp: int,
    num_groups: int,
) -> torch.Tensor:
    """
    Build group-wise correlation cost volume.
    Returns: [B, num_groups, maxdisp, H, W]
    """
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros(B, num_groups, maxdisp, H, W)
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = _groupwise_correlation(
                refimg_fea[:, :, :, i:],
                targetimg_fea[:, :, :, :-i],
                num_groups,
            )
        else:
            volume[:, :, i, :, :] = _groupwise_correlation(
                refimg_fea,
                targetimg_fea,
                num_groups,
            )
    return volume.contiguous()


def build_correlation_volume(
    refimg_fea: torch.Tensor,
    targetimg_fea: torch.Tensor,
    maxdisp: int,
    num_groups: int,
) -> torch.Tensor:
    """
    Build correlation volume with range [-maxdisp, maxdisp].
    Returns: [B, num_groups, 2*maxdisp+1, H, W]
    """
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros(B, num_groups, 2 * maxdisp + 1, H, W)
    for i in range(-maxdisp, maxdisp + 1):
        if i > 0:
            volume[:, :, i + maxdisp, :, i:] = _groupwise_correlation(
                refimg_fea[:, :, :, i:],
                targetimg_fea[:, :, :, :-i],
                num_groups,
            )
        elif i < 0:
            volume[:, :, i + maxdisp, :, :-i] = _groupwise_correlation(
                refimg_fea[:, :, :, :-i],
                targetimg_fea[:, :, :, i:],
                num_groups,
            )
        else:
            volume[:, :, i + maxdisp, :, :] = _groupwise_correlation(
                refimg_fea,
                targetimg_fea,
                num_groups,
            )
    return volume.contiguous()


# -----------------------------------------------------------------------------
# Feature warping
# -----------------------------------------------------------------------------


def warp_features(feature_right: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """
    Warp right feature map to left view using disparity.
    feature_right: [B, C, H, W], disp: [B, 1, H, W]
    Returns: [B, C, H, W]
    """
    B, C, H, W = feature_right.size()
    device = feature_right.device
    xx = torch.arange(0, W, device=device, dtype=torch.float32).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=device, dtype=torch.float32).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    xx_warp = xx - disp
    vgrid = torch.cat([xx_warp, yy], dim=1)
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(feature_right, vgrid, align_corners=True)
    mask = torch.ones_like(feature_right, device=device)
    mask = F.grid_sample(mask, vgrid, align_corners=True)
    mask = (mask >= 0.999).float()
    return output * mask


# -----------------------------------------------------------------------------
# Disparity regression
# -----------------------------------------------------------------------------


def disparity_regression(prob: torch.Tensor, maxdisp: int) -> torch.Tensor:
    """
    Expectation over disparity dimension.
    prob: [B, D, H, W] (e.g. after softmax), returns [B, H, W]
    """
    assert prob.dim() == 4
    disp_values = torch.arange(0, maxdisp, dtype=prob.dtype, device=prob.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(prob * disp_values, dim=1)


# -----------------------------------------------------------------------------
# 3D Hourglass & Cost aggregation
# -----------------------------------------------------------------------------


class Hourglass3D(nn.Module):
    """3D hourglass: down two strides, then up with skip connections."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels * 4,
                in_channels * 2,
                3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm3d(in_channels * 2),
        )
        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels * 2,
                in_channels,
                3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm3d(in_channels),
        )
        self.redir1 = convbn_3d(in_channels, in_channels, 1, 1, 0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = F.relu(self.conv5(c4) + self.redir2(c2), inplace=True)
        c6 = F.relu(self.conv6(c5) + self.redir1(x), inplace=True)
        return c6


class CostAggregation(nn.Module):
    """
    3D cost volume aggregation: initial convs + residual + hourglass + classifier.
    Input: [B, num_groups, D, H, W], output: [B, D, H, W] logits for disparity.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 32,
        hourglass_channels: int = 32,
    ) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            convbn_3d(in_channels, mid_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(mid_channels, mid_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(
            convbn_3d(mid_channels, mid_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(mid_channels, mid_channels, 3, 1, 1),
        )
        self.hourglass = Hourglass3D(hourglass_channels)
        self.classifier = nn.Sequential(
            convbn_3d(hourglass_channels, mid_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, 1, kernel_size=3, padding=1, stride=1, bias=False),
        )

    def forward(self, cost_volume: torch.Tensor) -> torch.Tensor:
        """
        cost_volume: [B, G, D, H, W]
        Returns: [B, D, H, W] (before softmax)
        """
        x = self.initial(cost_volume)
        x = self.residual(x) + x
        x = self.hourglass(x)
        x = self.classifier(x)
        x = torch.squeeze(x, dim=1)
        return x


# -----------------------------------------------------------------------------
# Learnable upsampling
# -----------------------------------------------------------------------------


class BasicBlock2D(nn.Module):
    """Residual block with convbn1 + Mish for refinement network. expansion=1."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        pad: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            convbn1(inplanes, planes, 3, stride, pad, dilation),
            Mish(),
        )
        self.conv2 = convbn1(planes, planes, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out = out + x
        return out


class SELayer(nn.Module):
    """Squeeze-and-Excitation: global pool -> FC -> sigmoid -> scale."""

    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LearnableUpsample(nn.Module):
    """Bilinear upsample by 2x + 3x3 conv (no activation for linear fusion)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            BasicConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm_layer=nn.BatchNorm2d,
                act_layer=None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)
