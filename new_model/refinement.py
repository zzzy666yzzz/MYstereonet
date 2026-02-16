"""
Refinement block for Stage 2 (1/2 resolution) and Stage 3 (full resolution).
Residual disparity correction with configurable in_channels and internal topology.
"""
from __future__ import annotations

from typing import List, Literal, Tuple

import torch
import torch.nn as nn

from .submodules import BasicBlock2D, SELayer, convbn1, Mish


StageKind = Literal["stage2", "stage3"]

# (out_ch, dilation) for the initial conv stack; first layer in_ch = in_channels
_STAGE_CONFIG: dict[StageKind, List[Tuple[int, int]]] = {
    "stage2": [
        (256, 1),
        (128, 1),
        (64, 2),
        (64, 4),
    ],
    "stage3": [
        (128, 1),
        (64, 1),
        (32, 2),
        (32, 4),
    ],
}

# (planes, blocks, dilation) for each BasicBlock2D layer
_STAGE_BLOCKS: dict[StageKind, List[Tuple[int, int, int]]] = {
    "stage2": [(64, 1, 8), (64, 1, 16), (32, 1, 1)],
    "stage3": [(32, 1, 8), (16, 1, 16), (16, 1, 1)],
}

# Final channel before 1x1 disp head (and SE)
_STAGE_FINAL_CH: dict[StageKind, int] = {
    "stage2": 32,
    "stage3": 16,
}

# 首层输出通道（与 _STAGE_CONFIG 首项一致，供扩展用）
_STAGE_FIRST_OUT: dict[StageKind, int] = {
    "stage2": 256,
    "stage3": 128,
}


class RefinementBlock(nn.Module):
    """
    2D CNN that takes concatenated features + current disparity and outputs
    a residual to add to the disparity. Configurable for Stage 2 or Stage 3
    via in_channels and stage argument.
    """

    def __init__(
        self,
        in_channels: int,
        stage: StageKind = "stage2",
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.stage = stage
        config = _STAGE_CONFIG[stage]
        block_config = _STAGE_BLOCKS[stage]
        first_out = _STAGE_FIRST_OUT[stage]
        final_ch = _STAGE_FINAL_CH[stage]

        self.conv_blocks = nn.ModuleList()
        in_ch = in_channels
        for out_ch, dilation in config:
            self.conv_blocks.append(
                nn.Sequential(
                    convbn1(in_ch, out_ch, 3, 1, 1, dilation),
                    Mish(),
                )
            )
            in_ch = out_ch

        self.res_blocks = nn.ModuleList()
        inplanes = in_ch
        for planes, num_blocks, dilation in block_config:
            downsample = None
            if inplanes != planes * BasicBlock2D.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes * BasicBlock2D.expansion, 1, 1, bias=False),
                    nn.BatchNorm2d(planes * BasicBlock2D.expansion),
                )
            layers: List[nn.Module] = [
                BasicBlock2D(inplanes, planes, 1, downsample, 1, dilation)
            ]
            inplanes = planes * BasicBlock2D.expansion
            for _ in range(1, num_blocks):
                layers.append(BasicBlock2D(inplanes, planes, 1, None, 1, dilation))
            self.res_blocks.append(nn.Sequential(*layers))

        self.se = SELayer(final_ch, reduction=reduction)
        self.disp_head = nn.Conv2d(final_ch, 1, kernel_size=3, padding=1, stride=1, bias=False)

    def forward(self, x: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W] concatenated features
            disp: [B, 1, H, W] current disparity
        Returns:
            [B, 1, H, W] refined disparity (disp + residual)
        """
        for blk in self.conv_blocks:
            x = blk(x)
        for blk in self.res_blocks:
            x = blk(x)
        x = self.se(x)
        residual = self.disp_head(x)
        return disp + residual
