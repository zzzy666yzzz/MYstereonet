"""
Main stereo matching network: Backbone -> Stage 1 (Coarse) -> Stage 2 (Refine 1/2) -> Stage 3 (Refine Full).
Coarse-to-fine cascade with GWC cost volume, 3D aggregation, and two 2D refinement stages.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Backbone
from .refinement import RefinementBlock
from .submodules import (
    CostAggregation,
    build_correlation_volume,
    build_gwc_volume,
    disparity_regression,
    warp_features,
)


def _compute_refinement_in_channels(
    feat_ch_half: int,
    feat_ch_quarter: int,
    disp_feature_channels: int,
    corr_maxdisp_half: int,
    corr_maxdisp_full: int,
    num_groups: int = 1,
) -> Tuple[int, int]:
    """Stage2 and Stage3 refinement block in_channels (for concatenated features)."""
    # Stage 2: (left - right_warp) + left + disp + disp_feature + correlation_volume
    corr_ch_half = num_groups * (2 * corr_maxdisp_half + 1)
    stage2_ch = feat_ch_half * 2 + 1 + disp_feature_channels + corr_ch_half

    # Stage 3: left + disp_full + disp_coarse_full + correlation_volume
    corr_ch_full = num_groups * (2 * corr_maxdisp_full + 1)
    stage3_ch = feat_ch_half + 1 + 1 + corr_ch_full

    return stage2_ch, stage3_ch


class StereoNet(nn.Module):
    """
    Coarse-to-fine stereo network:
    1. Backbone -> [feat_full, feat_half, feat_quarter]
    2. Stage 1: 1/4 GWC cost volume + 3D aggregation -> coarse disparity
    3. Stage 2: upsample to 1/2, warp, correlation, RefinementBlock -> disp_half
    4. Stage 3: upsample to full, warp, correlation, RefinementBlock -> disp_final
    """

    def __init__(
        self,
        max_disp: int,
        num_groups: int = 8,
        backbone: str = "MobileNetV2",
        pretrained_backbone: bool = False,
        coarse_max_disp_quarter: Optional[int] = None,
        corr_maxdisp_half: int = 32,
        corr_maxdisp_full: int = 16,
        disp_feature_channels: int = 32,
    ) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.num_groups = num_groups
        self.corr_maxdisp_half = corr_maxdisp_half
        self.corr_maxdisp_full = corr_maxdisp_full
        self.disp_quarter = coarse_max_disp_quarter if coarse_max_disp_quarter is not None else max_disp // 4

        self.backbone = Backbone(
            backbone=backbone,
            pretrained=pretrained_backbone,
        )
        chs = self.backbone.output_channels
        feat_ch_full = chs[0]
        feat_ch_half = chs[1]
        feat_ch_quarter = chs[2]

        self.cost_aggregation = CostAggregation(
            in_channels=num_groups,
            mid_channels=32,
            hourglass_channels=32,
        )

        self.disp_feature_extract = nn.Sequential(
            nn.Conv2d(1, disp_feature_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(disp_feature_channels),
            nn.ReLU(inplace=True),
        )

        stage2_in, stage3_in = _compute_refinement_in_channels(
            feat_ch_half=feat_ch_half,
            feat_ch_quarter=feat_ch_quarter,
            disp_feature_channels=disp_feature_channels,
            corr_maxdisp_half=corr_maxdisp_half,
            corr_maxdisp_full=corr_maxdisp_full,
            num_groups=1,
        )
        self.refinement_stage2 = RefinementBlock(in_channels=stage2_in, stage="stage2")
        self.refinement_stage3 = RefinementBlock(in_channels=stage3_in, stage="stage3")

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            left:  [B, 3, H, W]
            right: [B, 3, H, W]

        Returns:
            disp_coarse_full: [B, 1, H, W]  (1/4 coarse upsampled to full, for auxiliary loss)
            disp_half_upsampled: [B, 1, H, W] (Stage 2 refined, upsampled to full)
            disp_final: [B, 1, H, W] (Stage 3 refined, final output)
        """
        B, _, H, W = left.size()

        # -------- Backbone: multi-scale features --------
        feats_left = self.backbone(left)
        feats_right = self.backbone(right)
        # feats_*: [feat_full, feat_half, feat_quarter]
        # feat_full:   [B, C, H, W]
        # feat_half:   [B, C, H/2, W/2]
        # feat_quarter: [B, C, H/4, W/4]

        feat_left_quarter = feats_left[2]
        feat_right_quarter = feats_right[2]
        feat_left_half = feats_left[1]
        feat_right_half = feats_right[1]
        feat_left_full = feats_left[0]
        feat_right_full = feats_right[0]

        # -------- Stage 1: Coarse at 1/4 resolution --------
        cost_volume = build_gwc_volume(
            feat_left_quarter,
            feat_right_quarter,
            self.disp_quarter,
            self.num_groups,
        )
        # cost_volume: [B, num_groups, D, H/4, W/4]

        cost_logits = self.cost_aggregation(cost_volume)
        # cost_logits: [B, D, H/4, W/4]

        prob_quarter = F.softmax(cost_logits, dim=1)
        disp_coarse_quarter = disparity_regression(prob_quarter, self.disp_quarter)
        disp_coarse_quarter = disp_coarse_quarter.unsqueeze(1)
        # disp_coarse_quarter: [B, 1, H/4, W/4]

        disp_coarse_half = F.interpolate(
            disp_coarse_quarter * 2.0,
            size=[H // 2, W // 2],
            mode="bilinear",
        )
        disp_coarse_full = F.interpolate(
            disp_coarse_quarter * 4.0,
            size=[H, W],
            mode="bilinear",
        )
        # disp_coarse_half: [B, 1, H/2, W/2], disp_coarse_full: [B, 1, H, W]

        # -------- Stage 2: Refinement at 1/2 resolution --------
        right_warped_half = warp_features(feat_right_half, disp_coarse_half)
        # right_warped_half: [B, C, H/2, W/2]

        cost_vol_half = build_correlation_volume(
            feat_left_half,
            right_warped_half,
            self.corr_maxdisp_half,
            1,
        )
        cost_vol_half = cost_vol_half.squeeze(1)
        # cost_vol_half: [B, 2*24+1, H/2, W/2]

        disp_feat_half = self.disp_feature_extract(disp_coarse_half)
        # disp_feat_half: [B, 32, H/2, W/2]

        refinenet_input_half = torch.cat(
            [
                feat_left_half - right_warped_half,
                feat_left_half,
                disp_coarse_half,
                disp_feat_half,
                cost_vol_half,
            ],
            dim=1,
        )
        # refinenet_input_half: [B, 146, H/2, W/2]

        disp_refined_half = self.refinement_stage2(refinenet_input_half, disp_coarse_half)
        # disp_refined_half: [B, 1, H/2, W/2]

        disp_half_upsampled = F.interpolate(
            disp_refined_half * 2.0,
            size=[H, W],
            mode="bilinear",
        )
        # disp_half_upsampled: [B, 1, H, W]

        # -------- Stage 3: Refinement at full resolution --------
        # Use backbone's full-resolution features [B, C, H, W]
        right_warped_full = warp_features(feat_right_full, disp_half_upsampled)
        cost_vol_full = build_correlation_volume(
            feat_left_full,
            right_warped_full,
            self.corr_maxdisp_full,
            1,
        )
        cost_vol_full = cost_vol_full.squeeze(1)
        # cost_vol_full: [B, 25, H, W]

        refinenet_input_full = torch.cat(
            [
                feat_left_full,
                disp_half_upsampled,
                disp_coarse_full,
                cost_vol_full,
            ],
            dim=1,
        )
        # refinenet_input_full: [B, 59, H, W]

        disp_final = self.refinement_stage3(refinenet_input_full, disp_half_upsampled)
        # disp_final: [B, 1, H, W]

        return disp_coarse_full, disp_half_upsampled, disp_final
