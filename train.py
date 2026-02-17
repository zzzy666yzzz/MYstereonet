# -*- coding: utf-8 -*-
"""
StereoNet 训练脚本 (完美续训版)
集成特性：
1. 稳健的断点续训 (Resume)：保存/加载 模型、优化器、Scheduler、Epoch、Best_D1。
2. TensorBoard 可视化：完全对齐你提供的结构。
3. 鲁棒性修复：保留了之前修复的 OOM、维度对齐、Pad 问题。
"""

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler

# 优先使用新版 autocast，避免 FutureWarning
if hasattr(torch.amp, "autocast"):
    def autocast(*args, **kwargs):
        return torch.amp.autocast("cuda", *args, **kwargs)
else:
    from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 使用重构后的 StereoNet 与自定义数据集
from dataset import StereoDataset
from new_model.net import StereoNet


def setup_logging(save_dir: str) -> None:
    """配置 logging：同时输出到文件与控制台。"""
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def save_checkpoint(state, save_dir, filename="checkpoint.pth.tar", is_best=False):
    """保存检查点，包含模型权重、优化器状态等。"""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(save_dir, "best_model.pth")
        shutil.copyfile(filepath, best_path)
        logging.info(f"保存最佳模型 -> {best_path}")


class AverageMeter(object):
    """计算滑动平均值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def train_one_epoch(
        model: nn.Module,
        device: torch.device,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        scaler: GradScaler,
        epoch: int,
        writer: SummaryWriter,
        max_disp: int = 192,
        accumulation_steps: int = 4
) -> float:
    """训练一个 Epoch"""
    model.train()
    losses = AverageMeter()
    loop = tqdm(dataloader, desc=f"Epoch [{epoch}] Train")

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (imgL, imgR, disp_true) in enumerate(loop):
        imgL = imgL.to(device, non_blocking=True)
        imgR = imgR.to(device, non_blocking=True)
        disp_true = disp_true.to(device, non_blocking=True)

        # === 维度强制对齐 (鲁棒性修复) ===
        B, _, H, W = imgL.shape
        # 强制重塑 GT 为 [B, H, W]
        try:
            disp_true = disp_true.reshape(B, H, W)
        except Exception as e:
            logging.error(f"GT Reshape Error! Target: ({B},{H},{W}), Actual: {disp_true.shape}")
            raise e

        # Mask: 忽略无效值 (disp=0) 和过大值
        mask = (disp_true > 0) & (disp_true < max_disp)
        mask = mask.bool()

        if mask.sum() == 0:
            continue

        with autocast():
            # 前向传播
            output1, output2, output3 = model(imgL, imgR)

            # 强制重塑 Output 为 [B, H, W]
            output1 = output1.reshape(B, H, W)
            output2 = output2.reshape(B, H, W)
            output3 = output3.reshape(B, H, W)

            # 计算多尺度 Loss
            loss1 = F.smooth_l1_loss(output1[mask], disp_true[mask], reduction="mean")
            loss2 = F.smooth_l1_loss(output2[mask], disp_true[mask], reduction="mean")
            loss3 = F.smooth_l1_loss(output3[mask], disp_true[mask], reduction="mean")

            loss = 0.5 * loss1 + 0.7 * loss2 + 1.0 * loss3
            loss = loss / accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积更新
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # 记录日志
        current_loss = loss.item() * accumulation_steps
        losses.update(current_loss, imgL.size(0))
        loop.set_postfix(loss=f"{losses.avg:.4f}")

        # TensorBoard Step Loss
        global_step = (epoch - 1) * len(dataloader) + batch_idx
        writer.add_scalar("Train/Batch_Loss", current_loss, global_step)

    writer.add_scalar("Train/Epoch_Loss", losses.avg, epoch)
    logging.info(f"Epoch [{epoch}] Training Loss: {losses.avg:.4f}")
    return losses.avg


@torch.no_grad()
def validate_one_epoch(
        model: nn.Module,
        device: torch.device,
        dataloader: DataLoader,
        epoch: int,
        writer: SummaryWriter,
        max_disp: int = 192,
) -> tuple:
    """验证一个 Epoch"""
    model.eval()
    epe_meter = AverageMeter()
    d1_meter = AverageMeter()
    loop = tqdm(dataloader, desc=f"Epoch [{epoch}] Val")
    vis_done = False

    for batch_idx, (imgL, imgR, disp_true) in enumerate(loop):
        imgL = imgL.to(device, non_blocking=True)
        imgR = imgR.to(device, non_blocking=True)
        disp_true = disp_true.to(device, non_blocking=True)

        # 维度对齐
        B, _, H, W = imgL.shape
        disp_true = disp_true.reshape(B, H, W)

        with autocast():
            _, _, pred_disp = model(imgL, imgR)

        pred_disp = pred_disp.reshape(B, H, W)

        mask = (disp_true > 0) & (disp_true < max_disp)
        mask = mask.bool()

        if mask.sum() == 0:
            continue

        # 计算指标
        abs_diff = torch.abs(pred_disp[mask] - disp_true[mask])
        epe = abs_diff.mean().item()

        rel_error = abs_diff / (disp_true[mask].clamp(min=1e-3))
        bad_pixels = ((abs_diff > 3.0) & (rel_error > 0.05)).float()
        d1_error = bad_pixels.mean().item() * 100.0

        epe_meter.update(epe, mask.sum().item())
        d1_meter.update(d1_error, mask.sum().item())
        loop.set_postfix(EPE=f"{epe_meter.avg:.3f}", D1=f"{d1_meter.avg:.2f}%")

        # TensorBoard 可视化 (取 Batch 中的第一张图)
        if not vis_done and batch_idx == 0:
            vis_done = True
            # 可视化前需 squeeze(0) 去掉 Batch 维度
            left_vis = _denorm_for_vis(imgL[0:1]).squeeze(0)
            gt_vis = disp_true[0].unsqueeze(0).clamp(0, max_disp) / float(max_disp)
            pred_vis = pred_disp[0].unsqueeze(0).clamp(0, max_disp) / float(max_disp)

            writer.add_image("Val/Left", left_vis, epoch)
            writer.add_image("Val/GT", gt_vis, epoch)
            writer.add_image("Val/Pred", pred_vis, epoch)

    writer.add_scalar("Val/EPE", epe_meter.avg, epoch)
    writer.add_scalar("Val/D1_Error", d1_meter.avg, epoch)
    logging.info(f"Epoch [{epoch}] Val EPE: {epe_meter.avg:.4f}, D1: {d1_meter.avg:.2f}%")
    return epe_meter.avg, d1_meter.avg


def _denorm_for_vis(tensor: torch.Tensor) -> torch.Tensor:
    """反归一化，用于可视化"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description="StereoNet Training")
    parser.add_argument("--datapath", type=str, default="./dataset/", help="Data root")
    parser.add_argument("--savemodel", type=str, default="./checkpoints/", help="Save path")
    parser.add_argument("--maxdisp", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", type=str, default="./checkpoints/best_model.pth",
                        help="Path to checkpoint (e.g., ./checkpoints/checkpoint.pth.tar)")
    args = parser.parse_args()

    # 1. 初始化
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 日志与 TensorBoard 配置
    setup_logging(args.savemodel)
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(args.savemodel, "runs", current_time)
    writer = SummaryWriter(log_dir=log_dir)

    # 2. 数据加载 (Batch Size 1 + Accum 4)
    train_dataset = StereoDataset(
        args.datapath, train=True, train_ratio=0.9, seed=args.seed,
        crop_size=(256, 512), val_target_size=(704, 1280)
    )
    val_dataset = StereoDataset(
        args.datapath, train=False, train_ratio=0.9, seed=args.seed,
        val_target_size=(704, 1280)
    )

    TrainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, drop_last=True, pin_memory=True)
    ValLoader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                           num_workers=2, drop_last=False)

    logging.info(f"Dataset Loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 3. 模型与优化器
    model = StereoNet(max_disp=args.maxdisp,
                      pretrained_backbone=True).to(device)

    # === 修改后的代码 (差分学习率) ===
    # 1. 找出 Backbone (MobileNet) 的参数地址
    backbone_ids = list(map(id, model.backbone.parameters()))

    # 2. 找出其余部分 (Head, Cost Volume, Refinement) 的参数
    base_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())

    # 3. 分组设置学习率
    # === 微调阶段：学习率降 10 倍 ===
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': 1e-4},  # Head: 从 1e-3 改为 1e-4
        {'params': model.backbone.parameters(), 'lr': 1e-5}  # Backbone: 从 1e-4 改为 1e-5
    ], weight_decay=1e-4)

    logging.info(f"已启用差分学习率: Backbone LR={args.lr * 0.1:.6f}, Head LR={args.lr:.6f}")

    steps_per_epoch = len(TrainLoader) // args.accum_steps
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
        pct_start=0.1, div_factor=25.0
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 1
    best_d1 = 100.0

    # 4. 断点续训逻辑 / 微调逻辑
    if args.resume and os.path.isfile(args.resume):
        logging.info(f"Loading checkpoint '{args.resume}'")
        ckpt = torch.load(args.resume, map_location=device)

        # 1. 必须加载：模型权重
        # strict=True 保证权重完全匹配，更安全
        model.load_state_dict(ckpt["state_dict"], strict=True)

        # 2. 微调时【不要】加载以下内容：
        # 我们希望使用新的小学习率 (1e-4/1e-5) 和新的优化器，而不是沿用旧的
        # if "optimizer" in ckpt:
        #     optimizer.load_state_dict(ckpt["optimizer"])
        # if "scheduler" in ckpt:
        #     scheduler.load_state_dict(ckpt["scheduler"])
        # if "epoch" in ckpt:
        #     start_epoch = ckpt["epoch"] + 1

        # 3. 如果是微调，这里强制重置 start_epoch 为 1
        start_epoch = 1

        # 4. (可选) 加载之前的最佳指标，方便对比，或者直接重置为 100
        # if "best_d1" in ckpt:
        #     best_d1 = ckpt["best_d1"]
        best_d1 = 100.0  # 微调阶段我们想看它能不能再次创新低，重置比较好观察

        logging.info(f"已加载模型权重进行微调。Start Epoch 重置为 {start_epoch}")

    # 5. 训练循环
    for epoch in range(start_epoch, args.epochs + 1):
        logging.info(f"--- Epoch {epoch} / {args.epochs} ---")

        train_loss = train_one_epoch(
            model, device, TrainLoader, optimizer, scheduler, scaler, epoch, writer, args.maxdisp
        )

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Train/LR", current_lr, epoch)
        logging.info(f"Learning Rate: {current_lr:.6f}")

        val_epe, val_d1 = validate_one_epoch(
            model, device, ValLoader, epoch, writer, args.maxdisp
        )

        is_best = val_d1 < best_d1
        best_d1 = min(val_d1, best_d1)

        # 保存完整状态，包括 optimizer 和 scheduler
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_d1": best_d1,
            },
            args.savemodel,
            is_best=is_best,
        )

        logging.info(f"Current Best D1: {best_d1:.2f}%")

    writer.close()
    logging.info("Training Finished.")


if __name__ == "__main__":
    main()