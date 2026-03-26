#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAE 预训练脚本

实现 Masked Autoencoder (MAE) 自监督预训练，为 MAE-PatchGAN 模型提供
encoder 初始化权重。

训练流程：
1. 随机 mask 75% 的体素块（3D patch 级别遮挡）
2. 使用 InpaintingUNet 重建被遮挡区域
3. MSE 重建损失优化
4. 保存 encoder 权重供下游微调使用

参考文献：
- He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022

使用方式：
    python -m src.04_texture_synthesis.mae_pretrain --config config.yaml
"""

import json
import math
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
except ImportError:
    torch = None

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from .network import InpaintingUNet, create_model
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MAEMaskGenerator:
    """
    MAE 3D 块级别随机遮罩生成器

    将 3D 体积划分为小块（cube），随机遮挡指定比例的块。
    与标准 MAE 的 2D patch 遮挡类似，扩展到 3D。
    """

    def __init__(
        self,
        patch_size: int = 64,
        cube_size: int = 8,
        mask_ratio: float = 0.75
    ):
        """
        Args:
            patch_size: 输入 patch 的尺寸（假设各维度相同）
            cube_size: 遮挡块的尺寸
            mask_ratio: 遮挡比例（默认 75%，MAE 标准配置）
        """
        self.patch_size = patch_size
        self.cube_size = cube_size
        self.mask_ratio = mask_ratio
        self.num_cubes_per_dim = patch_size // cube_size
        self.total_cubes = self.num_cubes_per_dim ** 3

    def generate(self) -> np.ndarray:
        """
        生成一个 3D 随机遮挡 mask

        Returns:
            mask: (patch_size, patch_size, patch_size) 的 bool 数组
                  True = 被遮挡（待重建），False = 可见
        """
        # 随机选择要遮挡的块
        num_mask = int(self.total_cubes * self.mask_ratio)
        indices = np.random.permutation(self.total_cubes)
        mask_indices = indices[:num_mask]

        # 构建 3D mask
        mask = np.zeros(
            (self.patch_size, self.patch_size, self.patch_size),
            dtype=np.float32
        )

        for idx in mask_indices:
            # 将 1D 索引转换为 3D 块坐标
            z = (idx // (self.num_cubes_per_dim ** 2)) * self.cube_size
            y = ((idx % (self.num_cubes_per_dim ** 2)) // self.num_cubes_per_dim) * self.cube_size
            x = (idx % self.num_cubes_per_dim) * self.cube_size
            mask[z:z+self.cube_size, y:y+self.cube_size, x:x+self.cube_size] = 1.0

        return mask




class MAEPretrainer:
    """
    MAE 自监督预训练器

    使用 InpaintingUNet 作为 encoder-decoder，训练目标是重建被遮挡的体素块。
    训练完成后导出 encoder 权重，用于初始化 MAE-PatchGAN 的 Generator。
    """

    def __init__(
        self,
        model: 'nn.Module',
        config: Optional[dict] = None,
        device: str = "cuda"
    ):
        """
        Args:
            model: InpaintingUNet 模型
            config: 配置字典
            device: 计算设备
        """
        if torch is None:
            raise ImportError("PyTorch 未安装")

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"[MAEPretrainer] 使用设备: {self.device}")

        self.model = model.to(self.device)

        # 配置
        self.config = config or {}
        train_config = self.config.get('training', {})
        mae_config = train_config.get('mae', {})
        self.epochs = mae_config.get('pretrain_epochs', 300)

        # MAE 遮罩生成器
        patch_size = train_config.get('patch_size', [64, 64, 64])[0]
        self.mask_generator = MAEMaskGenerator(
            patch_size=patch_size,
            cube_size=mae_config.get('cube_size', 8),
            mask_ratio=mae_config.get('mask_ratio', 0.75)
        )

        # 优化器
        lr = mae_config.get('learning_rate', 1.5e-4)
        self.optimizer = Adam(
            self.model.parameters(), lr=lr,
            betas=(0.9, 0.999), weight_decay=0.05
        )

        # 学习率调度器
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )

        # TensorBoard
        self.writer = None
        if train_config.get('tensorboard', False) and HAS_TENSORBOARD:
            log_dir = Path(train_config.get('log_dir', 'logs/tensorboard_mae'))
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}

        logger.info(
            f"[MAEPretrainer] mask_ratio={self.mask_generator.mask_ratio}, "
            f"cube_size={self.mask_generator.cube_size}, epochs={self.epochs}"
        )

    def train_epoch(self, train_loader: 'DataLoader') -> Dict[str, float]:
        """
        预训练一个 epoch

        MAE 训练步骤：
        1. 从数据集获取 CT patch（target）
        2. 生成随机 75% 块级遮罩
        3. 用遮罩创建输入（被遮挡区域填 0）
        4. 模型预测完整 patch
        5. 仅在被遮挡区域计算 MSE 损失
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            target = batch['target'].to(self.device)
            batch_size = target.shape[0]

            # 为每个样本生成独立的 MAE 遮罩
            masks = []
            for _ in range(batch_size):
                mask = self.mask_generator.generate()
                masks.append(mask)
            mask_tensor = torch.from_numpy(
                np.stack(masks)[:, np.newaxis]
            ).float().to(self.device)

            # 创建输入：被遮挡区域填 0
            input_data = target.clone()
            input_data[mask_tensor > 0] = 0.0

            # 模型预测
            pred = self.model(input_data)

            # 仅在被遮挡区域计算 MSE 损失
            mask_bool = mask_tensor > 0
            if mask_bool.sum() > 0:
                loss = F.mse_loss(pred[mask_bool], target[mask_bool])
            else:
                loss = F.mse_loss(pred, target)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        return {'total': avg_loss, 'mse_recon': avg_loss}

    def validate(self, val_loader: 'DataLoader') -> Dict[str, float]:
        """验证（在验证集上计算 MAE 重建 MSE）"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                target = batch['target'].to(self.device)
                batch_size = target.shape[0]

                masks = []
                for _ in range(batch_size):
                    masks.append(self.mask_generator.generate())
                mask_tensor = torch.from_numpy(
                    np.stack(masks)[:, np.newaxis]
                ).float().to(self.device)

                input_data = target.clone()
                input_data[mask_tensor > 0] = 0.0

                pred = self.model(input_data)
                mask_bool = mask_tensor > 0
                if mask_bool.sum() > 0:
                    loss = F.mse_loss(pred[mask_bool], target[mask_bool])
                else:
                    loss = F.mse_loss(pred, target)
                total_loss += loss.item()

        avg_loss = total_loss / max(len(val_loader), 1)
        return {'total': avg_loss}

    def train(
        self,
        train_loader: 'DataLoader',
        val_loader: 'DataLoader',
        epochs: int = 300,
        checkpoint_dir: Union[str, Path] = "checkpoints/mae_pretrain",
        save_frequency: int = 50
    ) -> Dict:
        """
        完整 MAE 预训练流程

        训练完成后自动导出 encoder 权重到 checkpoint_dir/encoder_weights.pth
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[MAEPretrainer] 开始预训练: {epochs} epochs, "
            f"mask_ratio={self.mask_generator.mask_ratio}"
        )

        for epoch in range(epochs):
            self.current_epoch = epoch + 1

            train_losses = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_losses['total'])

            val_losses = self.validate(val_loader)
            self.history['val_loss'].append(val_losses['total'])

            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_scheduler.step()

            if self.writer:
                self.writer.add_scalar('MAE/train_loss', train_losses['total'], self.current_epoch)
                self.writer.add_scalar('MAE/val_loss', val_losses['total'], self.current_epoch)
                self.writer.add_scalar('MAE/lr', current_lr, self.current_epoch)

            logger.info(
                f"[MAE] Epoch {self.current_epoch}/{epochs} | "
                f"Train MSE: {train_losses['total']:.6f} | "
                f"Val MSE: {val_losses['total']:.6f} | "
                f"LR: {current_lr:.6f}"
            )

            if val_losses['total'] < self.best_loss:
                self.best_loss = val_losses['total']
                self._save_checkpoint(checkpoint_dir / "best.pth")
                logger.info(f"  ✓ 保存最佳预训练模型: MSE = {self.best_loss:.6f}")

            if self.current_epoch % save_frequency == 0:
                self._save_checkpoint(checkpoint_dir / "latest.pth")

        if self.writer:
            self.writer.close()

        # 保存训练历史
        with open(checkpoint_dir / "pretrain_log.json", 'w') as f:
            json.dump(self.history, f, indent=2)

        # 导出 encoder 权重
        encoder_path = checkpoint_dir / "encoder_weights.pth"
        self.export_encoder_weights(encoder_path)
        logger.info(f"[MAEPretrainer] 预训练完成! Encoder 权重: {encoder_path}")

        return self.history

    def _save_checkpoint(self, path: Union[str, Path]) -> None:
        """保存完整检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
            'model_type': 'mae_pretrain',
        }
        torch.save(checkpoint, path)

    def export_encoder_weights(self, path: Union[str, Path]) -> None:
        """
        导出 encoder 权重

        提取 InpaintingUNet 中 encoder 部分的权重：
        - input_conv: 输入卷积
        - down1, down2, down3: 下采样块
        - bottleneck: 瓶颈层

        下游微调时使用：
            model = InpaintingUNet()
            encoder_weights = torch.load(path)
            model.load_state_dict(encoder_weights, strict=False)
        """
        encoder_keys = []
        full_state = self.model.state_dict()

        for key in full_state:
            # 提取 encoder 相关的层：
            # InpaintingUNet 的 state_dict 顶层前缀:
            #   input_conv  — 输入卷积
            #   encoder     — 下采样块 (encoder.0/1/2, 即 DownBlock ×3)
            #   bottleneck  — 瓶颈层
            #   first_up    — 首个上采样 (属于 decoder, 不导出)
            #   decoder     — 解码器 (不导出)
            #   output_conv — 输出层 (不导出)
            if any(key.startswith(prefix) for prefix in [
                'input_conv', 'encoder', 'bottleneck'
            ]):
                encoder_keys.append(key)

        encoder_state = {k: full_state[k] for k in encoder_keys}
        torch.save(encoder_state, path)
        logger.info(
            f"[MAEPretrainer] 导出 encoder 权重: {len(encoder_keys)} 个参数张量 -> {path}"
        )