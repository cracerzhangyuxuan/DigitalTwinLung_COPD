#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDPM 扩散模型专用训练器

实现去噪扩散概率模型 (Denoising Diffusion Probabilistic Model) 的完整训练循环，
包含线性噪声调度器、前向加噪过程、噪声预测损失和 EMA 权重平均。

与标准 Trainer 的核心差异：
- 训练目标：预测噪声 ε_θ(x_t, t) 而非直接生成图像
- 损失函数：MSE(ε, ε_θ) 替代 L1+Perceptual+Adv+HU
- 前向传播：model(x_t, t) 需要额外的时间步参数

参考文献：
- Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
- Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021
"""

import json
import math
import copy
from pathlib import Path
from typing import Dict, Optional, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
except ImportError:
    torch = None

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from ..utils.logger import get_logger

logger = get_logger(__name__)


class NoiseScheduler:
    """
    线性噪声调度器

    实现 DDPM 的线性 β-schedule 及所有预计算量：
    - β_t: 噪声强度（线性从 β_start 到 β_end）
    - α_t = 1 - β_t
    - ᾱ_t = ∏_{i=1}^{t} α_i （累积乘积）
    - √ᾱ_t, √(1-ᾱ_t): 前向加噪公式的系数
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu"
    ):
        """
        Args:
            num_timesteps: 扩散步数 T
            beta_start: β 起始值
            beta_end: β 终止值
            device: 计算设备
        """
        self.num_timesteps = num_timesteps
        self.device = torch.device(device)

        # 线性 β-schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 前向加噪系数: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # 反向去噪系数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(
        self,
        x_0: 'torch.Tensor',
        noise: 'torch.Tensor',
        t: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """
        前向加噪: q(x_t | x_0) = N(x_t; √ᾱ_t * x_0, (1-ᾱ_t) * I)

        Args:
            x_0: 原始数据 (B, C, D, H, W)
            noise: 采样的高斯噪声 ε ~ N(0, I)
            t: 时间步 (B,) 整数张量

        Returns:
            x_t: 加噪后的数据
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def to(self, device: 'torch.device') -> 'NoiseScheduler':
        """迁移所有预计算张量到指定设备"""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self



class DiffusionTrainer:
    """
    DDPM 专用训练器

    与标准 Trainer 的关键差异：
    - 训练步骤：随机采样 t → 对 x_0 加噪得 x_t → 预测噪声 ε_θ(x_t, t) → MSE 损失
    - 无判别器（DDPM 不需要对抗训练）
    - 可选 EMA（指数移动平均权重，提升采样质量）
    """

    def __init__(
        self,
        model: 'nn.Module',
        config: Optional[dict] = None,
        device: str = "cuda"
    ):
        """
        Args:
            model: DiffusionUNet 模型
            config: 配置字典
            device: 计算设备
        """
        if torch is None:
            raise ImportError("PyTorch 未安装")

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"[DiffusionTrainer] 使用设备: {self.device}")

        # 模型
        self.model = model.to(self.device)

        # 配置
        self.config = config or {}
        train_config = self.config.get('training', {})
        self.epochs = train_config.get('epochs', 500)

        # 噪声调度器
        diffusion_config = train_config.get('diffusion', {})
        self.num_timesteps = diffusion_config.get('num_timesteps', 1000)
        self.scheduler = NoiseScheduler(
            num_timesteps=self.num_timesteps,
            beta_start=diffusion_config.get('beta_start', 1e-4),
            beta_end=diffusion_config.get('beta_end', 0.02),
        ).to(self.device)

        # 优化器
        lr = train_config.get('learning_rate', 0.0002)
        betas = (train_config.get('beta1', 0.9), train_config.get('beta2', 0.999))
        self.optimizer = Adam(self.model.parameters(), lr=lr, betas=betas)

        # 学习率调度器（余弦退火）
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )

        # EMA（指数移动平均）
        self.use_ema = diffusion_config.get('use_ema', True)
        self.ema_decay = diffusion_config.get('ema_decay', 0.999)
        self.ema_model = None
        if self.use_ema:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.eval()
            logger.info(f"[DiffusionTrainer] EMA 启用 (decay={self.ema_decay})")

        # TensorBoard
        self.writer = None
        if train_config.get('tensorboard', False) and HAS_TENSORBOARD:
            log_dir = Path(train_config.get('log_dir', 'logs/tensorboard_ddpm'))
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}

        logger.info(f"[DiffusionTrainer] T={self.num_timesteps}, epochs={self.epochs}")

    def _update_ema(self) -> None:
        """更新 EMA 模型权重"""
        if self.ema_model is None:
            return
        for ema_param, model_param in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_param.data.mul_(self.ema_decay).add_(
                model_param.data, alpha=1.0 - self.ema_decay
            )

    def train_epoch(self, train_loader: 'DataLoader') -> Dict[str, float]:
        """
        训练一个 epoch

        DDPM 训练步骤：
        1. 从数据集获取真实数据 x_0
        2. 随机采样时间步 t ~ Uniform(0, T)
        3. 采样噪声 ε ~ N(0, I)
        4. 计算加噪数据 x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        5. 用模型预测噪声 ε_θ = model(x_t, t)
        6. 计算损失 MSE(ε, ε_θ)
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # x_0: 真实的归一化 CT patch（target）
            x_0 = batch['target'].to(self.device)

            # 随机采样时间步
            t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=self.device)

            # 采样高斯噪声
            noise = torch.randn_like(x_0)

            # 前向加噪: q(x_t | x_0)
            x_t = self.scheduler.add_noise(x_0, noise, t)

            # 模型预测噪声
            noise_pred = self.model(x_t, t)

            # MSE 噪声预测损失
            loss = F.mse_loss(noise_pred, noise)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪（稳定训练）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 更新 EMA
            if self.use_ema:
                self._update_ema()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        return {'total': avg_loss, 'mse_noise': avg_loss}


    def validate(self, val_loader: 'DataLoader') -> Dict[str, float]:
        """验证（计算验证集上的噪声预测 MSE）"""
        model = self.ema_model if self.ema_model is not None else self.model
        model.eval()

        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x_0 = batch['target'].to(self.device)
                t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=self.device)
                noise = torch.randn_like(x_0)
                x_t = self.scheduler.add_noise(x_0, noise, t)

                noise_pred = model(x_t, t)
                loss = F.mse_loss(noise_pred, noise)
                total_loss += loss.item()

        avg_loss = total_loss / max(len(val_loader), 1)
        return {'total': avg_loss}

    def train(
        self,
        train_loader: 'DataLoader',
        val_loader: 'DataLoader',
        epochs: int = 500,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        save_frequency: int = 50
    ) -> Dict:
        """
        完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            checkpoint_dir: 检查点目录
            save_frequency: 保存频率（每 N 个 epoch 保存一次 latest）
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[DiffusionTrainer] 开始训练: {epochs} epochs, T={self.num_timesteps}")

        for epoch in range(epochs):
            self.current_epoch = epoch + 1

            # 训练
            train_losses = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_losses['total'])

            # 验证
            val_losses = self.validate(val_loader)
            self.history['val_loss'].append(val_losses['total'])

            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_scheduler.step()

            # TensorBoard
            if self.writer:
                self.writer.add_scalar('DDPM/train_loss', train_losses['total'], self.current_epoch)
                self.writer.add_scalar('DDPM/val_loss', val_losses['total'], self.current_epoch)
                self.writer.add_scalar('DDPM/lr', current_lr, self.current_epoch)

            # 日志
            logger.info(
                f"[DDPM] Epoch {self.current_epoch}/{epochs} | "
                f"Train MSE: {train_losses['total']:.6f} | "
                f"Val MSE: {val_losses['total']:.6f} | "
                f"LR: {current_lr:.6f}"
            )

            # 保存最佳模型
            if val_losses['total'] < self.best_loss:
                self.best_loss = val_losses['total']
                self.save_checkpoint(checkpoint_dir / "best.pth")
                logger.info(f"  ✓ 保存最佳模型: MSE = {self.best_loss:.6f}")

            # 定期保存
            if self.current_epoch % save_frequency == 0:
                self.save_checkpoint(checkpoint_dir / "latest.pth")

        # 关闭 TensorBoard
        if self.writer:
            self.writer.close()

        # 保存训练历史
        with open(checkpoint_dir / "training_log.json", 'w') as f:
            json.dump(self.history, f, indent=2)

        logger.info("[DiffusionTrainer] 训练完成!")
        return self.history

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """
        保存检查点

        兼容标准 Trainer 的检查点格式（使用 'generator_state_dict' 键），
        以便 load_model() 可以统一加载。
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
            'model_type': 'ddpm',
            'num_timesteps': self.num_timesteps,
        }

        if self.ema_model is not None:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['generator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})

        if self.ema_model is not None and 'ema_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])

        logger.info(
            f"[DiffusionTrainer] 加载检查点: epoch {self.current_epoch}, "
            f"best_loss {self.best_loss:.6f}"
        )