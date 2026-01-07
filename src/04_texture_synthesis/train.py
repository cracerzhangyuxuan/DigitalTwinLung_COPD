#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练模块

Inpainting 网络训练逻辑，支持三种模型架构：
- unet: 基线方案
- partial_conv: 进阶方案
- patchgan: 高级方案
"""

import json
import math
from pathlib import Path
from typing import Dict, Optional, Union
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR
except ImportError:
    torch = None

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from .network import InpaintingUNet, PatchDiscriminator, PartialConvUNet, create_model
from .losses import InpaintingLoss
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Inpainting 网络训练器

    支持三种模型架构和多种学习率调度策略
    """

    def __init__(
        self,
        generator: 'nn.Module',
        discriminator: Optional['nn.Module'] = None,
        config: Optional[dict] = None,
        device: str = "cuda"
    ):
        if torch is None:
            raise ImportError("PyTorch 未安装")

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        # 模型
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device) if discriminator else None

        # 默认配置
        self.config = config or {}
        train_config = self.config.get('training', {})
        self.epochs = train_config.get('epochs', 100)

        # 优化器
        lr = train_config.get('learning_rate', 0.0002)
        betas = (train_config.get('beta1', 0.5), train_config.get('beta2', 0.999))

        self.g_optimizer = Adam(self.generator.parameters(), lr=lr, betas=betas)
        if self.discriminator:
            self.d_optimizer = Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        # 学习率调度器
        self.g_scheduler = self._create_scheduler(
            self.g_optimizer,
            train_config.get('lr_scheduler', 'step'),
            train_config.get('warmup_epochs', 5)
        )

        # 损失函数
        loss_weights = train_config.get('loss_weights', {})
        self.criterion = InpaintingLoss(
            reconstruction_weight=loss_weights.get('reconstruction', 1.0),
            perceptual_weight=loss_weights.get('perceptual', 0.1),
            adversarial_weight=loss_weights.get('adversarial', 0.01),
        )

        # TensorBoard
        self.writer = None
        if train_config.get('tensorboard', False) and HAS_TENSORBOARD:
            log_dir = Path(train_config.get('log_dir', 'logs/tensorboard'))
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard 日志: {log_dir}")

        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'ssim': [], 'psnr': []}

    def _create_scheduler(self, optimizer, scheduler_type: str, warmup_epochs: int):
        """创建学习率调度器"""
        if scheduler_type == 'step':
            return StepLR(optimizer, step_size=50, gamma=0.5)
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        elif scheduler_type == 'warmup_cosine':
            def warmup_cosine_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                progress = (epoch - warmup_epochs) / (self.epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
            return LambdaLR(optimizer, warmup_cosine_lambda)
        else:
            return StepLR(optimizer, step_size=50, gamma=0.5)
    
    def train_epoch(self, train_loader: 'DataLoader') -> Dict[str, float]:
        """训练一个 epoch"""
        self.generator.train()
        if self.discriminator:
            self.discriminator.train()
        
        epoch_losses = {'reconstruction': 0, 'perceptual': 0, 'total': 0}
        
        for batch_idx, batch in enumerate(train_loader):
            input_data = batch['input'].to(self.device)
            target = batch['target'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # 生成器前向传播
            pred = self.generator(input_data)
            
            # 计算损失
            if self.discriminator:
                disc_pred = self.discriminator(pred)
                g_losses = self.criterion.generator_loss(pred, target, mask, disc_pred)
                
                # 判别器更新
                self.d_optimizer.zero_grad()
                real_pred = self.discriminator(target)
                fake_pred = self.discriminator(pred.detach())
                d_losses = self.criterion.discriminator_loss(real_pred, fake_pred)
                d_losses['total'].backward()
                self.d_optimizer.step()
            else:
                g_losses = self.criterion.generator_loss(pred, target, mask)
            
            # 生成器更新
            self.g_optimizer.zero_grad()
            g_losses['total'].backward()
            self.g_optimizer.step()
            
            # 累计损失
            for key in epoch_losses:
                if key in g_losses:
                    epoch_losses[key] += g_losses[key].item()
        
        # 平均损失
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self, val_loader: 'DataLoader') -> Dict[str, float]:
        """验证"""
        self.generator.eval()
        
        val_losses = {'reconstruction': 0, 'total': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                input_data = batch['input'].to(self.device)
                target = batch['target'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                pred = self.generator(input_data)
                losses = self.criterion.generator_loss(pred, target, mask)
                
                for key in val_losses:
                    if key in losses:
                        val_losses[key] += losses[key].item()
        
        num_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def train(
        self,
        train_loader: 'DataLoader',
        val_loader: 'DataLoader',
        epochs: int = 100,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        save_frequency: int = 10
    ) -> Dict:
        """完整训练流程"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"开始训练: {epochs} epochs")

        for epoch in range(epochs):
            self.current_epoch = epoch + 1

            # 训练
            train_losses = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_losses['total'])

            # 验证
            val_losses = self.validate(val_loader)
            self.history['val_loss'].append(val_losses['total'])

            # 更新学习率
            current_lr = self.g_optimizer.param_groups[0]['lr']
            self.g_scheduler.step()

            # TensorBoard 日志
            if self.writer:
                self.writer.add_scalar('Loss/train', train_losses['total'], self.current_epoch)
                self.writer.add_scalar('Loss/val', val_losses['total'], self.current_epoch)
                self.writer.add_scalar('Loss/reconstruction', train_losses.get('reconstruction', 0), self.current_epoch)
                self.writer.add_scalar('Loss/perceptual', train_losses.get('perceptual', 0), self.current_epoch)
                self.writer.add_scalar('LearningRate', current_lr, self.current_epoch)

            # 日志
            logger.info(
                f"Epoch {self.current_epoch}/{epochs} | "
                f"Train: {train_losses['total']:.4f} | "
                f"Val: {val_losses['total']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # 保存最佳模型
            if val_losses['total'] < self.best_loss:
                self.best_loss = val_losses['total']
                self.save_checkpoint(checkpoint_dir / "best.pth")
                logger.info(f"  ✓ 保存最佳模型: loss = {self.best_loss:.4f}")

            # 定期保存
            if self.current_epoch % save_frequency == 0:
                self.save_checkpoint(checkpoint_dir / "latest.pth")

        # 关闭 TensorBoard
        if self.writer:
            self.writer.close()

        # 保存训练历史
        with open(checkpoint_dir / "training_log.json", 'w') as f:
            json.dump(self.history, f, indent=2)

        logger.info("训练完成!")
        return self.history
    
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state_dict': self.generator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
        }
        
        if self.discriminator:
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()
            checkpoint['d_optimizer_state_dict'] = self.d_optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
        
        if self.discriminator and 'discriminator_state_dict' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        logger.info(f"加载检查点: epoch {self.current_epoch}, best_loss {self.best_loss:.4f}")


def main(config: dict) -> None:
    """主函数"""
    if torch is None:
        logger.error("PyTorch 未安装")
        return

    from .dataset import create_dataloader

    # 获取路径配置
    paths = config.get('paths', {})
    mapped_dir = Path(paths.get('mapped', 'data/03_mapped'))

    # 收集已配准的 CT 和病灶 mask
    ct_files = []
    mask_files = []

    for patient_dir in sorted(mapped_dir.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name == 'visualizations':
            continue

        warped_ct = patient_dir / f"{patient_dir.name}_warped.nii.gz"
        warped_mask = patient_dir / f"{patient_dir.name}_warped_lesion.nii.gz"

        if warped_ct.exists() and warped_mask.exists():
            ct_files.append(warped_ct)
            mask_files.append(warped_mask)

    if len(ct_files) == 0:
        logger.error("未找到已配准的数据，请先运行 Phase 3A")
        return

    logger.info(f"找到 {len(ct_files)} 例已配准数据")

    # 划分训练/验证集
    train_ratio = 0.8
    n_train = max(1, int(len(ct_files) * train_ratio))

    from .dataset import LungPatchDataset
    from torch.utils.data import DataLoader

    train_config = config.get('training', {})
    patch_size = tuple(train_config.get('patch_size', [64, 64, 64]))
    batch_size = train_config.get('batch_size', 4)
    num_workers = train_config.get('num_workers', 0)  # Windows 下建议设为 0

    train_dataset = LungPatchDataset(
        ct_paths=ct_files[:n_train],
        mask_paths=mask_files[:n_train],
        patch_size=patch_size,
        augment=True
    )

    val_dataset = LungPatchDataset(
        ct_paths=ct_files[n_train:] if n_train < len(ct_files) else ct_files[-1:],
        mask_paths=mask_files[n_train:] if n_train < len(mask_files) else mask_files[-1:],
        patch_size=patch_size,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    logger.info(f"训练集: {len(train_dataset)} patches, 验证集: {len(val_dataset)} patches")

    # 创建模型（根据 model_type 选择）
    model_type = train_config.get('model_type', 'unet')
    logger.info(f"创建模型: {model_type}")

    if model_type == 'patchgan':
        generator, discriminator = create_model('patchgan')
    elif model_type == 'partial_conv':
        generator = create_model('partial_conv')
        discriminator = None
    else:  # 默认 unet
        generator = create_model('unet')
        discriminator = None

    # 创建训练器
    trainer = Trainer(generator, discriminator, config)

    # 训练
    checkpoint_dir = Path(paths.get('checkpoints', 'checkpoints'))
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_config.get('epochs', 100),
        checkpoint_dir=checkpoint_dir,
        save_frequency=train_config.get('save_frequency', 10),
    )


if __name__ == "__main__":
    import yaml
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)

