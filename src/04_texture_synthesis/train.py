#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练模块

Inpainting 网络训练逻辑
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR
except ImportError:
    torch = None

from .network import InpaintingUNet, PatchDiscriminator
from .losses import InpaintingLoss
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Inpainting 网络训练器
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
        
        # 优化器
        self.g_optimizer = Adam(
            self.generator.parameters(),
            lr=train_config.get('learning_rate', 0.0002),
            betas=(train_config.get('beta1', 0.5), train_config.get('beta2', 0.999))
        )
        
        if self.discriminator:
            self.d_optimizer = Adam(
                self.discriminator.parameters(),
                lr=train_config.get('learning_rate', 0.0002),
                betas=(train_config.get('beta1', 0.5), train_config.get('beta2', 0.999))
            )
        
        # 学习率调度器
        self.g_scheduler = StepLR(self.g_optimizer, step_size=50, gamma=0.5)
        
        # 损失函数
        loss_weights = train_config.get('loss_weights', {})
        self.criterion = InpaintingLoss(
            reconstruction_weight=loss_weights.get('reconstruction', 1.0),
            perceptual_weight=loss_weights.get('perceptual', 0.1),
            adversarial_weight=loss_weights.get('adversarial', 0.01),
        )
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}
    
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
            self.g_scheduler.step()
            
            # 日志
            logger.info(
                f"Epoch {self.current_epoch}/{epochs} | "
                f"Train Loss: {train_losses['total']:.4f} | "
                f"Val Loss: {val_losses['total']:.4f}"
            )
            
            # 保存最佳模型
            if val_losses['total'] < self.best_loss:
                self.best_loss = val_losses['total']
                self.save_checkpoint(checkpoint_dir / "best.pth")
                logger.info(f"保存最佳模型: loss = {self.best_loss:.4f}")
            
            # 定期保存
            if self.current_epoch % save_frequency == 0:
                self.save_checkpoint(checkpoint_dir / "latest.pth")
        
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
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloader(
        ct_dir=Path(config['paths']['cleaned_data']) / 'copd_clean',
        mask_dir=Path(config['paths']['cleaned_data']) / 'copd_clean',
        batch_size=config['training']['batch_size'],
        patch_size=tuple(config['training']['patch_size']),
    )
    
    # 创建模型
    generator = InpaintingUNet()
    discriminator = PatchDiscriminator()
    
    # 创建训练器
    trainer = Trainer(generator, discriminator, config)
    
    # 训练
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        checkpoint_dir=config['paths']['checkpoints'],
        save_frequency=config['training']['save_frequency'],
    )


if __name__ == "__main__":
    import yaml
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)

