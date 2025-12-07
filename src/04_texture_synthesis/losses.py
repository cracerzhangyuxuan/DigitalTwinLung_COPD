#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
损失函数模块

包含:
- 重建损失 (L1, L2)
- 感知损失 (Perceptual Loss)
- 对抗损失 (Adversarial Loss)
"""

from typing import Optional, Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None


class ReconstructionLoss(nn.Module):
    """
    重建损失
    
    支持 L1 和 L2 损失，以及 mask 加权
    """
    
    def __init__(
        self,
        loss_type: str = "l1",
        mask_weight: float = 10.0
    ):
        """
        Args:
            loss_type: 损失类型 ("l1" 或 "l2")
            mask_weight: mask 区域的权重倍数
        """
        super().__init__()
        self.loss_type = loss_type
        self.mask_weight = mask_weight
    
    def forward(
        self,
        pred: 'torch.Tensor',
        target: 'torch.Tensor',
        mask: Optional['torch.Tensor'] = None
    ) -> 'torch.Tensor':
        """
        Args:
            pred: 预测输出 (B, C, D, H, W)
            target: 目标 (B, C, D, H, W)
            mask: 病灶 mask (B, 1, D, H, W)，可选
        """
        if self.loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction='none')
        else:
            loss = F.mse_loss(pred, target, reduction='none')
        
        if mask is not None:
            # mask 区域给予更高权重
            weight = 1.0 + (self.mask_weight - 1.0) * mask
            loss = loss * weight
        
        return loss.mean()


class PerceptualLoss(nn.Module):
    """
    感知损失 (简化版)
    
    使用多尺度特征差异作为感知损失
    """
    
    def __init__(self, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
    
    def forward(
        self,
        pred: 'torch.Tensor',
        target: 'torch.Tensor'
    ) -> 'torch.Tensor':
        loss = 0.0
        
        for scale in range(self.num_scales):
            if scale > 0:
                pred = F.avg_pool3d(pred, 2)
                target = F.avg_pool3d(target, 2)
            
            loss = loss + F.l1_loss(pred, target)
        
        return loss / self.num_scales


class AdversarialLoss(nn.Module):
    """
    对抗损失
    
    支持原始 GAN 和 LSGAN
    """
    
    def __init__(self, loss_type: str = "lsgan"):
        """
        Args:
            loss_type: "gan" (原始) 或 "lsgan" (最小二乘)
        """
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        pred: 'torch.Tensor',
        is_real: bool
    ) -> 'torch.Tensor':
        """
        Args:
            pred: 判别器输出
            is_real: 是否为真实样本
        """
        if self.loss_type == "lsgan":
            target = 1.0 if is_real else 0.0
            return F.mse_loss(pred, torch.full_like(pred, target))
        else:
            if is_real:
                return F.binary_cross_entropy_with_logits(
                    pred, torch.ones_like(pred)
                )
            else:
                return F.binary_cross_entropy_with_logits(
                    pred, torch.zeros_like(pred)
                )


class InpaintingLoss(nn.Module):
    """
    完整的 Inpainting 损失函数
    
    组合重建损失、感知损失和对抗损失
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        adversarial_weight: float = 0.01,
        mask_weight: float = 10.0,
        loss_type: str = "l1"
    ):
        super().__init__()
        
        self.reconstruction = ReconstructionLoss(loss_type, mask_weight)
        self.perceptual = PerceptualLoss()
        self.adversarial = AdversarialLoss("lsgan")
        
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
    
    def generator_loss(
        self,
        pred: 'torch.Tensor',
        target: 'torch.Tensor',
        mask: 'torch.Tensor',
        disc_pred: Optional['torch.Tensor'] = None
    ) -> Dict[str, 'torch.Tensor']:
        """
        生成器损失
        """
        losses = {}
        
        # 重建损失
        losses['reconstruction'] = self.reconstruction_weight * \
            self.reconstruction(pred, target, mask)
        
        # 感知损失
        losses['perceptual'] = self.perceptual_weight * \
            self.perceptual(pred, target)
        
        # 对抗损失 (如果使用 GAN)
        if disc_pred is not None:
            losses['adversarial'] = self.adversarial_weight * \
                self.adversarial(disc_pred, is_real=True)
        
        # 总损失
        losses['total'] = sum(losses.values())
        
        return losses
    
    def discriminator_loss(
        self,
        real_pred: 'torch.Tensor',
        fake_pred: 'torch.Tensor'
    ) -> Dict[str, 'torch.Tensor']:
        """
        判别器损失
        """
        losses = {}
        
        losses['real'] = self.adversarial(real_pred, is_real=True)
        losses['fake'] = self.adversarial(fake_pred, is_real=False)
        losses['total'] = (losses['real'] + losses['fake']) / 2
        
        return losses


class SSIMLoss(nn.Module):
    """
    SSIM 损失 (1 - SSIM)
    
    简化的 3D SSIM 实现
    """
    
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
    
    def forward(
        self,
        pred: 'torch.Tensor',
        target: 'torch.Tensor'
    ) -> 'torch.Tensor':
        # 简化实现：使用均值和方差
        mu_pred = F.avg_pool3d(pred, self.window_size, stride=1, padding=self.window_size // 2)
        mu_target = F.avg_pool3d(target, self.window_size, stride=1, padding=self.window_size // 2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.avg_pool3d(
            pred ** 2, self.window_size, stride=1, padding=self.window_size // 2
        ) - mu_pred_sq
        sigma_target_sq = F.avg_pool3d(
            target ** 2, self.window_size, stride=1, padding=self.window_size // 2
        ) - mu_target_sq
        sigma_pred_target = F.avg_pool3d(
            pred * target, self.window_size, stride=1, padding=self.window_size // 2
        ) - mu_pred_target
        
        ssim = ((2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2)) / \
               ((mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2))
        
        if self.size_average:
            return 1 - ssim.mean()
        else:
            return 1 - ssim

