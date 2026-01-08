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

    组合重建损失、感知损失、对抗损失和 HU 约束损失
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        adversarial_weight: float = 0.01,
        hu_constraint_weight: float = 0.5,  # 新增：HU 约束损失权重
        mask_weight: float = 10.0,
        loss_type: str = "l1",
        enable_hu_constraint: bool = True  # 新增：是否启用 HU 约束
    ):
        super().__init__()

        self.reconstruction = ReconstructionLoss(loss_type, mask_weight)
        self.perceptual = PerceptualLoss()
        self.adversarial = AdversarialLoss("lsgan")
        self.hu_constraint = HUConstraintLoss()  # 新增：HU 约束损失

        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.hu_constraint_weight = hu_constraint_weight  # 新增
        self.enable_hu_constraint = enable_hu_constraint  # 新增

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

        # HU 约束损失（鼓励病灶区域生成低 HU 值）
        if self.enable_hu_constraint and self.hu_constraint_weight > 0:
            losses['hu_constraint'] = self.hu_constraint_weight * \
                self.hu_constraint(pred, mask)

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


class HUConstraintLoss(nn.Module):
    """
    HU 约束损失

    鼓励病灶区域生成低 HU 值（模拟肺气肿特征，HU < -950）

    原理：
    - 肺气肿区域的 HU 值通常 < -950
    - 当模型在病灶区域生成 HU > target_hu + margin 时，施加惩罚
    - 这迫使模型学习生成低密度区域，而非健康肺组织纹理
    """

    def __init__(
        self,
        target_hu: float = -950,
        margin: float = 50,
        hu_min: float = -1000,
        hu_max: float = 400
    ):
        """
        Args:
            target_hu: 目标 HU 值（肺气肿标准 -950）
            margin: 容差范围（允许 HU 在 target_hu + margin 以下）
            hu_min: 归一化时的最小 HU 值
            hu_max: 归一化时的最大 HU 值
        """
        super().__init__()
        self.target_hu = target_hu
        self.margin = margin
        self.hu_min = hu_min
        self.hu_max = hu_max

        # 将目标 HU 转换为归一化值
        self.target_normalized = (target_hu + margin - hu_min) / (hu_max - hu_min)

    def forward(
        self,
        pred: 'torch.Tensor',
        mask: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """
        计算 HU 约束损失

        Args:
            pred: 模型输出（归一化后的 CT，范围 [0, 1]）
            mask: 病灶 mask（B, 1, D, H, W）

        Returns:
            loss: HU 约束损失
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        # 提取病灶区域的预测值
        mask_bool = mask > 0
        lesion_pred = pred[mask_bool]

        # 惩罚高于目标值的预测
        # 当 lesion_pred > target_normalized 时，penalty > 0
        penalty = F.relu(lesion_pred - self.target_normalized)

        return penalty.mean()


