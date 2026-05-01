#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CICI-FiLM v3 损失函数：多层次分布约束
==========================================

核心改进：
  1. Wasserstein-1 距离：约束整体分布形状，防止 KL 散度恶化
  2. Soft Histogram Loss：可导的直方图匹配
  3. HU Statistics Loss：归一化空间计算，避免量级失衡
  4. 多 Loss 平衡：所有 loss 在同一量级（~0.01-0.1）

物理含义：
  - Wasserstein：测量"搬运"一个分布到另一个分布的最小代价
  - 比 KL 散度更鲁棒，不要求分布有重叠支撑
  - 适合 COPD 病灶的多模态 HU 分布
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionLossesV3(nn.Module):
    """
    v3 条件损失：Wasserstein + Soft Histogram + HU Stats
    
    设计原则：
      1. 所有 loss 在归一化空间 [0,1] 计算
      2. 权重平衡：reconstruction ~0.01, wasserstein ~0.01, hu_stats ~0.01
      3. 可导性：所有 loss 支持梯度反传
    """
    
    def __init__(
        self,
        lambda_wasserstein=1.0,
        lambda_hu_stats=0.5,
        lambda_soft_histogram=0.3,
        lambda_ei=0.0,  # EI loss 由 Trainer 处理
        hu_min=-1000.0,
        hu_max=400.0,
    ):
        super().__init__()
        self.lambda_wasserstein = lambda_wasserstein
        self.lambda_hu_stats = lambda_hu_stats
        self.lambda_soft_histogram = lambda_soft_histogram
        self.lambda_ei = lambda_ei
        self.hu_min = hu_min
        self.hu_max = hu_max
    
    def compute_wasserstein_1d(self, pred, target, mask):
        """
        Wasserstein-1 距离（1D，基于排序）
        
        数学形式：
            W1(P, Q) = ∫|F_P^{-1}(u) - F_Q^{-1}(u)| du
        
        其中 F^{-1} 是累积分布函数的逆（分位数函数）
        
        实现：
            1. 对 pred 和 target 在 mask 区域排序
            2. 计算排序后的 L1 距离
        
        优点：
            - 可导
            - 对分布形状敏感
            - 不要求分布重叠
        
        Args:
            pred: (B, 1, D, H, W) 预测，归一化空间 [0,1]
            target: (B, 1, D, H, W) 目标，归一化空间 [0,1]
            mask: (B, 1, D, H, W) 病灶 mask
        
        Returns:
            scalar: Wasserstein-1 距离
        """
        B = pred.size(0)
        total_loss = torch.tensor(0.0, device=pred.device)
        count = 0
        
        for i in range(B):
            m = mask[i, 0] > 0
            if m.sum() < 10:
                continue
            
            p = pred[i, 0][m]  # (N,)
            t = target[i, 0][m]  # (N,)
            
            # 排序
            p_sorted, _ = torch.sort(p)
            t_sorted, _ = torch.sort(t)
            
            # Wasserstein-1 = L1 距离（排序后）
            w1 = torch.mean(torch.abs(p_sorted - t_sorted))
            total_loss = total_loss + w1
            count += 1
        
        return total_loss / max(count, 1)
    
    def compute_hu_stats_loss(self, pred, target, mask):
        """
        HU Statistics Loss（归一化空间）
        
        改进：
            - v2: 在 HU 空间计算，loss ~10-20（量级失衡）
            - v3: 在归一化空间计算，loss ~0.01-0.1（与 reconstruction 同量级）
        
        物理含义不变：
            - 归一化空间的 mean/std 与 HU 空间的 mean/std 仅差一个固定缩放
            - 优化归一化空间的统计量 = 优化 HU 空间的统计量
        
        Args:
            pred: (B, 1, D, H, W) 预测，归一化空间 [0,1]
            target: (B, 1, D, H, W) 目标，归一化空间 [0,1]
            mask: (B, 1, D, H, W) 病灶 mask
        
        Returns:
            scalar: HU 统计量 L1 距离
        """
        B = pred.size(0)
        total_loss = torch.tensor(0.0, device=pred.device)
        count = 0
        
        for i in range(B):
            m = mask[i, 0] > 0
            if m.sum() < 2:
                continue
            
            p = pred[i, 0][m]
            t = target[i, 0][m]
            
            # 归一化空间的均值和标准差
            mean_loss = F.l1_loss(p.mean(), t.mean())
            std_loss = F.l1_loss(p.std(), t.std())
            
            total_loss = total_loss + mean_loss + std_loss
            count += 1
        
        return total_loss / max(count, 1)
    
    def compute_soft_histogram_loss(self, pred, target, mask, num_bins=50):
        """
        Soft Histogram Loss（可导的直方图匹配）
        
        思路：
            - 传统直方图不可导（离散 binning）
            - Soft Histogram：用 Gaussian kernel 平滑 binning
        
        数学形式：
            h_soft(x) = Σ_i exp(-(x - x_i)^2 / (2σ^2))
        
        Args:
            pred: (B, 1, D, H, W) 预测
            target: (B, 1, D, H, W) 目标
            mask: (B, 1, D, H, W) mask
            num_bins: 直方图 bin 数量
        
        Returns:
            scalar: 直方图 L1 距离
        """
        B = pred.size(0)
        total_loss = torch.tensor(0.0, device=pred.device)
        count = 0
        
        # Bin centers（归一化空间 [0,1]）
        bin_centers = torch.linspace(0, 1, num_bins, device=pred.device)
        sigma = 1.0 / num_bins  # Gaussian kernel 宽度
        
        for i in range(B):
            m = mask[i, 0] > 0
            if m.sum() < 10:
                continue
            
            p = pred[i, 0][m]  # (N,)
            t = target[i, 0][m]  # (N,)
            
            # Soft histogram（使用 Gaussian kernel）
            # p_hist[j] = Σ_i exp(-(p_i - bin_j)^2 / (2σ^2))
            p_expanded = p.unsqueeze(1)  # (N, 1)
            t_expanded = t.unsqueeze(1)  # (N, 1)
            bins_expanded = bin_centers.unsqueeze(0)  # (1, num_bins)
            
            p_hist = torch.exp(-((p_expanded - bins_expanded) ** 2) / (2 * sigma ** 2))
            t_hist = torch.exp(-((t_expanded - bins_expanded) ** 2) / (2 * sigma ** 2))
            
            # 归一化
            p_hist = p_hist.sum(dim=0) / (p_hist.sum() + 1e-8)
            t_hist = t_hist.sum(dim=0) / (t_hist.sum() + 1e-8)
            
            # L1 距离
            hist_loss = F.l1_loss(p_hist, t_hist)
            total_loss = total_loss + hist_loss
            count += 1

        return total_loss / max(count, 1)

    def forward(self, pred, target, mask, condition=None):
        """
        计算所有条件化损失

        Args:
            pred: (B, 1, D, H, W) 预测，归一化空间 [0,1]
            target: (B, 1, D, H, W) 目标，归一化空间 [0,1]
            mask: (B, 1, D, H, W) 病灶 mask
            condition: (B, 5) 条件向量（当前版本未使用，保留接口）

        Returns:
            losses: dict，包含各项 loss 和 total
        """
        losses = {}
        total = torch.tensor(0.0, device=pred.device)

        # Wasserstein-1 距离
        if self.lambda_wasserstein > 0:
            w1 = self.compute_wasserstein_1d(pred, target, mask)
            losses['wasserstein'] = w1
            total = total + self.lambda_wasserstein * w1

        # HU Statistics Loss（归一化空间）
        if self.lambda_hu_stats > 0:
            hu_stats = self.compute_hu_stats_loss(pred, target, mask)
            losses['hu_stats'] = hu_stats
            total = total + self.lambda_hu_stats * hu_stats

        # Soft Histogram Loss
        if self.lambda_soft_histogram > 0:
            soft_hist = self.compute_soft_histogram_loss(pred, target, mask)
            losses['soft_histogram'] = soft_hist
            total = total + self.lambda_soft_histogram * soft_hist

        # EI Loss（由 Trainer 处理，这里保留接口）
        if self.lambda_ei > 0 and condition is not None:
            # 占位，实际由 Trainer 计算
            pass

        losses['condition_total'] = total
        return losses


