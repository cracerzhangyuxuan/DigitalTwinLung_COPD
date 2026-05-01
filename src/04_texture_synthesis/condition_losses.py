#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CICI-FiLM v2 条件化损失函数
============================

1. HU Statistics Loss: 均值 + 标准差匹配（可导，梯度强）
2. Soft EI Loss: 改进的 EI 损失
3. Histogram Loss: 直方图匹配（辅助监控）

设计原则：
- 避免 MMD 的 O(N²) 计算开销（64³ patch 有 262K 体素，cdist 会 OOM）
- 使用统计量匹配代替分布匹配，梯度信号更强
- histogram loss 用 torch.histc（不可导），仅作为监控指标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionLosses(nn.Module):
    """CICI-FiLM v2 条件化损失函数"""

    def __init__(
        self,
        lambda_hu_stats=1.0,
        lambda_ei=0.5,
        lambda_histogram=0.0,
        ei_temperature=10.0,
        hu_min=-1000.0,
        hu_max=400.0,
        histogram_bins=50,
    ):
        super().__init__()
        self.lambda_hu_stats = lambda_hu_stats
        self.lambda_ei = lambda_ei
        self.lambda_histogram = lambda_histogram
        self.ei_temperature = ei_temperature
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.histogram_bins = histogram_bins

    # ------------------------------------------------------------------
    # HU Statistics Loss（核心：可导，梯度强）
    # ------------------------------------------------------------------
    def compute_hu_stats_loss(self, pred, target, mask):
        """在 mask 区域内匹配 HU 均值和标准差

        比 MMD 高效得多（O(N) vs O(N²)），且梯度信号更强。

        Args:
            pred, target: (B, 1, D, H, W) 归一化空间 [0,1]
            mask: (B, 1, D, H, W)
        Returns:
            loss: 标量
        """
        B = pred.size(0)
        total_loss = torch.tensor(0.0, device=pred.device)
        count = 0

        # 反归一化到 HU 空间
        pred_hu = pred * (self.hu_max - self.hu_min) + self.hu_min
        target_hu = target * (self.hu_max - self.hu_min) + self.hu_min

        for i in range(B):
            m = mask[i, 0] > 0
            if m.sum() < 2:
                continue
            p = pred_hu[i, 0][m]
            t = target_hu[i, 0][m]

            # 均值匹配（HU 空间）
            mean_loss = F.l1_loss(p.mean(), t.mean())
            # 标准差匹配（HU 空间）
            std_loss = F.l1_loss(p.std(), t.std())

            total_loss = total_loss + mean_loss + std_loss
            count += 1

        return total_loss / max(count, 1)

    # ------------------------------------------------------------------
    # Soft EI Loss
    # ------------------------------------------------------------------
    def compute_ei_loss(self, pred, condition):
        """Soft EI Loss: sigmoid 近似的 EI 与条件向量中 EI 目标的 MSE

        Args:
            pred: (B, 1, D, H, W) 归一化空间
            condition: (B, 5)，c₁ = global_EI / 100
        """
        hu = pred * (self.hu_max - self.hu_min) + self.hu_min
        soft_ei = torch.sigmoid((-950.0 - hu) / self.ei_temperature)
        soft_ei_mean = soft_ei.view(pred.size(0), -1).mean(dim=1)
        ei_target = condition[:, 0]
        return F.mse_loss(soft_ei_mean, ei_target)

    # ------------------------------------------------------------------
    # Histogram Loss（不可导，仅监控）
    # ------------------------------------------------------------------
    def compute_histogram_loss(self, pred, target, mask):
        """直方图 L1 距离（torch.histc 不可导，仅用于日志监控）"""
        B = pred.size(0)
        pred_hu = pred * (self.hu_max - self.hu_min) + self.hu_min
        target_hu = target * (self.hu_max - self.hu_min) + self.hu_min

        total = 0.0
        for i in range(B):
            m = mask[i, 0] > 0
            if m.sum() == 0:
                continue
            ph = torch.histc(pred_hu[i, 0][m], self.histogram_bins,
                             self.hu_min, self.hu_max)
            th = torch.histc(target_hu[i, 0][m], self.histogram_bins,
                             self.hu_min, self.hu_max)
            ph = ph / (ph.sum() + 1e-8)
            th = th / (th.sum() + 1e-8)
            total += F.l1_loss(ph, th).item()

        return total / max(B, 1)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, pred, target, mask, condition=None):
        """计算所有条件化损失

        Returns:
            losses: dict，包含 'hu_stats', 'ei', 'histogram', 'condition_total'
        """
        losses = {}
        if condition is None:
            return losses

        total = torch.tensor(0.0, device=pred.device)

        # HU Statistics Loss
        if self.lambda_hu_stats > 0:
            hu_stats = self.compute_hu_stats_loss(pred, target, mask)
            losses['hu_stats'] = hu_stats
            total = total + self.lambda_hu_stats * hu_stats

        # EI Loss
        if self.lambda_ei > 0:
            ei = self.compute_ei_loss(pred, condition)
            losses['ei'] = ei
            total = total + self.lambda_ei * ei

        # Histogram Loss（仅监控，不参与梯度）
        if self.lambda_histogram > 0:
            with torch.no_grad():
                losses['histogram'] = self.compute_histogram_loss(
                    pred, target, mask)

        losses['condition_total'] = total
        return losses
