#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CICI-FiLM: Clinical-Imaging Conditioned Inpainting with Feature-wise Linear Modulation
=======================================================================================

Output-level FiLM Wrapper — 在已有 backbone 输出端添加条件驱动的
Feature-wise Linear Modulation，实现 COPD 病灶严重度可控生成。

核心设计原则:
  1. Wrapper 模式：backbone 完全不改、权重完全复用、检查点 100% 兼容
  2. Output-level FiLM：仅在最终输出层做一次 channel-wise affine 调制
  3. 冻结主干训练：只训练 FiLM 参数 (2,434 / 22,589,557 = 0.010%)
  4. 零初始化策略：γ≈0, β≈0，保证训练起点 = identity（不破坏预训练先验）
  5. 回退安全：condition=None 时自动退化为无条件模式（与原模型行为一致）

物理含义:
  - β（偏移）→ 控制生成纹理的整体 HU 基准线（密度中心）
  - γ（缩放）→ 控制生成纹理的内部对比度（HU 方差）
  - 5 维条件向量 c = [global_EI, lesion_vol_ratio, lesion_HU_mean, lesion_HU_std, GOLD]

使用方法:
    # 1. 加载预训练 backbone
    from src.04_texture_synthesis.network import create_model
    backbone, _ = create_model('patchgan')
    backbone.load_state_dict(torch.load('checkpoints/patchgan/best.pth')['generator_state_dict'])
    
    # 2. 包裹为条件化模型
    model = ConditionedGenerator(backbone, cond_dim=5, cond_emb_dim=64)
    model.freeze_backbone()  # 冻结主干，只训练 FiLM 分支
    
    # 3. 前向传播
    output = model(x, condition)  # condition: (B, 5) 或 None
    
    # 4. 回退到无条件模式
    output_uncond = model(x, condition=None)  # 等价于 backbone(x)

参考文献:
  - FiLM: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018
  - Wrapper Pattern: Gang of Four, "Design Patterns: Elements of Reusable Object-Oriented Software"
"""

import torch
import torch.nn as nn


class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for single-channel 3D output
    
    数学形式:
        y' = (γ + 1) ⊙ y + β
    
    其中:
        - y: backbone 输出 (B, 1, D, H, W)
        - γ, β: 从条件向量预测的调制参数 (B, 1)
        - γ + 1: 保证初始状态 γ=0 时缩放因子=1（identity）
    
    物理含义（COPD 病灶生成）:
        - β < 0: 整体 HU 下移（更暗，更接近空气 -1000 HU），对应重度气肿
        - β > 0: 整体 HU 上移（更亮，更接近正常组织），对应轻度气肿
        - γ > 0: 放大 HU 方差（更高对比度），对应非均匀气肿（小叶中心型）
        - γ < 0: 压缩 HU 方差（更低对比度），对应均匀弥漫性气肿
    
    零初始化策略:
        - 所有投影层权重和偏置初始化为 0
        - 保证训练起点: γ=0, β=0 → y'=y（完全等价于 Exp-0 baseline）
        - 避免引入 FiLM 后初始性能骤降
    """
    
    def __init__(self, cond_emb_dim, num_channels=1):
        """
        Args:
            cond_emb_dim (int): 条件编码器输出维度（默认 64）
            num_channels (int): 输出通道数（3D Inpainting 固定为 1）
        """
        super().__init__()
        self.gamma_proj = nn.Linear(cond_emb_dim, num_channels)
        self.beta_proj = nn.Linear(cond_emb_dim, num_channels)
        
        # 零初始化：保证训练起点 = identity
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
    
    def forward(self, h, cond_emb):
        """
        Args:
            h: backbone 输出 (B, C, D, H, W)，C=1
            cond_emb: 条件编码 (B, cond_emb_dim)
        
        Returns:
            调制后的输出 (B, C, D, H, W)
        """
        # 预测调制参数
        gamma = self.gamma_proj(cond_emb)  # (B, C)
        beta = self.beta_proj(cond_emb)    # (B, C)
        
        # reshape for broadcasting: (B, C) → (B, C, 1, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1.0  # +1 保证初始=1
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # FiLM 调制: y' = (γ + 1) * y + β
        return gamma * h + beta


class ConditionedGenerator(nn.Module):
    """
    条件化生成器 Wrapper
    
    架构设计:
        ┌─────────────────────────────────────────────────────────┐
        │  Input: x (B,1,D,H,W)    Condition: c (B,5)            │
        │         │                         │                     │
        │         ▼                         ▼                     │
        │  ┌──────────────┐         ┌────────────┐              │
        │  │   Backbone   │         │CondEncoder │              │
        │  │  (冻结)       │         │  5→32→64   │              │
        │  └──────┬───────┘         └─────┬──────┘              │
        │         │                       │                      │
        │    y (B,1,D,H,W)           e (B,64)                   │
        │         │                  ┌────┴────┐                │
        │         │                  │         │                │
        │         │            ┌─────┴──┐  ┌───┴────┐          │
        │         │            │γ_proj  │  │β_proj  │          │
        │         │            │ 64→1   │  │ 64→1   │          │
        │         │            └───┬────┘  └───┬────┘          │
        │         │                │           │                │
        │         │           γ (B,1)      β (B,1)             │
        │         │                └─────┬─────┘                │
        │         ▼                      ▼                      │
        │  ┌─────────────────────────────────────┐             │
        │  │  FiLM: y' = (γ+1)⊙y + β            │             │
        │  └──────────────┬──────────────────────┘             │
        │                 │                                     │
        │            y' (B,1,D,H,W)                            │
        └─────────────────────────────────────────────────────────┘
    
    参数量分析（5 维条件向量）:
        - Backbone (PatchGAN):     22,587,123  (冻结)
        - CondEncoder (5→32):            192  (可训练)
        - CondEncoder (32→64):         2,112  (可训练)
        - γ_proj (64→1):                  65  (可训练)
        - β_proj (64→1):                  65  (可训练)
        ─────────────────────────────────────
        总计:                      22,589,557
        可训练:                         2,434  (0.010%)
    
    Wrapper 模式的工程意义:
        1. 检查点兼容: backbone.* 的 key 完全不变，可直接 load_state_dict()
        2. 代码稳定: network.py 无需改动，新增逻辑局限于外层
        3. 回退安全: condition=None 时退化为原始生成器
    """
    
    def __init__(self, backbone, cond_dim=5, cond_emb_dim=64):
        """
        Args:
            backbone (nn.Module): 预训练的 3D Inpainting 生成器
            cond_dim (int): 条件向量维度（默认 5: EI, vol_ratio, HU_mean, HU_std, GOLD）
            cond_emb_dim (int): 条件编码器输出维度（默认 64）
        """
        super().__init__()
        
        # 持有原始 backbone（不修改任何内部结构）
        self.backbone = backbone
        
        # 条件编码器: 5 维 → 64 维
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, cond_emb_dim),
            nn.ReLU(inplace=True),
        )
        
        # FiLM 调制层（Output-level，仅作用于最终输出）
        self.film = FiLMBlock(cond_emb_dim, num_channels=1)
    
    def freeze_backbone(self):
        """
        冻结主干网络，只训练条件分支
        
        统计安全性:
            - 可训练参数: 2,434
            - 有效训练样本: 23 例 × ~50 patch = 1,150
            - 参数/样本比: 2.1（可控范围）
        
        对比从头重训:
            - 可训练参数: 22,589,557
            - 参数/样本比: ~19,643（高出三个数量级，过拟合风险极高）
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[CICI-FiLM] Trainable: {trainable:,} / {total:,} "
              f"({100*trainable/total:.3f}%)")
    
    def forward(self, x, condition=None):
        """
        前向传播
        
        Args:
            x: 输入图像 (B, 1, D, H, W)
            condition: 条件向量 (B, 5) 或 None
                      None 时退化为无条件模式（等价于 backbone(x)）
        
        Returns:
            生成结果 (B, 1, D, H, W)
        """
        # Backbone 前向（冻结，不参与梯度更新）
        y = self.backbone(x)
        
        # 条件化调制（仅当提供条件时）
        if condition is not None:
            cond_emb = self.cond_encoder(condition)  # (B, 5) → (B, 64)
            y = self.film(y, cond_emb)               # FiLM 调制
        
        return y

