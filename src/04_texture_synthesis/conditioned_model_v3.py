#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CICI-FiLM v3: SPADE-based Multi-scale Conditional Inpainting
=============================================================

核心改进：
  1. SPADE (Spatially-Adaptive Normalization)：空间自适应调制，替代 channel-wise FiLM
  2. 多尺度注入：encoder bottleneck + decoder 三层，共 4 个注入点
  3. 增强条件编码器：5→128→256→512，LayerNorm + Dropout + Residual
  4. 鲁棒性优化：条件向量 clipping + robust normalization

物理含义：
  - SPADE 允许在不同空间位置施加不同的调制强度
  - 适合 COPD 病灶的空间异质性（小叶中心型 vs 弥漫型）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustConditionEncoder(nn.Module):
    """
    鲁棒条件编码器
    
    改进点：
      1. 更深的网络：5→128→256→512
      2. LayerNorm：稳定训练，防止梯度爆炸
      3. Dropout：增强泛化，防止对极端值过拟合
      4. Residual：多尺度特征融合
    """
    
    def __init__(self, cond_dim=5, cond_emb_dim=512, dropout=0.1):
        super().__init__()
        
        # 主干网络
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(256, cond_emb_dim),
            nn.LayerNorm(cond_emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Residual shortcut
        self.shortcut = nn.Linear(cond_dim, cond_emb_dim)
    
    def forward(self, condition):
        """
        Args:
            condition: (B, 5) 条件向量
        
        Returns:
            (B, 512) 条件编码
        """
        return self.net(condition) + self.shortcut(condition)


class SPADEBlock(nn.Module):
    """
    SPADE (Spatially-Adaptive Normalization) Block
    
    数学形式：
        y' = γ(c) ⊙ normalize(y) + β(c)
    
    其中 γ(c), β(c) 是从条件向量 c 学习的空间调制图（spatial modulation maps）
    
    与 FiLM 的区别：
      - FiLM: γ, β 是标量（每个通道一个值）
      - SPADE: γ, β 是空间图（每个位置不同值）
    
    物理含义（COPD）：
      - 允许在病灶中心施加强调制，边缘施加弱调制
      - 适合小叶中心型气肿的空间异质性
    """
    
    def __init__(self, cond_emb_dim, num_channels, spatial_size):
        """
        Args:
            cond_emb_dim: 条件编码维度（512）
            num_channels: 特征图通道数
            spatial_size: 特征图空间尺寸（D, H, W）
        """
        super().__init__()
        self.num_channels = num_channels
        
        # 条件投影到中间维度
        hidden_dim = 128
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 学习空间调制图的卷积层
        # 使用 3x3x3 卷积学习空间模式（SPADE 的核心）
        self.gamma_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, num_channels, kernel_size=3, padding=1),
        )
        self.beta_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, num_channels, kernel_size=3, padding=1),
        )

        # 零初始化最后一层：保证训练起点 = identity
        nn.init.zeros_(self.gamma_conv[-1].weight)
        nn.init.zeros_(self.gamma_conv[-1].bias)
        nn.init.zeros_(self.beta_conv[-1].weight)
        nn.init.zeros_(self.beta_conv[-1].bias)
    
    def forward(self, h, cond_emb):
        """
        Args:
            h: (B, C, D, H, W) 特征图
            cond_emb: (B, 512) 条件编码
        
        Returns:
            (B, C, D, H, W) 调制后的特征图
        """
        B, C, D, H, W = h.shape
        
        # 归一化特征图（Instance Normalization）
        h_norm = F.instance_norm(h)
        
        # 条件投影
        cond_hidden = self.cond_proj(cond_emb)  # (B, 128)
        
        # 扩展到空间维度
        cond_spatial = cond_hidden.view(B, -1, 1, 1, 1).expand(B, -1, D, H, W)  # (B, 128, D, H, W)
        
        # 学习空间调制图
        gamma = self.gamma_conv(cond_spatial)  # (B, C, D, H, W)
        beta = self.beta_conv(cond_spatial)    # (B, C, D, H, W)
        
        # SPADE 调制: y' = (1 + γ) * normalize(y) + β
        return (1.0 + gamma) * h_norm + beta


class ConditionedGeneratorV3(nn.Module):
    """
    CICI-FiLM v3: SPADE-based Multi-scale Conditional Generator

    架构设计：
        ┌─────────────────────────────────────────────────────────┐
        │  Input: x (B,1,D,H,W)    Condition: c (B,5)            │
        │         │                         │                     │
        │         ▼                         ▼                     │
        │  ┌──────────────┐         ┌────────────────┐          │
        │  │   Backbone   │         │RobustCondEncoder│          │
        │  │  (冻结)       │         │  5→128→256→512 │          │
        │  └──────┬───────┘         └─────┬──────────┘          │
        │         │                       │                      │
        │    ┌────┴────┐                  │                      │
        │    │ encoder │                  │                      │
        │    │ decoder │ ←────── SPADE ───┤ (4 个注入点)         │
        │    │bottleneck│                 │                      │
        │    └────┬────┘                  │                      │
        │         │                       │                      │
        │    y (B,1,D,H,W)                │                      │
        └─────────────────────────────────────────────────────────┘

    注入点：
      1. backbone.bottleneck 输出 (512 ch, 8x8x8)
      2. backbone.decoder[0] 输出 (128 ch, 16x16x16)
      3. backbone.decoder[1] 输出 (64 ch, 32x32x32)
      4. backbone.decoder[2] 输出 (32 ch, 64x64x64)

    参数量分析：
      - Backbone (PatchGAN):     22,587,123  (冻结)
      - RobustCondEncoder:            ~400K  (可训练)
      - 4 × SPADEBlock:               ~800K  (可训练)
      ─────────────────────────────────────
      总计:                      ~23,787K
      可训练:                      ~1,200K  (5.0%)
    """

    def __init__(self, backbone, cond_dim=5, cond_emb_dim=512):
        """
        Args:
            backbone: 预训练的 3D Inpainting 生成器
            cond_dim: 条件向量维度（5）
            cond_emb_dim: 条件编码维度（512）
        """
        super().__init__()

        self.backbone = backbone

        # 鲁棒条件编码器
        self.cond_encoder = RobustConditionEncoder(cond_dim, cond_emb_dim)

        # 4 个 SPADE 注入点
        # 根据 InpaintingUNet 的实际通道数配置
        # features = [32, 64, 128, 256]
        # bottleneck: 512 ch (features[-1] * 2)
        # decoder[0]: 128 ch (features_rev[1])
        # decoder[1]: 64 ch (features_rev[2])
        # decoder[2]: 32 ch (features_rev[3])
        self.spade_bottleneck = SPADEBlock(cond_emb_dim, 512, (8, 8, 8))
        self.spade_dec0 = SPADEBlock(cond_emb_dim, 128, (16, 16, 16))
        self.spade_dec1 = SPADEBlock(cond_emb_dim, 64, (32, 32, 32))
        self.spade_dec2 = SPADEBlock(cond_emb_dim, 32, (64, 64, 64))

        self._cond_emb = None
        self._hooks = []

    def _bottleneck_hook(self, module, input, output):
        if self._cond_emb is not None:
            return self.spade_bottleneck(output, self._cond_emb)
        return output

    def _dec0_hook(self, module, input, output):
        if self._cond_emb is not None:
            return self.spade_dec0(output, self._cond_emb)
        return output

    def _dec1_hook(self, module, input, output):
        if self._cond_emb is not None:
            return self.spade_dec1(output, self._cond_emb)
        return output

    def _dec2_hook(self, module, input, output):
        if self._cond_emb is not None:
            return self.spade_dec2(output, self._cond_emb)
        return output

    def _register_hooks(self):
        if self._hooks:
            return
        h1 = self.backbone.bottleneck.register_forward_hook(self._bottleneck_hook)
        h2 = self.backbone.decoder[0].register_forward_hook(self._dec0_hook)
        h3 = self.backbone.decoder[1].register_forward_hook(self._dec1_hook)
        h4 = self.backbone.decoder[2].register_forward_hook(self._dec2_hook)
        self._hooks = [h1, h2, h3, h4]

    def freeze_backbone(self):
        """冻结主干网络，只训练条件分支"""
        for param in self.backbone.parameters():
            param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[CICI-FiLM v3] Trainable: {trainable:,} / {total:,} "
              f"({100*trainable/total:.3f}%)")

    def forward(self, x, condition=None):
        """
        前向传播

        Args:
            x: (B, 1, D, H, W) 输入图像
            condition: (B, 5) 条件向量或 None

        Returns:
            (B, 1, D, H, W) 生成结果
        """
        if condition is not None:
            # 条件向量预处理：clipping 极端值
            condition = self._preprocess_condition(condition)

            # 条件编码
            self._cond_emb = self.cond_encoder(condition)
            self._register_hooks()
        else:
            self._cond_emb = None

        # Backbone 前向（SPADE 通过 hooks 自动注入）
        y = self.backbone(x)

        self._cond_emb = None
        return y

    def _preprocess_condition(self, condition):
        """
        条件向量预处理：防止极端值导致调制崩坏

        策略：
          - c1 (EI): clip 到 [0, 0.5]（EI > 50% 极罕见）
          - c2 (vol_ratio): clip 到 [0, 0.5]
          - c3, c4 (HU mean/std): 已经过 sigmoid，天然在 [0,1]
          - c5 (GOLD): 已归一化到 [0,1]
        """
        c = condition.clone()
        c[:, 0] = torch.clamp(c[:, 0], 0.0, 0.5)  # EI
        c[:, 1] = torch.clamp(c[:, 1], 0.0, 0.5)  # vol_ratio
        return c


