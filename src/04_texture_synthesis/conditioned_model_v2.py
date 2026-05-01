#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CICI-FiLM v2: Multi-level FiLM with Enhanced Conditioning
==========================================================

通过 forward hooks 在 InpaintingUNet 的 bottleneck 和 decoder 各层
注入 FiLM 调制，无需修改 backbone 代码。

改进点（相比 v1）：
1. 多层 FiLM 注入（bottleneck + decoder[2] + output = 3 个注入点）
2. 更强的条件编码器（5→128→256，带残差连接）
3. 参数量从 2,434 增至 ~315K
4. 使用 forward hooks，backbone 代码零修改

通道配置（InpaintingUNet features=[32,64,128,256]）：
    bottleneck:  256→512  (ConvBlock3D)
    first_up:    512→256  (UpBlock)
    decoder[0]:  256→128  (UpBlock)
    decoder[1]:  128→64   (UpBlock)
    decoder[2]:  64→32    (UpBlock)
    output_conv: 32→1     (Conv3d)

FiLM 注入点与参数量：
    film_bottleneck (512ch): 256×512×2 + 512×2 = 263,168
    film_last_dec   (32ch):  256×32×2  + 32×2  = 16,448
    film_output     (1ch):   256×1×2   + 1×2   = 514
    cond_encoder:                                 ~35K

    选择注入点: bottleneck(512) + decoder[2](32) + output(1)
    → 总可训练参数 ~315K（合理范围，避免过拟合）
"""

import torch
import torch.nn as nn


class ResidualConditionEncoder(nn.Module):
    """残差式条件编码器: 5 → 128 → 256"""

    def __init__(self, cond_dim=5, cond_emb_dim=256, dropout=0.1):
        super().__init__()
        hidden_dim = cond_emb_dim // 2
        self.encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, cond_emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.residual = nn.Linear(cond_dim, cond_emb_dim)

    def forward(self, condition):
        return self.encoder(condition) + self.residual(condition)


class FiLMBlock(nn.Module):
    """FiLM: y' = (1 + γ) ⊙ y + β，零初始化保证 identity 起点"""

    def __init__(self, cond_emb_dim, num_channels):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_emb_dim, num_channels)
        self.beta_proj = nn.Linear(cond_emb_dim, num_channels)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, h, cond_emb):
        gamma = self.gamma_proj(cond_emb).view(h.size(0), -1, 1, 1, 1) + 1.0
        beta = self.beta_proj(cond_emb).view(h.size(0), -1, 1, 1, 1)
        return gamma * h + beta


class ConditionedGeneratorV2(nn.Module):
    """CICI-FiLM v2: 基于 forward hooks 的多层 FiLM 条件化生成器

    注入点（3 层）：
        1. backbone.bottleneck 输出后 (512 ch)
        2. backbone.decoder[2] 输出后  (32 ch，最后一个 decoder)
        3. backbone.output_conv 输出后 (1 ch)

    参数量：
        cond_encoder:       ~35K
        film_bottleneck:    256×512×2 + 512×2 ≈ 263K
        film_last_dec:      256×32×2  + 32×2  ≈ 16K
        film_output:        256×1×2   + 1×2   ≈ 0.5K
        ─────────────────────────────────────────
        总可训练:           ~315K
    """

    def __init__(self, backbone, cond_dim=5, cond_emb_dim=256):
        super().__init__()
        self.backbone = backbone

        # 条件编码器
        self.cond_encoder = ResidualConditionEncoder(cond_dim, cond_emb_dim)

        # 多层 FiLM blocks
        self.film_bottleneck = FiLMBlock(cond_emb_dim, 512)
        self.film_last_dec = FiLMBlock(cond_emb_dim, 32)
        self.film_output = FiLMBlock(cond_emb_dim, 1)

        # hook 状态
        self._cond_emb = None
        self._hooks = []

    def _bottleneck_hook(self, module, input, output):
        if self._cond_emb is not None:
            return self.film_bottleneck(output, self._cond_emb)
        return output

    def _last_decoder_hook(self, module, input, output):
        if self._cond_emb is not None:
            return self.film_last_dec(output, self._cond_emb)
        return output

    def _register_hooks(self):
        """注册 forward hooks 到 backbone 的指定层"""
        if self._hooks:
            return  # 已注册
        # bottleneck: ConvBlock3D, 输出 512ch
        h1 = self.backbone.bottleneck.register_forward_hook(self._bottleneck_hook)
        # decoder[2]: 最后一个 UpBlock, 输出 32ch (在 output_conv 之前)
        h2 = self.backbone.decoder[2].register_forward_hook(self._last_decoder_hook)
        self._hooks = [h1, h2]

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[CICI-FiLM v2] Trainable: {trainable:,} / {total:,} "
              f"({100*trainable/total:.3f}%)")

    def forward(self, x, condition=None):
        if condition is not None:
            self._cond_emb = self.cond_encoder(condition)
            self._register_hooks()
        else:
            self._cond_emb = None

        # backbone forward（hooks 会在内部自动触发）
        y = self.backbone(x)

        # output-level FiLM
        if self._cond_emb is not None:
            y = self.film_output(y, self._cond_emb)

        self._cond_emb = None
        return y

