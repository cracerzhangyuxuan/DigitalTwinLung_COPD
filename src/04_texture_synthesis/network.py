#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网络架构模块

定义 Inpainting U-Net、Patch Discriminator、Partial Conv U-Net、
Attention U-Net (AttGAN)、Diffusion U-Net (DDPM)

支持六种模型架构：
- 基线方案: InpaintingUNet (3D U-Net)
- 进阶方案: PartialConvUNet (3D Partial Convolution)
- 高级方案: PatchGAN (InpaintingUNet + PatchDiscriminator)
- 注意力增强方案: AttGAN (AttentionUNet + PatchDiscriminator)
- 自监督预训练方案: MAE-PatchGAN (InpaintingUNet + PatchDiscriminator, 预训练权重)
- 扩散模型方案: DDPM (DiffusionUNet)
"""

import math
from typing import List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None


class ConvBlock3D(nn.Module):
    """3D 卷积块: Conv -> BN -> ReLU"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """下采样块: MaxPool -> ConvBlock"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock3D(in_channels, out_channels)
    
    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """上采样块: ConvTranspose -> Concat -> ConvBlock"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = ConvBlock3D(in_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # 处理尺寸不匹配
        if x.shape != skip.shape:
            diff_d = skip.shape[2] - x.shape[2]
            diff_h = skip.shape[3] - x.shape[3]
            diff_w = skip.shape[4] - x.shape[4]
            x = F.pad(x, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
                diff_d // 2, diff_d - diff_d // 2
            ])
        
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class InpaintingUNet(nn.Module):
    """
    3D U-Net for Inpainting
    
    输入: (B, 1, D, H, W) - 带有空洞的 CT patch
    输出: (B, 1, D, H, W) - 填充后的 CT patch
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256]
    ):
        super().__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # 编码器
        self.input_conv = ConvBlock3D(in_channels, features[0])
        
        for i in range(len(features) - 1):
            self.encoder.append(DownBlock(features[i], features[i + 1]))
        
        # 瓶颈
        self.bottleneck = ConvBlock3D(features[-1], features[-1] * 2)
        
        # 解码器
        features_rev = features[::-1]
        self.first_up = UpBlock(features[-1] * 2, features[-1])
        
        for i in range(len(features_rev) - 1):
            self.decoder.append(UpBlock(features_rev[i], features_rev[i + 1]))
        
        # 输出层
        self.output_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        # 编码
        skips = []
        x = self.input_conv(x)
        skips.append(x)
        
        for down in self.encoder:
            x = down(x)
            skips.append(x)
        
        # 瓶颈
        x = self.bottleneck(x)
        
        # 解码
        x = self.first_up(x, skips[-1])
        
        for i, up in enumerate(self.decoder):
            x = up(x, skips[-(i + 2)])
        
        # 输出
        return self.output_conv(x)


class PatchDiscriminator(nn.Module):
    """
    Patch Discriminator for GAN training
    
    输入: (B, 1, D, H, W)
    输出: (B, 1, D', H', W') - patch-wise 判别结果
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        features: List[int] = [64, 128, 256, 512]
    ):
        super().__init__()
        
        layers = []
        prev_channels = in_channels
        
        for i, feat in enumerate(features):
            layers.append(
                nn.Conv3d(
                    prev_channels, feat,
                    kernel_size=4, stride=2, padding=1
                )
            )
            if i > 0:
                layers.append(nn.BatchNorm3d(feat))
            layers.append(nn.LeakyReLU(0.2, inplace=False))  # 改为 inplace=False 避免梯度问题
            prev_channels = feat
        
        # 最后一层
        layers.append(nn.Conv3d(prev_channels, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Partial Convolution 实现 (进阶方案)
# ============================================================================

class PartialConv3d(nn.Module):
    """
    3D Partial Convolution Layer

    参考: Liu et al., "Image Inpainting for Irregular Holes Using Partial Convolutions"

    特点: 只在有效区域（非 mask）进行卷积，自动更新 mask
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.mask_conv = nn.Conv3d(1, 1, kernel_size, stride, padding, bias=False)

        # mask 卷积权重固定为 1
        nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: 'torch.Tensor', mask: 'torch.Tensor') -> Tuple['torch.Tensor', 'torch.Tensor']:
        """
        Args:
            x: 输入特征 (B, C, D, H, W)
            mask: 有效区域 mask (B, 1, D, H, W), 1=有效, 0=空洞

        Returns:
            output: 输出特征
            updated_mask: 更新后的 mask
        """
        # 计算有效像素数
        with torch.no_grad():
            updated_mask = self.mask_conv(mask)
            # 避免除零
            mask_ratio = self.kernel_size ** 3 / (updated_mask + 1e-8)
            updated_mask = torch.clamp(updated_mask, 0, 1)
            updated_mask = (updated_mask > 0).float()

        # 只在有效区域卷积
        x_masked = x * mask
        output = self.conv(x_masked) * mask_ratio * updated_mask

        # 添加 bias（如果有）
        if self.conv.bias is not None:
            output = output + self.conv.bias.view(1, -1, 1, 1, 1) * updated_mask

        return output, updated_mask


class PartialConvBlock3D(nn.Module):
    """Partial Convolution 块: PConv -> BN -> ReLU"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pconv1 = PartialConv3d(in_channels, out_channels)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.pconv2 = PartialConv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: 'torch.Tensor', mask: 'torch.Tensor') -> Tuple['torch.Tensor', 'torch.Tensor']:
        x, mask = self.pconv1(x, mask)
        x = self.relu(self.bn1(x))
        x, mask = self.pconv2(x, mask)
        x = self.relu(self.bn2(x))
        return x, mask


class PartialConvUNet(nn.Module):
    """
    3D Partial Convolution U-Net

    进阶方案：处理不规则 Mask 更优

    输入:
        - x: (B, 1, D, H, W) 带空洞的 CT
        - mask: (B, 1, D, H, W) 有效区域 mask (1=有效, 0=空洞)
    输出: (B, 1, D, H, W) 填充后的 CT
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256]
    ):
        super().__init__()

        # 编码器
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_ch = in_channels
        for feat in features:
            self.enc_blocks.append(PartialConvBlock3D(prev_ch, feat))
            self.pools.append(nn.MaxPool3d(2))
            prev_ch = feat

        # 瓶颈
        self.bottleneck = PartialConvBlock3D(features[-1], features[-1] * 2)

        # 解码器
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        features_rev = features[::-1]
        prev_ch = features[-1] * 2

        for feat in features_rev:
            self.ups.append(nn.ConvTranspose3d(prev_ch, feat, kernel_size=2, stride=2))
            self.dec_blocks.append(PartialConvBlock3D(feat * 2, feat))
            prev_ch = feat

        # 输出层
        self.output_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: 'torch.Tensor', mask: 'torch.Tensor' = None) -> 'torch.Tensor':
        """
        Args:
            x: 输入 (B, 1, D, H, W)
            mask: 有效区域 mask (B, 1, D, H, W)，如果为 None 则从 x 推断
        """
        # 如果没有提供 mask，从输入推断（非零区域为有效）
        if mask is None:
            mask = (x != 0).float()

        # 编码
        skips = []
        skip_masks = []

        for enc, pool in zip(self.enc_blocks, self.pools):
            x, mask = enc(x, mask)
            skips.append(x)
            skip_masks.append(mask)
            x = pool(x)
            mask = F.max_pool3d(mask, 2)

        # 瓶颈
        x, mask = self.bottleneck(x, mask)

        # 解码
        for i, (up, dec) in enumerate(zip(self.ups, self.dec_blocks)):
            x = up(x)
            mask = F.interpolate(mask, scale_factor=2, mode='nearest')

            skip = skips[-(i+1)]
            skip_mask = skip_masks[-(i+1)]

            # 处理尺寸不匹配
            if x.shape != skip.shape:
                diff = [skip.shape[j] - x.shape[j] for j in range(2, 5)]
                x = F.pad(x, [d//2 for d in diff[::-1] for _ in range(2)])
                mask = F.pad(mask, [d//2 for d in diff[::-1] for _ in range(2)])

            x = torch.cat([x, skip], dim=1)
            mask = torch.cat([mask, skip_mask], dim=1)
            mask = (mask.sum(dim=1, keepdim=True) > 0).float()

            x, mask = dec(x, mask)

        return self.output_conv(x)


# ============================================================================
# Attention U-Net 实现 (AttGAN 方案 — 注意力增强对抗生成)
# ============================================================================

class AttentionGate3D(nn.Module):
    """
    3D 注意力门控模块

    作用于 Skip Connection：用解码器特征(gate)引导编码器特征(skip)，
    抑制与当前解码层级无关的冗余特征，增强纹理相关的高级语义。

    参考: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", MIDL 2018
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        """
        Args:
            gate_channels: 来自解码器的门控特征通道数
            skip_channels: 来自编码器的跳跃连接特征通道数
            inter_channels: 中间层通道数（通常为 skip_channels // 2）
        """
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(inter_channels)
        )
        self.W_skip = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: 'torch.Tensor', skip: 'torch.Tensor') -> 'torch.Tensor':
        """
        Args:
            gate: 解码器上采样特征 (B, gate_channels, D, H, W)
            skip: 编码器跳跃连接特征 (B, skip_channels, D, H, W)

        Returns:
            attended_skip: 注意力加权后的跳跃连接特征 (B, skip_channels, D, H, W)
        """
        g = self.W_gate(gate)
        x = self.W_skip(skip)
        alpha = self.psi(self.relu(g + x))
        return skip * alpha


class SelfAttention3D(nn.Module):
    """
    3D 自注意力模块

    在 Bottleneck 层后应用，建模全局空间依赖关系，
    使不同空间位置的纹理生成具有协调的差异化。

    参考: Zhang et al., "Self-Attention Generative Adversarial Networks", ICML 2019
    """

    def __init__(self, in_channels: int):
        """
        Args:
            in_channels: 输入特征通道数
        """
        super().__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """
        Args:
            x: 输入特征 (B, C, D, H, W)

        Returns:
            out: 自注意力增强后的特征 (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        N = D * H * W  # 空间位置总数

        q = self.query(x).view(B, -1, N).permute(0, 2, 1)  # (B, N, C//8)
        k = self.key(x).view(B, -1, N)                      # (B, C//8, N)
        v = self.value(x).view(B, -1, N)                     # (B, C, N)

        attention = torch.bmm(q, k)                          # (B, N, N)
        attention = F.softmax(attention / (C // 8) ** 0.5, dim=-1)

        out = torch.bmm(v, attention.permute(0, 2, 1))       # (B, C, N)
        out = out.view(B, C, D, H, W)

        return self.gamma * out + x


class AttentionUNet(nn.Module):
    """
    3D Attention U-Net for Inpainting (AttGAN Generator)

    在 InpaintingUNet 基础上增加：
    1. 每条 Skip Connection 上的 AttentionGate3D — 过滤冗余特征
    2. Bottleneck 后的 SelfAttention3D — 建模全局空间依赖

    输入: (B, 1, D, H, W) - 带有空洞的 CT patch
    输出: (B, 1, D, H, W) - 填充后的 CT patch
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256]
    ):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.attention_gates = nn.ModuleList()

        # 编码器
        self.input_conv = ConvBlock3D(in_channels, features[0])

        for i in range(len(features) - 1):
            self.encoder.append(DownBlock(features[i], features[i + 1]))

        # 瓶颈
        self.bottleneck = ConvBlock3D(features[-1], features[-1] * 2)

        # 自注意力模块（在 bottleneck 后）
        self.self_attention = SelfAttention3D(features[-1] * 2)

        # 解码器 + 注意力门控
        features_rev = features[::-1]
        self.first_up = nn.ConvTranspose3d(
            features[-1] * 2, features[-1], kernel_size=2, stride=2
        )
        self.first_att = AttentionGate3D(features[-1], features[-1], features[-1] // 2)
        self.first_conv = ConvBlock3D(features[-1] * 2, features[-1])

        for i in range(len(features_rev) - 1):
            in_ch = features_rev[i]
            out_ch = features_rev[i + 1]
            self.decoder.append(nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2))
            self.attention_gates.append(
                AttentionGate3D(out_ch, out_ch, out_ch // 2)
            )

        self.dec_convs = nn.ModuleList()
        for i in range(len(features_rev) - 1):
            out_ch = features_rev[i + 1]
            self.dec_convs.append(ConvBlock3D(out_ch * 2, out_ch))

        # 输出层
        self.output_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        # 编码
        skips = []
        x = self.input_conv(x)
        skips.append(x)

        for down in self.encoder:
            x = down(x)
            skips.append(x)

        # 瓶颈 + 自注意力
        x = self.bottleneck(x)
        x = self.self_attention(x)

        # 解码（带注意力门控的 skip connection）
        x = self.first_up(x)
        skip = skips[-1]
        if x.shape != skip.shape:
            diff_d = skip.shape[2] - x.shape[2]
            diff_h = skip.shape[3] - x.shape[3]
            diff_w = skip.shape[4] - x.shape[4]
            x = F.pad(x, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
                diff_d // 2, diff_d - diff_d // 2
            ])
        attended_skip = self.first_att(x, skip)
        x = torch.cat([x, attended_skip], dim=1)
        x = self.first_conv(x)

        for i, (up, att, conv) in enumerate(zip(self.decoder, self.attention_gates, self.dec_convs)):
            x = up(x)
            skip = skips[-(i + 2)]
            if x.shape != skip.shape:
                diff_d = skip.shape[2] - x.shape[2]
                diff_h = skip.shape[3] - x.shape[3]
                diff_w = skip.shape[4] - x.shape[4]
                x = F.pad(x, [
                    diff_w // 2, diff_w - diff_w // 2,
                    diff_h // 2, diff_h - diff_h // 2,
                    diff_d // 2, diff_d - diff_d // 2
                ])
            attended_skip = att(x, skip)
            x = torch.cat([x, attended_skip], dim=1)
            x = conv(x)

        # 输出
        return self.output_conv(x)


# ============================================================================
# Diffusion U-Net 实现 (DDPM 方案 — 去噪扩散概率模型)
# ============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """
    正弦位置编码 — 将离散时间步 t 编码为连续向量

    参考: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
    用于 DDPM 的时间步条件化注入
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: 编码向量维度
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: 'torch.Tensor') -> 'torch.Tensor':
        """
        Args:
            t: 时间步 (B,) 整数张量

        Returns:
            emb: 时间步编码 (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ConvBlock3D_T(nn.Module):
    """
    时间条件化 3D 卷积块: Conv -> BN -> ReLU + Time Embedding 注入

    在标准 ConvBlock3D 的基础上，将时间步编码通过可学习的线性层
    映射到与特征通道数相同的维度，然后以 channel-wise 偏置的方式注入。
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            time_emb_dim: 时间编码维度
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: 'torch.Tensor', t_emb: 'torch.Tensor') -> 'torch.Tensor':
        """
        Args:
            x: 输入特征 (B, C_in, D, H, W)
            t_emb: 时间步编码 (B, time_emb_dim)

        Returns:
            out: 输出特征 (B, C_out, D, H, W)
        """
        x = self.conv1(x)
        # 注入时间步条件：channel-wise bias
        t = self.time_mlp(t_emb)[:, :, None, None, None]  # (B, C_out, 1, 1, 1)
        x = x + t
        x = self.conv2(x)
        return x


class DiffusionUNet(nn.Module):
    """
    3D Diffusion U-Net for DDPM

    在 InpaintingUNet 骨架基础上增加：
    1. SinusoidalPositionEmbedding — 编码扩散时间步 t
    2. 每个卷积块接受 time embedding 作为额外条件输入

    训练时预测添加到 x_0 上的噪声 ε，损失为 MSE(ε, ε_θ(x_t, t))。
    Inpainting 推理时采用 RePaint 策略。

    输入: x (B, 1, D, H, W), t (B,)
    输出: noise_pred (B, 1, D, H, W)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256],
        time_emb_dim: int = 256
    ):
        super().__init__()

        self.time_emb = SinusoidalPositionEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # 编码器 — 结构对齐 InpaintingUNet:
        # input_conv(1→32) + 3个 DownBlock(pool+conv): 32→64, 64→128, 128→256
        self.input_conv = ConvBlock3D_T(in_channels, features[0], time_emb_dim)

        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(len(features) - 1):
            self.pools.append(nn.MaxPool3d(2))
            self.enc_blocks.append(ConvBlock3D_T(features[i], features[i + 1], time_emb_dim))

        # 瓶颈 — 与 InpaintingUNet 一致: 直接接 bottleneck，不额外下采样
        self.bottleneck = ConvBlock3D_T(features[-1], features[-1] * 2, time_emb_dim)

        # 解码器 — 与 InpaintingUNet 一致:
        # first_up(512→256) + 3个 UpBlock: 256→128, 128→64, 64→32
        # 共 4 次上采样对应 4 个 skip connections
        features_rev = features[::-1]  # [256, 128, 64, 32]
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        # first_up: bottleneck(512) → features[-1](256)
        self.ups.append(nn.ConvTranspose3d(features[-1] * 2, features[-1], kernel_size=2, stride=2))
        self.dec_blocks.append(ConvBlock3D_T(features[-1] * 2, features[-1], time_emb_dim))

        # 后续 up blocks: 与 InpaintingUNet 的 self.decoder 对齐
        for i in range(len(features_rev) - 1):
            in_ch = features_rev[i]
            out_ch = features_rev[i + 1]
            self.ups.append(nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock3D_T(out_ch * 2, out_ch, time_emb_dim))

        # 输出层
        self.output_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: 'torch.Tensor', t: 'torch.Tensor') -> 'torch.Tensor':
        """
        Args:
            x: 带噪声的输入 (B, 1, D, H, W)
            t: 扩散时间步 (B,) 整数张量

        Returns:
            noise_pred: 预测的噪声 (B, 1, D, H, W)
        """
        # 时间步编码
        t_emb = self.time_emb(t)
        t_emb = self.time_mlp(t_emb)

        # 编码 — 对齐 InpaintingUNet 的编码路径
        # input_conv → skips[0]
        # pool+enc ×3 → skips[1], skips[2], skips[3]
        skips = []
        x = self.input_conv(x, t_emb)
        skips.append(x)  # skips[0]: 32ch @ 64³

        for enc, pool in zip(self.enc_blocks, self.pools):
            x = pool(x)
            x = enc(x, t_emb)
            skips.append(x)  # skips[1]: 64ch@32³, skips[2]: 128ch@16³, skips[3]: 256ch@8³

        # 瓶颈 — 不额外下采样（与 InpaintingUNet 一致）
        x = self.bottleneck(x, t_emb)  # 512ch @ 8³

        # 解码 — 对齐 InpaintingUNet 的解码路径
        # ups[0] + skips[-1] → first_up
        # ups[1] + skips[-2], ups[2] + skips[-3], ups[3] + skips[-4]
        for i, (up, dec) in enumerate(zip(self.ups, self.dec_blocks)):
            x = up(x)
            skip = skips[-(i + 1)]

            # 处理尺寸不匹配
            if x.shape != skip.shape:
                diff_d = skip.shape[2] - x.shape[2]
                diff_h = skip.shape[3] - x.shape[3]
                diff_w = skip.shape[4] - x.shape[4]
                x = F.pad(x, [
                    diff_w // 2, diff_w - diff_w // 2,
                    diff_h // 2, diff_h - diff_h // 2,
                    diff_d // 2, diff_d - diff_d // 2
                ])

            x = torch.cat([x, skip], dim=1)
            x = dec(x, t_emb)

        # 输出
        return self.output_conv(x)


# ============================================================================
# 工厂函数
# ============================================================================

def create_model(
    model_type: str = "unet",
    in_channels: int = 1,
    out_channels: int = 1,
    features: List[int] = None,
    **kwargs
) -> nn.Module:
    """
    创建模型的工厂函数

    Args:
        model_type: 模型类型
            - "unet": 基线方案 - 3D U-Net Inpainting
            - "partial_conv": 进阶方案 - 3D Partial Convolution U-Net
            - "patchgan": 高级方案 - 返回 (generator, discriminator) 元组
            - "attgan": 注意力增强方案 - 返回 (AttentionUNet, PatchDiscriminator) 元组
            - "mae_patchgan": 自监督预训练方案 - 返回 (InpaintingUNet, PatchDiscriminator) 元组
                              训练时需单独加载 MAE 预训练权重到 Generator encoder
            - "ddpm": 扩散模型方案 - 返回 DiffusionUNet
        in_channels: 输入通道数
        out_channels: 输出通道数
        features: 特征通道列表

    Returns:
        model: 创建的模型
            - 对于 "patchgan"/"attgan"/"mae_patchgan"，返回 (generator, discriminator) 元组
            - 对于 "ddpm"，返回 DiffusionUNet（无 Discriminator）
    """
    if features is None:
        features = [32, 64, 128, 256]

    if model_type == "unet":
        return InpaintingUNet(in_channels, out_channels, features)

    elif model_type == "partial_conv":
        return PartialConvUNet(in_channels, out_channels, features)

    elif model_type == "patchgan":
        generator = InpaintingUNet(in_channels, out_channels, features)
        discriminator = PatchDiscriminator(in_channels)
        return generator, discriminator

    elif model_type == "attgan":
        generator = AttentionUNet(in_channels, out_channels, features)
        discriminator = PatchDiscriminator(in_channels)
        return generator, discriminator

    elif model_type == "mae_patchgan":
        # Generator 架构与 PatchGAN 完全相同（InpaintingUNet）
        # 差异在于训练时需先加载 MAE 自监督预训练权重到 encoder
        generator = InpaintingUNet(in_channels, out_channels, features)
        discriminator = PatchDiscriminator(in_channels)
        return generator, discriminator

    elif model_type == "ddpm":
        time_emb_dim = kwargs.get('time_emb_dim', 256)
        return DiffusionUNet(in_channels, out_channels, features, time_emb_dim)

    else:
        raise ValueError(
            f"未知的模型类型: {model_type}. "
            f"支持: unet, partial_conv, patchgan, attgan, mae_patchgan, ddpm"
        )


def test_network():
    """测试网络"""
    if torch is None:
        print("PyTorch 未安装")
        return

    print("=" * 60)
    print("测试 Phase 3B 网络架构 (6 种模型)")
    print("=" * 60)

    # 测试输入
    x = torch.randn(2, 1, 64, 64, 64)
    mask = (torch.rand(2, 1, 64, 64, 64) > 0.3).float()

    # 测试基线方案: U-Net
    print("\n[1] 基线方案: InpaintingUNet")
    unet = create_model("unet")
    print(f"    参数量: {count_parameters(unet):,}")
    out = unet(x)
    print(f"    输入: {x.shape} -> 输出: {out.shape}")

    # 测试进阶方案: Partial Conv
    print("\n[2] 进阶方案: PartialConvUNet")
    pconv = create_model("partial_conv")
    print(f"    参数量: {count_parameters(pconv):,}")
    out = pconv(x, mask)
    print(f"    输入: {x.shape} + mask -> 输出: {out.shape}")

    # 测试高级方案: PatchGAN
    print("\n[3] 高级方案: PatchGAN")
    gen, disc = create_model("patchgan")
    print(f"    Generator 参数量: {count_parameters(gen):,}")
    print(f"    Discriminator 参数量: {count_parameters(disc):,}")
    g_out = gen(x)
    d_out = disc(g_out)
    print(f"    Generator: {x.shape} -> {g_out.shape}")
    print(f"    Discriminator: {g_out.shape} -> {d_out.shape}")

    # 测试注意力增强方案: AttGAN
    print("\n[4] 注意力增强方案: AttGAN (AttentionUNet + PatchDiscriminator)")
    att_gen, att_disc = create_model("attgan")
    print(f"    Generator 参数量: {count_parameters(att_gen):,}")
    print(f"    Discriminator 参数量: {count_parameters(att_disc):,}")
    att_out = att_gen(x)
    att_d_out = att_disc(att_out)
    print(f"    Generator: {x.shape} -> {att_out.shape}")
    print(f"    Discriminator: {att_out.shape} -> {att_d_out.shape}")

    # 测试自监督预训练方案: MAE-PatchGAN
    print("\n[5] 自监督预训练方案: MAE-PatchGAN (InpaintingUNet + PatchDiscriminator)")
    mae_gen, mae_disc = create_model("mae_patchgan")
    print(f"    Generator 参数量: {count_parameters(mae_gen):,}")
    print(f"    Discriminator 参数量: {count_parameters(mae_disc):,}")
    mae_out = mae_gen(x)
    mae_d_out = mae_disc(mae_out)
    print(f"    Generator: {x.shape} -> {mae_out.shape}")
    print(f"    Discriminator: {mae_out.shape} -> {mae_d_out.shape}")
    print(f"    (注: 训练时需加载 MAE 预训练权重到 Generator encoder)")

    # 测试扩散模型方案: DDPM
    print("\n[6] 扩散模型方案: DDPM (DiffusionUNet)")
    ddpm = create_model("ddpm")
    print(f"    参数量: {count_parameters(ddpm):,}")
    t = torch.randint(0, 1000, (2,))  # 随机时间步
    ddpm_out = ddpm(x, t)
    print(f"    输入: {x.shape} + t={t.tolist()} -> 输出: {ddpm_out.shape}")

    print("\n" + "=" * 60)
    print("✅ 所有 6 种网络架构测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    test_network()

