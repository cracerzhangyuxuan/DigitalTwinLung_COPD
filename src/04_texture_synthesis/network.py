#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网络架构模块

定义 Inpainting U-Net 和 Patch Discriminator
"""

from typing import List, Tuple

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
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_channels = feat
        
        # 最后一层
        layers.append(nn.Conv3d(prev_channels, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_network():
    """测试网络"""
    if torch is None:
        print("PyTorch 未安装")
        return
    
    # 创建模型
    generator = InpaintingUNet()
    discriminator = PatchDiscriminator()
    
    print(f"Generator 参数量: {count_parameters(generator):,}")
    print(f"Discriminator 参数量: {count_parameters(discriminator):,}")
    
    # 测试前向传播
    x = torch.randn(2, 1, 64, 64, 64)
    
    g_out = generator(x)
    print(f"Generator 输出形状: {g_out.shape}")
    
    d_out = discriminator(x)
    print(f"Discriminator 输出形状: {d_out.shape}")


if __name__ == "__main__":
    test_network()

