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
        in_channels: 输入通道数
        out_channels: 输出通道数
        features: 特征通道列表

    Returns:
        model: 创建的模型
            - 对于 "patchgan"，返回 (InpaintingUNet, PatchDiscriminator) 元组
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

    else:
        raise ValueError(f"未知的模型类型: {model_type}. 支持: unet, partial_conv, patchgan")


def test_network():
    """测试网络"""
    if torch is None:
        print("PyTorch 未安装")
        return

    print("=" * 50)
    print("测试 Phase 3B 网络架构")
    print("=" * 50)

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

    print("\n" + "=" * 50)
    print("✅ 所有网络测试通过!")
    print("=" * 50)


if __name__ == "__main__":
    test_network()

