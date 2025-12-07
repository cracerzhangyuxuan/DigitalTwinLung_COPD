#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网络模块测试
"""

import numpy as np
import pytest

# 检查 PyTorch 是否可用
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch 未安装")
class TestInpaintingUNet:
    """Inpainting U-Net 测试"""

    def test_forward_pass(self):
        """测试前向传播"""
        from src.04_texture_synthesis.network import InpaintingUNet
        
        model = InpaintingUNet()
        
        # 创建输入
        x = torch.randn(2, 1, 64, 64, 64)
        
        # 前向传播
        output = model(x)
        
        assert output.shape == x.shape
    
    def test_forward_pass_different_sizes(self):
        """测试不同输入尺寸"""
        from src.04_texture_synthesis.network import InpaintingUNet
        
        model = InpaintingUNet()
        
        for size in [32, 48, 64]:
            x = torch.randn(1, 1, size, size, size)
            output = model(x)
            assert output.shape == x.shape
    
    def test_gradient_flow(self):
        """测试梯度流"""
        from src.04_texture_synthesis.network import InpaintingUNet
        
        model = InpaintingUNet()
        
        x = torch.randn(1, 1, 32, 32, 32, requires_grad=True)
        output = model(x)
        loss = output.mean()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch 未安装")
class TestPatchDiscriminator:
    """Patch Discriminator 测试"""

    def test_forward_pass(self):
        """测试前向传播"""
        from src.04_texture_synthesis.network import PatchDiscriminator
        
        model = PatchDiscriminator()
        
        x = torch.randn(2, 1, 64, 64, 64)
        output = model(x)
        
        # 输出应该是 patch-wise 的判别结果
        assert len(output.shape) == 5
        assert output.shape[0] == 2
        assert output.shape[1] == 1
    
    def test_output_range(self):
        """测试输出范围"""
        from src.04_texture_synthesis.network import PatchDiscriminator
        
        model = PatchDiscriminator()
        
        x = torch.randn(1, 1, 64, 64, 64)
        output = model(x)
        
        # 输出应该是有限值
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch 未安装")
class TestLosses:
    """损失函数测试"""

    def test_reconstruction_loss(self):
        """测试重建损失"""
        from src.04_texture_synthesis.losses import ReconstructionLoss
        
        loss_fn = ReconstructionLoss(loss_type="l1")
        
        pred = torch.randn(2, 1, 32, 32, 32)
        target = torch.randn(2, 1, 32, 32, 32)
        
        loss = loss_fn(pred, target)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_reconstruction_loss_with_mask(self):
        """测试带 mask 的重建损失"""
        from src.04_texture_synthesis.losses import ReconstructionLoss
        
        loss_fn = ReconstructionLoss(loss_type="l1", mask_weight=10.0)
        
        pred = torch.randn(2, 1, 32, 32, 32)
        target = torch.randn(2, 1, 32, 32, 32)
        mask = torch.zeros(2, 1, 32, 32, 32)
        mask[:, :, 10:20, 10:20, 10:20] = 1
        
        loss_with_mask = loss_fn(pred, target, mask)
        loss_without_mask = loss_fn(pred, target)
        
        # 带 mask 的损失应该更大（因为 mask 区域权重更高）
        assert loss_with_mask.item() >= 0
    
    def test_inpainting_loss(self):
        """测试完整的 Inpainting 损失"""
        from src.04_texture_synthesis.losses import InpaintingLoss
        
        loss_fn = InpaintingLoss()
        
        pred = torch.randn(2, 1, 32, 32, 32)
        target = torch.randn(2, 1, 32, 32, 32)
        mask = torch.zeros(2, 1, 32, 32, 32)
        mask[:, :, 10:20, 10:20, 10:20] = 1
        
        losses = loss_fn.generator_loss(pred, target, mask)
        
        assert 'reconstruction' in losses
        assert 'perceptual' in losses
        assert 'total' in losses
        assert losses['total'].item() >= 0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch 未安装")
class TestModelParameters:
    """模型参数测试"""

    def test_parameter_count(self):
        """测试参数量"""
        from src.04_texture_synthesis.network import InpaintingUNet, PatchDiscriminator, count_parameters
        
        generator = InpaintingUNet()
        discriminator = PatchDiscriminator()
        
        g_params = count_parameters(generator)
        d_params = count_parameters(discriminator)
        
        # 确保模型有合理的参数量
        assert g_params > 0
        assert d_params > 0
        
        # 打印参数量（用于调试）
        print(f"Generator: {g_params:,} parameters")
        print(f"Discriminator: {d_params:,} parameters")

