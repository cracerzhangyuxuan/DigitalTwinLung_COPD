#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估指标测试
"""

import numpy as np
import pytest


class TestImageMetrics:
    """图像质量指标测试"""
    
    def test_compute_ssim_identical(self):
        """测试相同图像的 SSIM"""
        from src.utils.metrics import compute_ssim
        
        img = np.random.rand(64, 64, 64).astype(np.float32)
        
        ssim = compute_ssim(img, img)
        
        assert ssim == pytest.approx(1.0, abs=0.01)
    
    def test_compute_ssim_different(self):
        """测试不同图像的 SSIM"""
        from src.utils.metrics import compute_ssim
        
        img1 = np.random.rand(64, 64, 64).astype(np.float32)
        img2 = np.random.rand(64, 64, 64).astype(np.float32)
        
        ssim = compute_ssim(img1, img2)
        
        assert 0 <= ssim <= 1
        assert ssim < 0.5  # 随机图像应该有较低的 SSIM
    
    def test_compute_psnr_identical(self):
        """测试相同图像的 PSNR"""
        from src.utils.metrics import compute_psnr
        
        img = np.random.rand(64, 64, 64).astype(np.float32)
        
        psnr = compute_psnr(img, img)
        
        # 相同图像的 PSNR 应该是无穷大或非常大
        assert psnr > 50 or np.isinf(psnr)
    
    def test_compute_psnr_different(self):
        """测试不同图像的 PSNR"""
        from src.utils.metrics import compute_psnr
        
        img1 = np.random.rand(64, 64, 64).astype(np.float32)
        img2 = img1 + np.random.randn(64, 64, 64).astype(np.float32) * 0.1
        
        psnr = compute_psnr(img1, img2)
        
        assert psnr > 0
        assert psnr < 50
    
    def test_compute_mse(self):
        """测试 MSE 计算"""
        from src.utils.metrics import compute_mse
        
        img1 = np.zeros((10, 10, 10), dtype=np.float32)
        img2 = np.ones((10, 10, 10), dtype=np.float32)
        
        mse = compute_mse(img1, img2)
        
        assert mse == pytest.approx(1.0, abs=0.001)


class TestSegmentationMetrics:
    """分割指标测试"""
    
    def test_compute_dice_identical(self):
        """测试相同 mask 的 Dice"""
        from src.utils.metrics import compute_dice
        
        mask = np.zeros((64, 64, 64), dtype=np.uint8)
        mask[20:40, 20:40, 20:40] = 1
        
        dice = compute_dice(mask, mask)
        
        assert dice == pytest.approx(1.0, abs=0.001)
    
    def test_compute_dice_no_overlap(self):
        """测试无重叠 mask 的 Dice"""
        from src.utils.metrics import compute_dice
        
        mask1 = np.zeros((64, 64, 64), dtype=np.uint8)
        mask1[10:20, 10:20, 10:20] = 1
        
        mask2 = np.zeros((64, 64, 64), dtype=np.uint8)
        mask2[40:50, 40:50, 40:50] = 1
        
        dice = compute_dice(mask1, mask2)
        
        assert dice == pytest.approx(0.0, abs=0.001)
    
    def test_compute_dice_partial_overlap(self):
        """测试部分重叠 mask 的 Dice"""
        from src.utils.metrics import compute_dice
        
        mask1 = np.zeros((64, 64, 64), dtype=np.uint8)
        mask1[20:40, 20:40, 20:40] = 1
        
        mask2 = np.zeros((64, 64, 64), dtype=np.uint8)
        mask2[30:50, 30:50, 30:50] = 1
        
        dice = compute_dice(mask1, mask2)
        
        assert 0 < dice < 1
    
    def test_volume_similarity(self):
        """测试体积相似度"""
        from src.utils.metrics import volume_similarity
        
        mask1 = np.zeros((64, 64, 64), dtype=np.uint8)
        mask1[20:40, 20:40, 20:40] = 1
        
        mask2 = np.zeros((64, 64, 64), dtype=np.uint8)
        mask2[20:40, 20:40, 20:40] = 1
        
        vs = volume_similarity(mask1, mask2)
        
        assert vs == pytest.approx(1.0, abs=0.001)


class TestEvaluateAll:
    """综合评估测试"""
    
    def test_evaluate_all(self):
        """测试综合评估函数"""
        from src.utils.metrics import evaluate_all
        
        pred = np.random.rand(64, 64, 64).astype(np.float32)
        target = pred + np.random.randn(64, 64, 64).astype(np.float32) * 0.1
        
        metrics = evaluate_all(pred, target)
        
        assert 'ssim' in metrics
        assert 'psnr' in metrics
        assert 'mse' in metrics
        
        assert 0 <= metrics['ssim'] <= 1
        assert metrics['psnr'] > 0
        assert metrics['mse'] >= 0

