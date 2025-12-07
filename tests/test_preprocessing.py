#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预处理模块测试
"""

import numpy as np
import pytest


class TestCleanBackground:
    """背景清洗测试"""

    def test_clean_background(self, sample_ct_data, sample_lung_mask):
        """测试背景清洗"""
        from src.01_preprocessing.clean_background import clean_background
        
        cleaned = clean_background(sample_ct_data, sample_lung_mask, background_hu=-1000)
        
        # 非肺部区域应该是 -1000
        assert np.all(cleaned[sample_lung_mask == 0] == -1000)
        
        # 肺部区域应该保持不变
        np.testing.assert_array_equal(
            cleaned[sample_lung_mask > 0],
            sample_ct_data[sample_lung_mask > 0]
        )
    
    def test_clean_background_custom_value(self, sample_ct_data, sample_lung_mask):
        """测试自定义背景值"""
        from src.01_preprocessing.clean_background import clean_background
        
        cleaned = clean_background(sample_ct_data, sample_lung_mask, background_hu=-500)
        
        assert np.all(cleaned[sample_lung_mask == 0] == -500)


class TestExtractEmphysema:
    """肺气肿提取测试"""

    def test_compute_laa950(self, sample_ct_data, sample_lung_mask):
        """测试 LAA-950 计算"""
        from src.01_preprocessing.extract_emphysema import compute_laa950
        
        # 修改数据，添加一些低密度区域
        ct_with_emphysema = sample_ct_data.copy()
        ct_with_emphysema[30:35, 30:35, 30:35] = -960
        
        mask, laa_percentage = compute_laa950(ct_with_emphysema, sample_lung_mask, threshold=-950)
        
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.uint8
        assert 0 <= laa_percentage <= 100
        
        # 应该检测到低密度区域
        assert np.sum(mask) > 0
    
    def test_compute_laa950_no_emphysema(self, sample_ct_data, sample_lung_mask):
        """测试无肺气肿情况"""
        from src.01_preprocessing.extract_emphysema import compute_laa950
        
        # 确保没有低于 -950 的区域
        ct_normal = np.clip(sample_ct_data, -900, 0)
        
        mask, laa_percentage = compute_laa950(ct_normal, sample_lung_mask, threshold=-950)
        
        assert np.sum(mask) == 0
        assert laa_percentage == 0.0
    
    def test_classify_severity(self):
        """测试严重程度分类"""
        from src.01_preprocessing.extract_emphysema import classify_emphysema_severity
        
        assert classify_emphysema_severity(3.0) == "正常"
        assert classify_emphysema_severity(10.0) == "轻度"
        assert classify_emphysema_severity(20.0) == "中度"
        assert classify_emphysema_severity(30.0) == "重度"

