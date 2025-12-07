#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估指标模块

包含:
- 图像质量指标: SSIM, PSNR, MSE, NCC
- 分割质量指标: Dice, HD95, 体积相似性
- 生成模型指标: FID (需额外库支持)
"""

from typing import Tuple, Optional, Union

import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import mean_squared_error as mse
except ImportError:
    ssim = psnr = mse = None

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    distance_transform_edt = None


def compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    data_range: Optional[float] = None,
    win_size: int = 7
) -> float:
    """
    计算结构相似性指数 (SSIM)
    
    Args:
        img1: 第一张图像
        img2: 第二张图像
        data_range: 数据范围，None 时自动计算
        win_size: 滑动窗口大小
        
    Returns:
        ssim_value: SSIM 值 [0, 1]
    """
    if ssim is None:
        raise ImportError("请安装 scikit-image: pip install scikit-image")
    
    if data_range is None:
        data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
    
    return float(ssim(img1, img2, data_range=data_range, win_size=win_size))


def compute_psnr(
    img1: np.ndarray,
    img2: np.ndarray,
    data_range: Optional[float] = None
) -> float:
    """
    计算峰值信噪比 (PSNR)
    
    Args:
        img1: 参考图像
        img2: 测试图像
        data_range: 数据范围
        
    Returns:
        psnr_value: PSNR 值 (dB)
    """
    if psnr is None:
        raise ImportError("请安装 scikit-image: pip install scikit-image")
    
    if data_range is None:
        data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
    
    return float(psnr(img1, img2, data_range=data_range))


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算均方误差 (MSE)
    
    Args:
        img1: 第一张图像
        img2: 第二张图像
        
    Returns:
        mse_value: MSE 值
    """
    return float(np.mean((img1 - img2) ** 2))


def compute_ncc(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算归一化互相关 (NCC)
    
    Args:
        img1: 第一张图像
        img2: 第二张图像
        
    Returns:
        ncc_value: NCC 值 [-1, 1]
    """
    img1_norm = img1 - img1.mean()
    img2_norm = img2 - img2.mean()
    
    numerator = np.sum(img1_norm * img2_norm)
    denominator = np.sqrt(np.sum(img1_norm ** 2) * np.sum(img2_norm ** 2))
    
    if denominator == 0:
        return 0.0
    
    return float(numerator / denominator)


def compute_dice(
    mask1: np.ndarray,
    mask2: np.ndarray,
    smooth: float = 1e-6
) -> float:
    """
    计算 Dice 系数
    
    Args:
        mask1: 第一个二值 mask
        mask2: 第二个二值 mask
        smooth: 平滑项，防止除零
        
    Returns:
        dice: Dice 系数 [0, 1]
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1) + np.sum(mask2)
    
    return float((2.0 * intersection + smooth) / (union + smooth))


def compute_hd95(
    mask1: np.ndarray,
    mask2: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> float:
    """
    计算 95% Hausdorff 距离 (HD95)
    
    Args:
        mask1: 第一个二值 mask
        mask2: 第二个二值 mask
        voxel_spacing: 体素间距 (mm)
        
    Returns:
        hd95: 95% Hausdorff 距离 (mm)
    """
    if distance_transform_edt is None:
        raise ImportError("请安装 scipy: pip install scipy")
    
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    # 边界提取
    from scipy.ndimage import binary_erosion
    border1 = mask1 ^ binary_erosion(mask1)
    border2 = mask2 ^ binary_erosion(mask2)
    
    # 距离变换
    dt1 = distance_transform_edt(~mask1, sampling=voxel_spacing)
    dt2 = distance_transform_edt(~mask2, sampling=voxel_spacing)
    
    # 获取边界点到另一个 mask 的距离
    dist1_to_2 = dt2[border1]
    dist2_to_1 = dt1[border2]
    
    if len(dist1_to_2) == 0 or len(dist2_to_1) == 0:
        return float('inf')
    
    # 95% 分位数
    hd95_1 = np.percentile(dist1_to_2, 95)
    hd95_2 = np.percentile(dist2_to_1, 95)
    
    return float(max(hd95_1, hd95_2))


def compute_volume_similarity(
    mask1: np.ndarray,
    mask2: np.ndarray
) -> float:
    """
    计算体积相似性
    
    Args:
        mask1: 第一个二值 mask
        mask2: 第二个二值 mask
        
    Returns:
        vs: 体积相似性 [0, 1]
    """
    v1 = np.sum(mask1 > 0)
    v2 = np.sum(mask2 > 0)
    
    if v1 + v2 == 0:
        return 1.0
    
    return float(1.0 - abs(v1 - v2) / (v1 + v2))


def evaluate_all(
    pred: np.ndarray,
    target: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    target_mask: Optional[np.ndarray] = None,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> dict:
    """
    计算所有评估指标
    
    Args:
        pred: 预测图像
        target: 目标图像
        pred_mask: 预测 mask (可选)
        target_mask: 目标 mask (可选)
        voxel_spacing: 体素间距
        
    Returns:
        metrics: 所有指标的字典
    """
    results = {
        'SSIM': compute_ssim(pred, target),
        'PSNR': compute_psnr(pred, target),
        'MSE': compute_mse(pred, target),
        'NCC': compute_ncc(pred, target),
    }
    
    if pred_mask is not None and target_mask is not None:
        results['Dice'] = compute_dice(pred_mask, target_mask)
        results['HD95'] = compute_hd95(pred_mask, target_mask, voxel_spacing)
        results['VS'] = compute_volume_similarity(pred_mask, target_mask)
    
    return results

