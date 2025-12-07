#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数学运算模块

包含:
- 归一化函数
- 裁剪函数
- 矩阵运算
- 形态学操作
"""

from typing import Tuple, Optional

import numpy as np

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


def normalize_ct(
    data: np.ndarray,
    min_hu: float = -1000,
    max_hu: float = 400,
    clip: bool = True
) -> np.ndarray:
    """
    归一化 CT 数据到 [0, 1] 范围
    
    Args:
        data: CT 体数据 (HU 单位)
        min_hu: 最小 HU 值
        max_hu: 最大 HU 值
        clip: 是否裁剪超出范围的值
        
    Returns:
        normalized: 归一化后的数据
    """
    if clip:
        data = np.clip(data, min_hu, max_hu)
    
    normalized = (data - min_hu) / (max_hu - min_hu)
    return normalized.astype(np.float32)


def denormalize_ct(
    data: np.ndarray,
    min_hu: float = -1000,
    max_hu: float = 400
) -> np.ndarray:
    """
    将 [0, 1] 归一化数据恢复到 HU 单位
    
    Args:
        data: 归一化数据
        min_hu: 最小 HU 值
        max_hu: 最大 HU 值
        
    Returns:
        hu_data: HU 单位数据
    """
    return data * (max_hu - min_hu) + min_hu


def crop_to_bbox(
    data: np.ndarray,
    mask: np.ndarray,
    margin: int = 10
) -> Tuple[np.ndarray, np.ndarray, Tuple[slice, ...]]:
    """
    根据 mask 裁剪数据到最小包围盒
    
    Args:
        data: 3D 体数据
        mask: 二值 mask
        margin: 边缘留白
        
    Returns:
        cropped_data: 裁剪后的数据
        cropped_mask: 裁剪后的 mask
        slices: 裁剪使用的切片对象
    """
    # 找到非零区域
    nonzero = np.where(mask > 0)
    
    slices = []
    for i in range(3):
        min_idx = max(0, nonzero[i].min() - margin)
        max_idx = min(data.shape[i], nonzero[i].max() + margin + 1)
        slices.append(slice(min_idx, max_idx))
    
    slices = tuple(slices)
    
    return data[slices], mask[slices], slices


def pad_to_shape(
    data: np.ndarray,
    target_shape: Tuple[int, int, int],
    pad_value: float = 0
) -> np.ndarray:
    """
    将数据填充到指定形状
    
    Args:
        data: 输入数据
        target_shape: 目标形状
        pad_value: 填充值
        
    Returns:
        padded: 填充后的数据
    """
    current_shape = data.shape
    
    pad_width = []
    for curr, target in zip(current_shape, target_shape):
        total_pad = max(0, target - curr)
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_width.append((pad_before, pad_after))
    
    return np.pad(data, pad_width, mode='constant', constant_values=pad_value)


def resample_volume(
    data: np.ndarray,
    current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    order: int = 1
) -> np.ndarray:
    """
    重采样体数据到新的体素间距
    
    Args:
        data: 输入体数据
        current_spacing: 当前体素间距
        target_spacing: 目标体素间距
        order: 插值阶数 (0=最近邻, 1=线性, 3=立方)
        
    Returns:
        resampled: 重采样后的数据
    """
    if ndimage is None:
        raise ImportError("请安装 scipy: pip install scipy")
    
    resize_factor = [c / t for c, t in zip(current_spacing, target_spacing)]
    new_shape = [int(s * f) for s, f in zip(data.shape, resize_factor)]
    
    return ndimage.zoom(data, resize_factor, order=order)


def binary_dilate(
    mask: np.ndarray,
    iterations: int = 1,
    structure: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    二值膨胀操作
    
    Args:
        mask: 二值 mask
        iterations: 迭代次数
        structure: 结构元素
        
    Returns:
        dilated: 膨胀后的 mask
    """
    if ndimage is None:
        raise ImportError("请安装 scipy: pip install scipy")
    
    return ndimage.binary_dilation(
        mask, structure=structure, iterations=iterations
    ).astype(mask.dtype)


def binary_erode(
    mask: np.ndarray,
    iterations: int = 1,
    structure: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    二值腐蚀操作
    """
    if ndimage is None:
        raise ImportError("请安装 scipy: pip install scipy")
    
    return ndimage.binary_erosion(
        mask, structure=structure, iterations=iterations
    ).astype(mask.dtype)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    填充 mask 中的孔洞
    """
    if ndimage is None:
        raise ImportError("请安装 scipy: pip install scipy")
    
    return ndimage.binary_fill_holes(mask).astype(mask.dtype)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    保留最大连通分量
    
    Args:
        mask: 二值 mask
        
    Returns:
        largest: 只包含最大连通分量的 mask
    """
    if ndimage is None:
        raise ImportError("请安装 scipy: pip install scipy")
    
    labeled, num_features = ndimage.label(mask)
    
    if num_features == 0:
        return mask
    
    # 找到最大连通分量
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(component_sizes) + 1
    
    return (labeled == largest_label).astype(mask.dtype)

