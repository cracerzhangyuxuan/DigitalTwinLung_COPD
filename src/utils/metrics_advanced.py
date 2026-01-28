# -*- coding: utf-8 -*-
"""
高级纹理评估指标模块

用于评估 AI Inpainting 与 Direct Warping 方法的纹理质量差异。

指标说明：
- Sharpness (清晰度): 使用拉普拉斯方差，值越高表示纹理越清晰
- Boundary Continuity (边界连续性): 边界梯度，值越低表示融合越平滑
- GLCM Features (灰度共生矩阵特征): 纹理统计特征，用于评估纹理真实性
"""

import numpy as np
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops


def compute_sharpness(image_slice: np.ndarray, mask_slice: np.ndarray) -> float:
    """
    使用拉普拉斯方差计算纹理清晰度。
    
    拉普拉斯算子突出快速强度变化（边缘/细节）。
    方差越高 = 高频细节越多 = 纹理越清晰。
    
    Args:
        image_slice: 2D numpy 数组（单个 CT 切片，HU 值）
        mask_slice: 2D 二值 mask（病灶区域，与 image_slice 形状相同）
    
    Returns:
        float: mask 区域内的拉普拉斯方差。值越高越清晰。
               如果 mask 为空则返回 0.0。
    
    预期行为：
        - AI Fused 应该比 Direct Warp 有更高的清晰度（更少模糊）
        - Real COPD 应该有最高的清晰度（真实参考）
    """
    if np.sum(mask_slice) == 0:
        return 0.0
    
    laplacian = ndimage.laplace(image_slice)
    focus_measure = np.var(laplacian[mask_slice > 0])
    return float(focus_measure)


def compute_boundary_continuity(image: np.ndarray, mask: np.ndarray, dilation_iter: int = 3) -> float:
    """
    计算病灶边界处的平均梯度幅值。
    
    检测病灶与健康组织交界处的"接缝"或不连续性。
    边界梯度高 = 可见接缝/伪影。
    
    Args:
        image: 2D 或 3D numpy 数组（CT 图像，HU 值）
        mask: 二值 mask（与 image 形状相同，病灶区域）
        dilation_iter: 形态学操作的迭代次数，定义边界宽度
    
    Returns:
        float: 边界区域的平均梯度幅值。值越低融合越平滑。
               如果边界区域为空则返回 0.0。
    
    预期行为：
        - AI Fused 应该比 Direct Warp 有更低的边界梯度（更平滑的融合）
        - Direct Warp 通常因插值伪影而显示高梯度
    """
    struct = ndimage.generate_binary_structure(image.ndim, 1)
    dilated = ndimage.binary_dilation(mask > 0, structure=struct, iterations=dilation_iter)
    eroded = ndimage.binary_erosion(mask > 0, structure=struct, iterations=dilation_iter)
    boundary_region = dilated ^ eroded  # XOR 获取边界带
    
    if np.sum(boundary_region) == 0:
        return 0.0
    
    gradients = np.gradient(image)
    gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
    
    mean_boundary_grad = np.mean(gradient_magnitude[boundary_region])
    return float(mean_boundary_grad)


def compute_glcm_features(image_slice: np.ndarray, mask_slice: np.ndarray) -> dict:
    """
    计算扩展的灰度共生矩阵 (GLCM) 纹理特征，用于放射组学分析。

    GLCM 捕获像素强度之间的空间关系，提供医学影像中常用的放射组学纹理统计。

    Args:
        image_slice: 2D numpy 数组（单个 CT 切片，HU 值，预期范围约 -1000 到 0）
        mask_slice: 2D 二值 mask（病灶区域）

    Returns:
        dict: {
            'glcm_contrast': float,     # 局部强度变化 (模糊时↓)
            'glcm_energy': float,       # 纹理均匀性 (均匀时↑)
            'glcm_entropy': float,      # 纹理随机性/复杂度 (模糊时↓)
            'glcm_correlation': float,  # 灰度级线性依赖性
            'glcm_homogeneity': float   # 分布接近对角线的程度
        }
        如果 mask 为空或无效则返回所有值为 0.0 的字典。

    预期行为：
        - AI Fused 的 GLCM 特征应该与 Real COPD 相似（纹理真实性）
        - Direct Warp 可能因模糊而显示不同的 GLCM（对比度低，熵低）
    """
    default_result = {
        'glcm_contrast': 0.0,
        'glcm_energy': 0.0,
        'glcm_entropy': 0.0,
        'glcm_correlation': 0.0,
        'glcm_homogeneity': 0.0
    }

    if np.sum(mask_slice) == 0:
        return default_result

    # 提取病灶区域的边界框用于 GLCM 计算
    coords = np.argwhere(mask_slice > 0)
    if len(coords) == 0:
        return default_result

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    crop = image_slice[y_min:y_max+1, x_min:x_max+1]

    # 检查退化情况
    if crop.size == 0 or crop.max() == crop.min():
        return default_result

    # 量化为 32 个灰度级 (0-31) 以提高 GLCM 效率
    # 假设肺窗：-1000 到 0 HU
    crop_normalized = (crop - (-1000)) / (0 - (-1000))  # 归一化到 0-1
    crop_int = (crop_normalized * 31).astype(np.uint8)
    crop_int = np.clip(crop_int, 0, 31)

    # 在 4 个角度计算 GLCM，距离=1
    glcm = graycomatrix(
        crop_int,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=32,
        symmetric=True,
        normed=True
    )

    # 计算标准 GLCM 属性
    contrast = graycoprops(glcm, 'contrast').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()

    # 手动计算熵 (graycoprops 不提供)
    # Entropy = -sum(P * log2(P))，其中 P 是归一化的 GLCM
    glcm_normalized = glcm / (glcm.sum() + 1e-10)
    entropy = -np.sum(glcm_normalized * np.log2(glcm_normalized + 1e-10))

    return {
        'glcm_contrast': float(contrast),
        'glcm_energy': float(energy),
        'glcm_entropy': float(entropy),
        'glcm_correlation': float(correlation),
        'glcm_homogeneity': float(homogeneity)
    }

