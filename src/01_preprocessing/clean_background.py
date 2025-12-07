#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
背景清洗模块

去除 CT 中的骨骼、气管等非肺部区域，保留纯净肺部
"""

from pathlib import Path
from typing import Union, Tuple

import numpy as np

from ..utils.logger import get_logger
from ..utils.io import load_nifti, save_nifti

logger = get_logger(__name__)


def clean_background(
    ct_data: np.ndarray,
    lung_mask: np.ndarray,
    background_hu: float = -1000
) -> np.ndarray:
    """
    使用肺部 mask 清洗 CT 背景
    
    Args:
        ct_data: CT 体数据 (HU 单位)
        lung_mask: 肺部二值 mask
        background_hu: 背景替换 HU 值 (默认 -1000，空气)
        
    Returns:
        cleaned: 清洗后的 CT 数据
    """
    cleaned = ct_data.copy()
    
    # 将非肺部区域替换为背景值
    cleaned[lung_mask == 0] = background_hu
    
    return cleaned


def replace_background(
    input_ct: Union[str, Path],
    input_mask: Union[str, Path],
    output_path: Union[str, Path],
    background_hu: float = -1000
) -> None:
    """
    从文件读取并清洗背景，保存结果
    
    Args:
        input_ct: 输入 CT 文件路径
        input_mask: 输入肺部 mask 文件路径
        output_path: 输出文件路径
        background_hu: 背景 HU 值
    """
    input_ct = Path(input_ct)
    input_mask = Path(input_mask)
    output_path = Path(output_path)
    
    logger.info(f"清洗背景: {input_ct.name}")
    
    # 加载数据
    ct_data, affine = load_nifti(input_ct, return_affine=True)
    lung_mask = load_nifti(input_mask)
    
    # 清洗
    cleaned = clean_background(ct_data, lung_mask, background_hu)
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(cleaned, output_path, affine=affine)
    
    # 统计
    lung_volume = np.sum(lung_mask > 0)
    total_volume = lung_mask.size
    logger.info(
        f"清洗完成: 肺部体素 {lung_volume} / {total_volume} "
        f"({lung_volume/total_volume*100:.1f}%)"
    )


def batch_clean_background(
    ct_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    output_dir: Union[str, Path],
    ct_pattern: str = "*.nii.gz",
    mask_suffix: str = "_mask",
    background_hu: float = -1000
) -> int:
    """
    批量清洗背景
    
    Args:
        ct_dir: CT 文件目录
        mask_dir: Mask 文件目录
        output_dir: 输出目录
        ct_pattern: CT 文件匹配模式
        mask_suffix: Mask 文件后缀
        background_hu: 背景 HU 值
        
    Returns:
        count: 成功处理的文件数
    """
    ct_dir = Path(ct_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ct_files = list(ct_dir.glob(ct_pattern))
    logger.info(f"找到 {len(ct_files)} 个 CT 文件")
    
    count = 0
    for ct_path in ct_files:
        # 构建对应的 mask 路径
        stem = ct_path.stem.replace('.nii', '')
        mask_path = mask_dir / f"{stem}{mask_suffix}.nii.gz"
        
        if not mask_path.exists():
            # 尝试其他可能的命名方式
            mask_path = mask_dir / ct_path.name
            if not mask_path.exists():
                logger.warning(f"未找到对应 mask: {ct_path.name}")
                continue
        
        output_path = output_dir / f"{stem}_clean.nii.gz"
        
        try:
            replace_background(ct_path, mask_path, output_path, background_hu)
            count += 1
        except Exception as e:
            logger.error(f"处理失败 {ct_path.name}: {e}")
    
    logger.info(f"批量清洗完成: {count}/{len(ct_files)} 成功")
    return count


def validate_cleaned_data(
    cleaned_path: Union[str, Path],
    expected_background: float = -1000,
    tolerance: float = 10
) -> Tuple[bool, dict]:
    """
    验证清洗后的数据质量
    
    Args:
        cleaned_path: 清洗后的文件路径
        expected_background: 期望的背景值
        tolerance: 容差
        
    Returns:
        valid: 是否有效
        stats: 统计信息
    """
    data = load_nifti(cleaned_path)
    
    # 统计
    stats = {
        'min': float(data.min()),
        'max': float(data.max()),
        'mean': float(data.mean()),
        'std': float(data.std()),
    }
    
    # 检查背景值
    background_mask = data < (expected_background + tolerance)
    stats['background_ratio'] = float(background_mask.sum() / data.size)
    
    # 检查是否有有效的肺部区域
    lung_region = (data > -950) & (data < -200)
    stats['lung_ratio'] = float(lung_region.sum() / data.size)
    
    valid = stats['lung_ratio'] > 0.05  # 至少 5% 是肺部区域
    
    return valid, stats


def main(config: dict) -> None:
    """
    主函数
    """
    # TODO: 根据配置实现批量清洗流程
    pass


if __name__ == "__main__":
    import yaml
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)

