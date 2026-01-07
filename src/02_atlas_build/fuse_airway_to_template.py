#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
气管树融合模块 (Airway Fusion)

将气管树的 CT 强度信息融合到标准模板中，生成完整的数字肺底座。

功能:
    - 读取标准模板和气管树 mask
    - 将气管树区域设置为合理的 HU 值（约 -995）
    - 边界平滑处理，避免生硬边界
    - 生成融合后的完整模板

使用方法:
    python -m src.02_atlas_build.fuse_airway_to_template
    
    或指定路径:
    python -m src.02_atlas_build.fuse_airway_to_template \
        --template data/02_atlas/standard_template.nii.gz \
        --trachea data/02_atlas/standard_trachea_mask.nii.gz \
        --output data/02_atlas/standard_template_with_airway.nii.gz

作者: DigitalTwinLung_COPD Team
日期: 2025-12-31
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import argparse

import numpy as np

try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
except ImportError:
    ndimage = None
    gaussian_filter = None
    binary_dilation = None
    binary_erosion = None

from ..utils.logger import get_logger
from ..utils.io import load_nifti, save_nifti

logger = get_logger(__name__)


def create_boundary_mask(
    binary_mask: np.ndarray,
    width: int = 2
) -> np.ndarray:
    """
    创建边界区域 mask
    
    Args:
        binary_mask: 二值 mask
        width: 边界宽度（体素数）
        
    Returns:
        boundary_mask: 边界区域 mask（布尔型）
    """
    if binary_dilation is None or binary_erosion is None:
        logger.warning("scipy.ndimage 不可用，跳过边界检测")
        return np.zeros_like(binary_mask, dtype=bool)
    
    # 膨胀
    dilated = binary_dilation(binary_mask, iterations=width)
    # 腐蚀
    eroded = binary_erosion(binary_mask, iterations=width)
    
    # 边界 = 膨胀 - 腐蚀
    boundary = dilated & (~eroded)
    
    return boundary


def fuse_airway_to_template(
    template_path: Union[str, Path],
    trachea_mask_path: Union[str, Path],
    output_path: Union[str, Path],
    airway_hu: float = -995.0,
    boundary_sigma: float = 0.8,
    boundary_width: int = 2,
    preserve_existing_low_hu: bool = True
) -> Path:
    """
    将气管树 CT 强度信息融合到标准模板
    
    算法流程:
    1. 加载模板和气管树 mask
    2. 对气管树 mask 进行边界平滑
    3. 将气管树核心区域设置为 airway_hu
    4. 在边界区域使用加权混合实现羽化过渡
    5. 可选：保留模板中已有的低 HU 区域
    
    Args:
        template_path: 标准模板路径
        trachea_mask_path: 气管树 mask 路径
        output_path: 输出路径
        airway_hu: 气道 HU 值（默认 -995，真实气道约 -1000 到 -950）
        boundary_sigma: 边界平滑高斯 sigma
        boundary_width: 边界羽化宽度（体素数）
        preserve_existing_low_hu: 是否保留模板中已有的低 HU 区域
        
    Returns:
        output_path: 生成的融合模板路径
        
    Note:
        - 真实 CT 中，气道内部是空气，HU 值约 -1000
        - 气道壁是软组织，HU 值约 -300 到 0
        - 我们设置 -995 作为气道内部的 HU 值，稍高于纯空气
    """
    template_path = Path(template_path)
    trachea_mask_path = Path(trachea_mask_path)
    output_path = Path(output_path)
    
    logger.info("=" * 60)
    logger.info("气管树融合到标准模板")
    logger.info("=" * 60)
    logger.info(f"  模板: {template_path}")
    logger.info(f"  气管树 mask: {trachea_mask_path}")
    logger.info(f"  输出: {output_path}")
    logger.info(f"  参数:")
    logger.info(f"    - 气道 HU 值: {airway_hu}")
    logger.info(f"    - 边界平滑 sigma: {boundary_sigma}")
    logger.info(f"    - 边界宽度: {boundary_width}")
    
    # 检查文件存在
    if not template_path.exists():
        raise FileNotFoundError(f"模板文件不存在: {template_path}")
    if not trachea_mask_path.exists():
        raise FileNotFoundError(f"气管树 mask 不存在: {trachea_mask_path}")
    
    # 加载数据
    logger.info("加载数据...")
    template_data, affine = load_nifti(template_path, return_affine=True)
    trachea_mask = load_nifti(trachea_mask_path)
    
    # 验证形状匹配
    if template_data.shape != trachea_mask.shape:
        raise ValueError(
            f"形状不匹配！模板: {template_data.shape}, "
            f"气管 mask: {trachea_mask.shape}"
        )
    
    logger.info(f"  模板形状: {template_data.shape}")
    logger.info(f"  模板 HU 范围: [{template_data.min():.0f}, {template_data.max():.0f}]")
    
    # 二值化气管 mask
    trachea_binary = (trachea_mask > 0).astype(np.uint8)
    trachea_voxels = np.sum(trachea_binary)
    logger.info(f"  气管树体素数: {trachea_voxels:,}")
    
    if trachea_voxels == 0:
        logger.warning("气管树 mask 为空！直接复制模板")
        save_nifti(template_data, output_path, affine=affine)
        return output_path
    
    # 创建融合后的模板
    fused_template = template_data.copy()
    
    # Step 1: 检查模板中气管区域的当前 HU 值
    current_airway_hu = template_data[trachea_binary > 0]
    logger.info(f"  模板中气管区域当前 HU 范围: "
                f"[{current_airway_hu.min():.0f}, {current_airway_hu.max():.0f}]")
    logger.info(f"  模板中气管区域平均 HU: {current_airway_hu.mean():.0f}")
    
    # Step 2: 创建平滑过渡的权重 mask
    if gaussian_filter is not None and boundary_sigma > 0:
        logger.info("创建边界平滑权重...")
        # 高斯平滑二值 mask，创建 0-1 过渡
        weight_mask = gaussian_filter(
            trachea_binary.astype(np.float32), 
            sigma=boundary_sigma
        )
        # 归一化到 0-1
        weight_mask = np.clip(weight_mask, 0, 1)
    else:
        weight_mask = trachea_binary.astype(np.float32)
    
    # Step 3: 加权融合
    # fused = original * (1 - weight) + airway_hu * weight
    logger.info("执行加权融合...")
    
    # 只在权重 > 0 的区域进行融合（节省计算）
    fusion_region = weight_mask > 0.01
    
    if preserve_existing_low_hu:
        # 保留已有的低 HU 区域（可能是模板中已经存在的气道信息）
        # 只在模板 HU 值高于 airway_hu 的地方进行替换
        should_replace = (template_data > airway_hu) & fusion_region
        fused_template[should_replace] = (
            template_data[should_replace] * (1 - weight_mask[should_replace]) +
            airway_hu * weight_mask[should_replace]
        )
        replaced_voxels = np.sum(should_replace)
    else:
        # 强制替换所有气管区域
        fused_template[fusion_region] = (
            template_data[fusion_region] * (1 - weight_mask[fusion_region]) +
            airway_hu * weight_mask[fusion_region]
        )
        replaced_voxels = np.sum(fusion_region)
    
    logger.info(f"  融合区域体素数: {replaced_voxels:,}")
    
    # Step 4: 验证融合结果
    fused_airway_hu = fused_template[trachea_binary > 0]
    logger.info(f"  融合后气管区域 HU 范围: "
                f"[{fused_airway_hu.min():.0f}, {fused_airway_hu.max():.0f}]")
    logger.info(f"  融合后气管区域平均 HU: {fused_airway_hu.mean():.0f}")
    
    # Step 5: 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(fused_template, output_path, affine=affine)
    
    logger.info("=" * 60)
    logger.info(f"✓ 融合完成: {output_path}")
    logger.info("=" * 60)
    
    return output_path


def verify_fusion(
    original_template_path: Union[str, Path],
    fused_template_path: Union[str, Path],
    trachea_mask_path: Union[str, Path],
    boundary_margin: int = 3
) -> dict:
    """
    验证融合结果

    Args:
        original_template_path: 原始模板路径
        fused_template_path: 融合后模板路径
        trachea_mask_path: 气管树 mask 路径
        boundary_margin: 边界检查时排除的体素数（因为羽化会影响边界）

    Returns:
        stats: 验证统计信息
    """
    logger.info("验证融合结果...")

    original = load_nifti(original_template_path)
    fused = load_nifti(fused_template_path)
    trachea_mask = load_nifti(trachea_mask_path) > 0

    # 创建扩展的气管区域（包含边界羽化区域）
    if binary_dilation is not None:
        extended_trachea = binary_dilation(trachea_mask, iterations=boundary_margin)
    else:
        extended_trachea = trachea_mask

    # 核心非气管区域（排除边界羽化区域）
    core_non_airway = ~extended_trachea

    # 计算核心非气管区域的变化
    core_diff = np.abs(original[core_non_airway] - fused[core_non_airway])
    max_core_diff = float(core_diff.max())
    mean_core_diff = float(core_diff.mean())

    stats = {
        'original_airway_mean_hu': float(original[trachea_mask].mean()),
        'fused_airway_mean_hu': float(fused[trachea_mask].mean()),
        'core_non_airway_unchanged': bool(max_core_diff < 0.01),  # 核心区域应完全不变
        'core_max_diff': max_core_diff,
        'core_mean_diff': mean_core_diff,
        'airway_hu_decreased': bool(
            fused[trachea_mask].mean() < original[trachea_mask].mean()
        ),
        'hu_reduction': float(
            original[trachea_mask].mean() - fused[trachea_mask].mean()
        )
    }

    logger.info(f"  原始模板气管区域平均 HU: {stats['original_airway_mean_hu']:.1f}")
    logger.info(f"  融合后模板气管区域平均 HU: {stats['fused_airway_mean_hu']:.1f}")
    logger.info(f"  气管区域 HU 降低值: {stats['hu_reduction']:.1f}")
    logger.info(f"  核心非气管区域未改变: {stats['core_non_airway_unchanged']} "
                f"(最大差异: {stats['core_max_diff']:.4f})")
    logger.info(f"  气管区域 HU 值已降低: {stats['airway_hu_decreased']}")
    
    if stats['airway_hu_decreased'] and stats['core_non_airway_unchanged']:
        logger.info("  ✓ 融合验证通过")
    else:
        if not stats['airway_hu_decreased']:
            logger.warning("  ⚠ 气管区域 HU 值未降低，请检查")
        if not stats['core_non_airway_unchanged']:
            logger.warning("  ⚠ 核心非气管区域发生变化，请检查")

    return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将气管树融合到标准模板')
    parser.add_argument(
        '--template', type=str,
        default='data/02_atlas/standard_template.nii.gz',
        help='标准模板路径'
    )
    parser.add_argument(
        '--trachea', type=str,
        default='data/02_atlas/standard_trachea_mask.nii.gz',
        help='气管树 mask 路径'
    )
    parser.add_argument(
        '--output', type=str,
        default='data/02_atlas/standard_template_with_airway.nii.gz',
        help='输出路径'
    )
    parser.add_argument(
        '--airway-hu', type=float, default=-995.0,
        help='气道 HU 值（默认 -995）'
    )
    parser.add_argument(
        '--boundary-sigma', type=float, default=0.8,
        help='边界平滑 sigma（默认 0.8）'
    )
    parser.add_argument(
        '--force-replace', action='store_true',
        help='强制替换所有气管区域（不保留已有的低 HU 值）'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='融合后执行验证'
    )

    args = parser.parse_args()

    # preserve_existing_low_hu 取反于 force_replace
    preserve_low_hu = not args.force_replace

    output_path = fuse_airway_to_template(
        template_path=args.template,
        trachea_mask_path=args.trachea,
        output_path=args.output,
        airway_hu=args.airway_hu,
        boundary_sigma=args.boundary_sigma,
        preserve_existing_low_hu=preserve_low_hu
    )
    
    if args.verify:
        verify_fusion(
            original_template_path=args.template,
            fused_template_path=output_path,
            trachea_mask_path=args.trachea
        )


if __name__ == "__main__":
    main()

