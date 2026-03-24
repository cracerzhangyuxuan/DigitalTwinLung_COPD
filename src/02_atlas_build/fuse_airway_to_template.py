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
    wall_hu: float = -400.0,
    wall_thickness: int = 1,
    boundary_sigma: float = 0.8,
    boundary_width: int = 2,
    preserve_existing_low_hu: bool = True
) -> Path:
    """
    将气管树 CT 强度信息融合到标准模板

    算法流程（2026-02 修复版）:
    1. 加载模板和气管树 mask
    2. 通过形态学腐蚀将 mask 分为「气道壁」和「气腔」两层
    3. 气道壁赋值 wall_hu (约 -400)，提供可视化对比度
    4. 气腔赋值 airway_hu (约 -995)，模拟管内空气
    5. 边界羽化平滑过渡到周围肺实质

    旧版 Bug 分析:
        模板中气管区域本身已是低 HU (平均 -981.6, 84% < -995)。
        旧算法用高斯权重把 -995 混合到已经是 -981 的区域，结果整片
        被压到 -1014 ~ -1024，与体外背景空气(-1024)完全混为一体，
        在 3D Slicer 中无法辨识。

    修复策略:
        真实 CT 中气管是「亮壁暗腔」管状结构——高密度气道壁 (-300~0 HU)
        包裹低密度气腔 (-1000 HU)。新算法通过腐蚀生成壁/腔分层，
        用壁层的高 HU 值在暗色肺底上"画出"可辨识的管状轮廓。

    Args:
        template_path: 标准模板路径
        trachea_mask_path: 气管树 mask 路径
        output_path: 输出路径
        airway_hu: 气腔 HU 值 (默认 -995, 模拟管内空气)
        wall_hu: 气道壁 HU 值 (默认 -400, 模拟软组织管壁)
        wall_thickness: 气道壁厚度, 体素数 (默认 1)
        boundary_sigma: 边界平滑高斯 sigma
        boundary_width: 边界羽化宽度（体素数）
        preserve_existing_low_hu: 已弃用, 保留以兼容旧调用

    Returns:
        output_path: 生成的融合模板路径
    """
    template_path = Path(template_path)
    trachea_mask_path = Path(trachea_mask_path)
    output_path = Path(output_path)

    logger.info("=" * 60)
    logger.info("气管树融合到标准模板 (壁-腔双层算法)")
    logger.info("=" * 60)
    logger.info(f"  模板: {template_path}")
    logger.info(f"  气管树 mask: {trachea_mask_path}")
    logger.info(f"  输出: {output_path}")
    logger.info(f"  参数:")
    logger.info(f"    - 气腔 HU 值: {airway_hu}")
    logger.info(f"    - 气道壁 HU 值: {wall_hu}")
    logger.info(f"    - 气道壁厚度: {wall_thickness} 体素")
    logger.info(f"    - 边界平滑 sigma: {boundary_sigma}")

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

    # 诊断: 模板中气管区域的当前 HU 值
    current_airway_hu = template_data[trachea_binary > 0]
    logger.info(f"  模板中气管区域当前 HU: "
                f"均值={current_airway_hu.mean():.0f}, "
                f"范围=[{current_airway_hu.min():.0f}, {current_airway_hu.max():.0f}]")

    # ================================================================
    # Step 1: 形态学分层 — 将 mask 拆分为「气道壁」和「气腔」
    # ================================================================
    logger.info("Step 1: 形态学分层 (壁 / 腔)...")

    if binary_erosion is not None:
        # 腐蚀得到气腔内核
        lumen_mask = binary_erosion(
            trachea_binary, iterations=wall_thickness
        ).astype(np.uint8)
    else:
        logger.warning("scipy.ndimage 不可用，跳过壁-腔分离，仅设置壁层")
        lumen_mask = np.zeros_like(trachea_binary)

    # 气道壁 = 原始 mask - 腐蚀后的内核
    wall_mask = (trachea_binary > 0) & (lumen_mask == 0)

    wall_voxels = int(np.sum(wall_mask))
    lumen_voxels = int(np.sum(lumen_mask))
    logger.info(f"    气道壁体素: {wall_voxels:,}")
    logger.info(f"    气腔体素:   {lumen_voxels:,}")

    # ================================================================
    # Step 2: 创建目标 HU 层
    # ================================================================
    logger.info("Step 2: 构建目标 HU 层...")

    fused_template = template_data.copy()

    # 2a: 气腔 → airway_hu (-995)
    if lumen_voxels > 0:
        fused_template[lumen_mask > 0] = airway_hu

    # 2b: 气道壁 → wall_hu (-400)
    #     壁层提供视觉对比度 (比肺实质 -650 亮, 比气腔 -995 暗得多)
    fused_template[wall_mask] = wall_hu

    logger.info(f"    气腔区域已设为 {airway_hu:.0f} HU")
    logger.info(f"    气道壁区域已设为 {wall_hu:.0f} HU")

    # ================================================================
    # Step 3: 边界羽化 — 壁层外缘到肺实质的平滑过渡
    # ================================================================
    if gaussian_filter is not None and boundary_sigma > 0:
        logger.info("Step 3: 边界羽化平滑...")

        # 只对壁层外缘做羽化 (膨胀 - 原始 mask = 外缘带)
        if binary_dilation is not None:
            outer_band = (
                binary_dilation(trachea_binary, iterations=boundary_width).astype(np.uint8)
                - trachea_binary
            )
            outer_band = (outer_band > 0)
        else:
            outer_band = np.zeros_like(trachea_binary, dtype=bool)

        if np.any(outer_band):
            # 创建从壁层边缘到外部的渐变权重
            distance_weight = gaussian_filter(
                wall_mask.astype(np.float32), sigma=boundary_sigma
            )
            # 只在外缘带内应用渐变
            distance_weight_outer = distance_weight[outer_band]
            # 归一化
            if distance_weight_outer.max() > 0:
                distance_weight_outer = distance_weight_outer / distance_weight_outer.max()
            fused_template[outer_band] = (
                template_data[outer_band] * (1 - distance_weight_outer) +
                wall_hu * distance_weight_outer
            )
            logger.info(f"    外缘羽化体素: {np.sum(outer_band):,}")
    else:
        logger.info("Step 3: 跳过边界羽化 (sigma=0 或 scipy 不可用)")

    # ================================================================
    # Step 4: 验证融合结果
    # ================================================================
    fused_airway_all = fused_template[trachea_binary > 0]
    fused_wall_values = fused_template[wall_mask]
    fused_lumen_values = fused_template[lumen_mask > 0] if lumen_voxels > 0 else np.array([])

    logger.info("Step 4: 融合结果验证")
    logger.info(f"  整体气管区域: 均值={fused_airway_all.mean():.0f}, "
                f"范围=[{fused_airway_all.min():.0f}, {fused_airway_all.max():.0f}]")
    logger.info(f"  气道壁: 均值={fused_wall_values.mean():.0f}")
    if len(fused_lumen_values) > 0:
        logger.info(f"  气腔:   均值={fused_lumen_values.mean():.0f}")

    # 对比度检查
    lung_region = (template_data > -950) & (template_data < -300) & (~(trachea_binary > 0))
    if np.any(lung_region):
        lung_mean = template_data[lung_region].mean()
        contrast = fused_wall_values.mean() - lung_mean
        logger.info(f"  壁-肺实质对比度: {contrast:+.0f} HU "
                    f"(壁={fused_wall_values.mean():.0f}, 肺={lung_mean:.0f})")

    # ================================================================
    # Step 5: 保存结果
    # ================================================================
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
    验证融合结果（壁-腔双层算法版本）

    验证标准:
    1. 核心非气管区域应完全不变
    2. 气管区域 HU 值应与原始模板有显著差异（壁层升高）
    3. 壁-肺实质对比度应 > 100 HU

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

    # 计算壁-肺实质对比度
    lung_region = (original > -950) & (original < -300) & core_non_airway
    lung_mean_hu = float(original[lung_region].mean()) if np.any(lung_region) else -650.0
    fused_airway_mean = float(fused[trachea_mask].mean())

    # HU 变化量（新算法壁层会升高 HU，所以取绝对差异）
    hu_change = abs(fused_airway_mean - float(original[trachea_mask].mean()))
    wall_contrast = fused_airway_mean - lung_mean_hu

    stats = {
        'original_airway_mean_hu': float(original[trachea_mask].mean()),
        'fused_airway_mean_hu': fused_airway_mean,
        'core_non_airway_unchanged': bool(max_core_diff < 0.01),
        'core_max_diff': max_core_diff,
        'core_mean_diff': mean_core_diff,
        'airway_hu_changed': bool(hu_change > 50),
        'hu_change': hu_change,
        'wall_lung_contrast': wall_contrast,
    }

    logger.info(f"  原始模板气管区域平均 HU: {stats['original_airway_mean_hu']:.1f}")
    logger.info(f"  融合后模板气管区域平均 HU: {stats['fused_airway_mean_hu']:.1f}")
    logger.info(f"  气管区域 HU 变化量: {stats['hu_change']:.1f}")
    logger.info(f"  壁-肺实质对比度: {stats['wall_lung_contrast']:+.1f} HU")
    logger.info(f"  核心非气管区域未改变: {stats['core_non_airway_unchanged']} "
                f"(最大差异: {stats['core_max_diff']:.4f})")

    passed = True
    if not stats['airway_hu_changed']:
        logger.warning("  ⚠ 气管区域 HU 变化不足，融合可能未生效")
        passed = False
    if not stats['core_non_airway_unchanged']:
        logger.warning("  ⚠ 核心非气管区域发生变化，请检查")
        passed = False
    if abs(stats['wall_lung_contrast']) < 100:
        logger.warning(f"  ⚠ 壁-肺实质对比度不足 ({stats['wall_lung_contrast']:+.0f} HU)，"
                       f"气管树可能仍不可辨识")
        passed = False

    if passed:
        logger.info("  ✓ 融合验证通过")

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
        '--suffix', type=str, default='',
        help='文件名后缀。示例：--suffix _exp 将自动映射到 *_exp.nii.gz'
    )
    parser.add_argument(
        '--airway-hu', type=float, default=-995.0,
        help='气腔 HU 值（默认 -995，模拟管内空气）'
    )
    parser.add_argument(
        '--wall-hu', type=float, default=-400.0,
        help='气道壁 HU 值（默认 -400，模拟软组织管壁）'
    )
    parser.add_argument(
        '--wall-thickness', type=int, default=1,
        help='气道壁厚度（体素数，默认 1）'
    )
    parser.add_argument(
        '--boundary-sigma', type=float, default=0.8,
        help='边界平滑 sigma（默认 0.8）'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='融合后执行验证'
    )

    args = parser.parse_args()

    template_path = args.template
    trachea_path = args.trachea
    output_path_arg = args.output
    if args.suffix:
        template_path = f"data/02_atlas/standard_template{args.suffix}.nii.gz"
        trachea_path = f"data/02_atlas/standard_trachea_mask{args.suffix}.nii.gz"
        output_path_arg = f"data/02_atlas/standard_template_with_airway{args.suffix}.nii.gz"

    output_path = fuse_airway_to_template(
        template_path=template_path,
        trachea_mask_path=trachea_path,
        output_path=output_path_arg,
        airway_hu=args.airway_hu,
        wall_hu=args.wall_hu,
        wall_thickness=args.wall_thickness,
        boundary_sigma=args.boundary_sigma,
    )

    if args.verify:
        verify_fusion(
            original_template_path=template_path,
            fused_template_path=output_path,
            trachea_mask_path=trachea_path
        )


if __name__ == "__main__":
    main()

