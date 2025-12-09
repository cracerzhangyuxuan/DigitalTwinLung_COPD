#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据验证模块

提供配准结果验证、可视化质量检查等功能。
与 data_quality.py 配合使用，data_quality.py 专注于输入数据质量，
本模块专注于处理流程中的验证和诊断。
"""

from pathlib import Path
from typing import Union, Optional, Dict, Tuple
from dataclasses import dataclass

import numpy as np

from .io import load_nifti
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class RegistrationValidationResult:
    """配准验证结果"""
    valid: bool
    original_voxels: int
    warped_voxels: int
    retention_rate: float
    z_coverage: float
    messages: list


def validate_registration_result(
    original_mask_path: Union[str, Path],
    warped_mask_path: Union[str, Path],
    template_mask_path: Optional[Union[str, Path]] = None,
    min_retention_rate: float = 0.5,
    min_z_coverage: float = 0.5
) -> RegistrationValidationResult:
    """
    验证配准结果的质量

    检查项:
    1. 病灶保留率是否足够（配准后病灶不应大幅减少）
    2. Z 轴覆盖率是否足够（病灶不应只集中在某一区域）
    3. 如果提供模板 mask，检查病灶是否在模板肺内

    Args:
        original_mask_path: 原始病灶 mask 路径
        warped_mask_path: 配准后的病灶 mask 路径
        template_mask_path: 模板肺 mask 路径（可选）
        min_retention_rate: 最小保留率阈值
        min_z_coverage: 最小 Z 轴覆盖率阈值

    Returns:
        result: 验证结果

    Example:
        >>> result = validate_registration_result(
        ...     original_mask_path="copd_001_emphysema.nii.gz",
        ...     warped_mask_path="copd_001_emphysema_warped.nii.gz",
        ...     min_retention_rate=0.5
        ... )
        >>> print(f"保留率: {result.retention_rate:.1%}")
    """
    messages = []

    original_mask_path = Path(original_mask_path)
    warped_mask_path = Path(warped_mask_path)

    if not original_mask_path.exists():
        return RegistrationValidationResult(
            valid=False, original_voxels=0, warped_voxels=0,
            retention_rate=0, z_coverage=0,
            messages=[f"原始 mask 不存在: {original_mask_path}"]
        )

    if not warped_mask_path.exists():
        return RegistrationValidationResult(
            valid=False, original_voxels=0, warped_voxels=0,
            retention_rate=0, z_coverage=0,
            messages=[f"配准后 mask 不存在: {warped_mask_path}"]
        )

    # 加载数据
    original_mask = load_nifti(original_mask_path)
    warped_mask = load_nifti(warped_mask_path)

    original_voxels = int(np.sum(original_mask > 0))
    warped_voxels = int(np.sum(warped_mask > 0))

    # 计算保留率
    if original_voxels > 0:
        retention_rate = warped_voxels / original_voxels
    else:
        retention_rate = 0.0
        messages.append("警告: 原始 mask 为空")

    # 检查保留率
    if retention_rate < min_retention_rate:
        messages.append(
            f"保留率过低: {retention_rate:.1%} < {min_retention_rate:.1%}"
        )
    else:
        messages.append(f"保留率正常: {retention_rate:.1%}")

    # 计算 Z 轴覆盖率
    warped_binary = (warped_mask > 0).astype(np.uint8)
    z_slices_with_lesion = np.any(warped_binary, axis=(0, 1))
    z_coverage = np.sum(z_slices_with_lesion) / warped_mask.shape[2]

    if z_coverage < min_z_coverage:
        messages.append(
            f"Z 轴覆盖率过低: {z_coverage:.1%} < {min_z_coverage:.1%}"
        )
    else:
        messages.append(f"Z 轴覆盖率正常: {z_coverage:.1%}")

    # 检查是否在模板肺内（如果提供了模板 mask）
    if template_mask_path is not None:
        template_mask_path = Path(template_mask_path)
        if template_mask_path.exists():
            template_mask = load_nifti(template_mask_path)
            template_mask_binary = (template_mask > 0).astype(np.uint8)

            # 检查配准后病灶是否在模板肺内
            outside_lung = warped_binary & (~template_mask_binary)
            outside_voxels = int(np.sum(outside_lung))

            if outside_voxels > 0:
                outside_ratio = outside_voxels / max(warped_voxels, 1)
                messages.append(
                    f"有 {outside_voxels} 个体素 ({outside_ratio:.1%}) 在模板肺外"
                )

    # 综合判断
    valid = (
        retention_rate >= min_retention_rate and
        z_coverage >= min_z_coverage
    )

    return RegistrationValidationResult(
        valid=valid,
        original_voxels=original_voxels,
        warped_voxels=warped_voxels,
        retention_rate=retention_rate,
        z_coverage=z_coverage,
        messages=messages
    )


def check_mask_coverage(
    mask_path: Union[str, Path],
    reference_shape: Optional[Tuple[int, int, int]] = None
) -> Dict[str, float]:
    """
    检查 mask 的覆盖统计信息

    Args:
        mask_path: mask 文件路径
        reference_shape: 参考形状（用于计算占比）

    Returns:
        stats: 统计信息字典，包含：
            - voxel_count: 体素数量
            - volume_ratio: 体积占比
            - x_coverage: X 轴覆盖率
            - y_coverage: Y 轴覆盖率
            - z_coverage: Z 轴覆盖率
            - center_of_mass: 质心坐标

    Example:
        >>> stats = check_mask_coverage("lung_mask.nii.gz")
        >>> print(f"体素数: {stats['voxel_count']:,}")
    """
    mask_path = Path(mask_path)

    if not mask_path.exists():
        logger.error(f"Mask 文件不存在: {mask_path}")
        return {}

    mask = load_nifti(mask_path)
    mask_binary = (mask > 0).astype(np.uint8)

    voxel_count = int(np.sum(mask_binary))
    total_voxels = mask.size if reference_shape is None else np.prod(reference_shape)
    volume_ratio = voxel_count / total_voxels

    # 计算各轴覆盖率
    x_coverage = np.sum(np.any(mask_binary, axis=(1, 2))) / mask.shape[0]
    y_coverage = np.sum(np.any(mask_binary, axis=(0, 2))) / mask.shape[1]
    z_coverage = np.sum(np.any(mask_binary, axis=(0, 1))) / mask.shape[2]

    # 计算质心
    if voxel_count > 0:
        indices = np.where(mask_binary > 0)
        center_of_mass = (
            float(np.mean(indices[0])),
            float(np.mean(indices[1])),
            float(np.mean(indices[2]))
        )
    else:
        center_of_mass = (0.0, 0.0, 0.0)

    return {
        'voxel_count': voxel_count,
        'volume_ratio': volume_ratio,
        'x_coverage': x_coverage,
        'y_coverage': y_coverage,
        'z_coverage': z_coverage,
        'center_of_mass': center_of_mass
    }


def compare_ct_shapes(
    moving_path: Union[str, Path],
    fixed_path: Union[str, Path],
    warped_path: Optional[Union[str, Path]] = None
) -> Dict[str, Tuple[int, int, int]]:
    """
    比较配准前后 CT 的形状，用于验证 ANTsPy 配准的尺寸变化

    Args:
        moving_path: 移动图像（COPD 患者 CT）路径
        fixed_path: 固定图像（模板）路径
        warped_path: 配准后图像路径（可选）

    Returns:
        shapes: 各图像的形状字典

    Example:
        >>> shapes = compare_ct_shapes(
        ...     moving_path="copd_001_clean.nii.gz",
        ...     fixed_path="template.nii.gz",
        ...     warped_path="copd_001_warped.nii.gz"
        ... )
        >>> print(f"配准后形状: {shapes['warped']}")
    """
    result = {}

    moving_path = Path(moving_path)
    fixed_path = Path(fixed_path)

    if moving_path.exists():
        moving = load_nifti(moving_path)
        result['moving'] = moving.shape
        logger.info(f"Moving (COPD): {moving.shape}")
    else:
        logger.warning(f"Moving 文件不存在: {moving_path}")

    if fixed_path.exists():
        fixed = load_nifti(fixed_path)
        result['fixed'] = fixed.shape
        logger.info(f"Fixed (Template): {fixed.shape}")
    else:
        logger.warning(f"Fixed 文件不存在: {fixed_path}")

    if warped_path is not None:
        warped_path = Path(warped_path)
        if warped_path.exists():
            warped = load_nifti(warped_path)
            result['warped'] = warped.shape
            logger.info(f"Warped (配准后): {warped.shape}")

            # 验证配准后形状是否与 fixed 一致
            if 'fixed' in result and warped.shape == result['fixed']:
                logger.info("✓ 配准后形状与模板一致")
            elif 'fixed' in result:
                logger.warning(f"⚠ 配准后形状与模板不一致: {warped.shape} vs {result['fixed']}")

    return result

