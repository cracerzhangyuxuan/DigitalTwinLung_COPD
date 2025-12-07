#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据质量检查模块

用于验证输入数据是否符合项目要求
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np

from .io import load_nifti, get_nifti_info
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityCheckResult:
    """质量检查结果"""
    passed: bool
    checks: Dict[str, bool]
    messages: List[str]
    stats: Dict[str, float]


def check_ct_quality(
    filepath: Union[str, Path],
    config: Optional[dict] = None
) -> QualityCheckResult:
    """
    检查 CT 数据质量
    
    检查项:
    1. 文件是否存在且可读
    2. 层数是否足够 (>=100)
    3. 层厚是否合适 (<=3mm)
    4. HU 值范围是否正常
    5. 肺部区域是否存在
    
    Args:
        filepath: NIfTI 文件路径
        config: 配置字典，包含阈值参数
        
    Returns:
        result: 质量检查结果
    """
    # 默认配置
    if config is None:
        config = {
            'min_slices': 100,
            'max_slice_thickness_mm': 3.0,
            'min_hu': -1100,
            'max_hu': -200,
            'min_lung_volume_ratio': 0.1,
        }
    
    checks = {}
    messages = []
    stats = {}
    
    filepath = Path(filepath)
    
    # 检查 1: 文件存在性
    checks['file_exists'] = filepath.exists()
    if not checks['file_exists']:
        messages.append(f"文件不存在: {filepath}")
        return QualityCheckResult(
            passed=False, checks=checks, messages=messages, stats=stats
        )
    
    try:
        # 加载数据
        data, affine = load_nifti(filepath, return_affine=True)
        info = get_nifti_info(filepath)
        
        # 检查 2: 层数
        num_slices = data.shape[2]
        stats['num_slices'] = num_slices
        checks['sufficient_slices'] = num_slices >= config['min_slices']
        if not checks['sufficient_slices']:
            messages.append(
                f"层数不足: {num_slices} < {config['min_slices']}"
            )
        
        # 检查 3: 层厚
        voxel_size = info['voxel_size']
        slice_thickness = voxel_size[2]
        stats['slice_thickness'] = slice_thickness
        checks['appropriate_thickness'] = slice_thickness <= config['max_slice_thickness_mm']
        if not checks['appropriate_thickness']:
            messages.append(
                f"层厚过大: {slice_thickness:.2f}mm > {config['max_slice_thickness_mm']}mm"
            )
        
        # 检查 4: HU 值范围
        min_hu = float(data.min())
        max_hu = float(data.max())
        mean_hu = float(data.mean())
        stats['min_hu'] = min_hu
        stats['max_hu'] = max_hu
        stats['mean_hu'] = mean_hu
        
        checks['valid_hu_range'] = (
            min_hu >= -1500 and max_hu <= 4000
        )
        if not checks['valid_hu_range']:
            messages.append(
                f"HU 值范围异常: [{min_hu:.0f}, {max_hu:.0f}]"
            )
        
        # 检查 5: 肺部区域存在性 (基于 HU 值分布)
        lung_mask = (data > config['min_hu']) & (data < config['max_hu'])
        lung_ratio = lung_mask.sum() / data.size
        stats['lung_volume_ratio'] = lung_ratio
        
        checks['lung_present'] = lung_ratio >= config['min_lung_volume_ratio']
        if not checks['lung_present']:
            messages.append(
                f"肺部区域占比过低: {lung_ratio:.2%} < {config['min_lung_volume_ratio']:.2%}"
            )
        
        # 汇总结果
        passed = all(checks.values())
        
        if passed:
            messages.append("所有质量检查通过")
            logger.info(f"质量检查通过: {filepath.name}")
        else:
            logger.warning(f"质量检查未通过: {filepath.name}")
        
        return QualityCheckResult(
            passed=passed, checks=checks, messages=messages, stats=stats
        )
        
    except Exception as e:
        messages.append(f"读取文件失败: {str(e)}")
        checks['readable'] = False
        logger.error(f"读取文件失败: {filepath}, 错误: {e}")
        return QualityCheckResult(
            passed=False, checks=checks, messages=messages, stats=stats
        )


def batch_quality_check(
    directory: Union[str, Path],
    pattern: str = "*.nii.gz",
    config: Optional[dict] = None
) -> Tuple[List[Path], List[Path], Dict[str, QualityCheckResult]]:
    """
    批量质量检查
    
    Args:
        directory: 数据目录
        pattern: 文件匹配模式
        config: 质量检查配置
        
    Returns:
        passed_files: 通过检查的文件列表
        failed_files: 未通过检查的文件列表
        results: 每个文件的检查结果
    """
    directory = Path(directory)
    files = list(directory.glob(pattern))
    
    passed_files = []
    failed_files = []
    results = {}
    
    logger.info(f"开始批量质量检查: {len(files)} 个文件")
    
    for filepath in files:
        result = check_ct_quality(filepath, config)
        results[str(filepath)] = result
        
        if result.passed:
            passed_files.append(filepath)
        else:
            failed_files.append(filepath)
    
    logger.info(
        f"批量检查完成: {len(passed_files)} 通过, {len(failed_files)} 未通过"
    )
    
    return passed_files, failed_files, results


def generate_quality_report(
    results: Dict[str, QualityCheckResult],
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """
    生成质量检查报告
    
    Args:
        results: 检查结果字典
        output_path: 报告输出路径
        
    Returns:
        report: 报告内容
    """
    lines = [
        "=" * 60,
        "数据质量检查报告",
        "=" * 60,
        "",
    ]
    
    passed_count = sum(1 for r in results.values() if r.passed)
    total_count = len(results)
    
    lines.append(f"总计: {total_count} 个文件")
    lines.append(f"通过: {passed_count} ({passed_count/total_count*100:.1f}%)")
    lines.append(f"未通过: {total_count - passed_count}")
    lines.append("")
    lines.append("-" * 60)
    
    for filepath, result in results.items():
        status = "✓ 通过" if result.passed else "✗ 未通过"
        lines.append(f"\n{Path(filepath).name}: {status}")
        
        for msg in result.messages:
            lines.append(f"  - {msg}")
        
        if result.stats:
            lines.append("  统计信息:")
            for key, value in result.stats.items():
                if isinstance(value, float):
                    lines.append(f"    {key}: {value:.2f}")
                else:
                    lines.append(f"    {key}: {value}")
    
    report = "\n".join(lines)
    
    if output_path:
        Path(output_path).write_text(report, encoding='utf-8')
        logger.info(f"报告已保存: {output_path}")
    
    return report

