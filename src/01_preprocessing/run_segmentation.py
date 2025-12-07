#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肺部分割模块

使用 TotalSegmentator 进行自动肺部分割
"""

import subprocess
from pathlib import Path
from typing import Optional, Union, List

import numpy as np

from ..utils.logger import get_logger
from ..utils.io import load_nifti, save_nifti

logger = get_logger(__name__)


def run_segmentation(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    task: str = "lung",
    fast: bool = False,
    device: str = "gpu"
) -> Path:
    """
    对单个 CT 文件运行 TotalSegmentator 分割
    
    Args:
        input_path: 输入 CT 文件路径 (NIfTI 格式)
        output_dir: 输出目录
        task: 分割任务 ("lung", "total", 等)
        fast: 是否使用快速模式
        device: 使用设备 ("gpu" 或 "cpu")
        
    Returns:
        output_path: 分割结果路径
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 输出路径
    output_path = output_dir / f"{input_path.stem}_segmentation"
    
    logger.info(f"开始分割: {input_path.name}")
    
    # 构建命令
    cmd = [
        "TotalSegmentator",
        "-i", str(input_path),
        "-o", str(output_path),
        "--task", task,
    ]
    
    if fast:
        cmd.append("--fast")
    
    if device == "cpu":
        cmd.extend(["--device", "cpu"])
    
    try:
        # 运行 TotalSegmentator
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"分割完成: {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"TotalSegmentator 运行失败: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("TotalSegmentator 未安装，请运行: pip install TotalSegmentator")
        raise


def batch_segmentation(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    pattern: str = "*.nii.gz",
    task: str = "lung",
    fast: bool = False
) -> List[Path]:
    """
    批量运行分割
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        pattern: 文件匹配模式
        task: 分割任务
        fast: 是否使用快速模式
        
    Returns:
        results: 分割结果路径列表
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    files = list(input_dir.glob(pattern))
    logger.info(f"找到 {len(files)} 个文件待分割")
    
    results = []
    for filepath in files:
        try:
            result = run_segmentation(
                filepath, output_dir, task=task, fast=fast
            )
            results.append(result)
        except Exception as e:
            logger.error(f"分割失败 {filepath.name}: {e}")
    
    logger.info(f"批量分割完成: {len(results)}/{len(files)} 成功")
    return results


def combine_lung_masks(
    segmentation_dir: Union[str, Path]
) -> np.ndarray:
    """
    合并左右肺 mask
    
    TotalSegmentator 输出的肺部分割包含:
    - lung_upper_lobe_left.nii.gz
    - lung_lower_lobe_left.nii.gz
    - lung_upper_lobe_right.nii.gz
    - lung_middle_lobe_right.nii.gz
    - lung_lower_lobe_right.nii.gz
    
    Args:
        segmentation_dir: TotalSegmentator 输出目录
        
    Returns:
        combined_mask: 合并后的肺部 mask
    """
    segmentation_dir = Path(segmentation_dir)
    
    lung_parts = [
        "lung_upper_lobe_left.nii.gz",
        "lung_lower_lobe_left.nii.gz",
        "lung_upper_lobe_right.nii.gz",
        "lung_middle_lobe_right.nii.gz",
        "lung_lower_lobe_right.nii.gz",
    ]
    
    combined_mask = None
    
    for part in lung_parts:
        part_path = segmentation_dir / part
        if part_path.exists():
            mask = load_nifti(part_path)
            if combined_mask is None:
                combined_mask = mask > 0
            else:
                combined_mask = combined_mask | (mask > 0)
    
    if combined_mask is None:
        raise FileNotFoundError(f"未找到肺部分割结果: {segmentation_dir}")
    
    return combined_mask.astype(np.uint8)


def main(config: dict) -> None:
    """
    主函数 - 从配置运行分割流程
    
    Args:
        config: 配置字典
    """
    input_dirs = [
        Path(config['paths']['raw_data']) / 'normal',
        Path(config['paths']['raw_data']) / 'copd',
    ]
    
    output_base = Path(config['paths']['cleaned_data'])
    
    for input_dir in input_dirs:
        if not input_dir.exists():
            logger.warning(f"目录不存在，跳过: {input_dir}")
            continue
        
        output_dir = output_base / f"{input_dir.name}_segmented"
        
        batch_segmentation(
            input_dir=input_dir,
            output_dir=output_dir,
            task=config.get('preprocessing', {}).get('segmentation', {}).get('task', 'lung'),
            fast=config.get('preprocessing', {}).get('segmentation', {}).get('fast_mode', False),
        )


if __name__ == "__main__":
    import yaml
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)

