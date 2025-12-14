#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肺部分割模块

使用 TotalSegmentator 进行自动肺部分割，支持：
- GPU 加速分割（TotalSegmentator）
- CPU 阈值分割（备选方案）
- 批量处理
- 环境检查

作者: DigitalTwinLung_COPD Team
日期: 2025-12-09
更新: 2025-12-14 - 整合 GPU 分割功能
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict

import numpy as np

from ..utils.logger import get_logger
from ..utils.io import load_nifti, save_nifti

logger = get_logger(__name__)


# =============================================================================
# 环境检查函数
# =============================================================================

def check_gpu_available() -> Tuple[bool, str]:
    """
    检查 GPU 是否可用

    Returns:
        (is_available, message): GPU 可用性和描述信息
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return True, f"GPU 可用: {gpu_name}"
        else:
            return False, "CUDA 不可用"
    except ImportError:
        return False, "PyTorch 未安装"


def check_totalsegmentator_available() -> Tuple[bool, str]:
    """
    检查 TotalSegmentator 是否可用

    Returns:
        (is_available, message): 可用性和描述信息
    """
    try:
        result = subprocess.run(
            ["TotalSegmentator", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return True, "TotalSegmentator 可用"
        return False, "TotalSegmentator 命令执行失败"
    except FileNotFoundError:
        return False, "TotalSegmentator 未安装"
    except subprocess.TimeoutExpired:
        return False, "TotalSegmentator 响应超时"
    except Exception as e:
        return False, f"检查失败: {e}"


def get_default_method() -> str:
    """
    获取默认分割方法

    Returns:
        method: "totalsegmentator" 或 "threshold"
    """
    ts_ok, _ = check_totalsegmentator_available()
    return "totalsegmentator" if ts_ok else "threshold"


def get_default_device() -> str:
    """
    获取默认设备

    Returns:
        device: "cuda:0" 或 "cpu"
    """
    gpu_ok, _ = check_gpu_available()
    return "cuda:0" if gpu_ok else "cpu"


# =============================================================================
# 核心分割函数
# =============================================================================


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


# =============================================================================
# 批量处理函数（GPU 加速版）
# =============================================================================

def run_totalsegmentator_batch(
    input_dir: Union[str, Path],
    mask_output_dir: Union[str, Path],
    clean_output_dir: Union[str, Path],
    device: str = "gpu",
    fast: bool = False,
    skip_existing: bool = True,
    limit: Optional[int] = None,
    background_hu: float = -1000
) -> Dict[str, List]:
    """
    使用 TotalSegmentator 批量分割

    Args:
        input_dir: 输入目录
        mask_output_dir: mask 输出目录
        clean_output_dir: 清洗后 CT 输出目录
        device: 设备 ("gpu", "cpu", "cuda:0", etc.)
        fast: 是否使用快速模式
        skip_existing: 是否跳过已处理的文件
        limit: 限制处理数量 (用于测试)
        background_hu: 背景 HU 值

    Returns:
        results: 处理结果字典 {"success": [], "failed": [], "skipped": []}
    """
    import nibabel as nib

    input_dir = Path(input_dir)
    mask_output_dir = Path(mask_output_dir)
    clean_output_dir = Path(clean_output_dir)

    mask_output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir.mkdir(parents=True, exist_ok=True)

    # 临时目录用于 TotalSegmentator 输出
    temp_dir = input_dir.parent.parent / ".temp_segmentation"
    temp_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(list(input_dir.glob("*.nii.gz")))
    if limit:
        nifti_files = nifti_files[:limit]

    logger.info(f"找到 {len(nifti_files)} 个文件待处理")

    results = {"success": [], "failed": [], "skipped": []}

    for i, nifti_path in enumerate(nifti_files, 1):
        stem = nifti_path.name.replace('.nii.gz', '').replace('.nii', '')
        mask_path = mask_output_dir / f"{stem}_mask.nii.gz"
        clean_path = clean_output_dir / f"{stem}_clean.nii.gz"

        # 检查是否已处理
        if skip_existing and mask_path.exists() and clean_path.exists():
            logger.info(f"[{i}/{len(nifti_files)}] 跳过已处理: {stem}")
            results["skipped"].append(stem)
            continue

        logger.info(f"[{i}/{len(nifti_files)}] 处理: {stem}")

        try:
            # 运行 TotalSegmentator
            seg_output = temp_dir / f"{stem}_seg"

            cmd = ["TotalSegmentator", "-i", str(nifti_path), "-o", str(seg_output)]

            # 仅分割肺部 (使用 -rs 参数指定 ROI)
            cmd.extend(["-rs",
                       "lung_upper_lobe_left", "lung_lower_lobe_left",
                       "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"])

            if fast:
                cmd.append("-f")

            # 设备选择
            if device == "cpu":
                cmd.extend(["-d", "cpu"])
            elif device.startswith("cuda:"):
                gpu_id = device.split(":")[1]
                cmd.extend(["-d", f"gpu:{gpu_id}"])
            elif device.startswith("gpu:"):
                cmd.extend(["-d", device])
            elif device == "gpu":
                cmd.extend(["-d", "gpu"])

            # 运行命令
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else result.stdout
                raise RuntimeError(f"TotalSegmentator 失败: {error_msg[:200]}")

            # 合并肺叶 mask
            lung_parts = [
                "lung_upper_lobe_left.nii.gz",
                "lung_lower_lobe_left.nii.gz",
                "lung_upper_lobe_right.nii.gz",
                "lung_middle_lobe_right.nii.gz",
                "lung_lower_lobe_right.nii.gz",
            ]

            combined_mask = None
            affine = None

            for part in lung_parts:
                part_path = seg_output / part
                if part_path.exists():
                    nii = nib.load(str(part_path))
                    mask = np.asanyarray(nii.dataobj) > 0
                    if combined_mask is None:
                        combined_mask = mask
                        affine = nii.affine
                    else:
                        combined_mask = combined_mask | mask

            if combined_mask is None:
                raise ValueError("未找到肺部分割结果")

            combined_mask = combined_mask.astype(np.uint8)

            # 保存 mask
            save_nifti(combined_mask, mask_path, affine=affine, dtype='uint8')

            # 加载原始 CT 并创建清洗后版本
            ct_data, ct_affine = load_nifti(nifti_path, return_affine=True)
            ct_clean = ct_data.copy()
            ct_clean[combined_mask == 0] = background_hu
            save_nifti(ct_clean, clean_path, affine=ct_affine)

            # 清理临时文件
            if seg_output.exists():
                shutil.rmtree(seg_output)

            lung_ratio = np.sum(combined_mask) / combined_mask.size * 100
            logger.info(f"    ✅ 完成 - 肺占比: {lung_ratio:.1f}%")
            results["success"].append(stem)

        except Exception as e:
            logger.error(f"    ❌ 失败: {e}")
            results["failed"].append((stem, str(e)))

    # 清理临时目录
    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()

    return results


def run_threshold_batch(
    input_dir: Union[str, Path],
    mask_output_dir: Union[str, Path],
    clean_output_dir: Union[str, Path],
    skip_existing: bool = True,
    limit: Optional[int] = None
) -> Dict[str, List]:
    """
    使用阈值方法批量分割

    Args:
        input_dir: 输入目录
        mask_output_dir: mask 输出目录
        clean_output_dir: 清洗后 CT 输出目录
        skip_existing: 是否跳过已处理的文件
        limit: 限制处理数量

    Returns:
        results: 处理结果字典
    """
    from .simple_lung_segment import segment_lung_from_file

    input_dir = Path(input_dir)
    mask_output_dir = Path(mask_output_dir)
    clean_output_dir = Path(clean_output_dir)

    mask_output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(list(input_dir.glob("*.nii.gz")))

    if limit:
        nifti_files = nifti_files[:limit]

    results = {"success": [], "failed": [], "skipped": []}

    for i, nifti_path in enumerate(nifti_files, 1):
        stem = nifti_path.name.replace('.nii.gz', '').replace('.nii', '')
        mask_path = mask_output_dir / f"{stem}_mask.nii.gz"
        clean_path = clean_output_dir / f"{stem}_clean.nii.gz"

        # 检查是否已处理
        if skip_existing and mask_path.exists() and clean_path.exists():
            logger.info(f"[{i}/{len(nifti_files)}] 跳过已处理: {stem}")
            results["skipped"].append(stem)
            continue

        logger.info(f"[{i}/{len(nifti_files)}] 处理: {stem}")

        try:
            result = segment_lung_from_file(
                nifti_path,
                mask_output_dir=mask_output_dir,
                clean_output_dir=clean_output_dir
            )
            if result.get('status') == 'success':
                lung_ratio = result.get('lung_ratio', 0) * 100
                logger.info(f"    ✅ 完成 - 肺占比: {lung_ratio:.1f}%")
                results["success"].append(stem)
            else:
                logger.error(f"    ❌ 失败: {result.get('error', '未知错误')}")
                results["failed"].append((stem, result.get('error', '未知错误')))
        except Exception as e:
            logger.error(f"    ❌ 异常: {e}")
            results["failed"].append((stem, str(e)))

    return results


# =============================================================================
# 主函数
# =============================================================================

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

