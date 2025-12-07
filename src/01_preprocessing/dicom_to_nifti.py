#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DICOM 转 NIfTI 模块

将 DICOM 序列转换为 NIfTI 格式
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

from ..utils.logger import get_logger

logger = get_logger(__name__)


def dicom_to_nifti_sitk(
    dicom_dir: Union[str, Path],
    output_path: Union[str, Path]
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    使用 SimpleITK 将 DICOM 序列转换为 NIfTI
    
    Args:
        dicom_dir: DICOM 文件目录
        output_path: 输出 NIfTI 文件路径
        
    Returns:
        volume: 3D 体数据
        affine: 仿射矩阵
        metadata: 元数据字典
    """
    if sitk is None:
        raise ImportError("请安装 SimpleITK: pip install SimpleITK")
    
    dicom_dir = Path(dicom_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"读取 DICOM 序列: {dicom_dir}")
    
    # 读取 DICOM 序列
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    
    if not dicom_files:
        raise FileNotFoundError(f"目录中没有找到 DICOM 序列: {dicom_dir}")
    
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    
    # 获取元数据
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()
    
    metadata = {
        'spacing': spacing,
        'origin': origin,
        'direction': direction,
        'size': image.GetSize(),
        'num_slices': len(dicom_files),
    }
    
    # 构建仿射矩阵
    direction_matrix = np.array(direction).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = direction_matrix * np.array(spacing)
    affine[:3, 3] = origin
    
    # 保存为 NIfTI
    sitk.WriteImage(image, str(output_path))
    logger.info(f"已保存 NIfTI: {output_path}")
    
    # 获取 numpy 数组
    volume = sitk.GetArrayFromImage(image)  # (Z, Y, X)
    volume = np.transpose(volume, (2, 1, 0))  # 转换为 (X, Y, Z)
    
    return volume.astype(np.float32), affine, metadata


def batch_dicom_to_nifti(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    prefix: str = ""
) -> dict:
    """
    批量将 DICOM 目录转换为 NIfTI
    
    Args:
        input_dir: 输入目录（包含多个患者子目录）
        output_dir: 输出目录
        prefix: 文件名前缀
        
    Returns:
        results: 转换结果字典
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 遍历子目录
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    logger.info(f"找到 {len(subdirs)} 个患者目录")
    
    for i, subdir in enumerate(subdirs, start=1):
        patient_name = subdir.name
        # 简化文件名（去除中文和特殊字符问题）
        safe_name = f"{prefix}_{i:03d}" if prefix else f"patient_{i:03d}"
        output_path = output_dir / f"{safe_name}.nii.gz"
        
        try:
            volume, affine, metadata = dicom_to_nifti_sitk(subdir, output_path)
            results[patient_name] = {
                'output_path': str(output_path),
                'shape': volume.shape,
                'metadata': metadata,
                'status': 'success'
            }
            logger.info(f"[{i}/{len(subdirs)}] {patient_name} -> {safe_name}.nii.gz")
        except Exception as e:
            logger.error(f"转换失败 {patient_name}: {e}")
            results[patient_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    logger.info(f"批量转换完成: {success_count}/{len(subdirs)} 成功")
    
    return results


def main():
    """主函数 - 将所有 DICOM 数据转换为 NIfTI"""
    import yaml
    
    # 加载配置
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    raw_dir = Path(config['paths']['raw_data'])
    cleaned_dir = Path(config['paths']['cleaned_data'])
    
    # 转换正常肺
    normal_input = raw_dir / 'normal'
    normal_output = cleaned_dir / 'normal_nifti'
    if normal_input.exists():
        logger.info("=" * 50)
        logger.info("转换正常肺 DICOM -> NIfTI")
        batch_dicom_to_nifti(normal_input, normal_output, prefix="normal")
    
    # 转换 COPD
    copd_input = raw_dir / 'copd'
    copd_output = cleaned_dir / 'copd_nifti'
    if copd_input.exists():
        logger.info("=" * 50)
        logger.info("转换 COPD DICOM -> NIfTI")
        batch_dicom_to_nifti(copd_input, copd_output, prefix="copd")
    
    logger.info("=" * 50)
    logger.info("全部转换完成!")


if __name__ == "__main__":
    main()

