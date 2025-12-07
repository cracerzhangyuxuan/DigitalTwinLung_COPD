#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据读写模块

支持格式:
- NIfTI (.nii, .nii.gz)
- DICOM (.dcm)
"""

from pathlib import Path
from typing import Tuple, Union, Optional

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    import pydicom
except ImportError:
    pydicom = None


def load_nifti(
    filepath: Union[str, Path],
    return_affine: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    加载 NIfTI 文件
    
    Args:
        filepath: NIfTI 文件路径
        return_affine: 是否返回仿射矩阵
        
    Returns:
        data: 3D 体数据 (numpy array)
        affine: 仿射矩阵 (可选)
    """
    if nib is None:
        raise ImportError("请安装 nibabel: pip install nibabel")
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    img = nib.load(str(filepath))
    data = img.get_fdata().astype(np.float32)
    
    if return_affine:
        return data, img.affine
    return data


def save_nifti(
    data: np.ndarray,
    filepath: Union[str, Path],
    affine: Optional[np.ndarray] = None,
    dtype: str = "float32"
) -> None:
    """
    保存 NIfTI 文件
    
    Args:
        data: 3D 体数据
        filepath: 保存路径
        affine: 仿射矩阵，默认为单位矩阵
        dtype: 数据类型
    """
    if nib is None:
        raise ImportError("请安装 nibabel: pip install nibabel")
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if affine is None:
        affine = np.eye(4)
    
    # 转换数据类型
    if dtype == "float32":
        data = data.astype(np.float32)
    elif dtype == "uint8":
        data = data.astype(np.uint8)
    elif dtype == "int16":
        data = data.astype(np.int16)
    
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(filepath))


def load_dicom_series(
    directory: Union[str, Path]
) -> Tuple[np.ndarray, dict]:
    """
    加载 DICOM 序列
    
    Args:
        directory: DICOM 文件目录
        
    Returns:
        volume: 3D 体数据
        metadata: DICOM 元数据
    """
    if pydicom is None:
        raise ImportError("请安装 pydicom: pip install pydicom")
    
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"不是有效目录: {directory}")
    
    # 读取所有 DICOM 文件
    dicom_files = sorted(directory.glob("*.dcm"))
    if not dicom_files:
        dicom_files = sorted(directory.glob("*"))  # 尝试无扩展名
    
    if not dicom_files:
        raise FileNotFoundError(f"目录中没有 DICOM 文件: {directory}")
    
    slices = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(str(f))
            slices.append(ds)
        except Exception:
            continue
    
    # 按层位置排序
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # 构建 3D 体
    volume = np.stack([s.pixel_array for s in slices], axis=-1)
    
    # 应用 Rescale 转换为 HU
    slope = float(slices[0].RescaleSlope) if hasattr(slices[0], 'RescaleSlope') else 1.0
    intercept = float(slices[0].RescaleIntercept) if hasattr(slices[0], 'RescaleIntercept') else 0.0
    volume = volume * slope + intercept
    
    # 提取元数据
    metadata = {
        'PatientID': getattr(slices[0], 'PatientID', 'Unknown'),
        'SliceThickness': float(getattr(slices[0], 'SliceThickness', 1.0)),
        'PixelSpacing': [float(x) for x in getattr(slices[0], 'PixelSpacing', [1.0, 1.0])],
        'Rows': int(slices[0].Rows),
        'Columns': int(slices[0].Columns),
        'NumSlices': len(slices),
    }
    
    return volume.astype(np.float32), metadata


def get_nifti_info(filepath: Union[str, Path]) -> dict:
    """
    获取 NIfTI 文件信息
    
    Args:
        filepath: NIfTI 文件路径
        
    Returns:
        info: 文件信息字典
    """
    if nib is None:
        raise ImportError("请安装 nibabel: pip install nibabel")
    
    img = nib.load(str(filepath))
    header = img.header
    
    return {
        'shape': img.shape,
        'dtype': str(img.get_data_dtype()),
        'affine': img.affine.tolist(),
        'voxel_size': header.get_zooms()[:3],
        'units': header.get_xyzt_units(),
    }

