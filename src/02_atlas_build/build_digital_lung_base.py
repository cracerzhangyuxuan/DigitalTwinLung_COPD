#!/usr/bin/env python3
"""
数字肺底座合成模块

将分散的模板文件合成为统一的数字肺底座：
- 融合标签文件：肺叶(1-5) + 气管树(6)
- 元数据文件：记录各组件信息

作者：Digital Twin Lung Team
日期：2025-01-07
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import nibabel as nib

logger = logging.getLogger(__name__)

# 标签值定义
LABEL_VALUES = {
    'left_upper_lobe': 1,    # 左上叶
    'left_lower_lobe': 2,    # 左下叶
    'right_upper_lobe': 3,   # 右上叶
    'right_middle_lobe': 4,  # 右中叶
    'right_lower_lobe': 5,   # 右下叶
    'trachea': 6,            # 气管树
}


def validate_spatial_consistency(
    images: Dict[str, nib.Nifti1Image],
    tolerance: float = 1e-4
) -> Tuple[bool, str]:
    """
    验证多个 NIfTI 图像的空间一致性
    
    Args:
        images: 图像字典 {名称: NIfTI图像}
        tolerance: 数值容差
        
    Returns:
        (is_valid, message)
    """
    if len(images) < 2:
        return True, "只有一个图像，无需验证"
    
    names = list(images.keys())
    ref_name = names[0]
    ref_img = images[ref_name]
    ref_shape = ref_img.shape
    ref_affine = ref_img.affine
    
    for name in names[1:]:
        img = images[name]
        
        # 检查形状
        if img.shape != ref_shape:
            return False, f"形状不匹配: {ref_name}={ref_shape}, {name}={img.shape}"
        
        # 检查 affine 矩阵
        if not np.allclose(img.affine, ref_affine, atol=tolerance):
            return False, f"Affine 矩阵不匹配: {ref_name} vs {name}"
    
    return True, f"空间一致性验证通过 ({len(images)} 个文件)"


def build_digital_lung_base(
    atlas_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    force_rebuild: bool = False
) -> Tuple[bool, Dict]:
    """
    构建数字肺底座文件
    
    将肺叶标签和气管树 mask 融合为单一标签文件。
    
    Args:
        atlas_dir: 模板文件目录
        output_dir: 输出目录（默认与 atlas_dir 相同）
        force_rebuild: 是否强制重建
        
    Returns:
        (success, info_dict)
    """
    atlas_dir = Path(atlas_dir)
    output_dir = Path(output_dir) if output_dir else atlas_dir
    
    # 定义源文件路径
    source_files = {
        'template': atlas_dir / 'standard_template.nii.gz',
        'mask': atlas_dir / 'standard_mask.nii.gz',
        'lobes': atlas_dir / 'standard_lung_lobes_labeled.nii.gz',
        'trachea': atlas_dir / 'standard_trachea_mask.nii.gz',
    }
    
    # 输出文件路径
    output_labels = output_dir / 'digital_lung_labels.nii.gz'
    output_meta = output_dir / 'digital_lung_base.json'
    
    # 检查是否需要重建
    if not force_rebuild and output_labels.exists() and output_meta.exists():
        logger.info("数字肺底座已存在，跳过构建")
        with open(output_meta, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        return True, {'status': 'exists', 'meta': meta}
    
    logger.info("开始构建数字肺底座...")
    
    # 检查源文件
    missing = [name for name, path in source_files.items() if not path.exists()]
    if missing:
        logger.error(f"缺少源文件: {missing}")
        return False, {'status': 'error', 'missing': missing}
    
    # 加载源文件
    logger.info("加载源文件...")
    images = {}
    for name, path in source_files.items():
        images[name] = nib.load(str(path))
        logger.info(f"  {name}: {images[name].shape}")

    # 验证空间一致性
    is_valid, msg = validate_spatial_consistency(images)
    if not is_valid:
        logger.error(f"空间一致性验证失败: {msg}")
        return False, {'status': 'error', 'message': msg}
    logger.info(f"  {msg}")

    # 获取参考图像信息
    ref_img = images['lobes']
    ref_affine = ref_img.affine
    ref_header = ref_img.header.copy()

    # 加载数据
    lobes_data = images['lobes'].get_fdata().astype(np.uint8)
    trachea_data = images['trachea'].get_fdata().astype(np.uint8)

    # 融合标签：肺叶(1-5) + 气管树(6)
    logger.info("融合标签...")
    fused_labels = lobes_data.copy()

    # 气管树区域设为 6（覆盖可能的重叠）
    trachea_mask = trachea_data > 0
    fused_labels[trachea_mask] = LABEL_VALUES['trachea']

    # 统计各标签体素数
    label_stats = {}
    for name, value in LABEL_VALUES.items():
        count = int(np.sum(fused_labels == value))
        label_stats[name] = count
        logger.info(f"  {name} (值={value}): {count:,} 体素")

    # 保存融合标签文件
    logger.info(f"保存融合标签: {output_labels}")
    fused_img = nib.Nifti1Image(fused_labels, ref_affine, ref_header)
    nib.save(fused_img, str(output_labels))

    # 生成元数据
    meta = {
        'version': '1.0',
        'created': datetime.now().isoformat(),
        'description': '数字肺底座 - 融合标签文件',
        'files': {
            'labels': str(output_labels.name),
            'template': str(source_files['template'].name),
        },
        'label_values': LABEL_VALUES,
        'label_stats': label_stats,
        'spatial_info': {
            'shape': list(ref_img.shape),
            'spacing': [float(x) for x in ref_img.header.get_zooms()],
        },
        'source_files': {k: str(v.name) for k, v in source_files.items()},
    }

    # 保存元数据
    logger.info(f"保存元数据: {output_meta}")
    with open(output_meta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("数字肺底座构建完成!")
    return True, {'status': 'created', 'meta': meta}


def load_digital_lung_base(
    atlas_dir: Union[str, Path]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    加载数字肺底座，提取肺叶和气管树数据

    Args:
        atlas_dir: 模板文件目录

    Returns:
        (lobes_data, trachea_data, meta) - 如果失败返回 (None, None, {})
    """
    atlas_dir = Path(atlas_dir)
    labels_path = atlas_dir / 'digital_lung_labels.nii.gz'
    meta_path = atlas_dir / 'digital_lung_base.json'

    if not labels_path.exists() or not meta_path.exists():
        logger.warning("数字肺底座文件不存在")
        return None, None, {}

    # 加载元数据
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    # 加载融合标签
    labels_img = nib.load(str(labels_path))
    labels_data = labels_img.get_fdata().astype(np.uint8)

    # 提取肺叶（值 1-5）
    lobes_data = labels_data.copy()
    lobes_data[lobes_data == LABEL_VALUES['trachea']] = 0

    # 提取气管树（值 6）
    trachea_data = (labels_data == LABEL_VALUES['trachea']).astype(np.uint8)

    return lobes_data, trachea_data, meta


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    atlas_dir = Path('data/02_atlas')
    success, info = build_digital_lung_base(atlas_dir, force_rebuild=True)

    if success:
        print(f"\n构建成功: {info['status']}")
    else:
        print(f"\n构建失败: {info}")

