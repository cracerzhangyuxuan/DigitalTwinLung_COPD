#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理与融合模块

使用训练好的 Inpainting 模型生成融合后的 COPD 数字孪生 CT
"""

from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .network import InpaintingUNet
from ..utils.io import load_nifti, save_nifti
from ..utils.math_ops import normalize_ct, denormalize_ct
from ..utils.logger import get_logger

logger = get_logger(__name__)


def load_model(
    checkpoint_path: Union[str, Path],
    device: str = "cuda"
) -> 'InpaintingUNet':
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
        
    Returns:
        model: 加载的模型
    """
    if torch is None:
        raise ImportError("PyTorch 未安装")
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    model = InpaintingUNet()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"模型加载完成: {checkpoint_path}")
    return model


def fuse_lesion(
    template_path: Union[str, Path],
    lesion_mask_path: Union[str, Path],
    model: 'InpaintingUNet',
    output_path: Union[str, Path],
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    overlap: int = 16,
    hu_min: float = -1000,
    hu_max: float = 400,
    device: str = "cuda"
) -> Path:
    """
    将病灶融合到模板中
    
    流程:
    1. 加载模板和病灶 mask
    2. 在 mask 区域挖空
    3. 使用 Inpainting 模型填充
    4. 保存融合结果
    
    Args:
        template_path: 模板 CT 路径
        lesion_mask_path: 病灶 mask 路径 (已配准到模板空间)
        model: Inpainting 模型
        output_path: 输出路径
        patch_size: 处理的 patch 大小
        overlap: patch 重叠
        hu_min: 归一化最小 HU
        hu_max: 归一化最大 HU
        device: 计算设备
        
    Returns:
        output_path: 融合后的 CT 路径
    """
    if torch is None:
        raise ImportError("PyTorch 未安装")
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    logger.info(f"开始融合: {Path(lesion_mask_path).name}")
    
    # 加载数据
    template, affine = load_nifti(template_path, return_affine=True)
    lesion_mask = load_nifti(lesion_mask_path)
    
    # 归一化
    template_norm = normalize_ct(template, hu_min, hu_max)
    
    # 创建输入 (mask 区域置为 0)
    input_volume = template_norm.copy()
    input_volume[lesion_mask > 0] = 0
    
    # 创建输出
    output_volume = template_norm.copy()
    weight_volume = np.zeros_like(template_norm)
    
    # 滑动窗口处理
    d, h, w = template.shape
    pd, ph, pw = patch_size
    step = pd - overlap
    
    with torch.no_grad():
        for z in range(0, d - pd + 1, step):
            for y in range(0, h - ph + 1, step):
                for x in range(0, w - pw + 1, step):
                    # 检查该 patch 是否包含 mask
                    mask_patch = lesion_mask[z:z+pd, y:y+ph, x:x+pw]
                    if np.sum(mask_patch) == 0:
                        continue
                    
                    # 提取 patch
                    input_patch = input_volume[z:z+pd, y:y+ph, x:x+pw]
                    
                    # 转换为 tensor
                    input_tensor = torch.from_numpy(
                        input_patch[np.newaxis, np.newaxis]
                    ).float().to(device)
                    
                    # 推理
                    output_patch = model(input_tensor)
                    output_patch = output_patch.cpu().numpy()[0, 0]
                    
                    # 只更新 mask 区域
                    mask_region = mask_patch > 0
                    output_volume[z:z+pd, y:y+ph, x:x+pw][mask_region] += \
                        output_patch[mask_region]
                    weight_volume[z:z+pd, y:y+ph, x:x+pw][mask_region] += 1
    
    # 处理重叠区域 (平均)
    weight_volume[weight_volume == 0] = 1
    output_volume = output_volume / weight_volume
    
    # 非 mask 区域保持原样
    output_volume[lesion_mask == 0] = template_norm[lesion_mask == 0]
    
    # 反归一化
    output_hu = denormalize_ct(output_volume, hu_min, hu_max)
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(output_hu, output_path, affine=affine)
    
    logger.info(f"融合完成: {output_path}")
    
    return output_path


def batch_fuse(
    template_path: Union[str, Path],
    mask_dir: Union[str, Path],
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path],
    pattern: str = "*_warped_lesion.nii.gz"
) -> int:
    """
    批量融合
    
    Args:
        template_path: 模板路径
        mask_dir: mask 目录
        checkpoint_path: 模型检查点路径
        output_dir: 输出目录
        pattern: mask 文件匹配模式
        
    Returns:
        count: 成功处理的数量
    """
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model = load_model(checkpoint_path)
    
    # 查找所有 mask
    mask_files = list(mask_dir.rglob(pattern))
    logger.info(f"找到 {len(mask_files)} 个 mask 文件")
    
    count = 0
    for mask_path in mask_files:
        try:
            patient_id = mask_path.parent.name
            output_path = output_dir / f"{patient_id}_fused.nii.gz"
            
            fuse_lesion(
                template_path=template_path,
                lesion_mask_path=mask_path,
                model=model,
                output_path=output_path
            )
            count += 1
            
        except Exception as e:
            logger.error(f"融合失败 {mask_path.name}: {e}")
    
    logger.info(f"批量融合完成: {count}/{len(mask_files)}")
    return count


def main(config: dict) -> None:
    """主函数"""
    paths = config.get('paths', {})
    
    template_path = Path(paths.get('atlas', 'data/02_atlas')) / 'standard_template.nii.gz'
    mask_dir = Path(paths.get('mapped', 'data/03_mapped'))
    checkpoint_path = Path(paths.get('checkpoints', 'checkpoints')) / 'best.pth'
    output_dir = Path(paths.get('final_viz', 'data/04_final_viz'))
    
    if not checkpoint_path.exists():
        logger.error(f"模型检查点不存在: {checkpoint_path}")
        return
    
    batch_fuse(template_path, mask_dir, checkpoint_path, output_dir)


if __name__ == "__main__":
    import yaml
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)

