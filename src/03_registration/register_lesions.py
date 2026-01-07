#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
病灶空间配准模块

使用 ANTsPy SyN 算法将 COPD 患者的病灶 mask 配准到标准模板空间
"""

import gc
from pathlib import Path
from typing import Union, Tuple, List, Optional, Dict

import numpy as np

try:
    import ants
except ImportError:
    ants = None

from ..utils.logger import get_logger
from ..utils.io import save_nifti

logger = get_logger(__name__)


def check_ants_available() -> bool:
    """检查 ANTsPy 是否可用"""
    return ants is not None


def register_to_template(
    moving_image_path: Union[str, Path],
    template_path: Union[str, Path],
    output_dir: Union[str, Path],
    type_of_transform: str = "SyNRA",
    reg_iterations: Tuple[int, ...] = (20, 10, 0),
    shrink_factors: Tuple[int, ...] = (4, 2, 1),
    smoothing_sigmas: Tuple[float, ...] = (2, 1, 0)
) -> Dict[str, Path]:
    """
    将移动图像配准到模板空间

    Args:
        moving_image_path: 待配准图像路径 (COPD 患者 CT)
        template_path: 模板图像路径 (标准底座)
        output_dir: 输出目录
        type_of_transform: 变换类型，默认 SyNRA（刚性+仿射+SyN）
        reg_iterations: 各分辨率级别的迭代次数
            - 默认 (20, 10, 0): 优化后的快速配准参数
            - 原始 (100, 70, 50, 20): 高精度但耗时较长
        shrink_factors: 缩放因子
            - 默认 (4, 2, 1): 3 级多分辨率
            - 原始 (8, 4, 2, 1): 4 级多分辨率
        smoothing_sigmas: 平滑 sigma

    Returns:
        outputs: 输出文件路径字典

    Note:
        优化后的默认参数可在约 5 分钟内完成配准（原始参数约 21 分钟）
        病灶保留率约 81%（原始参数约 77%）
    """
    if not check_ants_available():
        raise ImportError("ANTsPy 未安装")
    
    moving_image_path = Path(moving_image_path)
    template_path = Path(template_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patient_id = moving_image_path.stem.replace('.nii', '').replace('_clean', '')
    
    logger.info(f"开始配准: {patient_id}")
    logger.info(f"  移动图像: {moving_image_path.name}")
    logger.info(f"  目标模板: {template_path.name}")
    
    # 加载图像
    moving = ants.image_read(str(moving_image_path))
    fixed = ants.image_read(str(template_path))
    
    # 执行配准
    logger.info("执行 SyN 配准...")
    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=type_of_transform,
        reg_iterations=list(reg_iterations),
        shrink_factors=list(shrink_factors),
        smoothing_sigmas=list(smoothing_sigmas),
        verbose=True
    )
    
    # 保存结果
    outputs = {}
    
    # 保存配准后的图像
    warped_path = output_dir / f"{patient_id}_warped.nii.gz"
    ants.image_write(registration['warpedmovout'], str(warped_path))
    outputs['warped_image'] = warped_path
    
    # 保存变换矩阵
    import shutil
    for i, transform in enumerate(registration['fwdtransforms']):
        # 修复后缀问题：Path.suffix 只返回最后一个后缀（.gz），
        # 对于 .nii.gz 文件需要使用 suffixes 获取完整后缀
        transform_src = Path(transform)
        suffixes = ''.join(transform_src.suffixes)  # 例如 ".nii.gz" 或 ".mat"
        transform_path = output_dir / f"{patient_id}_transform_{i}{suffixes}"
        # 复制变换文件
        shutil.copy(transform, transform_path)
        outputs[f'transform_{i}'] = transform_path
        logger.debug(f"  保存变换文件: {transform_path.name}")
    
    logger.info(f"配准完成: {patient_id}")

    # 显式释放内存
    del moving, fixed, registration
    gc.collect()

    return outputs


def warp_mask(
    mask_path: Union[str, Path],
    template_path: Union[str, Path],
    transform_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    interpolation: str = "nearestNeighbor",
    template_mask_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    使用变换场将 mask 扭曲到模板空间（优化版）

    优化: 添加模板 mask 约束，确保变形后的病灶严格在目标肺内

    Args:
        mask_path: 原始 mask 路径
        template_path: 目标模板路径
        transform_paths: 变换文件路径列表
        output_path: 输出路径
        interpolation: 插值方法 (mask 应使用最近邻)
        template_mask_path: 可选，模板的肺 mask 路径，用于约束变形后的病灶

    Returns:
        output_path: 扭曲后的 mask 路径
    """
    if not check_ants_available():
        raise ImportError("ANTsPy 未安装")

    mask_path = Path(mask_path)
    template_path = Path(template_path)
    output_path = Path(output_path)

    logger.info(f"扭曲 mask: {mask_path.name}")

    # 验证变换文件
    if not transform_paths:
        raise ValueError("变换文件列表为空！请检查 register_to_template() 的输出")

    valid_transforms = []
    for tp in transform_paths:
        tp_path = Path(tp)
        if tp_path.exists():
            valid_transforms.append(str(tp_path))
            logger.debug(f"  变换文件: {tp_path.name}")
        else:
            logger.warning(f"  变换文件不存在: {tp_path}")

    if not valid_transforms:
        raise ValueError(f"所有变换文件都不存在！路径: {transform_paths}")

    # 加载
    mask = ants.image_read(str(mask_path))
    template = ants.image_read(str(template_path))

    logger.info(f"  原始 mask 形状: {mask.shape}")
    logger.info(f"  目标模板形状: {template.shape}")

    # 记录原始体素数
    original_voxels = int(np.sum(mask.numpy() > 0))
    logger.info(f"  原始 mask 体素数: {original_voxels}")

    # 应用变换（mask 必须使用最近邻插值！）
    logger.info(f"  应用 {len(valid_transforms)} 个变换...")
    warped_mask = ants.apply_transforms(
        fixed=template,
        moving=mask,
        transformlist=valid_transforms,
        interpolator=interpolation  # 默认 nearestNeighbor
    )

    # 验证变形结果的形状
    logger.info(f"  变形后形状: {warped_mask.shape}")
    if warped_mask.shape != template.shape:
        logger.warning(
            f"  ⚠️ 变形后形状 {warped_mask.shape} 与模板形状 {template.shape} 不匹配！"
        )

    # 二值化 (处理插值可能产生的非整数值)
    warped_array = warped_mask.numpy()
    warped_array = (warped_array > 0.5).astype(np.uint8)

    warped_voxels = int(np.sum(warped_array > 0))
    logger.info(f"  变形后体素数: {warped_voxels}")

    # 【关键优化】如果提供了模板的肺 mask，用它约束变形后的病灶
    if template_mask_path is not None:
        template_mask_path = Path(template_mask_path)
        if template_mask_path.exists():
            template_mask = ants.image_read(str(template_mask_path))
            template_mask_array = (template_mask.numpy() > 0).astype(np.uint8)

            # 检查维度是否匹配
            if warped_array.shape != template_mask_array.shape:
                logger.warning(
                    f"  维度不匹配! warped: {warped_array.shape}, "
                    f"template_mask: {template_mask_array.shape}"
                )
                logger.warning("  跳过模板 mask 约束（需要修复 standard_mask.nii.gz）")
            else:
                # 约束：病灶必须在模板肺内
                before_constrain = np.sum(warped_array)
                warped_array = warped_array & template_mask_array
                after_constrain = np.sum(warped_array)

                if before_constrain > after_constrain:
                    removed = before_constrain - after_constrain
                    logger.info(
                        f"  模板 mask 约束: 移除 {removed} 个肺外体素 "
                        f"({removed/max(before_constrain,1)*100:.1f}%)"
                    )
        else:
            logger.warning(f"  模板 mask 不存在: {template_mask_path}")

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    warped_mask_binary = ants.from_numpy(
        warped_array,
        origin=warped_mask.origin,
        spacing=warped_mask.spacing,
        direction=warped_mask.direction
    )
    ants.image_write(warped_mask_binary, str(output_path))

    final_voxels = int(np.sum(warped_array > 0))
    logger.info(f"Mask 扭曲完成: {output_path.name}, 最终体素数: {final_voxels}")

    return output_path


def batch_register(
    ct_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    template_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[dict] = None
) -> Dict[str, Dict]:
    """
    批量配准 COPD 患者数据
    
    Args:
        ct_dir: COPD CT 目录
        mask_dir: 病灶 mask 目录
        template_path: 标准模板路径
        output_dir: 输出目录
        config: 配准配置
        
    Returns:
        results: 每个患者的结果
    """
    ct_dir = Path(ct_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    
    # 默认配置
    if config is None:
        config = {}
    reg_config = config.get('registration', {}).get('lesion_registration', {})
    
    ct_files = list(ct_dir.glob("*.nii.gz"))
    logger.info(f"找到 {len(ct_files)} 个 COPD CT 文件")
    
    results = {}
    
    for ct_path in ct_files:
        patient_id = ct_path.stem.replace('.nii', '').replace('_clean', '')
        patient_output_dir = output_dir / patient_id
        
        # 查找对应的病灶 mask
        mask_path = mask_dir / f"{patient_id}_emphysema.nii.gz"
        if not mask_path.exists():
            mask_path = mask_dir / f"{patient_id}_lesion.nii.gz"
        
        if not mask_path.exists():
            logger.warning(f"未找到病灶 mask: {patient_id}")
            continue
        
        try:
            # 配准 CT
            reg_outputs = register_to_template(
                moving_image_path=ct_path,
                template_path=template_path,
                output_dir=patient_output_dir,
                type_of_transform=reg_config.get('type_of_transform', 'SyNRA'),
                reg_iterations=tuple(reg_config.get('reg_iterations', [100, 70, 50, 20])),
            )
            
            # 扭曲病灶 mask
            transform_paths = [
                reg_outputs.get(f'transform_{i}')
                for i in range(2)
                if reg_outputs.get(f'transform_{i}') is not None
            ]
            
            warped_mask_path = patient_output_dir / f"{patient_id}_warped_lesion.nii.gz"
            warp_mask(
                mask_path=mask_path,
                template_path=template_path,
                transform_paths=transform_paths,
                output_path=warped_mask_path
            )
            
            reg_outputs['warped_mask'] = warped_mask_path
            results[patient_id] = reg_outputs
            
        except Exception as e:
            logger.error(f"配准失败 {patient_id}: {e}")
            results[patient_id] = {'error': str(e)}
    
    logger.info(f"批量配准完成: {len([r for r in results.values() if 'error' not in r])}/{len(ct_files)} 成功")
    
    return results


def main(config: dict) -> None:
    """主函数"""
    paths = config.get('paths', {})
    
    ct_dir = Path(paths.get('cleaned_data', 'data/01_cleaned')) / 'copd_clean'
    mask_dir = ct_dir  # 假设 mask 和 CT 在同一目录
    template_path = Path(paths.get('atlas', 'data/02_atlas')) / 'standard_template.nii.gz'
    output_dir = Path(paths.get('mapped', 'data/03_mapped'))
    
    if not template_path.exists():
        logger.error(f"模板不存在: {template_path}")
        return
    
    batch_register(ct_dir, mask_dir, template_path, output_dir, config)


if __name__ == "__main__":
    import yaml
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)

