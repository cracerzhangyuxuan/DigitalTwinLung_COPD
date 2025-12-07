#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SimpleITK 配准模块

使用 SimpleITK 进行图像配准（替代 ANTsPy）
"""

from pathlib import Path
from typing import Union, Tuple, Dict, Optional
import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

from ..utils.logger import get_logger

logger = get_logger(__name__)


def check_sitk_available() -> bool:
    """检查 SimpleITK 是否可用"""
    return sitk is not None


def register_images(
    fixed_path: Union[str, Path],
    moving_path: Union[str, Path],
    output_dir: Union[str, Path],
    transform_type: str = "affine",
    use_mask: bool = False,
    fixed_mask_path: Optional[Union[str, Path]] = None,
    moving_mask_path: Optional[Union[str, Path]] = None
) -> Dict[str, Path]:
    """
    配准两个图像
    
    Args:
        fixed_path: 固定图像路径（模板/底座）
        moving_path: 移动图像路径（待配准）
        output_dir: 输出目录
        transform_type: 变换类型 ("rigid", "affine", "bspline")
        use_mask: 是否使用 mask
        fixed_mask_path: 固定图像 mask 路径
        moving_mask_path: 移动图像 mask 路径
        
    Returns:
        outputs: 输出文件路径字典
    """
    if not check_sitk_available():
        raise ImportError("SimpleITK 未安装")
    
    fixed_path = Path(fixed_path)
    moving_path = Path(moving_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"配准: {moving_path.name} -> {fixed_path.name}")
    
    # 读取图像
    fixed_image = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    moving_image = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    
    # 初始化配准方法
    registration_method = sitk.ImageRegistrationMethod()
    
    # 相似性度量 - 互信息
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    
    # 插值方法
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # 优化器
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # 多分辨率
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # 选择变换类型
    if transform_type == "rigid":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    elif transform_type == "affine":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    elif transform_type == "bspline":
        # 先做仿射配准
        affine_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration_method.SetInitialTransform(affine_transform, inPlace=False)
        affine_result = registration_method.Execute(fixed_image, moving_image)
        
        # 然后做 B-Spline
        mesh_size = [8, 8, 8]
        initial_transform = sitk.BSplineTransformInitializer(
            fixed_image, mesh_size, order=3
        )
        registration_method.SetInitialTransform(initial_transform, inPlace=True)
    else:
        raise ValueError(f"未知的变换类型: {transform_type}")
    
    if transform_type != "bspline":
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    # 执行配准
    logger.info("执行配准...")
    final_transform = registration_method.Execute(fixed_image, moving_image)
    
    logger.info(f"配准完成. 最终度量值: {registration_method.GetMetricValue():.4f}")
    logger.info(f"优化器停止条件: {registration_method.GetOptimizerStopConditionDescription()}")
    
    # 应用变换
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)
    resampler.SetTransform(final_transform)
    
    warped_image = resampler.Execute(moving_image)
    
    # 保存结果
    outputs = {}
    
    stem = moving_path.name.replace('.nii.gz', '').replace('.nii', '')
    
    # 保存配准后的图像
    warped_path = output_dir / f"{stem}_warped.nii.gz"
    sitk.WriteImage(warped_image, str(warped_path))
    outputs['warped_image'] = warped_path
    logger.info(f"已保存配准图像: {warped_path}")
    
    # 保存变换
    transform_path = output_dir / f"{stem}_transform.tfm"
    sitk.WriteTransform(final_transform, str(transform_path))
    outputs['transform'] = transform_path
    logger.info(f"已保存变换: {transform_path}")
    
    return outputs


def apply_transform_to_mask(
    mask_path: Union[str, Path],
    reference_path: Union[str, Path],
    transform_path: Union[str, Path],
    output_path: Union[str, Path]
) -> Path:
    """
    将变换应用到 mask
    
    Args:
        mask_path: 原始 mask 路径
        reference_path: 参考图像路径（目标空间）
        transform_path: 变换文件路径
        output_path: 输出路径
        
    Returns:
        output_path: 变换后的 mask 路径
    """
    if not check_sitk_available():
        raise ImportError("SimpleITK 未安装")
    
    mask_path = Path(mask_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"变换 mask: {mask_path.name}")
    
    # 读取
    mask_image = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    reference_image = sitk.ReadImage(str(reference_path))
    transform = sitk.ReadTransform(str(transform_path))
    
    # 应用变换（使用最近邻插值保持二值性）
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    
    warped_mask = resampler.Execute(mask_image)
    
    # 保存
    sitk.WriteImage(warped_mask, str(output_path))
    logger.info(f"已保存变换后的 mask: {output_path}")
    
    return output_path


def register_copd_to_template(
    template_path: Union[str, Path],
    copd_ct_path: Union[str, Path],
    copd_lesion_path: Union[str, Path],
    output_dir: Union[str, Path],
    transform_type: str = "affine"
) -> Dict[str, Path]:
    """
    将 COPD 配准到模板，并变换病灶 mask
    
    Args:
        template_path: 模板（底座）路径
        copd_ct_path: COPD CT 路径
        copd_lesion_path: COPD 病灶 mask 路径
        output_dir: 输出目录
        transform_type: 变换类型
        
    Returns:
        outputs: 输出文件路径字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 配准 CT
    reg_outputs = register_images(
        fixed_path=template_path,
        moving_path=copd_ct_path,
        output_dir=output_dir,
        transform_type=transform_type
    )
    
    # 2. 变换病灶 mask
    stem = Path(copd_lesion_path).name.replace('.nii.gz', '').replace('.nii', '')
    warped_lesion_path = output_dir / f"{stem}_warped.nii.gz"
    
    apply_transform_to_mask(
        mask_path=copd_lesion_path,
        reference_path=template_path,
        transform_path=reg_outputs['transform'],
        output_path=warped_lesion_path
    )
    
    reg_outputs['warped_lesion'] = warped_lesion_path
    
    return reg_outputs


def main():
    """主函数"""
    import yaml
    
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    paths = config['paths']
    cleaned_dir = Path(paths['cleaned_data'])
    mapped_dir = Path(paths['mapped'])
    atlas_dir = Path(paths['atlas'])
    
    # 使用第一个正常肺作为临时底座
    template_path = atlas_dir / 'temp_template.nii.gz'
    normal_clean_dir = cleaned_dir / 'normal_clean'
    
    if not template_path.exists():
        # 复制第一个正常肺作为临时模板
        normal_files = list(normal_clean_dir.glob("*.nii.gz"))
        if normal_files:
            import shutil
            atlas_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(normal_files[0], template_path)
            logger.info(f"创建临时模板: {normal_files[0].name} -> {template_path}")
    
    # 配准 COPD
    copd_clean_dir = cleaned_dir / 'copd_clean'
    copd_mask_dir = cleaned_dir / 'copd_mask'
    
    if copd_clean_dir.exists() and template_path.exists():
        copd_files = list(copd_clean_dir.glob("*.nii.gz"))
        for copd_file in copd_files:
            stem = copd_file.name.replace('_clean.nii.gz', '')
            lesion_file = copd_mask_dir / f"{stem}_mask_emphysema.nii.gz"
            
            if not lesion_file.exists():
                lesion_file = copd_mask_dir / f"{stem}_mask.nii.gz"
            
            if lesion_file.exists():
                patient_output = mapped_dir / stem
                register_copd_to_template(
                    template_path=template_path,
                    copd_ct_path=copd_file,
                    copd_lesion_path=lesion_file,
                    output_dir=patient_output
                )


if __name__ == "__main__":
    main()

