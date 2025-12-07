#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准底座构建模块

使用 ANTsPy 的 build_template 函数从多例正常肺 CT 构建标准模板
"""

from pathlib import Path
from typing import List, Union, Optional

import numpy as np

try:
    import ants
except ImportError:
    ants = None

from ..utils.logger import get_logger
from ..utils.io import load_nifti, save_nifti

logger = get_logger(__name__)


def check_ants_available() -> bool:
    """检查 ANTsPy 是否可用"""
    if ants is None:
        logger.error(
            "ANTsPy 未安装。请参考以下方式安装:\n"
            "  conda install -c aramislab antspyx\n"
            "  或 pip install antspyx"
        )
        return False
    return True


def load_images_for_template(
    image_paths: List[Union[str, Path]]
) -> List['ants.ANTsImage']:
    """
    加载多个图像用于模板构建
    
    Args:
        image_paths: 图像文件路径列表
        
    Returns:
        images: ANTsImage 列表
    """
    if not check_ants_available():
        raise ImportError("ANTsPy 不可用")
    
    images = []
    for path in image_paths:
        path = Path(path)
        if not path.exists():
            logger.warning(f"文件不存在，跳过: {path}")
            continue
        
        try:
            img = ants.image_read(str(path))
            images.append(img)
            logger.debug(f"加载图像: {path.name}, shape={img.shape}")
        except Exception as e:
            logger.error(f"加载失败 {path.name}: {e}")
    
    logger.info(f"成功加载 {len(images)} 个图像")
    return images


def build_template(
    image_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    type_of_transform: str = "SyN",
    iteration_limit: int = 5,
    gradient_step: float = 0.2,
    initial_template: Optional[Union[str, Path]] = None
) -> Path:
    """
    构建标准模板
    
    使用 ANTsPy 的 build_template 函数，通过迭代配准和平均
    从多例输入图像构建标准模板。
    
    Args:
        image_paths: 输入图像路径列表 (15-20例正常肺)
        output_path: 输出模板路径
        type_of_transform: 变换类型 ("SyN", "Affine", "Rigid" 等)
        iteration_limit: 迭代次数 (每次迭代包含配准到当前平均 + 更新平均)
        gradient_step: 梯度步长
        initial_template: 初始模板 (可选，默认使用输入图像的平均)
        
    Returns:
        output_path: 生成的模板路径
        
    Note:
        这个过程非常耗时，通常需要数小时到一整夜
    """
    if not check_ants_available():
        raise ImportError("ANTsPy 不可用")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("开始构建标准模板")
    logger.info(f"输入图像数量: {len(image_paths)}")
    logger.info(f"变换类型: {type_of_transform}")
    logger.info(f"迭代次数: {iteration_limit}")
    logger.info("警告: 此过程可能需要数小时，请耐心等待...")
    logger.info("=" * 60)
    
    # 加载图像
    images = load_images_for_template(image_paths)
    
    if len(images) < 2:
        raise ValueError(f"至少需要 2 个图像，当前只有 {len(images)} 个")
    
    # 加载初始模板
    if initial_template is not None:
        initial_template = ants.image_read(str(initial_template))
        logger.info(f"使用自定义初始模板: {initial_template}")
    
    # 构建模板
    logger.info("开始迭代构建...")
    template = ants.build_template(
        image_list=images,
        initial_template=initial_template,
        type_of_transform=type_of_transform,
        iterations=iteration_limit,
        gradient_step=gradient_step,
        verbose=True
    )
    
    # 保存模板
    ants.image_write(template, str(output_path))
    logger.info(f"模板已保存: {output_path}")
    
    # 输出统计信息
    logger.info(f"模板形状: {template.shape}")
    logger.info(f"模板间距: {template.spacing}")
    
    return output_path


def load_template(
    template_path: Union[str, Path]
) -> 'ants.ANTsImage':
    """
    加载已构建的模板
    
    Args:
        template_path: 模板文件路径
        
    Returns:
        template: ANTsImage 对象
    """
    if not check_ants_available():
        raise ImportError("ANTsPy 不可用")
    
    template_path = Path(template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"模板文件不存在: {template_path}")
    
    return ants.image_read(str(template_path))


def evaluate_template_quality(
    template_path: Union[str, Path],
    sample_image_paths: List[Union[str, Path]],
    output_report_path: Optional[Union[str, Path]] = None
) -> dict:
    """
    评估模板质量
    
    通过计算模板与各输入图像之间的相似度来评估模板质量
    
    Args:
        template_path: 模板路径
        sample_image_paths: 样本图像路径列表
        output_report_path: 报告输出路径
        
    Returns:
        metrics: 评估指标
    """
    if not check_ants_available():
        raise ImportError("ANTsPy 不可用")
    
    template = load_template(template_path)
    
    metrics = {
        'dice_scores': [],
        'correlation_scores': [],
    }
    
    for path in sample_image_paths:
        try:
            img = ants.image_read(str(path))
            
            # 配准到模板
            registration = ants.registration(
                fixed=template,
                moving=img,
                type_of_transform='Rigid'
            )
            
            warped = registration['warpedmovout']
            
            # 计算相关性
            corr = ants.image_similarity(template, warped, metric_type='Correlation')
            metrics['correlation_scores'].append(corr)
            
        except Exception as e:
            logger.warning(f"评估失败 {path}: {e}")
    
    # 汇总统计
    if metrics['correlation_scores']:
        metrics['mean_correlation'] = np.mean(metrics['correlation_scores'])
        metrics['std_correlation'] = np.std(metrics['correlation_scores'])
        
        logger.info(
            f"模板质量评估: 平均相关性 = {metrics['mean_correlation']:.4f} "
            f"± {metrics['std_correlation']:.4f}"
        )
    
    return metrics


def main(config: dict) -> None:
    """
    主函数 - 从配置运行模板构建
    """
    # 获取配置
    atlas_config = config.get('registration', {}).get('template_build', {})
    paths_config = config.get('paths', {})
    
    # 输入路径
    input_dir = Path(paths_config.get('cleaned_data', 'data/01_cleaned')) / 'normal_clean'
    
    # 输出路径
    output_dir = Path(paths_config.get('atlas', 'data/02_atlas'))
    output_path = output_dir / 'standard_template.nii.gz'
    
    # 获取所有输入图像
    image_paths = list(input_dir.glob('*.nii.gz'))
    
    if len(image_paths) < 2:
        logger.error(f"输入图像不足: {len(image_paths)} < 2")
        return
    
    # 构建模板
    build_template(
        image_paths=image_paths,
        output_path=output_path,
        type_of_transform=atlas_config.get('type_of_transform', 'SyN'),
        iteration_limit=atlas_config.get('iteration_limit', 5),
        gradient_step=atlas_config.get('gradient_step', 0.2),
    )


if __name__ == "__main__":
    import yaml
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)

