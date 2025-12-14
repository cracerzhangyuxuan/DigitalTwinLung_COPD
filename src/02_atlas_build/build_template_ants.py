#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准底座构建模块 (Phase 2: Atlas Construction)

使用 ANTsPy 的 build_template 函数从多例正常肺 CT 构建标准模板。

主要功能：
- build_template(): 从多例正常肺构建标准模板
- generate_template_mask(): 为模板生成肺部 mask
- evaluate_template_quality(): 评估模板质量（Dice >= 0.85）
- validate_atlas(): 完整的质量验证流程

使用方法：
    python -m src.02_atlas_build.build_template_ants
    或通过 run_phase2_atlas.py 入口脚本运行

验收标准（来自 Engineering_Edition.md）：
- standard_template.nii.gz 文件大小 > 10MB
- 与任一输入肺的 Dice >= 0.85
- 血管/气管结构可辨识

作者: DigitalTwinLung_COPD Team
更新: 2025-12-09
"""

from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict
import time
from datetime import datetime, timedelta

import numpy as np

try:
    import ants
except ImportError:
    ants = None

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

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


def generate_template_mask(
    template_path: Union[str, Path],
    output_mask_path: Union[str, Path],
    hu_min: float = -950,
    hu_max: float = -200,
    min_size: int = 10000
) -> Path:
    """
    为模板生成肺部 mask

    使用阈值分割和形态学操作从模板 CT 生成肺部 mask

    Args:
        template_path: 模板 CT 路径
        output_mask_path: 输出 mask 路径
        hu_min: 肺组织最小 HU 值
        hu_max: 肺组织最大 HU 值
        min_size: 最小连通域大小

    Returns:
        output_mask_path: 生成的 mask 路径
    """
    logger.info("生成模板肺部 mask...")

    # 加载模板
    data, affine = load_nifti(template_path, return_affine=True)

    # 阈值分割
    lung_mask = ((data >= hu_min) & (data <= hu_max)).astype(np.uint8)

    # 形态学操作
    if ndimage is not None:
        # 连通域分析，保留最大的两个（左右肺）
        labeled, num_features = ndimage.label(lung_mask)
        if num_features > 0:
            sizes = ndimage.sum(lung_mask, labeled, range(1, num_features + 1))
            # 保留最大的两个连通域
            sorted_indices = np.argsort(sizes)[::-1]
            cleaned_mask = np.zeros_like(lung_mask)
            for idx in sorted_indices[:2]:  # 保留最大两个
                if sizes[idx] >= min_size:
                    cleaned_mask[labeled == (idx + 1)] = 1
            lung_mask = cleaned_mask

        # 形态学闭操作填充空洞
        from scipy.ndimage import binary_closing
        lung_mask = binary_closing(lung_mask, iterations=3).astype(np.uint8)

    # 保存 mask
    output_mask_path = Path(output_mask_path)
    save_nifti(lung_mask, output_mask_path, affine=affine, dtype="uint8")

    voxel_count = np.sum(lung_mask)
    logger.info(f"模板 mask 已保存: {output_mask_path}")
    logger.info(f"  肺部体素数: {voxel_count:,}")

    return output_mask_path


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个 mask 的 Dice 系数"""
    intersection = np.sum((mask1 > 0) & (mask2 > 0))
    volume1 = np.sum(mask1 > 0)
    volume2 = np.sum(mask2 > 0)

    if volume1 + volume2 == 0:
        return 0.0

    return 2.0 * intersection / (volume1 + volume2)


def generate_template_mask_from_inputs(
    template_path: Union[str, Path],
    input_image_paths: List[Union[str, Path]],
    input_mask_paths: List[Union[str, Path]],
    output_mask_path: Union[str, Path],
    threshold: float = 0.5
) -> Path:
    """
    使用输入 mask 配准后投票生成模板 mask

    这种方法比简单阈值分割更精确，因为它利用了 TotalSegmentator
    分割的高质量 mask。

    Args:
        template_path: 模板 CT 路径
        input_image_paths: 输入图像路径列表
        input_mask_paths: 对应的输入 mask 路径列表
        output_mask_path: 输出 mask 路径
        threshold: 投票阈值（默认 0.5，即多数投票）

    Returns:
        output_mask_path: 生成的 mask 路径
    """
    if not check_ants_available():
        raise ImportError("ANTsPy 不可用")

    logger.info("从输入 mask 生成模板 mask...")
    logger.info(f"  输入图像数: {len(input_image_paths)}")
    logger.info(f"  输入 mask 数: {len(input_mask_paths)}")

    template = ants.image_read(str(template_path))
    template_shape = template.shape

    # 累积配准后的 mask
    accumulated_mask = np.zeros(template_shape, dtype=np.float32)
    valid_count = 0

    for i, (img_path, mask_path) in enumerate(zip(input_image_paths, input_mask_paths)):
        img_path = Path(img_path)
        mask_path = Path(mask_path)

        if not img_path.exists() or not mask_path.exists():
            continue

        try:
            # 加载图像和 mask
            img = ants.image_read(str(img_path))
            mask = ants.image_read(str(mask_path))

            # 配准图像到模板
            registration = ants.registration(
                fixed=template,
                moving=img,
                type_of_transform='SyN'  # 使用非线性配准
            )

            # 将 mask 应用相同的变换
            warped_mask = ants.apply_transforms(
                fixed=template,
                moving=mask,
                transformlist=registration['fwdtransforms'],
                interpolator='nearestNeighbor'
            )

            accumulated_mask += (warped_mask.numpy() > 0).astype(np.float32)
            valid_count += 1
            logger.info(f"  [{i+1}/{len(input_image_paths)}] 配准完成: {img_path.name}")

        except Exception as e:
            logger.warning(f"  [{i+1}/{len(input_image_paths)}] 配准失败: {e}")

    if valid_count == 0:
        logger.error("没有成功配准任何 mask，回退到阈值分割")
        return generate_template_mask(template_path, output_mask_path)

    # 投票生成最终 mask
    probability_map = accumulated_mask / valid_count
    final_mask = (probability_map >= threshold).astype(np.uint8)

    # 形态学清理
    if ndimage is not None:
        from scipy.ndimage import binary_closing, binary_opening
        final_mask = binary_opening(final_mask, iterations=2).astype(np.uint8)
        final_mask = binary_closing(final_mask, iterations=3).astype(np.uint8)

    # 保存
    output_mask_path = Path(output_mask_path)
    _, affine = load_nifti(template_path, return_affine=True)
    save_nifti(final_mask, output_mask_path, affine=affine, dtype="uint8")

    voxel_count = np.sum(final_mask)
    logger.info(f"模板 mask 已保存（从 {valid_count} 个输入生成）: {output_mask_path}")
    logger.info(f"  肺部体素数: {voxel_count:,}")

    return output_mask_path


def evaluate_template_quality(
    template_path: Union[str, Path],
    template_mask_path: Union[str, Path],
    sample_image_paths: List[Union[str, Path]],
    sample_mask_paths: Optional[List[Union[str, Path]]] = None,
    output_report_path: Optional[Union[str, Path]] = None,
    dice_threshold: float = 0.85
) -> Dict:
    """
    评估模板质量

    通过计算模板与各输入图像之间的 Dice 系数来评估模板质量

    Args:
        template_path: 模板路径
        template_mask_path: 模板 mask 路径
        sample_image_paths: 样本图像路径列表
        sample_mask_paths: 样本 mask 路径列表（可选）
        output_report_path: 报告输出路径
        dice_threshold: Dice 阈值（默认 0.85，快速测试可设为 0.50）

    Returns:
        metrics: 评估指标，包括 dice_scores, correlation_scores 等

    验收标准：
        - 平均 Dice >= dice_threshold
    """
    if not check_ants_available():
        raise ImportError("ANTsPy 不可用")

    logger.info("=" * 60)
    logger.info("开始评估模板质量...")
    logger.info(f"Dice 阈值: {dice_threshold}")
    logger.info("=" * 60)

    template = load_template(template_path)
    template_mask_data = load_nifti(template_mask_path)

    metrics = {
        'dice_scores': [],
        'correlation_scores': [],
        'sample_names': [],
        'passed': False,
        'threshold': dice_threshold
    }

    start_time = time.time()
    total = len(sample_image_paths)

    for i, path in enumerate(sample_image_paths, 1):
        path = Path(path)
        logger.info(f"  [{i}/{total}] 评估: {path.name}")

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
            metrics['correlation_scores'].append(abs(corr))  # 取绝对值

            # 如果提供了 mask，计算 Dice
            if sample_mask_paths and i <= len(sample_mask_paths):
                mask_path = sample_mask_paths[i - 1]
                if Path(mask_path).exists():
                    sample_mask = ants.image_read(str(mask_path))
                    warped_mask = ants.apply_transforms(
                        fixed=template,
                        moving=sample_mask,
                        transformlist=registration['fwdtransforms'],
                        interpolator='nearestNeighbor'
                    )

                    dice = compute_dice(template_mask_data, warped_mask.numpy())
                    metrics['dice_scores'].append(dice)
                    logger.info(f"    Dice = {dice:.4f}, Corr = {abs(corr):.4f}")
            else:
                # 如果没有 mask，使用阈值生成临时 mask 计算 Dice
                warped_data = warped.numpy()
                warped_lung = ((warped_data >= -950) & (warped_data <= -200)).astype(np.uint8)
                dice = compute_dice(template_mask_data, warped_lung)
                metrics['dice_scores'].append(dice)
                logger.info(f"    Dice = {dice:.4f}, Corr = {abs(corr):.4f}")

            metrics['sample_names'].append(path.name)

        except Exception as e:
            logger.warning(f"    评估失败: {e}")

    elapsed = time.time() - start_time

    # 汇总统计
    if metrics['dice_scores']:
        metrics['mean_dice'] = np.mean(metrics['dice_scores'])
        metrics['std_dice'] = np.std(metrics['dice_scores'])
        metrics['min_dice'] = np.min(metrics['dice_scores'])
        metrics['max_dice'] = np.max(metrics['dice_scores'])

    if metrics['correlation_scores']:
        metrics['mean_correlation'] = np.mean(metrics['correlation_scores'])
        metrics['std_correlation'] = np.std(metrics['correlation_scores'])

    # 检查是否通过验收
    if metrics.get('mean_dice', 0) >= metrics['threshold']:
        metrics['passed'] = True
        status = "✅ 通过"
    else:
        status = "❌ 未通过"

    logger.info("=" * 60)
    logger.info("模板质量评估结果:")
    logger.info(f"  评估样本数: {len(metrics['dice_scores'])}")
    logger.info(f"  平均 Dice: {metrics.get('mean_dice', 0):.4f} ± {metrics.get('std_dice', 0):.4f}")
    logger.info(f"  Dice 范围: [{metrics.get('min_dice', 0):.4f}, {metrics.get('max_dice', 0):.4f}]")
    logger.info(f"  平均相关性: {metrics.get('mean_correlation', 0):.4f}")
    logger.info(f"  验收标准: Dice >= {metrics['threshold']}")
    logger.info(f"  验收结果: {status}")
    logger.info(f"  评估耗时: {elapsed:.1f} 秒")
    logger.info("=" * 60)

    # 保存报告
    if output_report_path:
        import json
        output_report_path = Path(output_report_path)
        output_report_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'timestamp': datetime.now().isoformat(),
            'template_path': str(template_path),
            'num_samples': len(metrics['dice_scores']),
            'mean_dice': metrics.get('mean_dice', 0),
            'std_dice': metrics.get('std_dice', 0),
            'min_dice': metrics.get('min_dice', 0),
            'max_dice': metrics.get('max_dice', 0),
            'mean_correlation': metrics.get('mean_correlation', 0),
            'passed': metrics['passed'],
            'threshold': metrics['threshold'],
            'samples': [
                {'name': name, 'dice': dice}
                for name, dice in zip(metrics['sample_names'], metrics['dice_scores'])
            ]
        }

        with open(output_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"评估报告已保存: {output_report_path}")

    return metrics


def validate_atlas(
    template_path: Union[str, Path],
    mask_path: Union[str, Path]
) -> Dict:
    """
    验证 Atlas 文件的基本有效性

    检查项：
    - 文件存在
    - 文件大小 > 10MB
    - 形状合理
    - 数据范围正常

    Args:
        template_path: 模板路径
        mask_path: mask 路径

    Returns:
        validation_result: 验证结果
    """
    logger.info("验证 Atlas 文件...")

    result = {
        'valid': True,
        'checks': [],
        'errors': []
    }

    template_path = Path(template_path)
    mask_path = Path(mask_path)

    # 检查文件存在
    if not template_path.exists():
        result['valid'] = False
        result['errors'].append(f"模板文件不存在: {template_path}")
    else:
        result['checks'].append(f"✅ 模板文件存在")

        # 检查文件大小
        size_mb = template_path.stat().st_size / (1024 * 1024)
        if size_mb < 10:
            result['valid'] = False
            result['errors'].append(f"模板文件太小: {size_mb:.1f} MB < 10 MB")
        else:
            result['checks'].append(f"✅ 模板文件大小: {size_mb:.1f} MB")

    if not mask_path.exists():
        result['valid'] = False
        result['errors'].append(f"Mask 文件不存在: {mask_path}")
    else:
        result['checks'].append(f"✅ Mask 文件存在")

    # 检查数据
    if template_path.exists() and mask_path.exists():
        try:
            template_data = load_nifti(template_path)
            mask_data = load_nifti(mask_path)

            # 形状检查
            if template_data.shape != mask_data.shape:
                result['valid'] = False
                result['errors'].append(
                    f"形状不匹配: 模板 {template_data.shape} vs Mask {mask_data.shape}"
                )
            else:
                result['checks'].append(f"✅ 形状匹配: {template_data.shape}")

            # HU 值范围检查
            min_hu = template_data.min()
            max_hu = template_data.max()
            if min_hu < -1100 or max_hu > 500:
                result['checks'].append(f"⚠️ HU 范围异常: [{min_hu:.0f}, {max_hu:.0f}]")
            else:
                result['checks'].append(f"✅ HU 范围正常: [{min_hu:.0f}, {max_hu:.0f}]")

            # Mask 体素数检查
            mask_voxels = np.sum(mask_data > 0)
            if mask_voxels < 1000000:
                result['checks'].append(f"⚠️ Mask 体素数较少: {mask_voxels:,}")
            else:
                result['checks'].append(f"✅ Mask 体素数: {mask_voxels:,}")

        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"读取文件失败: {e}")

    # 输出结果
    logger.info("验证结果:")
    for check in result['checks']:
        logger.info(f"  {check}")
    for error in result['errors']:
        logger.error(f"  ❌ {error}")

    if result['valid']:
        logger.info("✅ Atlas 验证通过")
    else:
        logger.error("❌ Atlas 验证失败")

    return result


def main(
    config: Optional[Dict] = None,
    num_images: Optional[int] = None,
    skip_evaluation: bool = False,
    quick_test: bool = False
) -> Dict:
    """
    主函数 - 从配置运行完整的 Atlas 构建流程

    Args:
        config: 配置字典，如果为 None 则从 config.yaml 加载
        num_images: 使用的图像数量，如果为 None 则使用所有可用图像
        skip_evaluation: 是否跳过质量评估步骤
        quick_test: 快速测试模式（使用较少迭代和图像）

    Returns:
        result: 包含模板路径、mask 路径、验证结果等
    """
    import yaml

    # 加载配置
    if config is None:
        config_path = Path(__file__).parent.parent.parent / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            config = {}

    # 获取配置
    atlas_config = config.get('registration', {}).get('template_build', {})
    paths_config = config.get('paths', {})

    # 输入路径
    input_dir = Path(paths_config.get('cleaned_data', 'data/01_cleaned')) / 'normal_clean'
    mask_dir = Path(paths_config.get('cleaned_data', 'data/01_cleaned')) / 'normal_mask'

    # 输出路径
    output_dir = Path(paths_config.get('atlas', 'data/02_atlas'))
    output_dir.mkdir(parents=True, exist_ok=True)

    template_path = output_dir / 'standard_template.nii.gz'
    mask_path = output_dir / 'standard_mask.nii.gz'
    report_path = output_dir / 'atlas_evaluation_report.json'

    # 获取所有输入图像
    image_paths = sorted(list(input_dir.glob('*.nii.gz')))
    mask_paths = sorted(list(mask_dir.glob('*.nii.gz'))) if mask_dir.exists() else []

    logger.info("=" * 70)
    logger.info("Phase 2: Atlas Construction - 标准底座构建")
    logger.info("=" * 70)
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"发现 {len(image_paths)} 个输入图像")
    logger.info(f"输出目录: {output_dir}")

    if len(image_paths) < 2:
        logger.error(f"输入图像不足: {len(image_paths)} < 2")
        return {'success': False, 'error': '输入图像不足'}

    # 限制图像数量
    if num_images is not None and num_images < len(image_paths):
        image_paths = image_paths[:num_images]
        logger.info(f"限制使用前 {num_images} 个图像")

    # 快速测试模式
    if quick_test:
        logger.warning("⚡ 快速测试模式：使用最少迭代")
        iteration_limit = 2
        image_paths = image_paths[:3] if len(image_paths) >= 3 else image_paths
    else:
        iteration_limit = atlas_config.get('iteration_limit', 5)

    # 记录开始时间
    total_start = time.time()

    # Step 1: 构建模板
    logger.info("")
    logger.info("Step 1/3: 构建标准模板...")
    try:
        build_template(
            image_paths=image_paths,
            output_path=template_path,
            type_of_transform=atlas_config.get('type_of_transform', 'SyN'),
            iteration_limit=iteration_limit,
            gradient_step=atlas_config.get('gradient_step', 0.2),
        )
    except Exception as e:
        logger.error(f"模板构建失败: {e}")
        return {'success': False, 'error': str(e)}

    # Step 2: 生成模板 mask
    logger.info("")
    logger.info("Step 2/3: 生成模板肺部 mask...")
    try:
        # 优先使用输入 mask 配准生成（更精确），快速测试时用阈值分割（更快）
        if not quick_test and len(mask_paths) >= 3:
            logger.info("使用输入 mask 配准生成（高精度模式）...")
            generate_template_mask_from_inputs(
                template_path=template_path,
                input_image_paths=image_paths[:min(5, len(image_paths))],  # 最多用 5 个
                input_mask_paths=mask_paths[:min(5, len(mask_paths))],
                output_mask_path=mask_path
            )
        else:
            logger.info("使用阈值分割生成（快速模式）...")
            generate_template_mask(
                template_path=template_path,
                output_mask_path=mask_path
            )
    except Exception as e:
        logger.error(f"Mask 生成失败: {e}")
        return {'success': False, 'error': str(e)}

    # Step 3: 验证 Atlas
    logger.info("")
    logger.info("Step 3/3: 验证 Atlas 质量...")

    validation_result = validate_atlas(template_path, mask_path)

    # 质量评估（可选）
    evaluation_result = None
    if not skip_evaluation and len(image_paths) >= 3:
        logger.info("")
        logger.info("进行质量评估（抽取 3 个样本）...")

        # 抽取 3 个样本进行评估
        eval_images = image_paths[:3]
        eval_masks = mask_paths[:3] if len(mask_paths) >= 3 else None

        # 快速测试使用较低阈值，正常运行使用 0.85
        dice_threshold = 0.50 if quick_test else 0.85
        if quick_test:
            logger.warning("⚠️ 快速测试模式：使用较低的 Dice 阈值 (0.50)")

        try:
            evaluation_result = evaluate_template_quality(
                template_path=template_path,
                template_mask_path=mask_path,
                sample_image_paths=eval_images,
                sample_mask_paths=eval_masks,
                output_report_path=report_path,
                dice_threshold=dice_threshold
            )
        except Exception as e:
            logger.warning(f"质量评估失败: {e}")

    total_elapsed = time.time() - total_start

    # 结果汇总
    result = {
        'success': validation_result['valid'],
        'template_path': str(template_path),
        'mask_path': str(mask_path),
        'num_input_images': len(image_paths),
        'total_time_seconds': total_elapsed,
        'total_time_str': str(timedelta(seconds=int(total_elapsed))),
        'validation': validation_result,
        'evaluation': evaluation_result
    }

    logger.info("")
    logger.info("=" * 70)
    logger.info("Phase 2 完成!")
    logger.info("=" * 70)
    logger.info(f"模板文件: {template_path}")
    logger.info(f"Mask 文件: {mask_path}")
    logger.info(f"总耗时: {result['total_time_str']}")

    if evaluation_result and evaluation_result.get('passed'):
        logger.info(f"✅ 质量验收: 通过 (Dice = {evaluation_result.get('mean_dice', 0):.4f})")
    elif evaluation_result:
        logger.warning(f"⚠️ 质量验收: 未通过 (Dice = {evaluation_result.get('mean_dice', 0):.4f} < 0.85)")

    logger.info("=" * 70)

    return result


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='Phase 2: Atlas Construction')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--num-images', type=int, default=None, help='使用的图像数量')
    parser.add_argument('--skip-eval', action='store_true', help='跳过质量评估')
    parser.add_argument('--quick-test', action='store_true', help='快速测试模式')

    args = parser.parse_args()

    # 加载配置
    config = None
    if Path(args.config).exists():
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    result = main(
        config=config,
        num_images=args.num_images,
        skip_evaluation=args.skip_eval,
        quick_test=args.quick_test
    )

    # 返回退出码
    import sys
    sys.exit(0 if result.get('success') else 1)

