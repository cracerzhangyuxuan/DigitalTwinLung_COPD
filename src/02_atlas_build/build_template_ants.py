#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准底座构建模块 (Phase 2: Atlas Construction)

使用 ANTsPy 的 build_template 函数从多例正常肺 CT 构建标准模板。

主要功能：
- build_template(): 从多例正常肺构建标准模板
- generate_template_mask(): 为模板生成肺部 mask
- generate_template_trachea_mask(): 为模板生成气管树 mask
- evaluate_template_quality(): 评估模板质量（Dice >= 0.85）
- validate_atlas(): 完整的质量验证流程

使用方法：
    python -m src.02_atlas_build.build_template_ants
    或通过 run_phase2_atlas.py 入口脚本运行

验收标准（来自 Engineering_Edition.md）：
- standard_template.nii.gz 文件大小 > 10MB
- 与任一输入肺的 Dice >= 0.85
- 血管/气管结构可辨识
- standard_trachea_mask.nii.gz 气管树连续性验证

作者: DigitalTwinLung_COPD Team
更新: 2025-12-09
更新: 2025-12-22 - 添加气管树模板 mask 生成功能
更新: 2025-12-24 - 添加 Joint Label Fusion (JLF) 支持，更新分割模型说明
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

    这种方法比简单阈值分割更精确，因为它利用了 LungMask + Raidionicsrads
    分割的高质量 mask。（注：已于 2025-12-24 从 TotalSegmentator 替换）

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

    # 形态学清理（恢复原始参数）
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


def generate_template_trachea_mask(
    template_path: Union[str, Path],
    input_image_paths: List[Union[str, Path]],
    input_trachea_paths: List[Union[str, Path]],
    output_trachea_path: Union[str, Path],
    threshold: float = 0.25  # 降低阈值以保留更多分支结构
) -> Optional[Path]:
    """
    从输入气管树 mask 配准生成模板气管树 mask

    通过将多个输入样本的气管树 mask 配准到模板空间并投票，
    生成标准模板的气管树 mask。

    Args:
        template_path: 模板 CT 路径
        input_image_paths: 输入图像路径列表
        input_trachea_paths: 对应的输入气管树 mask 路径列表
        output_trachea_path: 输出气管树模板 mask 路径
        threshold: 投票阈值（默认 0.5）

    Returns:
        output_trachea_path: 生成的气管树模板 mask 路径，失败返回 None

    Note:
        气管树结构相对稳定，配准后应保持连续性。
        如果连通性检查失败，将输出警告但仍保存结果。
    """
    if not check_ants_available():
        raise ImportError("ANTsPy 不可用")

    logger.info("=" * 60)
    logger.info("生成模板气管树 mask...")
    logger.info(f"  输入图像数: {len(input_image_paths)}")
    logger.info(f"  输入气管 mask 数: {len(input_trachea_paths)}")
    logger.info("=" * 60)

    # 检查输入
    if len(input_trachea_paths) == 0:
        logger.warning("没有输入气管树 mask，跳过气管树模板生成")
        return None

    template = ants.image_read(str(template_path))
    template_shape = template.shape

    # 累积配准后的气管树 mask
    accumulated_mask = np.zeros(template_shape, dtype=np.float32)
    valid_count = 0

    for i, (img_path, trachea_path) in enumerate(zip(input_image_paths, input_trachea_paths)):
        img_path = Path(img_path)
        trachea_path = Path(trachea_path)

        if not img_path.exists() or not trachea_path.exists():
            logger.warning(f"  [{i+1}] 文件缺失，跳过")
            continue

        try:
            # 加载图像和气管 mask
            img = ants.image_read(str(img_path))
            trachea_mask = ants.image_read(str(trachea_path))

            # 配准图像到模板
            logger.info(f"  [{i+1}/{len(input_image_paths)}] 配准: {img_path.name}")
            registration = ants.registration(
                fixed=template,
                moving=img,
                type_of_transform='SyN'
            )

            # 将气管 mask 应用相同的变换
            warped_trachea = ants.apply_transforms(
                fixed=template,
                moving=trachea_mask,
                transformlist=registration['fwdtransforms'],
                interpolator='nearestNeighbor'  # 最近邻插值保持二值
            )

            accumulated_mask += (warped_trachea.numpy() > 0).astype(np.float32)
            valid_count += 1
            logger.info(f"    ✓ 配准成功")

        except Exception as e:
            logger.warning(f"  [{i+1}/{len(input_image_paths)}] 配准失败: {e}")

    if valid_count == 0:
        logger.error("没有成功配准任何气管树 mask")
        return None

    # 投票生成最终 mask
    probability_map = accumulated_mask / valid_count
    final_trachea_mask = (probability_map >= threshold).astype(np.uint8)

    # 形态学清理（保守处理以保留分支结构）
    if ndimage is not None:
        from scipy.ndimage import binary_closing, binary_dilation
        # 仅使用闭操作保持连续性，不使用开操作（会移除细小分支）
        final_trachea_mask = binary_closing(final_trachea_mask, iterations=1).astype(np.uint8)
        # 轻微膨胀以连接断裂的分支
        final_trachea_mask = binary_dilation(final_trachea_mask, iterations=1).astype(np.uint8)
        final_trachea_mask = binary_closing(final_trachea_mask, iterations=1).astype(np.uint8)

    # 连通性验证
    if ndimage is not None:
        labeled, num_features = ndimage.label(final_trachea_mask)
        if num_features > 1:
            # 保留最大连通域（主气管树）
            sizes = ndimage.sum(final_trachea_mask, labeled, range(1, num_features + 1))
            largest_idx = np.argmax(sizes) + 1
            final_trachea_mask = (labeled == largest_idx).astype(np.uint8)
            logger.warning(f"  气管树有 {num_features} 个连通域，保留最大的")
        else:
            logger.info(f"  ✓ 气管树连续性检查通过")

    # 保存
    output_trachea_path = Path(output_trachea_path)
    output_trachea_path.parent.mkdir(parents=True, exist_ok=True)
    _, affine = load_nifti(template_path, return_affine=True)
    save_nifti(final_trachea_mask, output_trachea_path, affine=affine, dtype="uint8")

    final_voxels = np.sum(final_trachea_mask)
    logger.info(f"气管树模板 mask 已保存（从 {valid_count} 个输入生成）: {output_trachea_path}")
    logger.info(f"  气管树体素数: {final_voxels:,}")

    return output_trachea_path


def generate_template_lung_lobes(
    template_path: Union[str, Path],
    input_image_paths: List[Union[str, Path]],
    input_lobes_paths: List[Union[str, Path]],
    output_lobes_path: Union[str, Path],
    threshold: float = 0.3
) -> Optional[Path]:
    """
    使用输入的肺叶标签 mask 配准后投票生成模板肺叶标签 mask

    标签定义:
        1 = 左上叶 (Left Upper Lobe)
        2 = 左下叶 (Left Lower Lobe)
        3 = 右上叶 (Right Upper Lobe)
        4 = 右中叶 (Right Middle Lobe)
        5 = 右下叶 (Right Lower Lobe)

    Args:
        template_path: 模板 CT 路径
        input_image_paths: 输入图像路径列表
        input_lobes_paths: 输入肺叶标签 mask 路径列表
        output_lobes_path: 输出肺叶标签模板路径
        threshold: 投票阈值（默认 0.3，因为边界区域可能重叠）

    Returns:
        output_lobes_path: 生成的肺叶模板路径，失败返回 None
    """
    if not check_ants_available():
        raise ImportError("ANTsPy 不可用")

    logger.info("=" * 60)
    logger.info("生成模板肺叶标签 mask...")
    logger.info(f"  输入图像数: {len(input_image_paths)}")
    logger.info(f"  输入肺叶标签数: {len(input_lobes_paths)}")
    logger.info("=" * 60)

    if len(input_lobes_paths) == 0:
        logger.warning("没有输入肺叶标签 mask，跳过生成")
        return None

    template = ants.image_read(str(template_path))
    template_shape = template.shape

    # 为每个肺叶标签（1-5）累积投票
    lobe_accumulators = {i: np.zeros(template_shape, dtype=np.float32) for i in range(1, 6)}
    valid_count = 0

    for i, (img_path, lobes_path) in enumerate(zip(input_image_paths, input_lobes_paths)):
        img_path = Path(img_path)
        lobes_path = Path(lobes_path)

        if not img_path.exists() or not lobes_path.exists():
            logger.warning(f"  [{i+1}] 文件缺失，跳过")
            continue

        try:
            img = ants.image_read(str(img_path))
            lobes_mask = ants.image_read(str(lobes_path))

            logger.info(f"  [{i+1}/{len(input_image_paths)}] 配准: {img_path.name}")
            registration = ants.registration(
                fixed=template,
                moving=img,
                type_of_transform='SyN'
            )

            # 对每个肺叶标签分别变换
            for label in range(1, 6):
                # 提取单个标签
                label_mask_data = (lobes_mask.numpy() == label).astype(np.float32)
                label_mask = ants.from_numpy(label_mask_data, origin=lobes_mask.origin,
                                             spacing=lobes_mask.spacing, direction=lobes_mask.direction)

                # 应用变换
                warped_label = ants.apply_transforms(
                    fixed=template,
                    moving=label_mask,
                    transformlist=registration['fwdtransforms'],
                    interpolator='linear'
                )

                lobe_accumulators[label] += warped_label.numpy()

            valid_count += 1

        except Exception as e:
            logger.warning(f"  [{i+1}] 配准失败: {e}")
            continue

    if valid_count == 0:
        logger.error("没有有效的配准结果")
        return None

    # 归一化投票
    for label in range(1, 6):
        lobe_accumulators[label] /= valid_count

    # 使用向量化操作进行投票（比三重循环快100倍以上）
    logger.info("  使用向量化投票算法...")

    # 将5个累积器堆叠成 (5, z, y, x) 的数组
    vote_stack = np.stack([lobe_accumulators[l] for l in range(1, 6)], axis=0)

    # 找到每个体素的最大投票值和对应标签
    max_votes = np.max(vote_stack, axis=0)
    argmax_labels = np.argmax(vote_stack, axis=0) + 1  # 标签从1开始

    # 应用阈值
    final_lobes = np.where(max_votes >= threshold, argmax_labels, 0).astype(np.uint8)

    # 注意：不对肺叶进行额外的形态学后处理，保持原始投票结果
    # 原始代码没有后处理，效果良好

    # 保存
    output_lobes_path = Path(output_lobes_path)
    output_lobes_path.parent.mkdir(parents=True, exist_ok=True)
    _, affine = load_nifti(template_path, return_affine=True)
    save_nifti(final_lobes, output_lobes_path, affine=affine, dtype="uint8")

    # 统计每个肺叶的体素数
    lobe_names = {1: "左上叶", 2: "左下叶", 3: "右上叶", 4: "右中叶", 5: "右下叶"}
    logger.info(f"肺叶模板已保存（从 {valid_count} 个输入生成）: {output_lobes_path}")
    for label in range(1, 6):
        count = np.sum(final_lobes == label)
        logger.info(f"  {lobe_names[label]}: {count:,} 体素")

    return output_lobes_path


def generate_template_lung_lobes_jlf(
    template_path: Union[str, Path],
    input_image_paths: List[Union[str, Path]],
    input_lobes_paths: List[Union[str, Path]],
    output_lobes_path: Union[str, Path],
    beta: float = 4,
    rad: int = 2,
    rho: float = 0.01,
    r_search: int = 3
) -> Optional[Path]:
    """
    使用 Joint Label Fusion (JLF) 算法生成模板肺叶标签 mask

    JLF 相比简单投票算法的优势:
    - 边界更平滑，减少碎片化
    - 对配准误差更鲁棒（通过加权融合）
    - 考虑局部强度相似性，不仅仅是投票数

    JLF 工作原理:
    1. 将所有 atlas 图像和标签配准到目标模板空间
    2. 对于每个目标体素，根据局部强度相似性计算每个 atlas 的权重
    3. 使用加权投票确定最终标签
    4. beta 参数控制权重锐度（越高相似 atlas 权重越大）

    Args:
        template_path: 模板 CT 路径
        input_image_paths: 输入图像路径列表
        input_lobes_paths: 输入肺叶标签 mask 路径列表
        output_lobes_path: 输出肺叶标签模板路径
        beta: 权重锐度参数，默认 4（越高相似样本权重越大）
        rad: 邻域半径，默认 2
        rho: 岭惩罚，增加对异常值的鲁棒性，默认 0.01
        r_search: 搜索半径，默认 3

    Returns:
        output_lobes_path: 生成的肺叶模板路径，失败返回 None

    Note:
        JLF 计算量较大，建议在 GPU 服务器上运行。
        可通过设置 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS 控制线程数。
    """
    if not check_ants_available():
        raise ImportError("ANTsPy 不可用")

    logger.info("=" * 60)
    logger.info("使用 Joint Label Fusion (JLF) 生成模板肺叶标签...")
    logger.info(f"  输入图像数: {len(input_image_paths)}")
    logger.info(f"  JLF 参数: beta={beta}, rad={rad}, rho={rho}, r_search={r_search}")
    logger.info("=" * 60)

    if len(input_lobes_paths) == 0:
        logger.warning("没有输入肺叶标签 mask，跳过生成")
        return None

    # 加载模板
    template = ants.image_read(str(template_path))

    # 生成模板 mask（JLF 需要 mask 参数）
    template_mask = ants.get_mask(template, low_thresh=template.min() + 100)

    # 配准所有图像到模板空间
    warped_images = []
    warped_labels = []

    for i, (img_path, lobes_path) in enumerate(zip(input_image_paths, input_lobes_paths)):
        img_path = Path(img_path)
        lobes_path = Path(lobes_path)

        if not img_path.exists() or not lobes_path.exists():
            logger.warning(f"  [{i+1}] 文件缺失，跳过")
            continue

        try:
            img = ants.image_read(str(img_path))
            lobes_mask = ants.image_read(str(lobes_path))

            logger.info(f"  [{i+1}/{len(input_image_paths)}] 配准: {img_path.name}")
            registration = ants.registration(
                fixed=template,
                moving=img,
                type_of_transform='SyN'
            )

            # 应用变换到图像
            warped_img = ants.apply_transforms(
                fixed=template,
                moving=img,
                transformlist=registration['fwdtransforms']
            )

            # 应用变换到标签（使用最近邻插值）
            warped_label = ants.apply_transforms(
                fixed=template,
                moving=lobes_mask,
                transformlist=registration['fwdtransforms'],
                interpolator='nearestNeighbor'
            )

            warped_images.append(warped_img)
            warped_labels.append(warped_label)
            logger.info(f"    ✓ 配准成功")

        except Exception as e:
            logger.warning(f"  [{i+1}] 配准失败: {e}")
            continue

    if len(warped_images) < 2:
        logger.error("有效的配准结果不足（需要至少 2 个），回退到简单投票")
        return generate_template_lung_lobes(
            template_path, input_image_paths, input_lobes_paths, output_lobes_path
        )

    # 调用 ANTsPy 的 Joint Label Fusion
    logger.info(f"执行 Joint Label Fusion（{len(warped_images)} 个 atlas）...")

    try:
        jlf_result = ants.joint_label_fusion(
            target_image=template,
            target_image_mask=template_mask,
            atlas_list=warped_images,
            label_list=warped_labels,
            beta=beta,
            rad=[rad] * template.dimension,
            rho=rho,
            r_search=r_search,
            verbose=True
        )

        final_lobes = jlf_result['segmentation']

    except Exception as e:
        logger.error(f"JLF 执行失败: {e}")
        logger.info("回退到简单投票算法...")
        return generate_template_lung_lobes(
            template_path, input_image_paths, input_lobes_paths, output_lobes_path
        )

    # 保存结果
    output_lobes_path = Path(output_lobes_path)
    output_lobes_path.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(final_lobes, str(output_lobes_path))

    # 统计每个肺叶的体素数
    final_lobes_np = final_lobes.numpy()
    lobe_names = {1: "左上叶", 2: "左下叶", 3: "右上叶", 4: "右中叶", 5: "右下叶"}
    logger.info(f"JLF 肺叶模板已保存: {output_lobes_path}")
    for label in range(1, 6):
        count = np.sum(final_lobes_np == label)
        logger.info(f"  {lobe_names[label]}: {count:,} 体素")

    return output_lobes_path


def validate_trachea_continuity(
    trachea_mask_path: Union[str, Path]
) -> Dict:
    """
    验证气管树 mask 的连续性和解剖合理性

    Args:
        trachea_mask_path: 气管树 mask 路径

    Returns:
        validation_result: 验证结果字典
    """
    logger.info("验证气管树连续性...")

    result = {
        'valid': True,
        'checks': [],
        'warnings': []
    }

    trachea_path = Path(trachea_mask_path)

    if not trachea_path.exists():
        result['valid'] = False
        result['warnings'].append(f"气管树 mask 不存在: {trachea_path}")
        return result

    try:
        trachea_data = load_nifti(trachea_path)
        voxel_count = np.sum(trachea_data > 0)

        # 检查体素数量（气管树通常在 5000-50000 体素之间）
        if voxel_count < 1000:
            result['warnings'].append(f"气管树体素数过少: {voxel_count}")
        elif voxel_count > 100000:
            result['warnings'].append(f"气管树体素数异常多: {voxel_count}")
        else:
            result['checks'].append(f"✓ 气管树体素数合理: {voxel_count:,}")

        # 连通性检查
        if ndimage is not None:
            _, num_features = ndimage.label(trachea_data > 0)
            if num_features == 1:
                result['checks'].append("✓ 气管树完全连续（单连通域）")
            elif num_features <= 3:
                result['warnings'].append(f"气管树有 {num_features} 个连通域（可能是分支）")
            else:
                result['warnings'].append(f"气管树碎片化: {num_features} 个连通域")

    except Exception as e:
        result['valid'] = False
        result['warnings'].append(f"读取气管树 mask 失败: {e}")

    # 输出结果
    for check in result['checks']:
        logger.info(f"  {check}")
    for warning in result['warnings']:
        logger.warning(f"  ⚠️ {warning}")

    return result


def validate_lung_lobes(
    lobes_mask_path: Union[str, Path],
    min_lobe_voxels: int = 100000,
    max_lobe_ratio: float = 5.0
) -> Dict:
    """
    验证肺叶标签 mask 的完整性和合理性

    检查项：
    1. 5 个肺叶是否都存在
    2. 每个肺叶的体素数是否合理
    3. 肺叶之间的大小比例是否正常

    Args:
        lobes_mask_path: 肺叶标签 mask 路径
        min_lobe_voxels: 最小肺叶体素数（默认 100,000）
        max_lobe_ratio: 最大/最小肺叶体积比上限（默认 5.0）

    Returns:
        validation_result: 验证结果字典
    """
    logger.info("验证肺叶标签完整性...")

    result = {
        'valid': True,
        'checks': [],
        'warnings': [],
        'lobe_stats': {}
    }

    lobe_names = {1: "左上叶", 2: "左下叶", 3: "右上叶", 4: "右中叶", 5: "右下叶"}
    lobes_path = Path(lobes_mask_path)

    if not lobes_path.exists():
        result['valid'] = False
        result['warnings'].append(f"肺叶标签 mask 不存在: {lobes_path}")
        return result

    try:
        lobes_data = load_nifti(lobes_path)

        # 统计每个肺叶的体素数
        lobe_counts = {}
        missing_lobes = []
        small_lobes = []

        for label in range(1, 6):
            count = np.sum(lobes_data == label)
            lobe_counts[label] = count
            result['lobe_stats'][lobe_names[label]] = count

            if count == 0:
                missing_lobes.append(lobe_names[label])
            elif count < min_lobe_voxels:
                small_lobes.append(f"{lobe_names[label]}({count:,})")

        # 检查 1: 是否所有肺叶都存在
        if missing_lobes:
            result['valid'] = False
            result['warnings'].append(f"缺失肺叶: {', '.join(missing_lobes)}")
        else:
            result['checks'].append("✓ 5 个肺叶全部存在")

        # 检查 2: 肺叶大小是否合理
        if small_lobes:
            result['warnings'].append(f"体积过小的肺叶: {', '.join(small_lobes)}")

        # 检查 3: 肺叶比例是否正常
        valid_counts = [c for c in lobe_counts.values() if c > 0]
        if len(valid_counts) >= 2:
            ratio = max(valid_counts) / min(valid_counts)
            if ratio > max_lobe_ratio:
                result['warnings'].append(f"肺叶大小差异过大: 比例 {ratio:.1f}x")
            else:
                result['checks'].append(f"✓ 肺叶大小比例正常: {ratio:.1f}x")

        # 检查 4: 左肺和右肺的分离（标签 1,2 为左肺，3,4,5 为右肺）
        left_lung = (lobes_data == 1) | (lobes_data == 2)
        right_lung = (lobes_data == 3) | (lobes_data == 4) | (lobes_data == 5)

        left_count = np.sum(left_lung)
        right_count = np.sum(right_lung)

        if left_count > 0 and right_count > 0:
            lr_ratio = max(left_count, right_count) / min(left_count, right_count)
            if lr_ratio > 2.5:
                result['warnings'].append(f"左右肺大小差异过大: {lr_ratio:.1f}x")
            else:
                result['checks'].append(f"✓ 左右肺比例正常: 左{left_count:,} / 右{right_count:,}")

    except Exception as e:
        result['valid'] = False
        result['warnings'].append(f"读取肺叶标签 mask 失败: {e}")

    # 输出结果
    for check in result['checks']:
        logger.info(f"  {check}")
    for warning in result['warnings']:
        logger.warning(f"  ⚠️ {warning}")

    return result


def validate_lung_separation(
    lung_mask_path: Union[str, Path],
    min_gap_ratio: float = 0.001
) -> Dict:
    """
    验证左右肺的分离度

    通过检查纵隔区域（肺的中间区域）是否有足够的间隙来判断分离度

    Args:
        lung_mask_path: 肺部 mask 路径
        min_gap_ratio: 中间切片最小间隙比例（默认 0.1%）

    Returns:
        validation_result: 验证结果字典
    """
    logger.info("验证左右肺分离度...")

    result = {
        'valid': True,
        'checks': [],
        'warnings': [],
        'separation_score': 0.0
    }

    mask_path = Path(lung_mask_path)

    if not mask_path.exists():
        result['valid'] = False
        result['warnings'].append(f"肺部 mask 不存在: {mask_path}")
        return result

    try:
        mask_data = load_nifti(mask_path)

        # 检查中间 1/3 区域的分离度
        # 假设 X 轴（第一个维度）是左右方向
        shape = mask_data.shape

        # 尝试不同的轴向来找到左右分离
        best_separation = 0.0
        best_axis = None

        for axis in range(3):
            mid_start = shape[axis] // 3
            mid_end = 2 * shape[axis] // 3

            # 提取中间切片
            if axis == 0:
                mid_region = mask_data[mid_start:mid_end, :, :]
            elif axis == 1:
                mid_region = mask_data[:, mid_start:mid_end, :]
            else:
                mid_region = mask_data[:, :, mid_start:mid_end]

            # 计算每个切片的间隙
            total_voxels = mid_region.size
            lung_voxels = np.sum(mid_region > 0)
            gap_voxels = total_voxels - lung_voxels

            # 计算间隙比例
            gap_ratio = gap_voxels / total_voxels if total_voxels > 0 else 0

            if gap_ratio > best_separation:
                best_separation = gap_ratio
                best_axis = axis

        result['separation_score'] = best_separation

        # 检查纵隔中心线的间隙
        center_axis = 0  # 假设 X 轴是左右方向
        center_slice_idx = shape[center_axis] // 2
        center_slice = mask_data[center_slice_idx, :, :]

        # 计算中心切片的肺部占比
        center_lung_ratio = np.sum(center_slice > 0) / center_slice.size if center_slice.size > 0 else 0

        if center_lung_ratio > 0.8:
            result['warnings'].append(f"中心切片肺部占比过高 ({center_lung_ratio:.1%})，可能存在粘连")
        elif center_lung_ratio < 0.1:
            result['checks'].append(f"✓ 纵隔间隙明显 (中心切片肺部占比 {center_lung_ratio:.1%})")
        else:
            result['checks'].append(f"✓ 左右肺分离度正常 (中心切片肺部占比 {center_lung_ratio:.1%})")

        # 使用连通域分析检查是否真的分离
        if ndimage is not None:
            labeled, num_features = ndimage.label(mask_data > 0)
            if num_features == 2:
                result['checks'].append("✓ 检测到 2 个独立连通域（左右肺完全分离）")
            elif num_features == 1:
                result['warnings'].append("仅检测到 1 个连通域（左右肺可能粘连）")
            else:
                result['warnings'].append(f"检测到 {num_features} 个连通域（异常）")

    except Exception as e:
        result['valid'] = False
        result['warnings'].append(f"读取肺部 mask 失败: {e}")

    # 输出结果
    for check in result['checks']:
        logger.info(f"  {check}")
    for warning in result['warnings']:
        logger.warning(f"  ⚠️ {warning}")

    return result


def run_all_quality_checks(
    output_dir: Union[str, Path]
) -> Dict:
    """
    运行所有质量检查

    Args:
        output_dir: Atlas 输出目录

    Returns:
        combined_result: 综合检查结果
    """
    output_dir = Path(output_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("运行完整质量检查...")
    logger.info("=" * 60)

    results = {
        'all_passed': True,
        'trachea': None,
        'lobes': None,
        'separation': None,
        'summary': []
    }

    # 1. 气管树检查
    trachea_path = output_dir / 'standard_trachea_mask.nii.gz'
    if trachea_path.exists():
        results['trachea'] = validate_trachea_continuity(trachea_path)
        if not results['trachea']['valid'] or results['trachea']['warnings']:
            results['all_passed'] = False

    # 2. 肺叶检查
    lobes_path = output_dir / 'standard_lung_lobes_labeled.nii.gz'
    if lobes_path.exists():
        results['lobes'] = validate_lung_lobes(lobes_path)
        if not results['lobes']['valid']:
            results['all_passed'] = False

    # 3. 肺分离度检查
    mask_path = output_dir / 'standard_mask.nii.gz'
    if mask_path.exists():
        results['separation'] = validate_lung_separation(mask_path)
        if not results['separation']['valid']:
            results['all_passed'] = False

    # 总结
    logger.info("")
    if results['all_passed']:
        logger.info("✅ 所有质量检查通过")
    else:
        logger.warning("⚠️ 部分质量检查存在警告，请查看详细信息")

    return results


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

            # 配准到模板（使用 SyN 非线性配准，与模板构建一致）
            # 注意：必须使用与 build_template() 和 generate_template_mask_from_inputs()
            # 相同的配准类型，否则 Dice 会显著偏低
            registration = ants.registration(
                fixed=template,
                moving=img,
                type_of_transform='SyN'  # 改为 SyN，与模板构建保持一致
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
    quick_test: bool = False,
    skip_template_build: bool = False
) -> Dict:
    """
    主函数 - 从配置运行完整的 Atlas 构建流程

    Args:
        config: 配置字典，如果为 None 则从 config.yaml 加载
        num_images: 使用的图像数量，如果为 None 则使用所有可用图像
        skip_evaluation: 是否跳过质量评估步骤
        quick_test: 快速测试模式（使用较少迭代和图像）
        skip_template_build: 是否跳过 Step 1（模板构建），直接从 Step 2 开始

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
    trachea_path = output_dir / 'standard_trachea_mask.nii.gz'
    lobes_path = output_dir / 'standard_lung_lobes_labeled.nii.gz'
    report_path = output_dir / 'atlas_evaluation_report.json'

    # 获取所有输入图像
    image_paths = sorted(list(input_dir.glob('*.nii.gz')))

    # 获取肺部 mask（排除气管树 mask 和肺叶标签 mask）
    # 肺部 mask 命名格式: *_mask.nii.gz（不包含 trachea 或 lung_lobes）
    all_mask_files = sorted(list(mask_dir.glob('*_mask.nii.gz'))) if mask_dir.exists() else []
    mask_paths = [p for p in all_mask_files
                  if not p.name.endswith('_trachea_mask.nii.gz')
                  and 'lung_lobes' not in p.name]

    # 获取气管树 mask（如果存在）
    trachea_mask_paths = sorted(list(mask_dir.glob('*_trachea_mask.nii.gz'))) if mask_dir.exists() else []

    # 获取肺叶标签 mask（如果存在）
    lobes_mask_paths = sorted(list(mask_dir.glob('*_lung_lobes_labeled.nii.gz'))) if mask_dir.exists() else []

    logger.info("=" * 70)
    logger.info("Phase 2: Atlas Construction - 标准底座构建")
    logger.info("=" * 70)
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"Mask 目录: {mask_dir}")
    logger.info(f"发现 {len(image_paths)} 个输入图像")
    logger.info(f"发现 {len(mask_paths)} 个输入肺部 mask")
    logger.info(f"发现 {len(trachea_mask_paths)} 个输入气管树 mask")
    logger.info(f"发现 {len(lobes_mask_paths)} 个输入肺叶标签 mask")
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

    # 确定步骤数（基于可用的输入类型）
    # 基础步骤: 模板构建(1) + 肺部mask(2) + 验证(N)
    # 可选步骤: 气管树(+1), 肺叶标签(+1)
    has_trachea = len(trachea_mask_paths) >= 3
    has_lobes = len(lobes_mask_paths) >= 3
    total_steps = 3  # 基础: 模板 + mask + 验证
    if has_trachea:
        total_steps += 1
    if has_lobes:
        total_steps += 1

    # Step 1: 构建模板
    if skip_template_build:
        logger.info("")
        logger.info(f"Step 1/{total_steps}: 跳过模板构建（使用已有模板）...")
        if not template_path.exists():
            error_msg = f"模板文件不存在: {template_path}\n请先运行完整的 Atlas 构建（不带 --skip-step1 参数）"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        logger.info(f"✓ 找到已有模板: {template_path}")
        # 获取模板信息
        try:
            import ants
            template = ants.image_read(str(template_path))
            logger.info(f"  模板形状: {template.shape}")
            logger.info(f"  模板间距: {template.spacing}")
        except Exception as e:
            logger.warning(f"  无法读取模板信息: {e}")
    else:
        logger.info("")
        logger.info(f"Step 1/{total_steps}: 构建标准模板...")
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

    # Step 2: 生成模板肺部 mask
    logger.info("")
    logger.info(f"Step 2/{total_steps}: 生成模板肺部 mask...")
    try:
        # 优先使用输入 mask 配准生成（更精确），快速测试时用阈值分割（更快）
        if not quick_test and len(mask_paths) >= 3:
            logger.info("✓ 条件满足: 使用输入 mask 配准生成（高精度模式）...")

            # 构建 image 和 mask 的对应关系（使用全部样本，而非只用 5 例）
            selected_images = []
            selected_masks = []
            for img_path in image_paths:  # 使用全部样本
                img_stem = img_path.stem.replace('.nii', '')
                sample_id = img_stem.replace('_clean', '')
                expected_mask = mask_dir / f"{sample_id}_mask.nii.gz"
                if expected_mask.exists():
                    selected_images.append(img_path)
                    selected_masks.append(expected_mask)

            logger.info(f"  配准样本数: {len(selected_images)}")

            if len(selected_images) >= 3:
                generate_template_mask_from_inputs(
                    template_path=template_path,
                    input_image_paths=selected_images,
                    input_mask_paths=selected_masks,
                    output_mask_path=mask_path
                )
            else:
                logger.warning(f"⚠ 配对的 mask 不足: 使用阈值分割生成...")
                generate_template_mask(
                    template_path=template_path,
                    output_mask_path=mask_path
                )
        else:
            if quick_test:
                logger.info("✓ 快速测试模式: 使用阈值分割生成...")
            else:
                logger.warning(f"⚠ mask 文件不足 ({len(mask_paths)} < 3): 使用阈值分割生成...")
            generate_template_mask(
                template_path=template_path,
                output_mask_path=mask_path
            )
    except Exception as e:
        logger.error(f"Mask 生成失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}

    # Step 3: 生成模板气管树 mask（如果有输入气管树 mask）
    # Step 3: 生成气管树模板（如果有输入）
    current_step = 3
    trachea_result = None
    if has_trachea:
        logger.info("")
        logger.info(f"Step {current_step}/{total_steps}: 生成模板气管树 mask...")
        logger.info(f"  配准样本数: {len(image_paths)} CT, {len(trachea_mask_paths)} 气管树")
        try:
            trachea_result = generate_template_trachea_mask(
                template_path=template_path,
                input_image_paths=image_paths,  # 使用全部样本
                input_trachea_paths=trachea_mask_paths,  # 使用全部样本
                output_trachea_path=trachea_path
            )
            if trachea_result:
                # 验证气管树连续性
                trachea_validation = validate_trachea_continuity(trachea_path)
                logger.info(f"  气管树验证: {'通过' if trachea_validation['valid'] else '警告'}")
        except Exception as e:
            logger.warning(f"气管树模板生成失败（非致命错误）: {e}")
        current_step += 1

    # Step 4: 生成肺叶标签模板（如果有输入）
    lobes_result = None
    if has_lobes:
        logger.info("")
        logger.info(f"Step {current_step}/{total_steps}: 生成模板肺叶标签 mask...")
        logger.info(f"  配准样本数: {len(image_paths)} CT, {len(lobes_mask_paths)} 肺叶标签")
        try:
            lobes_result = generate_template_lung_lobes(
                template_path=template_path,
                input_image_paths=image_paths,  # 使用全部样本
                input_lobes_paths=lobes_mask_paths,  # 使用全部样本
                output_lobes_path=lobes_path
            )
            if lobes_result:
                logger.info(f"  ✓ 肺叶标签模板已生成")
        except Exception as e:
            logger.warning(f"肺叶标签模板生成失败（非致命错误）: {e}")
        current_step += 1

    # Step N: 验证 Atlas
    final_step = total_steps
    logger.info("")
    logger.info(f"Step {final_step}/{total_steps}: 验证 Atlas 质量...")

    validation_result = validate_atlas(template_path, mask_path)

    # 质量评估（可选）
    evaluation_result = None
    if not skip_evaluation and len(image_paths) >= 3:
        logger.info("")
        logger.info("进行质量评估（抽取 3 个样本）...")

        # 抽取 3 个样本进行评估
        eval_images = image_paths[:3]

        # 为每个 image 找到对应的 mask（基于文件名匹配）
        eval_masks = []
        for img_path in eval_images:
            # 从 normal_001_clean.nii.gz 提取 normal_001
            img_stem = img_path.stem.replace('.nii', '')  # normal_001_clean
            sample_id = img_stem.replace('_clean', '')    # normal_001

            # 查找对应的 mask: normal_001_mask.nii.gz
            expected_mask_name = f"{sample_id}_mask.nii.gz"
            matching_mask = mask_dir / expected_mask_name

            if matching_mask.exists():
                eval_masks.append(matching_mask)
            else:
                logger.warning(f"未找到对应的 mask: {expected_mask_name}")

        if len(eval_masks) != len(eval_images):
            logger.warning(f"mask 数量不匹配: {len(eval_masks)} vs {len(eval_images)}，使用阈值分割")
            eval_masks = None

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

    # 运行完整质量检查
    quality_check_result = run_all_quality_checks(output_dir)

    total_elapsed = time.time() - total_start

    # 结果汇总
    result = {
        'success': validation_result['valid'],
        'template_path': str(template_path),
        'mask_path': str(mask_path),
        'trachea_path': str(trachea_path) if trachea_result else None,
        'lobes_path': str(lobes_path) if lobes_result else None,
        'num_input_images': len(image_paths),
        'total_time_seconds': total_elapsed,
        'total_time_str': str(timedelta(seconds=int(total_elapsed))),
        'validation': validation_result,
        'evaluation': evaluation_result,
        'quality_checks': quality_check_result,
        'has_trachea': trachea_result is not None,
        'has_lobes': lobes_result is not None
    }

    logger.info("")
    logger.info("=" * 70)
    logger.info("Phase 2 完成!")
    logger.info("=" * 70)
    logger.info(f"模板文件: {template_path}")
    logger.info(f"Mask 文件: {mask_path}")
    if trachea_result:
        logger.info(f"气管树 Mask: {trachea_path}")
    if lobes_result:
        logger.info(f"肺叶标签 Mask: {lobes_path}")
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

