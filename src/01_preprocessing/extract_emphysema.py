#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肺气肿病灶提取模块

使用 LAA-950 算法 (Low Attenuation Area at -950 HU) 提取肺气肿区域

优化记录 (2025-12-04):
- 添加形态学操作 (opening/closing) 去除噪声和平滑边界
- 增强连通域过滤，提高最小体积阈值
- 确保所有操作后 mask 严格约束在肺内
- 🆕 添加气道排除功能，避免气管/支气管被误标为病灶
"""

from pathlib import Path
from typing import Union, Tuple, Optional

import numpy as np

try:
    from scipy import ndimage
    from scipy.ndimage import binary_opening, binary_closing, binary_erosion, binary_dilation
except ImportError:
    ndimage = None
    binary_opening = None
    binary_closing = None
    binary_erosion = None
    binary_dilation = None

from ..utils.logger import get_logger
from ..utils.io import load_nifti, save_nifti

logger = get_logger(__name__)


def extract_airway_mask(
    ct_data: np.ndarray,
    lung_mask: np.ndarray,
    airway_hu_threshold: float = -980,
    min_airway_size: int = 1000,
    dilation_radius: int = 2
) -> np.ndarray:
    """
    从 CT 数据中提取气道区域（用于排除）

    原理:
    1. 气管/支气管内是空气，HU 值接近 -1000
    2. 气道是连续的管状结构，从气管一直延伸到肺内
    3. 我们识别极低 HU 值的大连通域作为气道

    Args:
        ct_data: CT 数据 (HU 单位)
        lung_mask: 肺部 mask
        airway_hu_threshold: 气道 HU 阈值（默认 -980，比 LAA-950 更严格）
        min_airway_size: 最小气道体素数（过滤小噪点）
        dilation_radius: 膨胀半径（扩大气道边界以确保完全覆盖）

    Returns:
        airway_mask: 气道区域 mask（用于从病灶中排除）
    """
    if ndimage is None:
        logger.warning("scipy.ndimage 不可用，跳过气道排除")
        return np.zeros_like(ct_data, dtype=np.uint8)

    logger.info("提取气道区域...")

    # Step 1: 找出极低 HU 值区域（气道候选）
    # 使用比 -950 更严格的阈值，因为正常肺组织不会低于 -980
    airway_candidate = (ct_data < airway_hu_threshold) & (lung_mask > 0)
    candidate_count = np.sum(airway_candidate)
    logger.debug(f"  气道候选区域: {candidate_count} 体素 (HU < {airway_hu_threshold})")

    if candidate_count == 0:
        logger.info("  未找到气道候选区域")
        return np.zeros_like(ct_data, dtype=np.uint8)

    # Step 2: 连通域分析 - 找出大的连通区域（气道）
    structure = ndimage.generate_binary_structure(3, 3)  # 26-连通
    labeled, num_features = ndimage.label(airway_candidate, structure=structure)
    logger.debug(f"  找到 {num_features} 个连通域")

    if num_features == 0:
        return np.zeros_like(ct_data, dtype=np.uint8)

    # 计算每个连通域的大小
    component_sizes = ndimage.sum(airway_candidate, labeled, range(1, num_features + 1))

    # Step 3: 选择大的连通域作为气道
    airway_mask = np.zeros_like(ct_data, dtype=np.uint8)
    selected_count = 0

    for i, size in enumerate(component_sizes):
        if size >= min_airway_size:
            airway_mask[labeled == (i + 1)] = 1
            selected_count += 1
            logger.debug(f"  选择连通域 {i+1}: {int(size)} 体素")

    if selected_count == 0:
        logger.info("  未找到足够大的气道区域")
        return np.zeros_like(ct_data, dtype=np.uint8)

    logger.info(f"  识别到 {selected_count} 个气道区域，共 {np.sum(airway_mask)} 体素")

    # Step 4: 膨胀气道边界（确保完全覆盖气道壁附近区域）
    if dilation_radius > 0:
        # 创建球形结构元素
        size = 2 * dilation_radius + 1
        struct = np.zeros((size, size, size), dtype=bool)
        center = dilation_radius
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    if (x - center)**2 + (y - center)**2 + (z - center)**2 <= dilation_radius**2:
                        struct[x, y, z] = True

        airway_mask_dilated = binary_dilation(airway_mask, structure=struct)
        # 仍然约束在 lung_mask 内
        airway_mask = (airway_mask_dilated & (lung_mask > 0)).astype(np.uint8)
        logger.debug(f"  膨胀后气道区域: {np.sum(airway_mask)} 体素")

    return airway_mask


def apply_morphological_cleaning(
    mask: np.ndarray,
    lung_mask: np.ndarray,
    opening_radius: int = 1,
    closing_radius: int = 2
) -> np.ndarray:
    """
    对 mask 应用形态学操作进行清理

    步骤:
    1. binary_opening: 移除小的突起和噪声点
    2. binary_closing: 填充小的空洞，平滑边界
    3. 再次用 lung_mask 约束，确保不超出肺部

    Args:
        mask: 输入的二值 mask
        lung_mask: 肺部 mask（用于约束）
        opening_radius: opening 操作的结构元素半径
        closing_radius: closing 操作的结构元素半径

    Returns:
        cleaned_mask: 清理后的 mask
    """
    if binary_opening is None:
        logger.warning("scipy.ndimage 不可用，跳过形态学清理")
        return mask

    # 创建 3D 球形结构元素
    def create_ball_structure(radius):
        size = 2 * radius + 1
        struct = np.zeros((size, size, size), dtype=bool)
        center = radius
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    if (x - center)**2 + (y - center)**2 + (z - center)**2 <= radius**2:
                        struct[x, y, z] = True
        return struct

    cleaned = mask.copy().astype(bool)

    # Step 1: Opening - 移除小的噪声点和细小突起
    if opening_radius > 0:
        struct_open = create_ball_structure(opening_radius)
        cleaned = binary_opening(cleaned, structure=struct_open)
        logger.debug(f"Opening 操作完成 (radius={opening_radius})")

    # Step 2: Closing - 填充小空洞，平滑边界
    if closing_radius > 0:
        struct_close = create_ball_structure(closing_radius)
        cleaned = binary_closing(cleaned, structure=struct_close)
        logger.debug(f"Closing 操作完成 (radius={closing_radius})")

    # Step 3: 关键！再次用 lung_mask 约束，确保不超出肺部边界
    cleaned = cleaned & (lung_mask > 0)

    return cleaned.astype(np.uint8)


def compute_laa950(
    ct_data: np.ndarray,
    lung_mask: np.ndarray,
    threshold: float = -950,
    apply_morphology: bool = True,
    opening_radius: int = 1,
    closing_radius: int = 2,
    exclude_airway: bool = True,
    airway_hu_threshold: float = -980,
    min_airway_size: int = 1000,
    airway_dilation_radius: int = 2
) -> Tuple[np.ndarray, float]:
    """
    计算 LAA-950 (肺气肿区域)（🆕 包含气道排除功能）

    LAA-950 定义：肺部区域内 HU 值低于 -950 的区域
    这些区域通常表示肺气肿（肺泡破坏，充满空气）

    🆕 优化：排除气道区域，避免气管/支气管被误标为肺气肿

    Args:
        ct_data: CT 数据 (HU 单位)
        lung_mask: 肺部 mask
        threshold: HU 阈值 (默认 -950)
        apply_morphology: 是否应用形态学清理
        opening_radius: opening 操作的结构元素半径
        closing_radius: closing 操作的结构元素半径
        exclude_airway: 🆕 是否排除气道区域
        airway_hu_threshold: 🆕 气道 HU 阈值
        min_airway_size: 🆕 最小气道体素数
        airway_dilation_radius: 🆕 气道膨胀半径

    Returns:
        emphysema_mask: 肺气肿区域 mask
        laa_percentage: LAA 百分比
    """
    # 确保 lung_mask 是二值的
    lung_mask_binary = (lung_mask > 0).astype(np.uint8)

    # 🆕 Step 0: 提取气道区域（用于排除）
    airway_mask = None
    if exclude_airway and ndimage is not None:
        airway_mask = extract_airway_mask(
            ct_data,
            lung_mask_binary,
            airway_hu_threshold=airway_hu_threshold,
            min_airway_size=min_airway_size,
            dilation_radius=airway_dilation_radius
        )
        airway_voxels = np.sum(airway_mask > 0)
        if airway_voxels > 0:
            logger.info(f"  排除气道区域: {airway_voxels} 体素")

    # Step 1: 在肺部区域内查找低密度区域（这是核心约束！）
    emphysema_mask = (ct_data < threshold) & (lung_mask_binary > 0)

    # 记录原始体积用于日志
    original_volume = np.sum(emphysema_mask)

    # 🆕 Step 1.5: 排除气道区域
    if airway_mask is not None and np.sum(airway_mask) > 0:
        before_count = np.sum(emphysema_mask)
        emphysema_mask = emphysema_mask & (airway_mask == 0)
        after_count = np.sum(emphysema_mask)
        logger.info(f"  气道排除: {before_count} -> {after_count} 体素 "
                    f"(移除 {before_count - after_count} 气道体素)")

    # Step 2: 应用形态学清理
    if apply_morphology and ndimage is not None:
        emphysema_mask = apply_morphological_cleaning(
            emphysema_mask.astype(np.uint8),
            lung_mask_binary,
            opening_radius=opening_radius,
            closing_radius=closing_radius
        )
        cleaned_volume = np.sum(emphysema_mask)
        logger.debug(
            f"形态学清理: {original_volume} -> {cleaned_volume} 体素 "
            f"({(1 - cleaned_volume/max(original_volume, 1))*100:.1f}% 减少)"
        )

    # 最终再次确保 mask 严格在肺内（双重保险）
    emphysema_mask = emphysema_mask & (lung_mask_binary > 0)

    # 计算 LAA 百分比
    lung_volume = np.sum(lung_mask_binary)
    emphysema_volume = np.sum(emphysema_mask)

    if lung_volume > 0:
        laa_percentage = emphysema_volume / lung_volume * 100
    else:
        laa_percentage = 0.0

    return emphysema_mask.astype(np.uint8), laa_percentage


def extract_emphysema_mask(
    ct_path: Union[str, Path],
    lung_mask_path: Union[str, Path],
    output_path: Union[str, Path],
    threshold: float = -950,
    min_volume_mm3: float = 100.0,  # 增加默认值：100 mm³ ≈ 0.1 mL
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    apply_morphology: bool = True,
    opening_radius: int = 1,
    closing_radius: int = 2,
    exclude_airway: bool = True,  # 🆕 默认开启气道排除
    airway_hu_threshold: float = -980,
    min_airway_size: int = 1000,
    airway_dilation_radius: int = 2
) -> Tuple[float, dict]:
    """
    从文件提取肺气肿 mask（优化版 + 气道排除）

    优化内容:
    1. 形态学操作去除噪声和平滑边界
    2. 连通域过滤去除小病灶
    3. 最终 lung_mask 再约束确保不超出肺部
    4. 🆕 气道排除：避免气管/支气管被误标为肺气肿

    Args:
        ct_path: CT 文件路径
        lung_mask_path: 肺部 mask 路径
        output_path: 输出 mask 路径
        threshold: HU 阈值
        min_volume_mm3: 最小病灶体积 (mm³)，建议 >= 100
        voxel_spacing: 体素间距
        apply_morphology: 是否应用形态学清理
        opening_radius: opening 操作半径
        closing_radius: closing 操作半径
        exclude_airway: 🆕 是否排除气道区域
        airway_hu_threshold: 🆕 气道 HU 阈值
        min_airway_size: 🆕 最小气道体素数
        airway_dilation_radius: 🆕 气道膨胀半径

    Returns:
        laa_percentage: LAA 百分比
        stats: 统计信息
    """
    ct_path = Path(ct_path)

    logger.info(f"提取肺气肿 mask: {ct_path.name}")
    logger.info(f"  阈值: {threshold} HU, 最小体积: {min_volume_mm3} mm³")
    logger.info(f"  气道排除: {'开启' if exclude_airway else '关闭'}")

    # 加载数据
    ct_data, affine = load_nifti(ct_path, return_affine=True)
    lung_mask = load_nifti(lung_mask_path)

    # 确保 lung_mask 是二值的
    lung_mask_binary = (lung_mask > 0).astype(np.uint8)

    # 检查 lung_mask 有效性
    lung_voxels = np.sum(lung_mask_binary)
    if lung_voxels == 0:
        logger.error("lung_mask 为空！请检查分割结果")
        raise ValueError("lung_mask 为空")
    logger.info(f"  肺部体积: {lung_voxels} 体素")

    # 计算 LAA-950（包含形态学操作 + 气道排除）
    emphysema_mask, laa_percentage = compute_laa950(
        ct_data,
        lung_mask_binary,
        threshold,
        apply_morphology=apply_morphology,
        opening_radius=opening_radius,
        closing_radius=closing_radius,
        exclude_airway=exclude_airway,
        airway_hu_threshold=airway_hu_threshold,
        min_airway_size=min_airway_size,
        airway_dilation_radius=airway_dilation_radius
    )

    # 移除小的连通区域
    if ndimage is not None and min_volume_mm3 > 0:
        before_count = np.sum(emphysema_mask)
        emphysema_mask = remove_small_components(
            emphysema_mask, min_volume_mm3, voxel_spacing
        )
        after_count = np.sum(emphysema_mask)
        logger.info(f"  连通域过滤: {before_count} -> {after_count} 体素")

    # 【关键】最终再次用 lung_mask 约束，确保 mask 绝对不超出肺部
    emphysema_mask = emphysema_mask & lung_mask_binary

    # 验证：检查是否有 mask 在肺外
    mask_outside_lung = np.sum(emphysema_mask & (~lung_mask_binary.astype(bool)))
    if mask_outside_lung > 0:
        logger.warning(f"仍有 {mask_outside_lung} 个体素在肺外！强制清除")
        emphysema_mask = emphysema_mask & lung_mask_binary

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(emphysema_mask, output_path, affine=affine, dtype='uint8')

    # 统计信息
    voxel_volume = np.prod(voxel_spacing)
    emphysema_volume_mm3 = np.sum(emphysema_mask) * voxel_volume

    stats = {
        'laa_percentage': laa_percentage,
        'emphysema_volume_mm3': emphysema_volume_mm3,
        'emphysema_volume_ml': emphysema_volume_mm3 / 1000,
        'threshold_hu': threshold,
        'num_voxels': int(np.sum(emphysema_mask)),
        'min_volume_mm3': min_volume_mm3,
        'apply_morphology': apply_morphology,
        'exclude_airway': exclude_airway,
    }

    logger.info(
        f"最终结果 - LAA-950: {laa_percentage:.2f}%, "
        f"体积: {stats['emphysema_volume_ml']:.1f} mL, "
        f"体素数: {stats['num_voxels']}"
    )

    return laa_percentage, stats


def remove_small_components(
    mask: np.ndarray,
    min_volume_mm3: float,
    voxel_spacing: Tuple[float, float, float],
    keep_largest_n: Optional[int] = None
) -> np.ndarray:
    """
    移除小于指定体积的连通分量（增强版）

    Args:
        mask: 二值 mask
        min_volume_mm3: 最小体积 (mm³)
        voxel_spacing: 体素间距
        keep_largest_n: 可选，只保留最大的 N 个连通域

    Returns:
        cleaned_mask: 清理后的 mask
    """
    if ndimage is None:
        raise ImportError("请安装 scipy: pip install scipy")

    if np.sum(mask) == 0:
        logger.warning("输入 mask 为空")
        return mask

    voxel_volume = np.prod(voxel_spacing)
    min_voxels = max(1, int(min_volume_mm3 / voxel_volume))

    logger.debug(f"连通域过滤: 最小体积 {min_volume_mm3} mm³ = {min_voxels} 体素")

    # 标记连通分量（使用 3D 26-连通性）
    structure = ndimage.generate_binary_structure(3, 3)  # 26-连通
    labeled, num_features = ndimage.label(mask, structure=structure)

    if num_features == 0:
        logger.warning("未找到连通分量")
        return np.zeros_like(mask)

    logger.debug(f"原始连通分量数: {num_features}")

    # 计算每个分量的体积
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))

    # 按体积排序（降序）
    sorted_indices = np.argsort(component_sizes)[::-1]

    # 决定保留哪些分量
    cleaned_mask = np.zeros_like(mask, dtype=np.uint8)
    kept_count = 0

    for rank, idx in enumerate(sorted_indices):
        component_label = idx + 1
        size = component_sizes[idx]

        # 检查体积阈值
        if size < min_voxels:
            continue

        # 检查最大保留数
        if keep_largest_n is not None and kept_count >= keep_largest_n:
            break

        cleaned_mask[labeled == component_label] = 1
        kept_count += 1

    removed_count = num_features - kept_count
    logger.info(
        f"连通域过滤结果: 保留 {kept_count}/{num_features} 个, "
        f"移除 {removed_count} 个小分量"
    )

    return cleaned_mask


def classify_emphysema_severity(laa_percentage: float) -> str:
    """
    根据 LAA-950 百分比分类肺气肿严重程度
    
    GOLD 标准:
    - 正常: LAA < 5%
    - 轻度: 5% <= LAA < 15%
    - 中度: 15% <= LAA < 25%
    - 重度: LAA >= 25%
    
    Args:
        laa_percentage: LAA 百分比
        
    Returns:
        severity: 严重程度等级
    """
    if laa_percentage < 5:
        return "正常"
    elif laa_percentage < 15:
        return "轻度"
    elif laa_percentage < 25:
        return "中度"
    else:
        return "重度"


def batch_extract_emphysema(
    ct_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    output_dir: Union[str, Path],
    threshold: float = -950,
    min_volume_mm3: float = 10
) -> dict:
    """
    批量提取肺气肿 mask
    
    Args:
        ct_dir: CT 文件目录
        mask_dir: 肺部 mask 目录
        output_dir: 输出目录
        threshold: HU 阈值
        min_volume_mm3: 最小体积
        
    Returns:
        results: 每个文件的结果
    """
    ct_dir = Path(ct_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ct_files = list(ct_dir.glob("*.nii.gz"))
    results = {}
    
    for ct_path in ct_files:
        stem = ct_path.stem.replace('.nii', '')
        mask_path = mask_dir / f"{stem}_mask.nii.gz"
        
        if not mask_path.exists():
            mask_path = mask_dir / ct_path.name
        
        if not mask_path.exists():
            logger.warning(f"未找到 mask: {ct_path.name}")
            continue
        
        output_path = output_dir / f"{stem}_emphysema.nii.gz"
        
        try:
            laa, stats = extract_emphysema_mask(
                ct_path, mask_path, output_path,
                threshold=threshold,
                min_volume_mm3=min_volume_mm3
            )
            stats['severity'] = classify_emphysema_severity(laa)
            results[str(ct_path)] = stats
        except Exception as e:
            logger.error(f"处理失败 {ct_path.name}: {e}")
    
    return results


def main(config: dict) -> None:
    """主函数"""
    threshold = config.get('preprocessing', {}).get('laa950', {}).get('threshold', -950)
    min_volume = config.get('preprocessing', {}).get('laa950', {}).get('min_volume_mm3', 10)
    
    # TODO: 实现批量处理
    logger.info(f"LAA 阈值: {threshold} HU, 最小体积: {min_volume} mm³")


if __name__ == "__main__":
    import yaml
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)

