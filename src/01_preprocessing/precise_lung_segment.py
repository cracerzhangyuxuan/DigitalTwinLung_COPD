#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精确肺部分割模块

使用 SimpleITK 实现高质量的肺部分割，排除：
- 骨骼（肋骨、胸骨、脊柱）
- 肌肉组织
- 心脏、大血管
- 气管/支气管
- 其他非肺组织

基于以下策略：
1. 使用更严格的 HU 阈值范围（-950 ~ -200）排除纯空气区域
2. 连通域分析选择左右肺
3. 气管排除：从顶部切片识别并去除气管
4. 边界清理：使用形态学操作平滑边界
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    from scipy import ndimage
    from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
    from scipy.ndimage import binary_opening, binary_closing, label
except ImportError:
    ndimage = None

from ..utils.logger import get_logger
from ..utils.io import load_nifti, save_nifti

logger = get_logger(__name__)


def remove_trachea(lung_mask: np.ndarray, ct_data: np.ndarray, 
                   trachea_hu_threshold: float = -950,
                   min_trachea_size: int = 500) -> np.ndarray:
    """
    从肺 mask 中移除气管和主支气管
    
    策略：
    1. 气管位于肺的中央顶部
    2. 气管内 HU 值接近 -1000（纯空气）
    3. 气管是从上到下连续的管状结构
    
    Args:
        lung_mask: 肺部 mask
        ct_data: CT 数据
        trachea_hu_threshold: 气管 HU 阈值
        min_trachea_size: 最小气管连通域大小
    
    Returns:
        lung_mask_no_trachea: 移除气管后的肺 mask
    """
    if ndimage is None:
        return lung_mask
    
    # 找出肺 mask 内的极低 HU 区域（气管候选）
    trachea_candidate = (ct_data < trachea_hu_threshold) & (lung_mask > 0)
    
    # 连通域分析
    structure = ndimage.generate_binary_structure(3, 3)  # 26-连通
    labeled, num_features = label(trachea_candidate, structure=structure)
    
    if num_features == 0:
        return lung_mask
    
    # 计算每个连通域的属性
    trachea_mask = np.zeros_like(lung_mask, dtype=np.uint8)
    shape = lung_mask.shape
    
    for i in range(1, num_features + 1):
        component = (labeled == i)
        size = np.sum(component)
        
        if size < min_trachea_size:
            continue
        
        # 检查是否接触顶部（气管从顶部开始）
        top_slices = shape[2] - 20  # 顶部 20 层
        touches_top = np.any(component[:, :, top_slices:])
        
        # 检查是否位于中央区域
        coords = np.where(component)
        center_x = np.mean(coords[0])
        center_y = np.mean(coords[1])
        is_central = (0.3 * shape[0] < center_x < 0.7 * shape[0]) and \
                     (0.3 * shape[1] < center_y < 0.7 * shape[1])
        
        # 气管特征：接触顶部 + 位于中央 + 足够大
        if touches_top and is_central and size >= min_trachea_size:
            trachea_mask[component] = 1
            logger.debug(f"  识别气管区域: {size} 体素")
    
    # 膨胀气管 mask 以确保完全覆盖
    if np.sum(trachea_mask) > 0:
        trachea_mask = binary_dilation(trachea_mask, iterations=2)
        logger.info(f"  移除气管区域: {np.sum(trachea_mask)} 体素")
    
    # 从肺 mask 中排除气管
    lung_mask_clean = lung_mask.copy()
    lung_mask_clean[trachea_mask > 0] = 0
    
    return lung_mask_clean.astype(np.uint8)


def precise_lung_segmentation(
    ct_data: np.ndarray,
    lower_threshold: float = -950,  # 排除纯空气 (< -950)
    upper_threshold: float = -200,  # 排除软组织 (> -200)
    min_lung_size_ratio: float = 0.02,  # 最小肺占比
    max_lung_size_ratio: float = 0.35,  # 最大肺占比
    remove_trachea_flag: bool = True,
    fill_holes: bool = True
) -> np.ndarray:
    """
    精确肺部分割
    
    Args:
        ct_data: CT 数据 (HU 单位)
        lower_threshold: 下限阈值（排除纯空气）
        upper_threshold: 上限阈值（排除软组织）
        min_lung_size_ratio: 最小肺占比
        max_lung_size_ratio: 最大肺占比
        remove_trachea_flag: 是否移除气管
        fill_holes: 是否填充孔洞
    
    Returns:
        lung_mask: 精确的肺部 mask
    """
    if ndimage is None:
        raise ImportError("请安装 scipy: pip install scipy")
    
    total_voxels = ct_data.size
    shape = ct_data.shape
    
    logger.info("=" * 60)
    logger.info("开始精确肺部分割...")
    logger.info(f"  CT 形状: {shape}")
    logger.info(f"  HU 阈值: {lower_threshold} < HU < {upper_threshold}")
    
    # =========================================================================
    # Step 1: 阈值分割 - 提取肺组织候选区域
    # 关键：使用 -950 作为下限，排除气管内的纯空气
    # =========================================================================
    lung_candidate = (ct_data > lower_threshold) & (ct_data < upper_threshold)
    logger.info(f"  阈值分割: {np.sum(lung_candidate):,} 体素 "
                f"({np.sum(lung_candidate)/total_voxels*100:.1f}%)")
    
    # =========================================================================
    # Step 2: 创建身体轮廓 mask（排除体外空气）
    # =========================================================================
    # 身体组织 HU > -200
    body_tissue = ct_data > -200
    
    # 膨胀形成连续的身体边界
    struct = ndimage.generate_binary_structure(3, 1)
    body_dilated = binary_dilation(body_tissue, structure=struct, iterations=5)
    
    # 逐层填充孔洞，形成体腔
    body_cavity = np.zeros_like(ct_data, dtype=bool)
    for z in range(shape[2]):
        body_cavity[:, :, z] = binary_fill_holes(body_dilated[:, :, z])
    
    logger.info(f"  体腔区域: {np.sum(body_cavity):,} 体素 "
                f"({np.sum(body_cavity)/total_voxels*100:.1f}%)")
    
    # =========================================================================
    # Step 3: 肺候选 = 阈值区域 AND 体腔内部
    # =========================================================================
    lung_in_body = lung_candidate & body_cavity
    logger.info(f"  体腔内肺候选: {np.sum(lung_in_body):,} 体素")
    
    # =========================================================================
    # Step 4: 连通域分析 - 选择左右肺
    # =========================================================================
    labeled, num_features = label(lung_in_body)
    logger.info(f"  找到 {num_features} 个连通域")
    
    if num_features == 0:
        logger.warning("未找到肺候选区域！")
        return np.zeros_like(ct_data, dtype=np.uint8)
    
    # 计算每个连通域的大小
    component_sizes = ndimage.sum(lung_in_body, labeled, range(1, num_features + 1))
    component_info = [(i + 1, int(component_sizes[i]), component_sizes[i] / total_voxels)
                      for i in range(len(component_sizes))]
    component_info.sort(key=lambda x: x[1], reverse=True)
    
    # 选择符合条件的连通域
    # 策略：选择最大的 2 个连通域（左右肺），但每个必须足够大
    lung_mask = np.zeros_like(ct_data, dtype=np.uint8)
    selected_count = 0
    selected_total_ratio = 0

    # 显示前 3 个最大的连通域
    logger.info("  候选连通域（前 3）:")
    for label_id, size, ratio in component_info[:3]:
        logger.info(f"    - 区域 {label_id}: {size:,} 体素 ({ratio*100:.2f}%)")

    for label_id, size, ratio in component_info:
        # 跳过太小的区域（小于最大区域的 1/10）
        if len(component_info) > 0:
            max_size = component_info[0][1]
            if size < max_size * 0.1:  # 放宽到 1/10
                continue

        if ratio > max_lung_size_ratio:
            logger.info(f"  跳过过大区域 {label_id}: {ratio*100:.1f}%")
            continue  # 太大

        # 至少要 1% 的体积（一个肺约 5-15%，所以 1% 是合理的下限）
        if ratio < 0.01:
            continue

        lung_mask[labeled == label_id] = 1
        selected_count += 1
        selected_total_ratio += ratio
        logger.info(f"  ✓ 选择肺区域 {label_id}: {size:,} 体素 ({ratio*100:.2f}%)")

        if selected_count >= 2:  # 最多 2 个（左右肺）
            break

    logger.info(f"  共选择 {selected_count} 个区域，总占比 {selected_total_ratio*100:.2f}%")

    if selected_count == 0:
        # 使用最大的区域
        if len(component_info) > 0:
            label_id, size, ratio = component_info[0]
            lung_mask[labeled == label_id] = 1
            logger.warning(f"  强制选择最大区域 {label_id}: {size:,} 体素")
    
    # =========================================================================
    # Step 5: 移除气管
    # =========================================================================
    if remove_trachea_flag and np.sum(lung_mask) > 0:
        lung_mask = remove_trachea(lung_mask, ct_data)
    
    # =========================================================================
    # Step 6: 形态学后处理
    # =========================================================================
    if np.sum(lung_mask) > 0:
        # 开运算：去除小突起
        lung_mask = binary_opening(lung_mask, structure=struct, iterations=1)
        
        # 填充孔洞（肺内血管等）
        if fill_holes:
            for z in range(shape[2]):
                lung_mask[:, :, z] = binary_fill_holes(lung_mask[:, :, z])
        
        # 闭运算：平滑边界
        lung_mask = binary_closing(lung_mask, structure=struct, iterations=1)
    
    # =========================================================================
    # Step 7: 最终验证
    # =========================================================================
    final_ratio = np.sum(lung_mask) / total_voxels
    logger.info(f"  最终肺体积: {np.sum(lung_mask):,} 体素 ({final_ratio*100:.1f}%)")
    
    # 验证 HU 分布
    if np.sum(lung_mask) > 0:
        lung_hu = ct_data[lung_mask > 0]
        mean_hu = np.mean(lung_hu)
        logger.info(f"  肺区域平均 HU: {mean_hu:.1f}")
        
        # 检查肺组织占比
        lung_tissue = np.sum((lung_hu > -900) & (lung_hu < -500))
        airway_like = np.sum(lung_hu < -950)
        soft_tissue = np.sum(lung_hu > -200)
        
        logger.info(f"  HU 分布:")
        logger.info(f"    - 肺实质 (-900~-500): {lung_tissue/len(lung_hu)*100:.1f}%")
        logger.info(f"    - 气道样 (< -950): {airway_like/len(lung_hu)*100:.1f}%")
        logger.info(f"    - 软组织 (> -200): {soft_tissue/len(lung_hu)*100:.1f}%")
    
    logger.info("=" * 60)
    return lung_mask.astype(np.uint8)


def segment_lung_precise(
    input_path: Union[str, Path],
    output_mask_path: Union[str, Path],
    output_clean_path: Optional[Union[str, Path]] = None,
    background_hu: float = -1000,
    **kwargs
) -> dict:
    """
    从文件进行精确肺部分割
    """
    input_path = Path(input_path)
    output_mask_path = Path(output_mask_path)
    
    logger.info(f"处理: {input_path.name}")
    
    # 加载数据
    ct_data, affine = load_nifti(input_path, return_affine=True)
    
    # 精确分割
    lung_mask = precise_lung_segmentation(ct_data, **kwargs)
    
    # 保存 mask
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(lung_mask, output_mask_path, affine=affine, dtype='uint8')
    logger.info(f"已保存 mask: {output_mask_path}")
    
    # 统计
    lung_hu = ct_data[lung_mask > 0] if np.sum(lung_mask) > 0 else np.array([])
    stats = {
        'total_voxels': int(ct_data.size),
        'lung_voxels': int(np.sum(lung_mask)),
        'lung_ratio': float(np.sum(lung_mask) / ct_data.size),
        'mean_hu': float(np.mean(lung_hu)) if len(lung_hu) > 0 else 0,
    }
    
    # 可选：保存清洗后的 CT
    if output_clean_path is not None:
        output_clean_path = Path(output_clean_path)
        output_clean_path.parent.mkdir(parents=True, exist_ok=True)
        ct_clean = ct_data.copy()
        ct_clean[lung_mask == 0] = background_hu
        save_nifti(ct_clean, output_clean_path, affine=affine)
        logger.info(f"已保存清洗后 CT: {output_clean_path}")
    
    return stats

