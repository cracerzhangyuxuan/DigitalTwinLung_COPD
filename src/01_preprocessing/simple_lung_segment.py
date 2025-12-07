#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单肺部分割模块（优化版 2025-12-04）

使用阈值方法和形态学操作进行肺部分割
（替代 TotalSegmentator 的轻量方案，适用于 MVP 阶段）

优化内容：
1. 收紧阈值范围：-900 < HU < -200（排除纯空气）
2. 增强边界排除逻辑
3. 添加体积约束（肺体积应该是总体积的 5-35%）
4. 移除过度的形态学膨胀
5. 添加气管排除逻辑
"""

from pathlib import Path
from typing import Union, Tuple
import numpy as np

try:
    from scipy import ndimage
    from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation, binary_opening, binary_closing
except ImportError:
    ndimage = None

try:
    from skimage import measure, morphology
except ImportError:
    measure = None
    morphology = None

from ..utils.logger import get_logger
from ..utils.io import load_nifti, save_nifti

logger = get_logger(__name__)


def threshold_lung_segmentation(
    ct_volume: np.ndarray,
    lower_threshold: float = -1000,  # 恢复宽阈值
    upper_threshold: float = -300,   # 略微收紧上限
    min_lung_size: int = 500000,     # 增加最小肺体积（肺至少 50 万体素）
    max_lung_ratio: float = 0.40,    # 肺最大占比
    min_lung_ratio: float = 0.03     # 肺最小占比
) -> np.ndarray:
    """
    基于阈值的肺部分割（优化版 v2 2025-12-04）

    核心策略改变：
    1. 使用宽阈值捕获所有低密度区域
    2. 通过"体腔填充"来区分肺和背景空气
    3. 利用体组织（高密度）边界来定义体腔

    Args:
        ct_volume: CT 体数据 (HU 单位)
        lower_threshold: 下限阈值
        upper_threshold: 上限阈值
        min_lung_size: 最小肺体积（体素数）
        max_lung_ratio: 最大肺占比
        min_lung_ratio: 最小肺占比

    Returns:
        lung_mask: 肺部二值 mask
    """
    if ndimage is None:
        raise ImportError("请安装 scipy: pip install scipy")

    total_voxels = ct_volume.size
    shape = ct_volume.shape
    logger.info("=" * 50)
    logger.info("开始改进版阈值肺部分割 v2...")
    logger.info(f"CT 形状: {shape}")
    logger.info(f"阈值范围: {lower_threshold} < HU < {upper_threshold}")

    # Step 1: 阈值分割 - 提取低密度区域（包含背景空气和肺）
    binary = (ct_volume > lower_threshold) & (ct_volume < upper_threshold)
    logger.info(f"阈值分割: {np.sum(binary)} 体素 ({np.sum(binary)/total_voxels*100:.1f}%)")

    # Step 2: 使用体组织（高密度区域）创建"体腔"边界
    # 体组织 HU 通常 > -200，使用这个来找到身体边界
    struct = ndimage.generate_binary_structure(3, 1)
    body_tissue = ct_volume > -200
    logger.info(f"体组织区域: {np.sum(body_tissue)} 体素 ({np.sum(body_tissue)/total_voxels*100:.1f}%)")

    # 膨胀体组织区域以创建连续的"体壁"
    body_tissue_dilated = binary_dilation(body_tissue, structure=struct, iterations=5)

    # 逐层填充体腔（从体壁外部填充，剩下的就是体腔内部）
    body_cavity = np.zeros_like(ct_volume, dtype=bool)
    for z in range(shape[2]):
        # 填充孔洞 = 体腔内部
        body_cavity[:, :, z] = binary_fill_holes(body_tissue_dilated[:, :, z])

    logger.info(f"体腔区域: {np.sum(body_cavity)} 体素 ({np.sum(body_cavity)/total_voxels*100:.1f}%)")

    # Step 3: 肺候选 = 低密度区域 AND 体腔内部
    lung_candidate = binary & body_cavity
    logger.info(f"肺候选区域: {np.sum(lung_candidate)} 体素 ({np.sum(lung_candidate)/total_voxels*100:.1f}%)")

    # Step 4: 标记连通分量
    labeled, num_features = ndimage.label(lung_candidate)
    logger.info(f"找到 {num_features} 个连通分量")

    if num_features == 0:
        logger.warning("未找到连通分量，返回空 mask")
        return np.zeros_like(ct_volume, dtype=np.uint8)

    # Step 5: 高效计算每个分量的大小（使用 ndimage.sum，一次性计算）
    component_sizes = ndimage.sum(lung_candidate, labeled, range(1, num_features + 1))
    component_info = [(i + 1, int(component_sizes[i]), component_sizes[i] / total_voxels)
                      for i in range(len(component_sizes))]

    # 按大小排序
    component_info.sort(key=lambda x: x[1], reverse=True)

    # 显示前 5 个最大分量
    logger.info("最大的 5 个分量:")
    for idx, (label_id, size, ratio) in enumerate(component_info[:5]):
        logger.info(f"  {idx+1}. 分量 {label_id}: {size} 体素 ({ratio*100:.2f}%)")

    # Step 6: 选择分量
    lung_mask = np.zeros_like(ct_volume, dtype=np.uint8)
    selected_count = 0

    for label_id, size, ratio in component_info:
        # 跳过太小的分量
        if size < min_lung_size:
            continue
        # 跳过太大的分量（可能是背景）
        if ratio > max_lung_ratio:
            logger.warning(f"分量 {label_id} 太大 ({ratio*100:.1f}%)，跳过")
            continue

        lung_mask[labeled == label_id] = 1
        selected_count += 1
        logger.info(f"选择分量 {label_id}: {size} 体素 ({ratio*100:.2f}%)")

        if selected_count >= 2:  # 最多选择 2 个（左右肺）
            break

    # 如果没有选中任何分量，使用最大的分量
    if selected_count == 0 and len(component_info) > 0:
        logger.warning("未找到符合条件的分量！使用最大的分量")
        label_id, size, ratio = component_info[0]
        lung_mask[labeled == label_id] = 1
        logger.info(f"强制选择分量 {label_id}: {size} 体素 ({ratio*100:.2f}%)")

    # Step 7: 形态学后处理
    if np.sum(lung_mask) > 0:
        # 填充孔洞
        for z in range(shape[2]):
            lung_mask[:, :, z] = binary_fill_holes(lung_mask[:, :, z])

        # 轻微开运算去除毛刺
        lung_mask = binary_opening(lung_mask, structure=struct, iterations=1)

    # Step 8: 最终检查
    final_ratio = np.sum(lung_mask) / total_voxels
    logger.info(f"最终肺体积: {np.sum(lung_mask)} 体素 ({final_ratio*100:.1f}%)")

    if final_ratio < min_lung_ratio:
        logger.warning(f"警告：肺体积占比 {final_ratio*100:.1f}% 过低")

    logger.info("=" * 50)
    return lung_mask.astype(np.uint8)


def segment_lung_from_file(
    input_path: Union[str, Path],
    output_mask_path: Union[str, Path],
    output_clean_path: Union[str, Path] = None,
    background_hu: float = -1000
) -> dict:
    """
    从文件进行肺部分割
    
    Args:
        input_path: 输入 NIfTI 文件路径
        output_mask_path: 输出 mask 路径
        output_clean_path: 输出清洗后 CT 路径（可选）
        background_hu: 背景 HU 值
        
    Returns:
        stats: 统计信息
    """
    input_path = Path(input_path)
    output_mask_path = Path(output_mask_path)
    
    logger.info(f"处理: {input_path.name}")
    
    # 加载数据
    ct_data, affine = load_nifti(input_path, return_affine=True)
    
    # 分割
    lung_mask = threshold_lung_segmentation(ct_data)
    
    # 保存 mask
    save_nifti(lung_mask, output_mask_path, affine=affine, dtype='uint8')
    logger.info(f"已保存 mask: {output_mask_path}")
    
    # 统计信息
    stats = {
        'total_voxels': int(ct_data.size),
        'lung_voxels': int(np.sum(lung_mask)),
        'lung_ratio': float(np.sum(lung_mask) / ct_data.size),
        'ct_min': float(ct_data.min()),
        'ct_max': float(ct_data.max()),
    }
    
    # 可选：保存清洗后的 CT
    if output_clean_path is not None:
        output_clean_path = Path(output_clean_path)
        ct_clean = ct_data.copy()
        ct_clean[lung_mask == 0] = background_hu
        save_nifti(ct_clean, output_clean_path, affine=affine)
        logger.info(f"已保存清洗后 CT: {output_clean_path}")
    
    return stats


def batch_segment_lungs(
    input_dir: Union[str, Path],
    mask_output_dir: Union[str, Path],
    clean_output_dir: Union[str, Path] = None,
    pattern: str = "*.nii.gz"
) -> dict:
    """
    批量肺部分割
    
    Args:
        input_dir: 输入目录
        mask_output_dir: mask 输出目录
        clean_output_dir: 清洗后 CT 输出目录（可选）
        pattern: 文件匹配模式
        
    Returns:
        results: 结果字典
    """
    input_dir = Path(input_dir)
    mask_output_dir = Path(mask_output_dir)
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    
    if clean_output_dir:
        clean_output_dir = Path(clean_output_dir)
        clean_output_dir.mkdir(parents=True, exist_ok=True)
    
    nifti_files = list(input_dir.glob(pattern))
    logger.info(f"找到 {len(nifti_files)} 个 NIfTI 文件")
    
    results = {}
    
    for i, nifti_path in enumerate(nifti_files, start=1):
        stem = nifti_path.name.replace('.nii.gz', '').replace('.nii', '')
        mask_path = mask_output_dir / f"{stem}_mask.nii.gz"
        clean_path = clean_output_dir / f"{stem}_clean.nii.gz" if clean_output_dir else None
        
        try:
            stats = segment_lung_from_file(
                nifti_path, mask_path, clean_path
            )
            stats['status'] = 'success'
            results[stem] = stats
            logger.info(f"[{i}/{len(nifti_files)}] 成功 - 肺占比: {stats['lung_ratio']*100:.1f}%")
        except Exception as e:
            logger.error(f"[{i}/{len(nifti_files)}] 失败: {e}")
            results[stem] = {'status': 'failed', 'error': str(e)}
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    logger.info(f"批量分割完成: {success_count}/{len(nifti_files)} 成功")
    
    return results


def main():
    """主函数"""
    import yaml
    
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    cleaned_dir = Path(config['paths']['cleaned_data'])
    
    # 分割正常肺
    normal_nifti = cleaned_dir / 'normal_nifti'
    if normal_nifti.exists():
        logger.info("=" * 50)
        logger.info("分割正常肺...")
        batch_segment_lungs(
            normal_nifti,
            mask_output_dir=cleaned_dir / 'normal_mask',
            clean_output_dir=cleaned_dir / 'normal_clean'
        )
    
    # 分割 COPD
    copd_nifti = cleaned_dir / 'copd_nifti'
    if copd_nifti.exists():
        logger.info("=" * 50)
        logger.info("分割 COPD...")
        batch_segment_lungs(
            copd_nifti,
            mask_output_dir=cleaned_dir / 'copd_mask',
            clean_output_dir=cleaned_dir / 'copd_clean'
        )
    
    logger.info("=" * 50)
    logger.info("全部分割完成!")


if __name__ == "__main__":
    main()

