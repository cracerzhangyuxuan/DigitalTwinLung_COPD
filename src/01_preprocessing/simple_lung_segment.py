#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肺部分割模块（集成版 2025-12-09）

支持两种分割方法：
1. TotalSegmentator (GPU 加速，高精度) - 推荐用于生产环境
2. 阈值分割 (CPU，轻量) - 用于无 GPU 环境或快速测试

默认行为：
- 自动检测 GPU 和 TotalSegmentator 可用性
- GPU 可用时自动使用 TotalSegmentator
- GPU 不可用时回退到阈值方法

使用方法：
    # 自动选择方法 (推荐)
    python -m src.01_preprocessing.simple_lung_segment

    # 强制使用 TotalSegmentator
    python -m src.01_preprocessing.simple_lung_segment --method totalsegmentator

    # 强制使用阈值方法
    python -m src.01_preprocessing.simple_lung_segment --method threshold

    # 使用特定 GPU
    python -m src.01_preprocessing.simple_lung_segment --device cuda:1
"""

import subprocess
import shutil
import argparse
from pathlib import Path
from typing import Union, Tuple, Optional

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

# ============================================================================
# GPU 和 TotalSegmentator 可用性检测
# ============================================================================

def check_gpu_available() -> Tuple[bool, str]:
    """检查 GPU 是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return True, f"GPU 可用: {gpu_name}"
        else:
            return False, "CUDA 不可用"
    except ImportError:
        return False, "PyTorch 未安装"


def check_totalsegmentator_available() -> Tuple[bool, str]:
    """检查 TotalSegmentator 是否可用"""
    try:
        result = subprocess.run(
            ["TotalSegmentator", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return True, "TotalSegmentator 可用"
        return False, "TotalSegmentator 命令执行失败"
    except FileNotFoundError:
        return False, "TotalSegmentator 未安装"
    except subprocess.TimeoutExpired:
        return False, "TotalSegmentator 响应超时"
    except Exception as e:
        return False, f"检查失败: {e}"


def get_default_method() -> str:
    """获取默认分割方法（根据环境自动选择）"""
    ts_ok, _ = check_totalsegmentator_available()
    gpu_ok, _ = check_gpu_available()

    if ts_ok and gpu_ok:
        return "totalsegmentator"
    elif ts_ok:
        return "totalsegmentator"  # 即使没有 GPU，TotalSegmentator 也可以用 CPU
    else:
        return "threshold"


# ============================================================================
# TotalSegmentator 分割方法
# ============================================================================

def totalsegmentator_lung_segmentation(
    input_path: Union[str, Path],
    output_mask_path: Union[str, Path],
    output_clean_path: Union[str, Path] = None,
    device: str = "gpu",
    fast: bool = False,
    background_hu: float = -1000,
    temp_dir: Path = None
) -> dict:
    """
    使用 TotalSegmentator 进行肺部分割

    TotalSegmentator 输出的肺部分割包含 5 个肺叶:
    - lung_upper_lobe_left.nii.gz
    - lung_lower_lobe_left.nii.gz
    - lung_upper_lobe_right.nii.gz
    - lung_middle_lobe_right.nii.gz
    - lung_lower_lobe_right.nii.gz

    Args:
        input_path: 输入 CT 文件路径 (NIfTI 格式)
        output_mask_path: 输出 mask 路径
        output_clean_path: 输出清洗后 CT 路径（可选）
        device: 使用设备 ("gpu" 或 "cpu")
        fast: 是否使用快速模式（精度略低，速度快 5 倍）
        background_hu: 背景 HU 值
        temp_dir: 临时目录（用于存放 TotalSegmentator 输出）

    Returns:
        stats: 统计信息字典
    """
    import nibabel as nib

    input_path = Path(input_path)
    output_mask_path = Path(output_mask_path)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)

    if output_clean_path:
        output_clean_path = Path(output_clean_path)
        output_clean_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建临时目录
    if temp_dir is None:
        temp_dir = Path("data/.temp_totalsegmentator")
    temp_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.name.replace('.nii.gz', '').replace('.nii', '')
    seg_output = temp_dir / f"{stem}_seg"

    logger.info(f"[TotalSegmentator] 处理: {input_path.name}")

    try:
        # 构建 TotalSegmentator 命令
        cmd = [
            "TotalSegmentator",
            "-i", str(input_path),
            "-o", str(seg_output),
            "--roi_subset",
            "lung_upper_lobe_left", "lung_lower_lobe_left",
            "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"
        ]

        if fast:
            cmd.append("--fast")

        if device == "cpu":
            cmd.extend(["--device", "cpu"])

        # 运行 TotalSegmentator
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # 合并所有肺叶 mask
        lung_parts = [
            "lung_upper_lobe_left.nii.gz",
            "lung_lower_lobe_left.nii.gz",
            "lung_upper_lobe_right.nii.gz",
            "lung_middle_lobe_right.nii.gz",
            "lung_lower_lobe_right.nii.gz",
        ]

        combined_mask = None
        affine = None

        for part in lung_parts:
            part_path = seg_output / part
            if part_path.exists():
                nii = nib.load(str(part_path))
                mask = np.asanyarray(nii.dataobj) > 0
                if combined_mask is None:
                    combined_mask = mask
                    affine = nii.affine
                else:
                    combined_mask = combined_mask | mask

        if combined_mask is None:
            raise ValueError("TotalSegmentator 未生成肺部分割结果")

        combined_mask = combined_mask.astype(np.uint8)

        # 保存 mask
        save_nifti(combined_mask, output_mask_path, affine=affine, dtype='uint8')
        logger.info(f"[TotalSegmentator] 已保存 mask: {output_mask_path}")

        # 统计信息
        total_voxels = combined_mask.size
        lung_voxels = int(np.sum(combined_mask))
        lung_ratio = lung_voxels / total_voxels

        stats = {
            'method': 'totalsegmentator',
            'total_voxels': total_voxels,
            'lung_voxels': lung_voxels,
            'lung_ratio': lung_ratio,
            'status': 'success'
        }

        # 可选：保存清洗后的 CT
        if output_clean_path is not None:
            ct_nii = nib.load(str(input_path))
            ct_data = np.asanyarray(ct_nii.dataobj)
            ct_clean = ct_data.copy()
            ct_clean[combined_mask == 0] = background_hu
            save_nifti(ct_clean, output_clean_path, affine=ct_nii.affine)
            logger.info(f"[TotalSegmentator] 已保存清洗后 CT: {output_clean_path}")
            stats['ct_min'] = float(ct_data.min())
            stats['ct_max'] = float(ct_data.max())

        logger.info(f"[TotalSegmentator] 肺占比: {lung_ratio*100:.1f}%")

        return stats

    except subprocess.CalledProcessError as e:
        logger.error(f"[TotalSegmentator] 运行失败: {e.stderr}")
        raise RuntimeError(f"TotalSegmentator 执行失败: {e.stderr}")
    except FileNotFoundError:
        logger.error("[TotalSegmentator] 未安装，请运行: pip install TotalSegmentator")
        raise RuntimeError("TotalSegmentator 未安装")
    finally:
        # 清理临时文件
        if seg_output.exists():
            shutil.rmtree(seg_output)


# ============================================================================
# 阈值分割方法（CPU 轻量备选）
# ============================================================================

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
    mask_output_dir: Union[str, Path] = None,
    clean_output_dir: Union[str, Path] = None,
    output_mask_path: Union[str, Path] = None,
    output_clean_path: Union[str, Path] = None,
    method: str = "auto",
    device: str = "gpu",
    fast: bool = False,
    background_hu: float = -1000
) -> dict:
    """
    从文件进行肺部分割

    Args:
        input_path: 输入 NIfTI 文件路径
        mask_output_dir: mask 输出目录（与 output_mask_path 二选一）
        clean_output_dir: 清洗后 CT 输出目录（与 output_clean_path 二选一）
        output_mask_path: 输出 mask 路径（与 mask_output_dir 二选一）
        output_clean_path: 输出清洗后 CT 路径（与 clean_output_dir 二选一）
        method: 分割方法 ("auto", "totalsegmentator", "threshold")
        device: 设备 ("gpu", "cpu")
        fast: 是否使用快速模式（仅 TotalSegmentator）
        background_hu: 背景 HU 值

    Returns:
        stats: 统计信息
    """
    input_path = Path(input_path)
    stem = input_path.name.replace('.nii.gz', '').replace('.nii', '')

    # 确定输出路径
    if output_mask_path is None:
        if mask_output_dir is None:
            raise ValueError("必须提供 mask_output_dir 或 output_mask_path")
        mask_output_dir = Path(mask_output_dir)
        mask_output_dir.mkdir(parents=True, exist_ok=True)
        output_mask_path = mask_output_dir / f"{stem}_mask.nii.gz"
    else:
        output_mask_path = Path(output_mask_path)
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)

    if output_clean_path is None and clean_output_dir is not None:
        clean_output_dir = Path(clean_output_dir)
        clean_output_dir.mkdir(parents=True, exist_ok=True)
        output_clean_path = clean_output_dir / f"{stem}_clean.nii.gz"
    elif output_clean_path is not None:
        output_clean_path = Path(output_clean_path)
        output_clean_path.parent.mkdir(parents=True, exist_ok=True)

    # 自动选择方法
    if method == "auto":
        method = get_default_method()
        logger.info(f"自动选择分割方法: {method}")

    # 根据方法调用不同的分割函数
    if method == "totalsegmentator":
        try:
            stats = totalsegmentator_lung_segmentation(
                input_path=input_path,
                output_mask_path=output_mask_path,
                output_clean_path=output_clean_path,
                device=device,
                fast=fast,
                background_hu=background_hu
            )
            return stats
        except (RuntimeError, FileNotFoundError) as e:
            logger.warning(f"TotalSegmentator 失败: {e}，回退到阈值方法")
            method = "threshold"

    # 阈值方法
    logger.info(f"[阈值方法] 处理: {input_path.name}")

    # 加载数据
    ct_data, affine = load_nifti(input_path, return_affine=True)

    # 分割
    lung_mask = threshold_lung_segmentation(ct_data)

    # 保存 mask
    save_nifti(lung_mask, output_mask_path, affine=affine, dtype='uint8')
    logger.info(f"已保存 mask: {output_mask_path}")

    # 统计信息
    stats = {
        'method': 'threshold',
        'total_voxels': int(ct_data.size),
        'lung_voxels': int(np.sum(lung_mask)),
        'lung_ratio': float(np.sum(lung_mask) / ct_data.size),
        'ct_min': float(ct_data.min()),
        'ct_max': float(ct_data.max()),
        'status': 'success'
    }

    # 可选：保存清洗后的 CT
    if output_clean_path is not None:
        ct_clean = ct_data.copy()
        ct_clean[lung_mask == 0] = background_hu
        save_nifti(ct_clean, output_clean_path, affine=affine)
        logger.info(f"已保存清洗后 CT: {output_clean_path}")

    return stats


def batch_segment_lungs(
    input_dir: Union[str, Path],
    mask_output_dir: Union[str, Path],
    clean_output_dir: Union[str, Path] = None,
    pattern: str = "*.nii.gz",
    method: str = "auto",
    device: str = "gpu",
    fast: bool = False
) -> dict:
    """
    批量肺部分割

    Args:
        input_dir: 输入目录
        mask_output_dir: mask 输出目录
        clean_output_dir: 清洗后 CT 输出目录（可选）
        pattern: 文件匹配模式
        method: 分割方法 ("auto", "totalsegmentator", "threshold")
        device: 设备 ("gpu", "cpu")
        fast: 是否使用快速模式

    Returns:
        results: 结果字典
    """
    input_dir = Path(input_dir)
    mask_output_dir = Path(mask_output_dir)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    if clean_output_dir:
        clean_output_dir = Path(clean_output_dir)
        clean_output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(list(input_dir.glob(pattern)))
    logger.info(f"找到 {len(nifti_files)} 个 NIfTI 文件")

    # 确定实际使用的方法
    actual_method = method if method != "auto" else get_default_method()
    logger.info(f"使用分割方法: {actual_method}")

    results = {}

    for i, nifti_path in enumerate(nifti_files, start=1):
        stem = nifti_path.name.replace('.nii.gz', '').replace('.nii', '')

        try:
            stats = segment_lung_from_file(
                input_path=nifti_path,
                mask_output_dir=mask_output_dir,
                clean_output_dir=clean_output_dir,
                method=method,
                device=device,
                fast=fast
            )
            results[stem] = stats
            logger.info(f"[{i}/{len(nifti_files)}] 成功 - 肺占比: {stats['lung_ratio']*100:.1f}%")
        except Exception as e:
            logger.error(f"[{i}/{len(nifti_files)}] 失败: {e}")
            results[stem] = {'status': 'failed', 'error': str(e)}

    success_count = sum(1 for r in results.values() if r.get('status') == 'success')
    logger.info(f"批量分割完成: {success_count}/{len(nifti_files)} 成功")

    return results


def main():
    """主函数

    支持命令行参数：
        --method: 分割方法 (auto/totalsegmentator/threshold)
        --device: 设备 (gpu/cpu)
        --type: 数据类型 (all/normal/copd)
        --fast: 快速模式
        --check-only: 仅检查环境

    数据流说明（Phase 2 优化后）：
        输入: data/00_raw/{normal,copd}/*.nii.gz (已转换的 NIfTI)
        输出: data/01_cleaned/{normal,copd}_{mask,clean}/*.nii.gz
    """
    import yaml

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='肺部分割（支持 TotalSegmentator GPU 加速）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python -m src.01_preprocessing.simple_lung_segment
  python -m src.01_preprocessing.simple_lung_segment --method totalsegmentator --device cuda:0
  python -m src.01_preprocessing.simple_lung_segment --method threshold
  python -m src.01_preprocessing.simple_lung_segment --type normal --fast
        """
    )

    parser.add_argument(
        '--method', choices=['auto', 'totalsegmentator', 'threshold'],
        default='auto',
        help='分割方法: auto=自动选择, totalsegmentator=GPU深度学习, threshold=CPU阈值 (默认: auto)'
    )
    parser.add_argument(
        '--device', type=str, default='gpu',
        help='设备: gpu, cpu, cuda:0, cuda:1 等 (默认: gpu)'
    )
    parser.add_argument(
        '--type', choices=['all', 'normal', 'copd'],
        default='all',
        help='处理的数据类型 (默认: all)'
    )
    parser.add_argument(
        '--fast', action='store_true',
        help='快速模式（TotalSegmentator 精度略低但速度快 5 倍）'
    )
    parser.add_argument(
        '--check-only', action='store_true',
        help='仅检查环境配置，不执行分割'
    )

    args = parser.parse_args()

    # 环境检查
    logger.info("=" * 60)
    logger.info("环境检查")
    logger.info("=" * 60)

    gpu_ok, gpu_msg = check_gpu_available()
    ts_ok, ts_msg = check_totalsegmentator_available()

    logger.info(f"GPU: {'✓' if gpu_ok else '✗'} {gpu_msg}")
    logger.info(f"TotalSegmentator: {'✓' if ts_ok else '✗'} {ts_msg}")

    default_method = get_default_method()
    logger.info(f"默认方法: {default_method}")
    logger.info(f"请求方法: {args.method}")

    if args.check_only:
        logger.info("环境检查完成。")
        return

    # 加载配置
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config['paths']['raw_data'])
    cleaned_dir = Path(config['paths']['cleaned_data'])

    logger.info("=" * 60)
    logger.info(f"分割方法: {args.method}")
    logger.info(f"设备: {args.device}")
    logger.info(f"数据类型: {args.type}")
    logger.info(f"快速模式: {args.fast}")
    logger.info("=" * 60)

    # 分割正常肺
    if args.type in ['all', 'normal']:
        normal_input = raw_dir / 'normal'
        if normal_input.exists():
            logger.info("=" * 50)
            logger.info("分割正常肺...")
            logger.info(f"  输入: {normal_input}")
            batch_segment_lungs(
                normal_input,
                mask_output_dir=cleaned_dir / 'normal_mask',
                clean_output_dir=cleaned_dir / 'normal_clean',
                method=args.method,
                device=args.device,
                fast=args.fast
            )
        else:
            logger.warning(f"正常肺数据目录不存在: {normal_input}")

    # 分割 COPD
    if args.type in ['all', 'copd']:
        copd_input = raw_dir / 'copd'
        if copd_input.exists():
            logger.info("=" * 50)
            logger.info("分割 COPD...")
            logger.info(f"  输入: {copd_input}")
            batch_segment_lungs(
                copd_input,
                mask_output_dir=cleaned_dir / 'copd_mask',
                clean_output_dir=cleaned_dir / 'copd_clean',
                method=args.method,
                device=args.device,
                fast=args.fast
            )
        else:
            logger.warning(f"COPD 数据目录不存在: {copd_input}")

    logger.info("=" * 60)
    logger.info("全部分割完成!")


if __name__ == "__main__":
    main()

