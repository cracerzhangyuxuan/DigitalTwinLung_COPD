#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肺部分割模块

=============================================================================
重要更新 (2025-12-24):
=============================================================================
由于 TotalSegmentator 在气管树和肺叶分割上存在以下问题：
1. 气管树分割质量差：仅能分割主气管，缺少分支结构
2. 肺叶分割边界碎片化：5 个肺叶之间的边界出现不连续的碎片

因此本模块已替换为专用模型：
- 气管树分割：Raidionicsrads (AGU-Net) - 分支检测能力强，可达 3-4 级支气管
- 肺叶分割：LungMask (LTRCLobes_R231) - 边界质量高，支持病理肺

原 TotalSegmentator 代码已保留但标记为"已弃用"，不再调用。
=============================================================================

支持功能：
- GPU 加速分割（LungMask + Raidionicsrads）
- CPU 阈值分割（备选方案）
- 气管树分割（Raidionicsrads AGU-Net）
- 肺叶精细标记（LungMask LTRCLobes，5个肺叶独立标签）
- 批量处理
- 环境检查

作者: DigitalTwinLung_COPD Team
日期: 2025-12-09
更新: 2025-12-14 - 整合 GPU 分割功能
更新: 2025-12-22 - 添加气管树分割和肺叶精细标记功能
更新: 2025-12-24 - 替换 TotalSegmentator 为 LungMask + Raidionicsrads
"""

import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict

import numpy as np

from ..utils.logger import get_logger
from ..utils.io import load_nifti, save_nifti

logger = get_logger(__name__)


# =============================================================================
# 环境检查函数
# =============================================================================

def check_gpu_available() -> Tuple[bool, str]:
    """
    检查 GPU 是否可用

    Returns:
        (is_available, message): GPU 可用性和描述信息
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return True, f"GPU 可用: {gpu_name}"
        else:
            return False, "CUDA 不可用"
    except ImportError:
        return False, "PyTorch 未安装"


def check_lungmask_available() -> Tuple[bool, str]:
    """
    检查 LungMask 是否可用（用于肺叶分割）

    Returns:
        (is_available, message): 可用性和描述信息
    """
    try:
        from lungmask import LMInferer
        return True, "LungMask 可用"
    except ImportError:
        return False, "LungMask 未安装，请运行: pip install lungmask"
    except Exception as e:
        return False, f"LungMask 检查失败: {e}"


def check_raidionicsrads_available() -> Tuple[bool, str]:
    """
    检查 Raidionicsrads 是否可用（用于气管树分割）

    Returns:
        (is_available, message): 可用性和描述信息
    """
    try:
        from raidionicsrads.compute import run_model
        return True, "Raidionicsrads 可用"
    except ImportError:
        return False, "Raidionicsrads 未安装，请运行: pip install raidionicsrads"
    except Exception as e:
        return False, f"Raidionicsrads 检查失败: {e}"


def check_totalsegmentator_available() -> Tuple[bool, str]:
    """
    [已弃用] 检查 TotalSegmentator 是否可用

    注意：TotalSegmentator 已不再用于气管树和肺叶分割，
    但保留此函数用于兼容性检查。

    Returns:
        (is_available, message): 可用性和描述信息
    """
    try:
        result = subprocess.run(
            ["TotalSegmentator", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return True, "TotalSegmentator 可用 (已弃用，不再使用)"
        return False, "TotalSegmentator 命令执行失败"
    except FileNotFoundError:
        return False, "TotalSegmentator 未安装"
    except subprocess.TimeoutExpired:
        return False, "TotalSegmentator 响应超时"
    except Exception as e:
        return False, f"检查失败: {e}"


def get_default_method() -> str:
    """
    获取默认分割方法

    优先级：
    1. lungmask + raidionicsrads（推荐）
    2. threshold（备选）

    Returns:
        method: "lungmask" 或 "threshold"
    """
    lm_ok, _ = check_lungmask_available()
    return "lungmask" if lm_ok else "threshold"


def get_default_device() -> str:
    """
    获取默认设备

    Returns:
        device: "cuda:0" 或 "cpu"
    """
    gpu_ok, _ = check_gpu_available()
    return "cuda:0" if gpu_ok else "cpu"


def check_segmentation_environment() -> Dict[str, Tuple[bool, str]]:
    """
    检查分割环境的完整状态

    Returns:
        环境检查结果字典
    """
    results = {
        "gpu": check_gpu_available(),
        "lungmask": check_lungmask_available(),
        "raidionicsrads": check_raidionicsrads_available(),
        "totalsegmentator": check_totalsegmentator_available(),  # 仅用于兼容性检查
    }

    # 输出检查结果
    logger.info("=" * 50)
    logger.info("分割环境检查结果:")
    for name, (ok, msg) in results.items():
        status = "✅" if ok else "❌"
        logger.info(f"  {status} {name}: {msg}")
    logger.info("=" * 50)

    return results


# =============================================================================
# 核心分割函数
# =============================================================================


def run_segmentation(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    task: str = "lung",
    fast: bool = False,
    device: str = "gpu"
) -> Path:
    """
    对单个 CT 文件运行 TotalSegmentator 分割
    
    Args:
        input_path: 输入 CT 文件路径 (NIfTI 格式)
        output_dir: 输出目录
        task: 分割任务 ("lung", "total", 等)
        fast: 是否使用快速模式
        device: 使用设备 ("gpu" 或 "cpu")
        
    Returns:
        output_path: 分割结果路径
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 输出路径
    output_path = output_dir / f"{input_path.stem}_segmentation"
    
    logger.info(f"开始分割: {input_path.name}")
    
    # 构建命令
    cmd = [
        "TotalSegmentator",
        "-i", str(input_path),
        "-o", str(output_path),
        "--task", task,
    ]
    
    if fast:
        cmd.append("--fast")
    
    if device == "cpu":
        cmd.extend(["--device", "cpu"])
    
    try:
        # 运行 TotalSegmentator
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"分割完成: {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"TotalSegmentator 运行失败: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("TotalSegmentator 未安装，请运行: pip install TotalSegmentator")
        raise


def batch_segmentation(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    pattern: str = "*.nii.gz",
    task: str = "lung",
    fast: bool = False
) -> List[Path]:
    """
    批量运行分割
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        pattern: 文件匹配模式
        task: 分割任务
        fast: 是否使用快速模式
        
    Returns:
        results: 分割结果路径列表
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    files = list(input_dir.glob(pattern))
    logger.info(f"找到 {len(files)} 个文件待分割")
    
    results = []
    for filepath in files:
        try:
            result = run_segmentation(
                filepath, output_dir, task=task, fast=fast
            )
            results.append(result)
        except Exception as e:
            logger.error(f"分割失败 {filepath.name}: {e}")
    
    logger.info(f"批量分割完成: {len(results)}/{len(files)} 成功")
    return results


def combine_lung_masks(
    segmentation_dir: Union[str, Path]
) -> np.ndarray:
    """
    合并左右肺 mask
    
    TotalSegmentator 输出的肺部分割包含:
    - lung_upper_lobe_left.nii.gz
    - lung_lower_lobe_left.nii.gz
    - lung_upper_lobe_right.nii.gz
    - lung_middle_lobe_right.nii.gz
    - lung_lower_lobe_right.nii.gz
    
    Args:
        segmentation_dir: TotalSegmentator 输出目录
        
    Returns:
        combined_mask: 合并后的肺部 mask
    """
    segmentation_dir = Path(segmentation_dir)
    
    lung_parts = [
        "lung_upper_lobe_left.nii.gz",
        "lung_lower_lobe_left.nii.gz",
        "lung_upper_lobe_right.nii.gz",
        "lung_middle_lobe_right.nii.gz",
        "lung_lower_lobe_right.nii.gz",
    ]
    
    combined_mask = None
    
    for part in lung_parts:
        part_path = segmentation_dir / part
        if part_path.exists():
            mask = load_nifti(part_path)
            if combined_mask is None:
                combined_mask = mask > 0
            else:
                combined_mask = combined_mask | (mask > 0)
    
    if combined_mask is None:
        raise FileNotFoundError(f"未找到肺部分割结果: {segmentation_dir}")

    return combined_mask.astype(np.uint8)


# =============================================================================
# 肺叶标记常量定义
# =============================================================================

# 肺叶标签值定义（符合解剖学标准）
LOBE_LABELS = {
    "lung_upper_lobe_left": 1,      # 左上叶 (Left Upper Lobe)
    "lung_lower_lobe_left": 2,      # 左下叶 (Left Lower Lobe)
    "lung_upper_lobe_right": 3,     # 右上叶 (Right Upper Lobe)
    "lung_middle_lobe_right": 4,    # 右中叶 (Right Middle Lobe)
    "lung_lower_lobe_right": 5,     # 右下叶 (Right Lower Lobe)
}

# 标签值到中文名称的映射
LOBE_NAMES = {
    1: "左上叶 (Left Upper)",
    2: "左下叶 (Left Lower)",
    3: "右上叶 (Right Upper)",
    4: "右中叶 (Right Middle)",
    5: "右下叶 (Right Lower)",
}

# 气管树相关结构（保留用于兼容性）
TRACHEA_STRUCTURES = [
    "trachea",              # 气管
    "bronchus_left",        # 左主支气管
    "bronchus_right",       # 右主支气管
]


# =============================================================================
# 新版分割函数（LungMask + Raidionicsrads）
# =============================================================================

def segment_lung_lobes_lungmask(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    use_fusion: bool = True,
    force_cpu: bool = False,
    batch_size: int = 20
) -> Tuple[np.ndarray, Dict[int, float], np.ndarray]:
    """
    使用 LungMask 进行肺叶分割

    LungMask 输出标签值：
        1 = 左上叶 (Left Upper Lobe)
        2 = 左下叶 (Left Lower Lobe)
        3 = 右上叶 (Right Upper Lobe)
        4 = 右中叶 (Right Middle Lobe)
        5 = 右下叶 (Right Lower Lobe)

    注意：LungMask 的标签值与项目定义完全一致，无需转换！

    Args:
        input_path: 输入 CT 文件路径 (NIfTI 格式)
        output_path: 可选，保存分割结果的路径
        use_fusion: 是否使用 LTRCLobes_R231 融合模型（推荐，边界更清晰）
        force_cpu: 是否强制使用 CPU
        batch_size: 批处理大小（GPU 显存不足时减小）

    Returns:
        labeled_mask: 带标签的肺叶 mask (uint8, 值为 0-5)
        volume_stats: 每个肺叶的体积统计 (单位: mm³)
        affine: NIfTI affine 矩阵
    """
    import nibabel as nib
    import SimpleITK as sitk
    from lungmask import LMInferer

    input_path = Path(input_path)
    start_time = time.time()

    logger.info(f"[LungMask] 开始肺叶分割: {input_path.name}")

    # 初始化 LungMask 推理器
    # use_fusion=True 时使用 LTRCLobes + R231 融合，边界质量更高
    if use_fusion:
        inferer = LMInferer(
            modelname='LTRCLobes',
            fillmodel='R231',
            force_cpu=force_cpu,
            batch_size=batch_size
        )
        logger.info("  使用融合模型: LTRCLobes_R231")
    else:
        inferer = LMInferer(
            modelname='LTRCLobes',
            force_cpu=force_cpu,
            batch_size=batch_size
        )
        logger.info("  使用单一模型: LTRCLobes")

    # 读取 CT 图像
    input_image = sitk.ReadImage(str(input_path))

    # 执行分割
    segmentation = inferer.apply(input_image)
    # segmentation 是 numpy array，形状为 (Z, Y, X)，值为 0-5

    # 获取 affine 矩阵
    nii = nib.load(str(input_path))
    affine = nii.affine

    # 计算体素体积
    voxel_dims = np.abs(np.diag(affine)[:3])
    voxel_volume = float(np.prod(voxel_dims))

    # 计算各肺叶体积统计
    volume_stats = {}
    for label_value in range(1, 6):
        voxel_count = np.sum(segmentation == label_value)
        volume_mm3 = voxel_count * voxel_volume
        volume_stats[label_value] = volume_mm3

    # 输出体积统计日志
    logger.info("=" * 50)
    logger.info("[LungMask] 肺叶体积统计:")
    total_volume = 0.0
    for label, volume in sorted(volume_stats.items()):
        lobe_name = LOBE_NAMES.get(label, f"未知({label})")
        volume_ml = volume / 1000  # 转换为 mL
        logger.info(f"  {lobe_name}: {volume_ml:.1f} mL ({volume:.0f} mm³)")
        total_volume += volume
    logger.info(f"  总肺容积: {total_volume/1000:.1f} mL")
    logger.info("=" * 50)

    # 转换为 uint8
    labeled_mask = segmentation.astype(np.uint8)

    # 保存结果
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_nifti(labeled_mask, output_path, affine=affine, dtype='uint8')
        logger.info(f"[LungMask] 肺叶标签 mask 已保存: {output_path}")

    elapsed = time.time() - start_time
    logger.info(f"[LungMask] 分割完成，耗时: {elapsed:.1f}s")

    return labeled_mask, volume_stats, affine


def segment_airway_raidionics(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    temp_dir: Optional[Union[str, Path]] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    使用 Raidionicsrads (AGU-Net) 进行气管树分割

    Raidionicsrads 特点：
    - 可分割到 3-4 级支气管
    - 分支结构完整
    - 支持正常肺和病理肺

    Args:
        input_path: 输入 CT 文件路径 (NIfTI 格式)
        output_path: 可选，保存气管树 mask 的路径
        temp_dir: 临时目录（用于存放中间结果）

    Returns:
        trachea_mask: 气管树 mask (uint8)
        affine: NIfTI affine 矩阵
    """
    import nibabel as nib
    import tempfile

    input_path = Path(input_path)
    start_time = time.time()

    logger.info(f"[Raidionicsrads] 开始气管树分割: {input_path.name}")

    # 创建临时目录
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="raidionics_"))
    else:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 导入 raidionicsrads
        from raidionicsrads.compute import run_model

        # 运行气管树分割
        # task="airways" 指定气管树分割任务
        run_model(
            input_filename=str(input_path),
            output_folder=str(temp_dir),
            task="airways"
        )

        # 查找输出文件
        # Raidionicsrads 输出文件名可能是 *_airways.nii.gz 或 airways_mask.nii.gz
        possible_outputs = [
            temp_dir / "airways_mask.nii.gz",
            temp_dir / f"{input_path.stem}_airways.nii.gz",
        ]

        airway_file = None
        for p in possible_outputs:
            if p.exists():
                airway_file = p
                break

        # 如果没找到，搜索目录
        if airway_file is None:
            for f in temp_dir.glob("*airway*.nii.gz"):
                airway_file = f
                break

        if airway_file is None:
            logger.error(f"[Raidionicsrads] 未找到气管树分割结果")
            return None, None

        # 加载结果
        nii = nib.load(str(airway_file))
        trachea_mask = np.asanyarray(nii.dataobj) > 0
        trachea_mask = trachea_mask.astype(np.uint8)
        affine = nii.affine

        # 统计信息
        voxel_count = np.sum(trachea_mask)
        voxel_dims = np.abs(np.diag(affine)[:3])
        voxel_volume = float(np.prod(voxel_dims))
        volume_ml = voxel_count * voxel_volume / 1000

        logger.info(f"[Raidionicsrads] 气管树体素数: {voxel_count}, 体积: {volume_ml:.1f} mL")

        # 保存结果
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_nifti(trachea_mask, output_path, affine=affine, dtype='uint8')
            logger.info(f"[Raidionicsrads] 气管树 mask 已保存: {output_path}")

        elapsed = time.time() - start_time
        logger.info(f"[Raidionicsrads] 分割完成，耗时: {elapsed:.1f}s")

        return trachea_mask, affine

    except ImportError as e:
        logger.error(f"[Raidionicsrads] 导入失败: {e}")
        logger.error("请安装: pip install raidionicsrads")
        return None, None
    except Exception as e:
        logger.error(f"[Raidionicsrads] 分割失败: {e}")
        return None, None
    finally:
        # 清理临时文件
        if temp_dir.exists() and temp_dir.name.startswith("raidionics_"):
            shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# 旧版气管树分割函数（TotalSegmentator，已弃用）
# =============================================================================

def extract_trachea_mask(
    segmentation_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    [已弃用] 从 TotalSegmentator 输出中提取气管树 mask

    警告：此函数已弃用！TotalSegmentator 的气管树分割质量差，
    仅能分割主气管，缺少分支结构。请使用 segment_airway_raidionics() 替代。

    保留此函数仅用于兼容性。

    Args:
        segmentation_dir: TotalSegmentator 输出目录
        output_path: 可选，保存气管树 mask 的路径

    Returns:
        trachea_mask: 气管树 mask (uint8)
        affine: NIfTI affine 矩阵（如果有）
    """
    import warnings
    warnings.warn(
        "extract_trachea_mask() 已弃用，TotalSegmentator 气管树分割质量差。"
        "请使用 segment_airway_raidionics() 替代。",
        DeprecationWarning,
        stacklevel=2
    )

    import nibabel as nib

    segmentation_dir = Path(segmentation_dir)
    trachea_mask = None
    affine = None

    # 尝试加载气管 mask
    trachea_path = segmentation_dir / "trachea.nii.gz"
    if trachea_path.exists():
        nii = nib.load(str(trachea_path))
        trachea_mask = np.asanyarray(nii.dataobj) > 0
        affine = nii.affine
        logger.debug(f"加载气管 mask: {trachea_path.name}")
    else:
        logger.warning(f"气管 mask 文件不存在: {trachea_path}")
        return None, None

    trachea_mask = trachea_mask.astype(np.uint8)

    # 保存气管树 mask
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_nifti(trachea_mask, output_path, affine=affine, dtype='uint8')
        logger.info(f"气管树 mask 已保存: {output_path}")

    return trachea_mask, affine


def create_labeled_lung_lobes(
    segmentation_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[np.ndarray, Dict[int, float], Optional[np.ndarray]]:
    """
    [已弃用] 从 TotalSegmentator 输出创建带标签的肺叶 mask

    警告：此函数已弃用！TotalSegmentator 的肺叶分割边界碎片化严重。
    请使用 segment_lung_lobes_lungmask() 替代。

    保留此函数仅用于兼容性。

    标签定义:
        1 = 左上叶 (Left Upper Lobe)
        2 = 左下叶 (Left Lower Lobe)
        3 = 右上叶 (Right Upper Lobe)
        4 = 右中叶 (Right Middle Lobe)
        5 = 右下叶 (Right Lower Lobe)

    Args:
        segmentation_dir: TotalSegmentator 输出目录
        output_path: 可选，保存带标签 mask 的路径

    Returns:
        labeled_mask: 带标签的肺叶 mask (uint8, 值为 0-5)
        volume_stats: 每个肺叶的体积统计 (单位: mm³)
        affine: NIfTI affine 矩阵
    """
    import warnings
    warnings.warn(
        "create_labeled_lung_lobes() 已弃用，TotalSegmentator 肺叶分割边界碎片化。"
        "请使用 segment_lung_lobes_lungmask() 替代。",
        DeprecationWarning,
        stacklevel=2
    )

    import nibabel as nib

    segmentation_dir = Path(segmentation_dir)
    labeled_mask = None
    affine = None
    voxel_volume = 1.0  # 默认体素体积 (mm³)
    volume_stats = {}

    # 遍历所有肺叶结构
    for lobe_file, label_value in LOBE_LABELS.items():
        lobe_path = segmentation_dir / f"{lobe_file}.nii.gz"

        if lobe_path.exists():
            nii = nib.load(str(lobe_path))
            lobe_mask = np.asanyarray(nii.dataobj) > 0

            if labeled_mask is None:
                labeled_mask = np.zeros(lobe_mask.shape, dtype=np.uint8)
                affine = nii.affine
                # 计算体素体积 (mm³)
                voxel_dims = np.abs(np.diag(affine)[:3])
                voxel_volume = float(np.prod(voxel_dims))

            # 分配标签值
            labeled_mask[lobe_mask] = label_value

            # 计算体积
            voxel_count = np.sum(lobe_mask)
            volume_mm3 = voxel_count * voxel_volume
            volume_stats[label_value] = volume_mm3

            logger.debug(f"加载 {LOBE_NAMES[label_value]}: {voxel_count} voxels, {volume_mm3:.1f} mm³")
        else:
            logger.warning(f"肺叶 mask 文件不存在: {lobe_path}")
            volume_stats[label_value] = 0.0

    if labeled_mask is None:
        raise FileNotFoundError(f"未找到任何肺叶分割结果: {segmentation_dir}")

    # 输出体积统计日志
    logger.info("=" * 50)
    logger.info("肺叶体积统计:")
    total_volume = 0.0
    for label, volume in sorted(volume_stats.items()):
        lobe_name = LOBE_NAMES.get(label, f"未知({label})")
        volume_ml = volume / 1000  # 转换为 mL
        logger.info(f"  {lobe_name}: {volume_ml:.1f} mL ({volume:.0f} mm³)")
        total_volume += volume
    logger.info(f"  总肺容积: {total_volume/1000:.1f} mL")
    logger.info("=" * 50)

    # 保存带标签的 mask
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_nifti(labeled_mask, output_path, affine=affine, dtype='uint8')
        logger.info(f"肺叶标签 mask 已保存: {output_path}")

    return labeled_mask, volume_stats, affine


# =============================================================================
# 新版批量处理函数（LungMask + Raidionicsrads）
# =============================================================================

def run_lungmask_batch(
    input_dir: Union[str, Path],
    mask_output_dir: Union[str, Path],
    clean_output_dir: Union[str, Path],
    force_cpu: bool = False,
    skip_existing: bool = True,
    limit: Optional[int] = None,
    background_hu: float = -1000,
    extract_trachea: bool = True,
    create_labeled_lobes: bool = True,
    use_fusion: bool = True
) -> Dict[str, List]:
    """
    使用 LungMask + Raidionicsrads 批量分割（推荐方案）

    替代 TotalSegmentator 的原因：
    1. 气管树分割：TotalSegmentator 仅能分割主气管，缺少分支结构
    2. 肺叶分割：TotalSegmentator 边界碎片化严重

    新方案优势：
    - LungMask LTRCLobes_R231：肺叶边界清晰，支持病理肺
    - Raidionicsrads AGU-Net：气管树分支完整，可达 3-4 级支气管

    Args:
        input_dir: 输入目录
        mask_output_dir: mask 输出目录
        clean_output_dir: 清洗后 CT 输出目录
        force_cpu: 是否强制使用 CPU
        skip_existing: 是否跳过已处理的文件
        limit: 限制处理数量 (用于测试)
        background_hu: 背景 HU 值
        extract_trachea: 是否提取气管树 mask (使用 Raidionicsrads)
        create_labeled_lobes: 是否创建带标签的肺叶 mask (使用 LungMask)
        use_fusion: 是否使用 LungMask 融合模型 (LTRCLobes_R231)

    Returns:
        results: 处理结果字典 {"success": [], "failed": [], "skipped": []}

    Output files:
        - {stem}_mask.nii.gz: 二值肺部 mask
        - {stem}_clean.nii.gz: 清洗后的 CT
        - {stem}_trachea_mask.nii.gz: 气管树 mask (Raidionicsrads)
        - {stem}_lung_lobes_labeled.nii.gz: 带标签的肺叶 mask (LungMask)
    """
    import nibabel as nib

    input_dir = Path(input_dir)
    mask_output_dir = Path(mask_output_dir)
    clean_output_dir = Path(clean_output_dir)

    mask_output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(list(input_dir.glob("*.nii.gz")))
    if limit:
        nifti_files = nifti_files[:limit]

    logger.info("=" * 60)
    logger.info("批量分割配置 (LungMask + Raidionicsrads)")
    logger.info("=" * 60)
    logger.info(f"  输入目录: {input_dir}")
    logger.info(f"  文件数量: {len(nifti_files)}")
    logger.info(f"  肺叶分割: {'启用 (LungMask)' if create_labeled_lobes else '禁用'}")
    logger.info(f"  气管树分割: {'启用 (Raidionicsrads)' if extract_trachea else '禁用'}")
    logger.info(f"  融合模型: {'LTRCLobes_R231' if use_fusion else 'LTRCLobes'}")
    logger.info(f"  设备: {'CPU' if force_cpu else 'GPU (如可用)'}")
    logger.info("=" * 60)

    results = {"success": [], "failed": [], "skipped": []}

    for i, nifti_path in enumerate(nifti_files, 1):
        stem = nifti_path.name.replace('.nii.gz', '').replace('.nii', '')
        mask_path = mask_output_dir / f"{stem}_mask.nii.gz"
        clean_path = clean_output_dir / f"{stem}_clean.nii.gz"
        trachea_path = mask_output_dir / f"{stem}_trachea_mask.nii.gz"
        lobes_path = mask_output_dir / f"{stem}_lung_lobes_labeled.nii.gz"

        # 检查是否已处理（需要检查所有输出文件）
        all_exist = mask_path.exists() and clean_path.exists()
        if extract_trachea:
            all_exist = all_exist and trachea_path.exists()
        if create_labeled_lobes:
            all_exist = all_exist and lobes_path.exists()

        if skip_existing and all_exist:
            logger.info(f"[{i}/{len(nifti_files)}] 跳过已处理: {stem}")
            results["skipped"].append(stem)
            continue

        logger.info(f"[{i}/{len(nifti_files)}] 处理: {stem}")
        start_time = time.time()

        try:
            # ===== 步骤 1: 使用 LungMask 进行肺叶分割 =====
            if create_labeled_lobes:
                labeled_mask, volume_stats, affine = segment_lung_lobes_lungmask(
                    input_path=nifti_path,
                    output_path=lobes_path,
                    use_fusion=use_fusion,
                    force_cpu=force_cpu
                )

                # 从肺叶标签生成二值 mask
                binary_mask = (labeled_mask > 0).astype(np.uint8)
            else:
                # 如果不需要肺叶标签，使用 LungMask R231 进行左右肺分割
                import SimpleITK as sitk
                from lungmask import LMInferer

                inferer = LMInferer(modelname='R231', force_cpu=force_cpu)
                input_image = sitk.ReadImage(str(nifti_path))
                segmentation = inferer.apply(input_image)

                # R231 输出: 1=右肺, 2=左肺
                binary_mask = (segmentation > 0).astype(np.uint8)
                labeled_mask = None

                nii = nib.load(str(nifti_path))
                affine = nii.affine

            # 保存二值 mask
            save_nifti(binary_mask, mask_path, affine=affine, dtype='uint8')
            logger.info(f"    二值 mask 已保存: {mask_path.name}")

            # ===== 步骤 2: 使用 Raidionicsrads 进行气管树分割 =====
            trachea_mask = None
            if extract_trachea:
                trachea_mask, _ = segment_airway_raidionics(
                    input_path=nifti_path,
                    output_path=trachea_path
                )

            # ===== 步骤 3: 创建清洗后的 CT =====
            ct_data, ct_affine = load_nifti(nifti_path, return_affine=True)
            ct_clean = ct_data.copy()

            # 构建保留区域 mask：肺叶 + 气管树
            keep_mask = binary_mask.copy()
            if trachea_mask is not None:
                keep_mask = keep_mask | (trachea_mask > 0)
                logger.debug(f"    保留区域已包含气管树")

            ct_clean[keep_mask == 0] = background_hu
            save_nifti(ct_clean, clean_path, affine=ct_affine)
            logger.info(f"    清洗后 CT 已保存: {clean_path.name}")

            # 统计信息
            lung_ratio = np.sum(binary_mask) / binary_mask.size * 100
            elapsed = time.time() - start_time
            logger.info(f"    ✅ 完成 - 肺占比: {lung_ratio:.1f}%, 耗时: {elapsed:.1f}s")
            results["success"].append(stem)

        except Exception as e:
            logger.error(f"    ❌ 失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            results["failed"].append((stem, str(e)))

    # 输出汇总
    logger.info("=" * 60)
    logger.info("批量分割完成汇总:")
    logger.info(f"  成功: {len(results['success'])}")
    logger.info(f"  失败: {len(results['failed'])}")
    logger.info(f"  跳过: {len(results['skipped'])}")
    logger.info("=" * 60)

    return results


# =============================================================================
# 旧版批量处理函数（TotalSegmentator，已弃用）
# =============================================================================

def run_totalsegmentator_batch(
    input_dir: Union[str, Path],
    mask_output_dir: Union[str, Path],
    clean_output_dir: Union[str, Path],
    device: str = "gpu",
    fast: bool = False,
    skip_existing: bool = True,
    limit: Optional[int] = None,
    background_hu: float = -1000,
    extract_trachea: bool = True,
    create_labeled_lobes: bool = True
) -> Dict[str, List]:
    """
    [已弃用] 使用 TotalSegmentator 批量分割

    ⚠️ 警告：此函数已弃用！
    TotalSegmentator 存在以下问题：
    1. 气管树分割质量差：仅能分割主气管，缺少分支结构
    2. 肺叶分割边界碎片化：5 个肺叶之间的边界出现不连续的碎片

    请使用 run_lungmask_batch() 替代，该函数使用：
    - LungMask LTRCLobes_R231：肺叶边界清晰
    - Raidionicsrads AGU-Net：气管树分支完整

    保留此函数仅用于兼容性和回退测试。

    Args:
        input_dir: 输入目录
        mask_output_dir: mask 输出目录
        clean_output_dir: 清洗后 CT 输出目录
        device: 设备 ("gpu", "cpu", "cuda:0", etc.)
        fast: 是否使用快速模式
        skip_existing: 是否跳过已处理的文件
        limit: 限制处理数量 (用于测试)
        background_hu: 背景 HU 值
        extract_trachea: 是否提取气管树 mask
        create_labeled_lobes: 是否创建带标签的肺叶 mask

    Returns:
        results: 处理结果字典 {"success": [], "failed": [], "skipped": []}

    Output files:
        - {stem}_mask.nii.gz: 二值肺部 mask
        - {stem}_clean.nii.gz: 清洗后的 CT
        - {stem}_trachea_mask.nii.gz: 气管树 mask (如果 extract_trachea=True)
        - {stem}_lung_lobes_labeled.nii.gz: 带标签的肺叶 mask (如果 create_labeled_lobes=True)
    """
    import warnings
    warnings.warn(
        "run_totalsegmentator_batch() 已弃用！"
        "TotalSegmentator 气管树分割质量差，肺叶边界碎片化。"
        "请使用 run_lungmask_batch() 替代。",
        DeprecationWarning,
        stacklevel=2
    )

    import nibabel as nib

    input_dir = Path(input_dir)
    mask_output_dir = Path(mask_output_dir)
    clean_output_dir = Path(clean_output_dir)

    mask_output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir.mkdir(parents=True, exist_ok=True)

    # 临时目录用于 TotalSegmentator 输出
    temp_dir = input_dir.parent.parent / ".temp_segmentation"
    temp_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(list(input_dir.glob("*.nii.gz")))
    if limit:
        nifti_files = nifti_files[:limit]

    logger.info(f"找到 {len(nifti_files)} 个文件待处理")
    logger.info(f"气管树分割: {'启用' if extract_trachea else '禁用'}")
    logger.info(f"肺叶标记: {'启用' if create_labeled_lobes else '禁用'}")

    results = {"success": [], "failed": [], "skipped": []}

    for i, nifti_path in enumerate(nifti_files, 1):
        stem = nifti_path.name.replace('.nii.gz', '').replace('.nii', '')
        mask_path = mask_output_dir / f"{stem}_mask.nii.gz"
        clean_path = clean_output_dir / f"{stem}_clean.nii.gz"
        trachea_path = mask_output_dir / f"{stem}_trachea_mask.nii.gz"
        lobes_path = mask_output_dir / f"{stem}_lung_lobes_labeled.nii.gz"

        # 检查是否已处理（需要检查所有输出文件）
        all_exist = mask_path.exists() and clean_path.exists()
        if extract_trachea:
            all_exist = all_exist and trachea_path.exists()
        if create_labeled_lobes:
            all_exist = all_exist and lobes_path.exists()

        if skip_existing and all_exist:
            logger.info(f"[{i}/{len(nifti_files)}] 跳过已处理: {stem}")
            results["skipped"].append(stem)
            continue

        logger.info(f"[{i}/{len(nifti_files)}] 处理: {stem}")

        try:
            # 运行 TotalSegmentator
            seg_output = temp_dir / f"{stem}_seg"

            cmd = ["TotalSegmentator", "-i", str(nifti_path), "-o", str(seg_output)]

            # 构建 ROI 列表：肺叶 + 气管（如果需要）
            roi_list = [
                "lung_upper_lobe_left", "lung_lower_lobe_left",
                "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"
            ]
            if extract_trachea:
                roi_list.append("trachea")

            cmd.extend(["-rs"] + roi_list)

            if fast:
                cmd.append("-f")

            # 设备选择
            if device == "cpu":
                cmd.extend(["-d", "cpu"])
            elif device.startswith("cuda:"):
                gpu_id = device.split(":")[1]
                cmd.extend(["-d", f"gpu:{gpu_id}"])
            elif device.startswith("gpu:"):
                cmd.extend(["-d", device])
            elif device == "gpu":
                cmd.extend(["-d", "gpu"])

            # 运行命令
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else result.stdout
                raise RuntimeError(f"TotalSegmentator 失败: {error_msg[:200]}")

            # 合并肺叶 mask（二值）
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
                raise ValueError("未找到肺部分割结果")

            combined_mask = combined_mask.astype(np.uint8)

            # 保存二值 mask
            save_nifti(combined_mask, mask_path, affine=affine, dtype='uint8')

            # 提取气管树 mask
            if extract_trachea:
                trachea_mask, _ = extract_trachea_mask(seg_output, output_path=trachea_path)
                if trachea_mask is not None:
                    trachea_voxels = np.sum(trachea_mask)
                    logger.info(f"    气管树体素数: {trachea_voxels}")
                else:
                    logger.warning(f"    气管树分割失败或未检测到")

            # 创建带标签的肺叶 mask
            if create_labeled_lobes:
                # volume_stats 已在函数内部通过日志输出
                create_labeled_lung_lobes(seg_output, output_path=lobes_path)

            # 加载原始 CT 并创建清洗后版本
            # 注意：保留区域 = 肺叶 + 气管树，确保配准时气管树可见
            ct_data, ct_affine = load_nifti(nifti_path, return_affine=True)
            ct_clean = ct_data.copy()

            # 构建保留区域 mask：肺叶 + 气管树
            keep_mask = combined_mask.copy()
            if extract_trachea and trachea_mask is not None:
                # 将气管树也加入保留区域
                keep_mask = keep_mask | (trachea_mask > 0)
                logger.debug(f"    保留区域已包含气管树")

            ct_clean[keep_mask == 0] = background_hu
            save_nifti(ct_clean, clean_path, affine=ct_affine)

            # 清理临时文件
            if seg_output.exists():
                shutil.rmtree(seg_output)

            lung_ratio = np.sum(combined_mask) / combined_mask.size * 100
            logger.info(f"    ✅ 完成 - 肺占比: {lung_ratio:.1f}%")
            results["success"].append(stem)

        except Exception as e:
            logger.error(f"    ❌ 失败: {e}")
            results["failed"].append((stem, str(e)))

    # 清理临时目录
    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()

    return results


def run_threshold_batch(
    input_dir: Union[str, Path],
    mask_output_dir: Union[str, Path],
    clean_output_dir: Union[str, Path],
    skip_existing: bool = True,
    limit: Optional[int] = None
) -> Dict[str, List]:
    """
    使用阈值方法批量分割

    Args:
        input_dir: 输入目录
        mask_output_dir: mask 输出目录
        clean_output_dir: 清洗后 CT 输出目录
        skip_existing: 是否跳过已处理的文件
        limit: 限制处理数量

    Returns:
        results: 处理结果字典
    """
    from .simple_lung_segment import segment_lung_from_file

    input_dir = Path(input_dir)
    mask_output_dir = Path(mask_output_dir)
    clean_output_dir = Path(clean_output_dir)

    mask_output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(list(input_dir.glob("*.nii.gz")))

    if limit:
        nifti_files = nifti_files[:limit]

    results = {"success": [], "failed": [], "skipped": []}

    for i, nifti_path in enumerate(nifti_files, 1):
        stem = nifti_path.name.replace('.nii.gz', '').replace('.nii', '')
        mask_path = mask_output_dir / f"{stem}_mask.nii.gz"
        clean_path = clean_output_dir / f"{stem}_clean.nii.gz"

        # 检查是否已处理
        if skip_existing and mask_path.exists() and clean_path.exists():
            logger.info(f"[{i}/{len(nifti_files)}] 跳过已处理: {stem}")
            results["skipped"].append(stem)
            continue

        logger.info(f"[{i}/{len(nifti_files)}] 处理: {stem}")

        try:
            result = segment_lung_from_file(
                nifti_path,
                mask_output_dir=mask_output_dir,
                clean_output_dir=clean_output_dir
            )
            if result.get('status') == 'success':
                lung_ratio = result.get('lung_ratio', 0) * 100
                logger.info(f"    ✅ 完成 - 肺占比: {lung_ratio:.1f}%")
                results["success"].append(stem)
            else:
                logger.error(f"    ❌ 失败: {result.get('error', '未知错误')}")
                results["failed"].append((stem, result.get('error', '未知错误')))
        except Exception as e:
            logger.error(f"    ❌ 异常: {e}")
            results["failed"].append((stem, str(e)))

    return results


# =============================================================================
# 主函数
# =============================================================================

def main(config: dict) -> None:
    """
    主函数 - 从配置运行分割流程

    Args:
        config: 配置字典
    """
    input_dirs = [
        Path(config['paths']['raw_data']) / 'normal',
        Path(config['paths']['raw_data']) / 'copd',
    ]

    output_base = Path(config['paths']['cleaned_data'])

    for input_dir in input_dirs:
        if not input_dir.exists():
            logger.warning(f"目录不存在，跳过: {input_dir}")
            continue

        output_dir = output_base / f"{input_dir.name}_segmented"

        batch_segmentation(
            input_dir=input_dir,
            output_dir=output_dir,
            task=config.get('preprocessing', {}).get('segmentation', {}).get('task', 'lung'),
            fast=config.get('preprocessing', {}).get('segmentation', {}).get('fast_mode', False),
        )


if __name__ == "__main__":
    import yaml

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    main(config)

