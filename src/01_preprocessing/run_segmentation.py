#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肺部分割模块

=============================================================================
重要更新 (2025-12-25):
=============================================================================
分割方案：
- 肺叶分割：LungMask (LTRCLobes_R231) - 边界质量高，支持病理肺 ✅
- 气管树分割：TotalSegmentator --task lung_vessels ✅
  (使用 lung_vessels 任务而非默认 total 任务，可获得完整支气管树 3-4 级分支)

TotalSegmentator 任务对比：
- --task total：仅输出 trachea.nii.gz（主气管）
- --task lung_vessels：输出 lung_trachea_bronchia.nii.gz（完整支气管树）

已移除 Raidionicsrads：该包仅支持 neuro_diagnosis 和 mediastinum_diagnosis，
不支持 airways_segmentation 任务。
=============================================================================

支持功能：
- GPU 加速分割（LungMask + TotalSegmentator）
- CPU 阈值分割（备选方案）
- 气管树分割（TotalSegmentator lung_vessels 任务）
- 肺叶精细标记（LungMask LTRCLobes，5个肺叶独立标签）
- 批量处理
- 环境检查

作者: DigitalTwinLung_COPD Team
日期: 2025-12-09
更新: 2025-12-14 - 整合 GPU 分割功能
更新: 2025-12-22 - 添加气管树分割和肺叶精细标记功能
更新: 2025-12-24 - 替换 TotalSegmentator 为 LungMask + Raidionicsrads
更新: 2025-12-25 - 气管树分割改用 TotalSegmentator lung_vessels 任务
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


# LungMask 模型文件信息（用于校验）
LUNGMASK_MODELS = {
    "unet_ltrclobes-3a07043d.pth": {
        "url": "https://github.com/JoHof/lungmask/releases/download/v0.0/unet_ltrclobes-3a07043d.pth",
        "expected_size_mb": 119,  # 约 119 MB
        "min_size_bytes": 100_000_000,  # 最小 100 MB
    },
    "unet_r231-d5d2fc3d.pth": {
        "url": "https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231-d5d2fc3d.pth",
        "expected_size_mb": 30,  # 约 30 MB
        "min_size_bytes": 25_000_000,  # 最小 25 MB
    },
}


def get_torch_cache_dir() -> Path:
    """获取 PyTorch hub 缓存目录"""
    import torch
    # PyTorch 默认缓存目录
    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    return cache_dir


def verify_lungmask_models(auto_fix: bool = True) -> Tuple[bool, str]:
    """
    验证 LungMask 模型文件的完整性

    检查缓存目录中的模型文件是否存在且大小正确。
    如果发现损坏的文件（大小不足），可以自动删除以便重新下载。

    Args:
        auto_fix: 是否自动删除损坏的文件

    Returns:
        (is_valid, message): 验证结果和详细信息
    """
    try:
        cache_dir = get_torch_cache_dir()
    except Exception as e:
        return False, f"无法获取缓存目录: {e}"

    issues = []
    fixed = []

    for model_name, info in LUNGMASK_MODELS.items():
        model_path = cache_dir / model_name

        if model_path.exists():
            file_size = model_path.stat().st_size
            min_size = info["min_size_bytes"]
            expected_mb = info["expected_size_mb"]

            if file_size < min_size:
                # 文件太小，可能是下载中断
                actual_mb = file_size / 1_000_000
                issues.append(
                    f"  ❌ {model_name}: 文件损坏（{actual_mb:.1f} MB < 预期 {expected_mb} MB）"
                )

                if auto_fix:
                    try:
                        model_path.unlink()
                        fixed.append(f"  🔧 已删除损坏文件: {model_name}")
                    except Exception as e:
                        issues.append(f"  ⚠️ 无法删除损坏文件 {model_name}: {e}")
            else:
                logger.debug(f"  ✓ {model_name}: {file_size / 1_000_000:.1f} MB (正常)")
        else:
            # 文件不存在，首次运行时会自动下载
            logger.debug(f"  ⏳ {model_name}: 未缓存（首次运行时将下载）")

    if issues:
        msg = "LungMask 模型文件校验失败:\n" + "\n".join(issues)
        if fixed:
            msg += "\n\n已自动修复:\n" + "\n".join(fixed)
            msg += "\n\n请重新运行，将自动下载完整的模型文件。"
        return False, msg

    return True, "LungMask 模型文件校验通过"


def ensure_lungmask_models_ready() -> bool:
    """
    确保 LungMask 模型已准备就绪

    调用此函数会：
    1. 检查模型文件是否存在且完整
    2. 如果发现损坏文件，自动删除
    3. 返回是否可以安全调用 LungMask

    Returns:
        is_ready: 模型是否已准备就绪
    """
    is_valid, msg = verify_lungmask_models(auto_fix=True)

    if not is_valid:
        logger.warning(msg)
        logger.info("")
        logger.info("=" * 60)
        logger.info("如果下载速度慢，可以手动下载模型文件：")
        logger.info("=" * 60)

        cache_dir = get_torch_cache_dir()
        for model_name, info in LUNGMASK_MODELS.items():
            logger.info(f"  wget -c {info['url']}")
            logger.info(f"       -O {cache_dir / model_name}")
            logger.info("")

        return False

    return True


def check_totalsegmentator_lung_vessels_available() -> Tuple[bool, str]:
    """
    检查 TotalSegmentator lung_vessels 任务是否可用（用于气管树分割）

    lung_vessels 任务输出包含完整的支气管树（3-4 级分支），
    比默认的 total 任务（仅主气管）质量更高。

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
            return True, "TotalSegmentator 可用 (支持 lung_vessels 任务)"
        return False, "TotalSegmentator 命令执行失败"
    except FileNotFoundError:
        return False, "TotalSegmentator 未安装，请运行: pip install TotalSegmentator"
    except subprocess.TimeoutExpired:
        return False, "TotalSegmentator 响应超时"
    except Exception as e:
        return False, f"检查失败: {e}"


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
        "totalsegmentator_lung_vessels": check_totalsegmentator_lung_vessels_available(),
        "totalsegmentator": check_totalsegmentator_available(),
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

    Raises:
        RuntimeError: 如果模型文件损坏或下载失败
    """
    import nibabel as nib
    import SimpleITK as sitk

    # 在导入 LungMask 之前验证模型文件完整性
    # 这可以提前发现下载中断导致的损坏文件
    if not ensure_lungmask_models_ready():
        raise RuntimeError(
            "LungMask 模型文件不完整或已损坏。\n"
            "已自动删除损坏文件，请重新运行以下载完整模型。\n"
            "如果下载速度慢，请参考日志中的手动下载说明。"
        )

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
    # 需要转置为 nibabel 的 (X, Y, Z) 顺序以与原始 CT 数据对齐
    segmentation = np.transpose(segmentation, (2, 1, 0))
    logger.debug(f"  分割结果形状（转置后）: {segmentation.shape}")

    # 获取 affine 矩阵
    nii = nib.load(str(input_path))
    affine = nii.affine
    logger.debug(f"  原始 CT 形状: {nii.shape}")

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


def segment_lung_lobes_totalsegmentator(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    device: str = "gpu"
) -> Tuple[Optional[np.ndarray], Optional[Dict[int, float]], Optional[np.ndarray]]:
    """
    使用 TotalSegmentator 进行肺叶分割

    命令：TotalSegmentator -i input.nii.gz -o output_dir/ --task total --device gpu

    输出文件映射（标签值与 LungMask 一致）：
        lung_upper_lobe_left.nii.gz  → 标签 1 (左上叶)
        lung_lower_lobe_left.nii.gz  → 标签 2 (左下叶)
        lung_upper_lobe_right.nii.gz → 标签 3 (右上叶)
        lung_middle_lobe_right.nii.gz → 标签 4 (右中叶)
        lung_lower_lobe_right.nii.gz → 标签 5 (右下叶)

    Args:
        input_path: 输入 CT 文件路径 (NIfTI 格式)
        output_path: 可选，保存肺叶标签 mask 的路径
        device: 设备选择 ("gpu" 或 "cpu")

    Returns:
        lobes_labeled: 肺叶标签 mask (uint8, 值 0-5)
        volume_stats: 每个肺叶的体积统计 (mm³)
        affine: NIfTI affine 矩阵
    """
    import nibabel as nib
    import tempfile

    input_path = Path(input_path)
    start_time = time.time()

    logger.info(f"[TotalSegmentator] 开始肺叶分割 (--task total): {input_path.name}")

    # 创建临时目录存放分割结果
    temp_dir = Path(tempfile.mkdtemp(prefix="totalseg_lobes_"))

    # TotalSegmentator 输出文件到标签的映射
    lobe_file_mapping = {
        "lung_upper_lobe_left.nii.gz": 1,    # 左上叶
        "lung_lower_lobe_left.nii.gz": 2,    # 左下叶
        "lung_upper_lobe_right.nii.gz": 3,   # 右上叶
        "lung_middle_lobe_right.nii.gz": 4,  # 右中叶
        "lung_lower_lobe_right.nii.gz": 5,   # 右下叶
    }

    try:
        # 构建 TotalSegmentator 命令
        # 使用 -rs (roi_subset) 仅分割肺叶，加速处理
        roi_list = [
            "lung_upper_lobe_left", "lung_lower_lobe_left",
            "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"
        ]

        cmd = [
            "TotalSegmentator",
            "-i", str(input_path),
            "-o", str(temp_dir),
            "-rs",
        ] + roi_list

        # 设备选择
        if device.lower() == "cpu":
            cmd.extend(["--device", "cpu"])
        else:
            cmd.extend(["--device", "gpu"])

        logger.info(f"[TotalSegmentator] 执行命令: {' '.join(cmd[:6])}... (共 {len(roi_list)} 个 ROI)")

        # 运行 TotalSegmentator
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 分钟超时
        )

        if result.returncode != 0:
            logger.error(f"[TotalSegmentator] 命令执行失败")
            logger.error(f"[TotalSegmentator] stderr: {result.stderr[:500]}")
            return None, None, None

        # 合并所有肺叶为带标签的 mask
        labeled_mask = None
        affine = None
        volume_stats = {}

        for lobe_file, label_value in lobe_file_mapping.items():
            lobe_path = temp_dir / lobe_file

            if lobe_path.exists():
                nii = nib.load(str(lobe_path))
                lobe_mask = np.asanyarray(nii.dataobj) > 0

                if labeled_mask is None:
                    labeled_mask = np.zeros(lobe_mask.shape, dtype=np.uint8)
                    affine = nii.affine

                labeled_mask[lobe_mask] = label_value

                # 计算体积
                voxel_dims = np.abs(np.diag(nii.affine)[:3])
                voxel_volume = float(np.prod(voxel_dims))  # mm³
                lobe_volume = np.sum(lobe_mask) * voxel_volume
                volume_stats[label_value] = lobe_volume

                lobe_name = LOBE_NAMES.get(label_value, f"Lobe {label_value}")
                logger.info(f"    肺叶 {label_value} ({lobe_name}): {lobe_volume/1000:.1f} mL")
            else:
                logger.warning(f"[TotalSegmentator] 未找到肺叶文件: {lobe_file}")

        if labeled_mask is None:
            logger.error("[TotalSegmentator] 没有找到任何肺叶分割结果")
            return None, None, None

        # 保存结果
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_nifti(labeled_mask, output_path, affine=affine, dtype='uint8')
            logger.info(f"[TotalSegmentator] 肺叶标签 mask 已保存: {output_path}")

        elapsed = time.time() - start_time
        logger.info(f"[TotalSegmentator] 肺叶分割完成，耗时: {elapsed:.1f}s")

        return labeled_mask, volume_stats, affine

    except subprocess.TimeoutExpired:
        logger.error("[TotalSegmentator] 执行超时（>30分钟）")
        return None, None, None
    except FileNotFoundError:
        logger.error("[TotalSegmentator] 未找到 TotalSegmentator 命令")
        logger.error("[TotalSegmentator] 请安装: pip install TotalSegmentator")
        return None, None, None
    except Exception as e:
        logger.error(f"[TotalSegmentator] 肺叶分割失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None, None, None
    finally:
        # 清理临时文件
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def segment_airway_totalsegmentator(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    device: str = "gpu",
    fast: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    使用 TotalSegmentator lung_vessels 任务进行气管树分割

    重要：使用 --task lung_vessels 而非默认的 --task total
    - total 任务：仅输出 trachea.nii.gz（主气管）
    - lung_vessels 任务：输出 lung_trachea_bronchia.nii.gz（完整支气管树 3-4 级分支）

    Args:
        input_path: 输入 CT 文件路径 (NIfTI 格式)
        output_path: 可选，保存气管树 mask 的路径
        device: 设备选择 ("gpu" 或 "cpu")
        fast: 是否使用快速模式（精度略低但速度更快）

    Returns:
        trachea_mask: 气管树 mask (uint8)
        affine: NIfTI affine 矩阵
    """
    import nibabel as nib
    import tempfile

    input_path = Path(input_path)
    start_time = time.time()

    logger.info(f"[TotalSegmentator] 开始气管树分割 (lung_vessels 任务): {input_path.name}")

    # 创建临时目录存放分割结果
    temp_dir = Path(tempfile.mkdtemp(prefix="totalseg_airways_"))

    try:
        # 构建 TotalSegmentator 命令
        # 关键：使用 --task lung_vessels 获取完整支气管树
        cmd = [
            "TotalSegmentator",
            "-i", str(input_path),
            "-o", str(temp_dir),
            "--task", "lung_vessels",  # 关键参数！获取完整支气管树
        ]

        # 设备选择
        if device.lower() == "cpu":
            cmd.extend(["--device", "cpu"])
        else:
            cmd.extend(["--device", "gpu"])

        # 快速模式
        if fast:
            cmd.append("--fast")

        logger.info(f"[TotalSegmentator] 执行命令: {' '.join(cmd)}")

        # 运行 TotalSegmentator
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 分钟超时
        )

        if result.returncode != 0:
            logger.error(f"[TotalSegmentator] 命令执行失败")
            logger.error(f"[TotalSegmentator] stderr: {result.stderr[:500]}")
            return None, None

        # 查找支气管树输出文件
        # lung_vessels 任务输出 lung_trachea_bronchia.nii.gz
        bronchia_path = temp_dir / "lung_trachea_bronchia.nii.gz"

        if not bronchia_path.exists():
            # 列出实际输出的文件帮助调试
            output_files = list(temp_dir.glob("*.nii.gz"))
            logger.warning(f"[TotalSegmentator] 未找到 lung_trachea_bronchia.nii.gz")
            logger.warning(f"[TotalSegmentator] 实际输出文件: {[f.name for f in output_files]}")

            # 尝试其他可能的文件名
            for alt_name in ["trachea.nii.gz", "bronchi.nii.gz", "airways.nii.gz"]:
                alt_path = temp_dir / alt_name
                if alt_path.exists():
                    bronchia_path = alt_path
                    logger.info(f"[TotalSegmentator] 使用替代文件: {alt_name}")
                    break
            else:
                return None, None

        # 加载结果
        logger.info(f"[TotalSegmentator] 加载输出文件: {bronchia_path.name}")
        nii = nib.load(str(bronchia_path))
        trachea_mask = np.asanyarray(nii.dataobj) > 0
        trachea_mask = trachea_mask.astype(np.uint8)
        affine = nii.affine

        # 统计信息
        voxel_count = np.sum(trachea_mask)
        voxel_dims = np.abs(np.diag(affine)[:3])
        voxel_volume = float(np.prod(voxel_dims))
        volume_ml = voxel_count * voxel_volume / 1000

        logger.info(f"[TotalSegmentator] 气管树体素数: {voxel_count:,}, 体积: {volume_ml:.1f} mL")

        # 保存结果
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_nifti(trachea_mask, output_path, affine=affine, dtype='uint8')
            logger.info(f"[TotalSegmentator] 气管树 mask 已保存: {output_path}")

        elapsed = time.time() - start_time
        logger.info(f"[TotalSegmentator] 分割完成，耗时: {elapsed:.1f}s")

        return trachea_mask, affine

    except subprocess.TimeoutExpired:
        logger.error("[TotalSegmentator] 执行超时（>30分钟）")
        return None, None
    except FileNotFoundError:
        logger.error("[TotalSegmentator] 未找到 TotalSegmentator 命令")
        logger.error("[TotalSegmentator] 请安装: pip install TotalSegmentator")
        return None, None
    except Exception as e:
        logger.error(f"[TotalSegmentator] 分割失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None, None
    finally:
        # 清理临时文件
        if temp_dir.exists():
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
# 新版批量处理函数（LungMask + TotalSegmentator lung_vessels）
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
    使用 LungMask + TotalSegmentator 批量分割（推荐方案）

    分割方案：
    - 肺叶分割：LungMask LTRCLobes_R231（边界清晰，支持病理肺）
    - 气管树分割：TotalSegmentator --task lung_vessels（完整支气管树 3-4 级分支）

    关于 TotalSegmentator 任务选择：
    - --task total（默认）：仅输出 trachea.nii.gz（主气管）
    - --task lung_vessels：输出 lung_trachea_bronchia.nii.gz（完整支气管树）

    Args:
        input_dir: 输入目录
        mask_output_dir: mask 输出目录
        clean_output_dir: 清洗后 CT 输出目录
        force_cpu: 是否强制使用 CPU
        skip_existing: 是否跳过已处理的文件
        limit: 限制处理数量 (用于测试)
        background_hu: 背景 HU 值
        extract_trachea: 是否提取气管树 mask (使用 TotalSegmentator lung_vessels)
        create_labeled_lobes: 是否创建带标签的肺叶 mask (使用 LungMask)
        use_fusion: 是否使用 LungMask 融合模型 (LTRCLobes_R231)

    Returns:
        results: 处理结果字典 {"success": [], "failed": [], "skipped": []}

    Output files:
        - {stem}_mask.nii.gz: 二值肺部 mask
        - {stem}_clean.nii.gz: 清洗后的 CT
        - {stem}_trachea_mask.nii.gz: 气管树 mask (TotalSegmentator lung_vessels)
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
    logger.info("批量分割配置 (LungMask + TotalSegmentator lung_vessels)")
    logger.info("=" * 60)
    logger.info(f"  输入目录: {input_dir}")
    logger.info(f"  文件数量: {len(nifti_files)}")
    logger.info(f"  肺叶分割: {'启用 (LungMask)' if create_labeled_lobes else '禁用'}")
    logger.info(f"  气管树分割: {'启用 (TotalSegmentator lung_vessels)' if extract_trachea else '禁用'}")
    logger.info(f"  融合模型: {'LTRCLobes_R231' if use_fusion else 'LTRCLobes'}")
    logger.info(f"  设备: {'CPU' if force_cpu else 'GPU (如可用)'}")
    logger.info("=" * 60)

    # ===== 预检查：验证模型文件完整性 =====
    # 在开始批量处理前检查，避免所有样本都失败
    if create_labeled_lobes:
        logger.info("")
        logger.info("正在验证 LungMask 模型文件...")
        is_valid, msg = verify_lungmask_models(auto_fix=True)
        if not is_valid:
            logger.error(msg)
            logger.error("")
            logger.error("=" * 60)
            logger.error("模型文件校验失败！请执行以下步骤修复：")
            logger.error("=" * 60)
            logger.error("")
            logger.error("方案 1：清除缓存后重新运行")
            try:
                cache_dir = get_torch_cache_dir()
                logger.error(f"  rm -rf {cache_dir}/unet_*.pth")
            except Exception:
                logger.error("  rm -rf ~/.cache/torch/hub/checkpoints/unet_*.pth")
            logger.error("  python run_phase2_pipeline.py --step1-only --force")
            logger.error("")
            logger.error("方案 2：手动下载模型文件（如果网络慢）")
            for model_name, info in LUNGMASK_MODELS.items():
                logger.error(f"  wget -c {info['url']}")
            logger.error("")
            return {"success": [], "failed": [f.name for f in nifti_files], "skipped": []}
        else:
            logger.info("  ✅ 模型文件校验通过")
        logger.info("")

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
            logger.info(f"[{i}/{len(nifti_files)}] {stem} - 跳过（已存在）")
            results["skipped"].append(stem)
            continue

        # 精简日志：显示当前进度和文件名
        logger.info(f"[{i}/{len(nifti_files)}] {stem}")
        start_time = time.time()

        try:
            # ===== 步骤 1: 使用 LungMask 进行肺叶分割 =====
            if create_labeled_lobes:
                labeled_mask, _, affine = segment_lung_lobes_lungmask(
                    input_path=nifti_path,
                    output_path=lobes_path,
                    use_fusion=use_fusion,
                    force_cpu=force_cpu
                )

                # 从肺叶标签生成二值 mask
                binary_mask = (labeled_mask > 0).astype(np.uint8)
                logger.info(f"  ├─ [1/3] 肺叶分割 (LungMask) ✅")
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
                logger.info(f"  ├─ [1/3] 二值分割 (LungMask R231) ✅")

            # 保存二值 mask
            save_nifti(binary_mask, mask_path, affine=affine, dtype='uint8')

            # ===== 步骤 2: 使用 TotalSegmentator lung_vessels 进行气管树分割 =====
            trachea_mask = None
            if extract_trachea:
                device_str = "cpu" if force_cpu else "gpu"
                trachea_mask, _ = segment_airway_totalsegmentator(
                    input_path=nifti_path,
                    output_path=trachea_path,
                    device=device_str
                )
                if trachea_mask is not None:
                    logger.info(f"  ├─ [2/3] 气管树分割 (TotalSegmentator lung_vessels) ✅")
                else:
                    logger.warning(f"  ├─ [2/3] 气管树分割 ⚠️ 跳过")
            else:
                logger.info(f"  ├─ [2/3] 气管树分割 - 已禁用")

            # ===== 步骤 3: 创建清洗后的 CT =====
            ct_data, ct_affine = load_nifti(nifti_path, return_affine=True)
            ct_clean = ct_data.copy()

            # 构建保留区域 mask：肺叶 + 气管树
            keep_mask = binary_mask.copy()
            if trachea_mask is not None:
                keep_mask = keep_mask | (trachea_mask > 0)

            ct_clean[keep_mask == 0] = background_hu
            save_nifti(ct_clean, clean_path, affine=ct_affine)

            # 统计信息
            lung_ratio = np.sum(binary_mask) / binary_mask.size * 100
            elapsed = time.time() - start_time
            logger.info(f"  └─ [3/3] 清洗 CT ✅ - 肺占比: {lung_ratio:.1f}%, 耗时: {elapsed:.1f}s")
            results["success"].append(stem)

        except Exception as e:
            logger.error(f"  └─ ❌ 失败: {e}")
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
# TotalSegmentator 肺叶分割批量处理函数（可选方案）
# =============================================================================

def run_totalsegmentator_lobes_batch(
    input_dir: Union[str, Path],
    mask_output_dir: Union[str, Path],
    clean_output_dir: Union[str, Path],
    device: str = "gpu",
    skip_existing: bool = True,
    limit: Optional[int] = None,
    background_hu: float = -1000,
    extract_trachea: bool = True
) -> Dict[str, List]:
    """
    使用 TotalSegmentator 批量进行肺叶分割

    分割方案：
    - 肺叶分割：TotalSegmentator --task total（默认任务）
    - 气管树分割：TotalSegmentator --task lung_vessels（如启用）

    Args:
        input_dir: 输入目录
        mask_output_dir: mask 输出目录
        clean_output_dir: 清洗后 CT 输出目录
        device: 设备 ("gpu" 或 "cpu")
        skip_existing: 是否跳过已处理的文件
        limit: 限制处理数量
        background_hu: 背景 HU 值
        extract_trachea: 是否提取气管树 mask

    Returns:
        results: 处理结果字典 {"success": [], "failed": [], "skipped": []}

    Output files:
        - {stem}_mask.nii.gz: 二值肺部 mask
        - {stem}_clean.nii.gz: 清洗后的 CT
        - {stem}_trachea_mask.nii.gz: 气管树 mask（如启用）
        - {stem}_lung_lobes_labeled.nii.gz: 带标签的肺叶 mask (1-5)
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

    total_files = len(nifti_files)

    logger.info("=" * 60)
    logger.info("批量分割配置 (TotalSegmentator 肺叶分割)")
    logger.info("=" * 60)
    logger.info(f"  输入目录: {input_dir}")
    logger.info(f"  文件数量: {total_files}")
    logger.info(f"  [肺叶分割] TotalSegmentator --task total")
    logger.info(f"  [气管树分割] {'TotalSegmentator --task lung_vessels' if extract_trachea else '禁用'}")
    logger.info(f"  设备: {device.upper()}")
    logger.info("=" * 60)

    results = {"success": [], "failed": [], "skipped": []}

    for i, nifti_path in enumerate(nifti_files, 1):
        stem = nifti_path.name.replace('.nii.gz', '').replace('.nii', '')
        mask_path = mask_output_dir / f"{stem}_mask.nii.gz"
        clean_path = clean_output_dir / f"{stem}_clean.nii.gz"
        trachea_path = mask_output_dir / f"{stem}_trachea_mask.nii.gz"
        lobes_path = mask_output_dir / f"{stem}_lung_lobes_labeled.nii.gz"

        # 检查是否已存在
        all_exist = mask_path.exists() and clean_path.exists() and lobes_path.exists()
        if extract_trachea:
            all_exist = all_exist and trachea_path.exists()

        if skip_existing and all_exist:
            logger.info(f"[{i}/{total_files}] {stem} - 跳过（已存在）")
            results["skipped"].append(stem)
            continue

        logger.info(f"[{i}/{total_files}] {stem}")
        start_time = time.time()

        try:
            # ===== 步骤 1: 使用 TotalSegmentator 进行肺叶分割 =====
            labeled_mask, volume_stats, affine = segment_lung_lobes_totalsegmentator(
                input_path=nifti_path,
                output_path=lobes_path,
                device=device
            )

            if labeled_mask is None:
                raise RuntimeError("TotalSegmentator 肺叶分割失败")

            # 创建二值 mask
            binary_mask = (labeled_mask > 0).astype(np.uint8)
            save_nifti(binary_mask, mask_path, affine=affine, dtype='uint8')
            logger.info(f"  ├─ [1/3] 肺叶分割 ✅ - 二值 mask 已保存")

            # ===== 步骤 2: 气管树分割（如启用）=====
            trachea_mask = None
            if extract_trachea:
                trachea_mask, _ = segment_airway_totalsegmentator(
                    input_path=nifti_path,
                    output_path=trachea_path,
                    device=device
                )
                if trachea_mask is not None:
                    logger.info(f"  ├─ [2/3] 气管树分割 ✅")
                else:
                    logger.warning(f"  ├─ [2/3] 气管树分割 ⚠️ 跳过")
            else:
                logger.info(f"  ├─ [2/3] 气管树分割 - 已禁用")

            # ===== 步骤 3: 创建清洗后的 CT =====
            ct_data, ct_affine = load_nifti(nifti_path, return_affine=True)
            ct_clean = ct_data.copy()

            # 构建保留区域 mask：肺叶 + 气管树
            keep_mask = binary_mask.copy()
            if trachea_mask is not None:
                keep_mask = keep_mask | (trachea_mask > 0)

            ct_clean[keep_mask == 0] = background_hu
            save_nifti(ct_clean, clean_path, affine=ct_affine)

            elapsed = time.time() - start_time
            logger.info(f"  └─ [3/3] 清洗 CT ✅ - 耗时: {elapsed:.1f}s")
            results["success"].append(stem)

        except Exception as e:
            logger.error(f"  └─ ❌ 失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            results["failed"].append((stem, str(e)))

    # 输出汇总
    logger.info("=" * 60)
    logger.info("批量分割完成汇总 (TotalSegmentator):")
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

