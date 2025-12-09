#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2 数据准备脚本

功能：
1. 清理 Phase 1 临时数据（子任务 5.1）
2. 从外部数据源复制和转换数据（子任务 5.2）

数据源：
- AE_37: 37 例正常肺（对照组），DICOM 格式 → data/00_raw/normal/
- AECOPD_29: 29 例 COPD 患者，DICOM 格式 → data/00_raw/copd/

使用方法：
    # 查看帮助
    python prepare_phase2_data.py --help
    
    # 仅清理 Phase 1 数据
    python prepare_phase2_data.py --clean-only
    
    # 仅准备新数据（不清理）
    python prepare_phase2_data.py --prepare-only
    
    # 执行完整流程（清理 + 准备）
    python prepare_phase2_data.py --full
    
    # 干运行（仅显示将执行的操作）
    python prepare_phase2_data.py --dry-run --full

作者: DigitalTwinLung_COPD Team
日期: 2025-12-09
"""

import argparse
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据源路径
SOURCE_NORMAL = Path(r"D:\lung-data\Feature_Embedding\COPD\2_aecopd_biphasic_ct_images\AE_37\origin")
SOURCE_COPD = Path(r"D:\lung-data\Feature_Embedding\COPD\2_aecopd_biphasic_ct_images\AECOPD_29\origin")

# 目标路径
DATA_ROOT = PROJECT_ROOT / "data"
RAW_NORMAL = DATA_ROOT / "00_raw" / "normal"
RAW_COPD = DATA_ROOT / "00_raw" / "copd"


def setup_logging():
    """设置日志"""
    import logging
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(
                log_dir / f"data_prepare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                encoding='utf-8'
            ),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def clean_phase1_data(dry_run: bool = False) -> Tuple[int, int]:
    """
    清理 Phase 1 临时数据
    
    保留目录结构，删除所有数据文件
    
    Args:
        dry_run: 是否仅显示将执行的操作
        
    Returns:
        (deleted_count, total_size_mb): 删除的文件数和总大小
    """
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("子任务 5.1: 清理 Phase 1 临时数据")
    logger.info("=" * 60)
    
    # 要清理的目录
    # 注意: Phase 2 优化后已移除 *_nifti/ 中间目录
    # 00_raw/ 现在直接存储 NIfTI 格式的 CT 数据
    clean_targets = [
        DATA_ROOT / "00_raw" / "normal",
        DATA_ROOT / "00_raw" / "copd",
        DATA_ROOT / "01_cleaned" / "normal_clean",
        DATA_ROOT / "01_cleaned" / "normal_mask",
        DATA_ROOT / "01_cleaned" / "copd_clean",
        DATA_ROOT / "01_cleaned" / "copd_mask",
        DATA_ROOT / "01_cleaned" / "copd_emphysema",
        DATA_ROOT / "02_atlas",
        DATA_ROOT / "03_mapped",
        DATA_ROOT / "04_final_viz",
        DATA_ROOT / "04_final_viz" / "renders",
    ]
    
    deleted_count = 0
    total_size = 0
    
    for target_dir in clean_targets:
        if not target_dir.exists():
            continue
            
        # 查找所有文件
        files = list(target_dir.glob("*"))
        for f in files:
            if f.is_file():
                size = f.stat().st_size
                if dry_run:
                    logger.info(f"  [DRY-RUN] 将删除: {f} ({size / 1024 / 1024:.2f} MB)")
                else:
                    try:
                        f.unlink()
                        logger.info(f"  已删除: {f.name}")
                        deleted_count += 1
                        total_size += size
                    except Exception as e:
                        logger.error(f"  删除失败: {f} - {e}")
            elif f.is_dir() and f.name not in ['renders', 'normal', 'copd']:
                # 递归删除子目录（如 patient_001, copd_001）
                if dry_run:
                    logger.info(f"  [DRY-RUN] 将删除目录: {f}")
                else:
                    try:
                        shutil.rmtree(f)
                        logger.info(f"  已删除目录: {f.name}")
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"  删除目录失败: {f} - {e}")
    
    total_size_mb = total_size / 1024 / 1024
    
    logger.info("")
    if dry_run:
        logger.info(f"[DRY-RUN] 将删除 {deleted_count} 个项目")
    else:
        logger.info(f"清理完成: 删除 {deleted_count} 个项目, 释放 {total_size_mb:.2f} MB")
    
    return deleted_count, total_size_mb


def find_phase_directory(
    patient_dir: Path,
    use_inspiration: bool = True
) -> Tuple[Optional[Path], str]:
    """
    智能识别呼吸相位目录

    支持的命名模式：
    - AE_37 标准: ThorRoutine_1_Inspiration, ThorRoutine_2_Expiration
    - AE_37 反转: ThorRoutine_1_Expiration, ThorRoutine_2_Inspiration
    - AE_37 变体: xxxx_Inspiration, xxxx_Expiration
    - AECOPD_29 标准: THORROUTINE_1_1_0_I31S_3_0002_I (I=吸气), THORROUTINE_2_1_0_I31S_3_0003_E (E=呼气)
    - AECOPD_29 简写: 2_I (I=吸气), 3_E (E=呼气)
    - AECOPD_29 变体: 2_I31_I, 3_I50_I_, 7_I31_E, 8_I50_E_

    Args:
        patient_dir: 患者目录
        use_inspiration: 是否选择吸气相

    Returns:
        (phase_dir, phase_type): 相位目录和识别到的类型
    """
    subdirs = [d for d in patient_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None, "no_subdirs"

    # 目标相位标记
    if use_inspiration:
        target_markers = ['inspiration', '_i', '_i_']  # 吸气相标记
        avoid_markers = ['expiration', '_e', '_e_']     # 呼气相标记
    else:
        target_markers = ['expiration', '_e', '_e_']    # 呼气相标记
        avoid_markers = ['inspiration', '_i', '_i_']    # 吸气相标记

    # 定义识别规则（按优先级排序）
    patterns = [
        # 规则1: 标准 AE_37 命名 (ThorRoutine_1_Inspiration)
        lambda d, insp: 'inspiration' in d.name.lower() if insp else 'expiration' in d.name.lower(),
        # 规则2: AECOPD_29 标准命名 (末尾 _I 或 _E)
        lambda d, insp: d.name.upper().endswith('_I') if insp else d.name.upper().endswith('_E'),
        # 规则3: AECOPD_29 变体 (包含 _I_ 或 _E_)
        lambda d, insp: '_I_' in d.name.upper() or d.name.upper().endswith('_I') if insp else '_E_' in d.name.upper() or d.name.upper().endswith('_E'),
    ]

    # 尝试每个规则
    for pattern in patterns:
        matches = [d for d in subdirs if pattern(d, use_inspiration)]
        if matches:
            # 优先选择包含 "1" 或 "2" 的目录（通常表示序列号）
            for m in matches:
                if '1' in m.name or '2' in m.name:
                    return m, f"matched_{m.name}"
            return matches[0], f"matched_{matches[0].name}"

    # 规则4: 如果只有两个目录，按照命名排序选择
    if len(subdirs) == 2:
        sorted_dirs = sorted(subdirs, key=lambda d: d.name.lower())
        # 通常第一个是吸气相，第二个是呼气相
        idx = 0 if use_inspiration else 1
        return sorted_dirs[idx], f"fallback_order_{sorted_dirs[idx].name}"

    # 规则5: 默认选择第一个目录
    return subdirs[0], f"fallback_first_{subdirs[0].name}"


def convert_dicom_to_nifti(
    dicom_dir: Path,
    output_path: Path,
    use_inspiration: bool = True
) -> Tuple[bool, str]:
    """
    将 DICOM 序列转换为 NIfTI 格式

    Args:
        dicom_dir: DICOM 文件夹（患者目录）
        output_path: 输出 NIfTI 文件路径
        use_inspiration: 使用吸气相（True）或呼气相（False）

    Returns:
        (success, info): 是否成功和附加信息
    """
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as e:
        return False, f"缺少依赖: {e}"

    # 使用智能识别找到相位目录
    phase_dir, phase_info = find_phase_directory(dicom_dir, use_inspiration)
    if phase_dir is None:
        return False, "未找到相位目录"

    # 尝试导入 pydicom
    try:
        import pydicom
    except ImportError:
        try:
            import SimpleITK as sitk

            # 读取 DICOM 序列
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(str(phase_dir))
            if not dicom_files:
                return False, f"DICOM 文件为空: {phase_dir}"

            reader.SetFileNames(dicom_files)
            image = reader.Execute()

            # 保存为 NIfTI
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(image, str(output_path))
            return True, f"SimpleITK|{phase_info}|slices={image.GetSize()[2]}"

        except ImportError:
            return False, "需要安装 pydicom 或 SimpleITK"

    # 使用 pydicom + nibabel
    # 读取所有 DICOM 文件
    dcm_files = sorted(phase_dir.glob("*.dcm"))
    if not dcm_files:
        return False, f"DICOM 文件为空: {phase_dir}"

    slices = []
    for dcm_file in dcm_files:
        ds = pydicom.dcmread(str(dcm_file))
        slices.append(ds)

    # 按位置排序
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except (AttributeError, TypeError):
        # 如果没有 ImagePositionPatient，按文件名排序
        pass

    # 构建 3D 数组
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape, dtype=np.float32)

    for i, s in enumerate(slices):
        img2d = s.pixel_array.astype(np.float32)
        # 转换为 HU 值
        intercept = float(getattr(s, 'RescaleIntercept', 0))
        slope = float(getattr(s, 'RescaleSlope', 1))
        img2d = img2d * slope + intercept
        img3d[:, :, i] = img2d

    # 构建仿射矩阵
    ds = slices[0]
    pixel_spacing = [float(x) for x in getattr(ds, 'PixelSpacing', [1.0, 1.0])]
    slice_thickness = float(getattr(ds, 'SliceThickness', 1.0))

    affine = np.eye(4)
    affine[0, 0] = pixel_spacing[0]
    affine[1, 1] = pixel_spacing[1]
    affine[2, 2] = slice_thickness

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nii_img = nib.Nifti1Image(img3d, affine)
    nib.save(nii_img, str(output_path))

    return True, f"pydicom|{phase_info}|slices={len(slices)}"


def prepare_phase2_data(
    dry_run: bool = False,
    max_normal: Optional[int] = None,
    max_copd: Optional[int] = None
) -> Tuple[int, int]:
    """
    准备 Phase 2 数据
    
    从外部数据源转换 DICOM 到 NIfTI 并复制到项目目录
    
    Args:
        dry_run: 是否仅显示将执行的操作
        max_normal: 最大正常肺数量（None = 全部）
        max_copd: 最大 COPD 数量（None = 全部）
        
    Returns:
        (normal_count, copd_count): 处理的数据数量
    """
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("子任务 5.2: 准备 Phase 2 正式数据")
    logger.info("=" * 60)
    
    # 检查数据源
    if not SOURCE_NORMAL.exists():
        logger.error(f"正常肺数据源不存在: {SOURCE_NORMAL}")
        return 0, 0
    
    if not SOURCE_COPD.exists():
        logger.error(f"COPD 数据源不存在: {SOURCE_COPD}")
        return 0, 0
    
    # 获取患者目录列表
    normal_patients = sorted([d for d in SOURCE_NORMAL.iterdir() if d.is_dir()])
    copd_patients = sorted([d for d in SOURCE_COPD.iterdir() if d.is_dir()])
    
    logger.info(f"发现正常肺数据: {len(normal_patients)} 例")
    logger.info(f"发现 COPD 数据: {len(copd_patients)} 例")
    
    # 限制数量
    if max_normal:
        normal_patients = normal_patients[:max_normal]
    if max_copd:
        copd_patients = copd_patients[:max_copd]
    
    logger.info(f"将处理: {len(normal_patients)} 例正常肺, {len(copd_patients)} 例 COPD")
    
    normal_count = 0
    copd_count = 0
    
    # 处理正常肺数据
    logger.info("")
    logger.info("处理正常肺数据...")
    for i, patient_dir in enumerate(normal_patients, 1):
        output_name = f"normal_{i:03d}.nii.gz"
        output_path = RAW_NORMAL / output_name

        if dry_run:
            # 干运行模式：检测相位目录
            phase_dir, _ = find_phase_directory(patient_dir, use_inspiration=True)
            phase_name = phase_dir.name if phase_dir else "未找到"
            logger.info(f"  [DRY-RUN] {patient_dir.name} → {output_name} (相位: {phase_name})")
            normal_count += 1
        else:
            logger.info(f"  [{i}/{len(normal_patients)}] 转换: {patient_dir.name}")
            success, info = convert_dicom_to_nifti(patient_dir, output_path)
            if success:
                logger.info(f"    → 已保存: {output_name} ({info})")
                normal_count += 1
            else:
                logger.warning(f"    → 转换失败: {info}")

    # 处理 COPD 数据
    logger.info("")
    logger.info("处理 COPD 数据...")
    for i, patient_dir in enumerate(copd_patients, 1):
        output_name = f"copd_{i:03d}.nii.gz"
        output_path = RAW_COPD / output_name

        if dry_run:
            # 干运行模式：检测相位目录
            phase_dir, _ = find_phase_directory(patient_dir, use_inspiration=True)
            phase_name = phase_dir.name if phase_dir else "未找到"
            logger.info(f"  [DRY-RUN] {patient_dir.name} → {output_name} (相位: {phase_name})")
            copd_count += 1
        else:
            logger.info(f"  [{i}/{len(copd_patients)}] 转换: {patient_dir.name}")
            success, info = convert_dicom_to_nifti(patient_dir, output_path)
            if success:
                logger.info(f"    → 已保存: {output_name} ({info})")
                copd_count += 1
            else:
                logger.warning(f"    → 转换失败: {info}")
    
    logger.info("")
    logger.info(f"数据准备完成: {normal_count} 例正常肺, {copd_count} 例 COPD")
    
    return normal_count, copd_count


def verify_data_quality() -> bool:
    """验证准备的数据质量"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("数据质量验证")
    logger.info("=" * 60)
    
    try:
        from src.utils.data_quality import batch_quality_check, generate_quality_report
    except ImportError:
        logger.warning("无法导入数据质量检查模块，跳过验证")
        return True
    
    # 检查正常肺数据
    normal_files = list(RAW_NORMAL.glob("*.nii.gz"))
    copd_files = list(RAW_COPD.glob("*.nii.gz"))
    
    logger.info(f"正常肺文件: {len(normal_files)}")
    logger.info(f"COPD 文件: {len(copd_files)}")
    
    all_passed = True
    
    if normal_files:
        passed, failed, results = batch_quality_check(RAW_NORMAL)
        if failed:
            logger.warning(f"正常肺数据: {len(passed)} 通过, {len(failed)} 未通过")
            all_passed = False
        else:
            logger.info(f"正常肺数据: 全部 {len(passed)} 个通过质量检查")
    
    if copd_files:
        passed, failed, results = batch_quality_check(RAW_COPD)
        if failed:
            logger.warning(f"COPD 数据: {len(passed)} 通过, {len(failed)} 未通过")
            all_passed = False
        else:
            logger.info(f"COPD 数据: 全部 {len(passed)} 个通过质量检查")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Phase 2 数据准备脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--clean-only', action='store_true',
                        help='仅清理 Phase 1 临时数据')
    parser.add_argument('--prepare-only', action='store_true',
                        help='仅准备新数据（不清理）')
    parser.add_argument('--full', action='store_true',
                        help='执行完整流程（清理 + 准备）')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅显示将执行的操作，不实际执行')
    parser.add_argument('--max-normal', type=int, default=None,
                        help='最大正常肺数量（默认全部）')
    parser.add_argument('--max-copd', type=int, default=None,
                        help='最大 COPD 数量（默认全部）')
    parser.add_argument('--skip-verify', action='store_true',
                        help='跳过数据质量验证')
    
    args = parser.parse_args()
    
    # 默认显示帮助
    if not (args.clean_only or args.prepare_only or args.full):
        parser.print_help()
        print("\n请指定 --clean-only, --prepare-only 或 --full 选项")
        sys.exit(0)
    
    print("=" * 60)
    print("Phase 2 数据准备脚本")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模式: {'干运行' if args.dry_run else '实际执行'}")
    print()
    
    if args.clean_only or args.full:
        clean_phase1_data(dry_run=args.dry_run)
        print()
    
    if args.prepare_only or args.full:
        prepare_phase2_data(
            dry_run=args.dry_run,
            max_normal=args.max_normal,
            max_copd=args.max_copd
        )
        print()
        
        if not args.dry_run and not args.skip_verify:
            verify_data_quality()
    
    print()
    print("=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

