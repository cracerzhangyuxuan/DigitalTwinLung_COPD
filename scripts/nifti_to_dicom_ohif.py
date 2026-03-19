#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NIfTI 转 DICOM 导出脚本 (OHIF Viewer 兼容版)

功能：将数字孪生肺项目的 NIfTI 文件转换为 DICOM 序列，
      确保 OHIF Viewer 能够实现 1x3 网格布局下的空间同步滚动。

核心约束：
- Patient 层级：三个序列共享相同的 PatientID 和 PatientName
- Study 层级：三个序列共享相同的 StudyInstanceUID
- Series 层级：三个序列拥有不同的 SeriesInstanceUID 和 SeriesDescription
- 空间信息：从 NIfTI 继承准确的 Spacing, Origin, Direction

依赖：
    pip install SimpleITK pydicom numpy

作者：Digital Twin Lung Project
日期：2026-01-21
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import uuid

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    print("错误: 请安装 SimpleITK: pip install SimpleITK")
    sys.exit(1)

try:
    import pydicom
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
except ImportError:
    print("错误: 请安装 pydicom: pip install pydicom")
    sys.exit(1)


# =============================================================================
# DICOM UID 生成
# =============================================================================

def generate_study_uid() -> str:
    """生成全局唯一的 StudyInstanceUID"""
    return generate_uid()


def generate_series_uid() -> str:
    """生成全局唯一的 SeriesInstanceUID"""
    return generate_uid()


def generate_sop_uid() -> str:
    """生成全局唯一的 SOPInstanceUID"""
    return generate_uid()


# =============================================================================
# NIfTI 加载与空间信息提取
# =============================================================================

def load_nifti_with_metadata(nifti_path: str) -> tuple:
    """
    加载 NIfTI 文件并提取空间元数据
    
    Args:
        nifti_path: NIfTI 文件路径
        
    Returns:
        tuple: (image_array, spacing, origin, direction, sitk_image)
    """
    print(f"  加载: {nifti_path}")
    
    sitk_image = sitk.ReadImage(str(nifti_path))
    
    # 提取空间信息
    spacing = sitk_image.GetSpacing()      # (x, y, z) in mm
    origin = sitk_image.GetOrigin()        # (x, y, z) in mm
    direction = sitk_image.GetDirection()  # 9 元素的方向余弦矩阵
    
    # 转换为 numpy 数组 (注意: SimpleITK 使用 (x, y, z) 顺序)
    image_array = sitk.GetArrayFromImage(sitk_image)  # 返回 (z, y, x) 顺序
    
    print(f"    形状: {image_array.shape}")
    print(f"    间距: {spacing} mm")
    print(f"    原点: {origin} mm")
    print(f"    数据范围: [{image_array.min():.1f}, {image_array.max():.1f}]")
    
    return image_array, spacing, origin, direction, sitk_image


# =============================================================================
# DICOM 切片生成
# =============================================================================

def create_dicom_slice(
    pixel_data: np.ndarray,
    slice_index: int,
    total_slices: int,
    spacing: tuple,
    origin: tuple,
    direction: tuple,
    patient_id: str,
    patient_name: str,
    study_uid: str,
    series_uid: str,
    series_description: str,
    series_number: int,
    study_date: str,
    study_time: str,
    is_mask: bool = False
) -> FileDataset:
    """
    创建单个 DICOM 切片
    
    Args:
        pixel_data: 2D 切片数据 (y, x)
        slice_index: 切片索引 (0-based)
        total_slices: 总切片数
        spacing: (x_spacing, y_spacing, z_spacing) in mm
        origin: (x_origin, y_origin, z_origin) in mm
        direction: 方向余弦矩阵 (9 元素)
        patient_id: 患者 ID
        patient_name: 患者姓名
        study_uid: StudyInstanceUID
        series_uid: SeriesInstanceUID
        series_description: 序列描述
        series_number: 序列编号
        study_date: 检查日期
        study_time: 检查时间
        is_mask: 是否为掩膜图像
        
    Returns:
        FileDataset: DICOM 数据集
    """
    # 生成唯一的 SOP Instance UID
    sop_uid = generate_sop_uid()

    # 创建文件元信息
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    # 创建 DICOM 数据集
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    # ========== 【新增这两行，解决 Orthanc 拒收的关键】 ==========
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    # =========================================================

    # =========================================================================
    # Patient 层级 (所有序列必须相同)

    # =========================================================================
    # Patient 层级 (所有序列必须相同)
    # =========================================================================
    ds.PatientID = patient_id
    ds.PatientName = patient_name
    ds.PatientBirthDate = ''
    ds.PatientSex = ''

    # =========================================================================
    # Study 层级 (所有序列必须相同)
    # =========================================================================
    ds.StudyInstanceUID = study_uid
    ds.StudyDate = study_date
    ds.StudyTime = study_time
    ds.StudyDescription = 'Digital Twin Lung COPD Study'
    ds.AccessionNumber = ''
    ds.ReferringPhysicianName = ''
    ds.StudyID = '1'

    # =========================================================================
    # Series 层级 (每个序列不同)
    # =========================================================================
    ds.SeriesInstanceUID = series_uid
    ds.SeriesDescription = series_description
    ds.SeriesNumber = series_number
    ds.Modality = 'CT'
    ds.BodyPartExamined = 'CHEST'
    ds.PatientPosition = 'HFS'  # Head First Supine

    # =========================================================================
    # Instance 层级 (每个切片不同)
    # =========================================================================
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    ds.SOPInstanceUID = sop_uid
    ds.InstanceNumber = slice_index + 1  # 1-based

    # =========================================================================
    # 空间信息 (关键：确保 OHIF 同步滚动)
    # =========================================================================

    # 像素间距 (行间距, 列间距) - 注意顺序
    ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]  # [row, col] = [y, x]

    # 切片厚度
    ds.SliceThickness = float(spacing[2])
    ds.SpacingBetweenSlices = float(spacing[2])

    # 计算当前切片的 Image Position Patient (IPP)
    # NIfTI 的 origin 是第一个体素的中心位置
    # 方向矩阵: [Xx, Xy, Xz, Yx, Yy, Yz, Zx, Zy, Zz]
    dir_matrix = np.array(direction).reshape(3, 3)

    # 计算当前切片在物理空间中的位置
    slice_offset = dir_matrix[:, 2] * spacing[2] * slice_index
    slice_position = np.array(origin) + slice_offset

    ds.ImagePositionPatient = [float(slice_position[0]),
                               float(slice_position[1]),
                               float(slice_position[2])]

    # Image Orientation Patient (IOP): 行方向和列方向的方向余弦
    # 行方向 = X 轴方向, 列方向 = Y 轴方向
    ds.ImageOrientationPatient = [
        float(dir_matrix[0, 0]), float(dir_matrix[1, 0]), float(dir_matrix[2, 0]),  # 行方向 (X)
        float(dir_matrix[0, 1]), float(dir_matrix[1, 1]), float(dir_matrix[2, 1])   # 列方向 (Y)
    ]

    # Slice Location (用于排序)
    ds.SliceLocation = float(slice_position[2])

    # =========================================================================
    # 图像数据
    # =========================================================================
    ds.Rows = pixel_data.shape[0]
    ds.Columns = pixel_data.shape[1]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # 有符号整数 (CT 值可能为负)

    # 处理像素数据
    if is_mask:
        # 掩膜：将 0/1 映射到 0/1000 以便在 OHIF 中可视化
        pixel_array = (pixel_data.astype(np.float32) * 1000).astype(np.int16)
        ds.WindowCenter = 500
        ds.WindowWidth = 1000
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1
    else:
        # CT 图像：保持原始 HU 值
        pixel_array = pixel_data.astype(np.int16)
        ds.WindowCenter = -600  # 肺窗
        ds.WindowWidth = 1500
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1

    ds.PixelData = pixel_array.tobytes()

    # =========================================================================
    # 其他必要标签
    # =========================================================================
    ds.ImageType = ['DERIVED', 'SECONDARY']
    ds.ContentDate = study_date
    ds.ContentTime = study_time
    ds.AcquisitionDate = study_date
    ds.AcquisitionTime = study_time
    ds.Manufacturer = 'Digital Twin Lung Project'
    ds.InstitutionName = 'COPD Research Lab'
    ds.KVP = ''
    ds.FrameOfReferenceUID = study_uid  # 使用 StudyUID 作为 Frame of Reference
    ds.PositionReferenceIndicator = ''

    return ds


# =============================================================================
# NIfTI 转 DICOM 序列
# =============================================================================

def convert_nifti_to_dicom_series(
    nifti_path: str,
    output_dir: str,
    patient_id: str,
    patient_name: str,
    study_uid: str,
    series_uid: str,
    series_description: str,
    series_number: int,
    study_date: str,
    study_time: str,
    is_mask: bool = False
) -> int:
    """
    将单个 NIfTI 文件转换为 DICOM 序列

    Args:
        nifti_path: NIfTI 文件路径
        output_dir: 输出目录
        patient_id: 患者 ID
        patient_name: 患者姓名
        study_uid: StudyInstanceUID (所有序列共享)
        series_uid: SeriesInstanceUID (每个序列唯一)
        series_description: 序列描述
        series_number: 序列编号
        study_date: 检查日期
        study_time: 检查时间
        is_mask: 是否为掩膜图像

    Returns:
        int: 生成的切片数量
    """
    # 加载 NIfTI
    image_array, spacing, origin, direction, _ = load_nifti_with_metadata(nifti_path)

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取切片数量 (z 轴)
    num_slices = image_array.shape[0]

    print(f"  转换 {num_slices} 个切片到: {output_dir}")

    # 逐切片转换
    for slice_idx in range(num_slices):
        # 提取当前切片 (z, y, x) -> (y, x)
        slice_data = image_array[slice_idx, :, :]

        # 创建 DICOM 切片
        ds = create_dicom_slice(
            pixel_data=slice_data,
            slice_index=slice_idx,
            total_slices=num_slices,
            spacing=spacing,
            origin=origin,
            direction=direction,
            patient_id=patient_id,
            patient_name=patient_name,
            study_uid=study_uid,
            series_uid=series_uid,
            series_description=series_description,
            series_number=series_number,
            study_date=study_date,
            study_time=study_time,
            is_mask=is_mask
        )

        # 保存 DICOM 文件
        dcm_filename = f"slice_{slice_idx + 1:04d}.dcm"
        dcm_path = output_path / dcm_filename
        ds.save_as(str(dcm_path), write_like_original=False)

    print(f"  ✓ 完成: {num_slices} 个 DICOM 文件")
    return num_slices


# =============================================================================
# 主转换函数
# =============================================================================

def convert_digital_twin_to_dicom(
    healthy_baseline_path: str,
    copd_generated_path: str,
    lesion_mask_path: str,
    output_base_dir: str,
    patient_id: str = "DigitalTwin_001"
) -> dict:
    """
    将数字孪生肺的三个 NIfTI 文件转换为 DICOM 序列

    确保 OHIF Viewer 能够实现 1x3 网格布局下的空间同步滚动

    Args:
        healtine_path: 健康基线 CT 路径
        copd_generated_path: AI 生成的病变 CT 路径
        lesion_mask_path: 病灶掩膜路径
        output_base_dir: 输出基础目录
        patient_id: 患者 ID

    Returns:
        dict: 转换结果统计
    """
    print("=" * 60)
    print("  NIfTI 转 DICOM (OHIF Viewer 兼容版)")
    print("=" * 60)

    # 生成共享的 StudyInstanceUID (关键：确保 OHIF 识别为同一检查)
    study_uid = generate_study_uid()
    print(f"\n共享 StudyInstanceUID: {study_uid}")

    # 生成各自的 SeriesInstanceUID
    series_uid_baseline = generate_series_uid()
    series_uid_copd = generate_series_uid()
    series_uid_mask = generate_series_uid()

    # 当前日期时间
    now = datetime.now()
    study_date = now.strftime("%Y%m%d")
    study_time = now.strftime("%H%M%S")

    # 患者信息
    patient_name = patient_id.replace("_", "^")

    results = {}

    # =========================================================================
    # 1. 转换健康基线 CT
    # =========================================================================
    print(f"\n[1/3] 转换健康基线 CT...")
    output_dir_baseline = Path(output_base_dir) / "Series_01_Healthy_Baseline"
    results['healthy_baseline'] = convert_nifti_to_dicom_series(
        nifti_path=healthy_baseline_path,
        output_dir=str(output_dir_baseline),
        patient_id=patient_id,
        patient_name=patient_name,
        study_uid=study_uid,
        series_uid=series_uid_baseline,
        series_description="Healthy_Baseline",
        series_number=1,
        study_date=study_date,
        study_time=study_time,
        is_mask=False
    )

    # =========================================================================
    # 2. 转换 AI 生成的病变 CT
    # =========================================================================
    print(f"\n[2/3] 转换 AI 生成的病变 CT...")
    output_dir_copd = Path(output_base_dir) / "Series_02_COPD_Generated"
    results['copd_generated'] = convert_nifti_to_dicom_series(
        nifti_path=copd_generated_path,
        output_dir=str(output_dir_copd),
        patient_id=patient_id,
        patient_name=patient_name,
        study_uid=study_uid,
        series_uid=series_uid_copd,
        series_description="COPD_Generated",
        series_number=2,
        study_date=study_date,
        study_time=study_time,
        is_mask=False
    )

    # =========================================================================
    # 3. 转换病灶掩膜
    # =========================================================================
    print(f"\n[3/3] 转换病灶掩膜...")
    output_dir_mask = Path(output_base_dir) / "Series_03_Lesion_Mask"
    results['lesion_mask'] = convert_nifti_to_dicom_series(
        nifti_path=lesion_mask_path,
        output_dir=str(output_dir_mask),
        patient_id=patient_id,
        patient_name=patient_name,
        study_uid=study_uid,
        series_uid=series_uid_mask,
        series_description="Lesion_Mask",
        series_number=3,
        study_date=study_date,
        study_time=study_time,
        is_mask=True
    )

    # =========================================================================
    # 输出汇总
    # =========================================================================
    print("\n" + "=" * 60)
    print("  转换完成!")
    print("=" * 60)
    print(f"\n输出目录: {output_base_dir}")
    print(f"  - Series_01_Healthy_Baseline: {results['healthy_baseline']} 切片")
    print(f"  - Series_02_COPD_Generated: {results['copd_generated']} 切片")
    print(f"  - Series_03_Lesion_Mask: {results['lesion_mask']} 切片")
    print(f"\n共享 StudyInstanceUID: {study_uid}")
    print(f"PatientID: {patient_id}")
    print("\n提示: 将输出目录上传到 OHIF Viewer 即可实现 1x3 同步显示")

    return results


# =============================================================================
# 命令行入口
# =============================================================================

def find_default_files(patient_id: str = "copd_001") -> dict:
    """
    自动查找默认的输入文件

    Args:
        patient_id: 患者 ID

    Returns:
        dict: 文件路径字典
    """
    base_dir = Path(__file__).parent.parent

    # 健康基线: 标准模板
    healthy_baseline = base_dir / "data" / "02_atlas" / "standard_template.nii.gz"

    # AI 生成的病变 CT: patchgan 模型输出
    copd_generated = base_dir / "data" / "04_final_viz" / "patchgan" / f"{patient_id}_fused.nii.gz"

    # 如果 patchgan 不存在，尝试 partial_conv
    if not copd_generated.exists():
        copd_generated = base_dir / "data" / "04_final_viz" / "partial_conv" / f"{patient_id}_fused.nii.gz"

    # 病灶掩膜: 配准后的病灶 mask
    lesion_mask = base_dir / "data" / "03_mapped" / patient_id / f"{patient_id}_warped_lesion.nii.gz"

    return {
        'healthy_baseline': str(healthy_baseline),
        'copd_generated': str(copd_generated),
        'lesion_mask': str(lesion_mask)
    }


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(
        description='NIfTI 转 DICOM (OHIF Viewer 兼容版)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认文件 (copd_001)
  python nifti_to_dicom_ohif.py

  # 指定患者 ID
  python nifti_to_dicom_ohif.py --patient copd_002

  # 指定所有输入文件
  python nifti_to_dicom_ohif.py \\
      --healthy data/02_atlas/standard_template.nii.gz \\
      --copd data/04_final_viz/patchgan/copd_001_fused.nii.gz \\
      --mask data/03_mapped/copd_001/copd_001_warped_lesion.nii.gz \\
      --output dicom_output
        """
    )

    parser.add_argument('--patient', type=str, default='copd_001',
                        help='患者 ID (默认: copd_001)')
    parser.add_argument('--healthy', type=str, default=None,
                        help='健康基线 CT 路径')
    parser.add_argument('--copd', type=str, default=None,
                        help='AI 生成的病变 CT 路径')
    parser.add_argument('--mask', type=str, default=None,
                        help='病灶掩膜路径')
    parser.add_argument('--output', type=str, default='dicom_output',
                        help='输出目录 (默认: dicom_output)')
    parser.add_argument('--patient-id', type=str, default=None,
                        help='DICOM PatientID (默认: DigitalTwin_{patient})')

    args = parser.parse_args()

    # 查找默认文件
    default_files = find_default_files(args.patient)

    # 使用命令行参数或默认值
    healthy_path = args.healthy or default_files['healthy_baseline']
    copd_path = args.copd or default_files['copd_generated']
    mask_path = args.mask or default_files['lesion_mask']

    # 检查文件是否存在
    for name, path in [('健康基线', healthy_path), ('病变CT', copd_path), ('病灶掩膜', mask_path)]:
        if not Path(path).exists():
            print(f"错误: {name}文件不存在: {path}")
            sys.exit(1)

    # 设置 PatientID
    patient_id = args.patient_id or f"DigitalTwin_{args.patient.replace('copd_', '')}"

    # 执行转换
    convert_digital_twin_to_dicom(
        healthy_baseline_path=healthy_path,
        copd_generated_path=copd_path,
        lesion_mask_path=mask_path,
        output_base_dir=args.output,
        patient_id=patient_id
    )


if __name__ == "__main__":
    main()

