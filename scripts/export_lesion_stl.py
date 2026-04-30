#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
病灶区域与正常肺部 STL 3D 网格导出脚本

功能：将 PatchGAN 模型生成的 COPD 病灶区域和正常肺部区域导出为 STL 3D 网格文件
      - 病灶区域：仅提取病灶阴影/病变区域
      - 正常肺部：肺部掩膜区域减去病灶区域

核心特性：
- 使用 Marching Cubes 算法从二值掩膜生成平滑的 3D 表面网格
- 保留原始 NIfTI 文件中的物理空间位置（基于 Origin、Spacing、Direction）
- 输出的 STL 可与其他医学影像数据在同一坐标系中对齐

依赖：
    pip install numpy nibabel scikit-image trimesh

使用方法：
    # 导出单个患者（病灶+肺部）
    python export_lesion_stl.py --patient copd_001

    # 仅导出病灶区域
    python export_lesion_stl.py --patient copd_001 --lesion-only

    # 仅导出正常肺部区域
    python export_lesion_stl.py --patient copd_001 --lung-only

    # 导出所有患者
    python export_lesion_stl.py --all

作者：Digital Twin Lung Project
日期：2026-02-23
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional
import warnings

import numpy as np

# 检查依赖
try:
    import nibabel as nib
except ImportError:
    print("错误: 请安装 nibabel: pip install nibabel")
    sys.exit(1)

try:
    from skimage import measure
except ImportError:
    print("错误: 请安装 scikit-image: pip install scikit-image")
    sys.exit(1)

try:
    import trimesh
except ImportError:
    print("错误: 请安装 trimesh: pip install trimesh")
    sys.exit(1)


# =============================================================================
# NIfTI 加载与空间信息提取
# =============================================================================

def load_nifti_with_spatial_info(nifti_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 NIfTI 文件并提取空间变换信息
    
    Args:
        nifti_path: NIfTI 文件路径
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            (体素数据, 仿射变换矩阵, 体素间距)
    """
    print(f"  加载: {nifti_path}")
    
    nii = nib.load(str(nifti_path))
    data = nii.get_fdata()
    affine = nii.affine  # 4x4 仿射变换矩阵（体素坐标 -> 物理坐标）
    
    # 从仿射矩阵提取体素间距
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    
    print(f"    形状: {data.shape}")
    print(f"    间距: {spacing} mm")
    print(f"    数据范围: [{data.min():.2f}, {data.max():.2f}]")
    
    return data, affine, spacing


def voxel_to_physical(vertices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    将体素坐标转换为物理空间坐标
    
    Args:
        vertices: 体素坐标数组 (N, 3)
        affine: 4x4 仿射变换矩阵
        
    Returns:
        np.ndarray: 物理空间坐标数组 (N, 3)
    """
    # 添加齐次坐标
    ones = np.ones((vertices.shape[0], 1))
    vertices_homo = np.hstack([vertices, ones])
    
    # 应用仿射变换
    physical_coords = vertices_homo @ affine.T
    
    # 返回 xyz 坐标（去掉齐次坐标）
    return physical_coords[:, :3]


# =============================================================================
# Marching Cubes 表面提取
# =============================================================================

def extract_surface_marching_cubes(
    mask_data: np.ndarray,
    spacing: np.ndarray,
    level: float = 0.5,
    step_size: int = 1,
    region_name: str = "区域"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 Marching Cubes 算法从二值掩膜提取 3D 表面

    Args:
        mask_data: 二值掩膜数据 (0=背景, 1=目标区域)
        spacing: 体素间距 (x, y, z)
        level: 等值面阈值（默认 0.5，适用于二值掩膜）
        step_size: 采样步长（1=最高精度，2=降采样）
        region_name: 区域名称（用于日志输出）

    Returns:
        Tuple[np.ndarray, np.ndarray]: (顶点数组, 面片数组)
    """
    # 确保数据是浮点型
    mask_float = mask_data.astype(np.float32)

    # 检查是否有目标区域
    if mask_float.max() < level:
        raise ValueError(f"{region_name}中没有有效区域（所有值都低于阈值）")

    voxel_count = np.sum(mask_float > level)
    print(f"  {region_name}体素数: {voxel_count}")

    if voxel_count < 10:
        raise ValueError(f"{region_name}太小（仅 {voxel_count} 个体素）")

    # 使用 Marching Cubes 提取等值面
    print(f"  执行 Marching Cubes (level={level}, step_size={step_size})...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vertices, faces, _, _ = measure.marching_cubes(
            mask_float,
            level=level,
            spacing=tuple(spacing),
            step_size=step_size,
            allow_degenerate=False
        )

    print(f"    顶点数: {len(vertices)}")
    print(f"    面片数: {len(faces)}")

    return vertices, faces


# =============================================================================
# STL 导出
# =============================================================================

def export_stl(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str
) -> None:
    """
    将网格导出为 STL 文件

    Args:
        vertices: 顶点数组 (N, 3)
        faces: 面片数组 (M, 3)
        output_path: 输出文件路径
    """
    print(f"  导出 STL: {output_path}")

    # 创建 trimesh 网格对象
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # 修复网格（移除退化面片、修复法向量等）
    mesh.fix_normals()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())

    # 输出网格统计信息
    print(f"    网格体积: {abs(mesh.volume):.2f} mm³")
    print(f"    表面积: {mesh.area:.2f} mm²")
    print(f"    边界框: {mesh.bounds[0]} -> {mesh.bounds[1]}")

    # 确保输出目录存在
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 导出 STL
    mesh.export(str(output_path), file_type='stl')

    # 验证文件大小
    file_size = Path(output_path).stat().st_size / 1024
    print(f"    文件大小: {file_size:.1f} KB")


# =============================================================================
# 主转换函数
# =============================================================================

def convert_mask_to_stl(
    mask_data: np.ndarray,
    affine: np.ndarray,
    spacing: np.ndarray,
    output_path: str,
    region_name: str = "区域",
    level: float = 0.5,
    step_size: int = 1
) -> dict:
    """
    将掩膜数据转换为 STL 3D 网格（通用函数）

    Args:
        mask_data: 掩膜数据
        affine: 仿射变换矩阵
        spacing: 体素间距
        output_path: 输出 STL 文件路径
        region_name: 区域名称
        level: 等值面阈值
        step_size: 采样步长

    Returns:
        dict: 转换结果统计
    """
    # 1. 提取 3D 表面
    print(f"\n  提取 {region_name} 3D 表面...")
    vertices, faces = extract_surface_marching_cubes(
        mask_data, spacing, level=level, step_size=step_size, region_name=region_name
    )

    # 2. 转换到物理空间坐标
    print(f"\n  转换到物理空间坐标...")
    vertices_voxel = vertices / spacing  # 还原到体素坐标
    vertices_physical = voxel_to_physical(vertices_voxel, affine)

    print(f"    物理坐标范围:")
    print(f"      X: [{vertices_physical[:, 0].min():.2f}, {vertices_physical[:, 0].max():.2f}] mm")
    print(f"      Y: [{vertices_physical[:, 1].min():.2f}, {vertices_physical[:, 1].max():.2f}] mm")
    print(f"      Z: [{vertices_physical[:, 2].min():.2f}, {vertices_physical[:, 2].max():.2f}] mm")

    # 3. 导出 STL
    print(f"\n  导出 STL 文件...")
    export_stl(vertices_physical, faces, output_path)

    return {
        'output_file': output_path,
        'voxel_count': int(np.sum(mask_data > level)),
        'vertex_count': len(vertices),
        'face_count': len(faces),
        'bounds': {
            'min': vertices_physical.min(axis=0).tolist(),
            'max': vertices_physical.max(axis=0).tolist()
        }
    }


def convert_lesion_to_stl(
    mask_path: str,
    output_path: str,
    level: float = 0.5,
    step_size: int = 1
) -> dict:
    """
    将病灶掩膜转换为 STL 3D 网格

    Args:
        mask_path: 病灶掩膜 NIfTI 文件路径
        output_path: 输出 STL 文件路径
        level: 等值面阈值
        step_size: 采样步长

    Returns:
        dict: 转换结果统计
    """
    print("=" * 60)
    print("  病灶区域 STL 3D 网格导出")
    print("=" * 60)

    # 加载 NIfTI 文件
    print("\n[1/2] 加载病灶掩膜...")
    mask_data, affine, spacing = load_nifti_with_spatial_info(mask_path)

    # 转换
    print("\n[2/2] 生成 STL 网格...")
    results = convert_mask_to_stl(
        mask_data, affine, spacing, output_path,
        region_name="病灶", level=level, step_size=step_size
    )
    results['input_file'] = mask_path
    results['spacing'] = spacing.tolist()

    print("\n" + "=" * 60)
    print("  ✓ 病灶区域导出完成!")
    print("=" * 60)

    return results


def convert_lesion_and_lung_to_stl(
    lesion_mask_path: str,
    lung_mask_path: str,
    lesion_output_path: str,
    lung_output_path: str,
    level: float = 0.5,
    step_size: int = 1,
    export_lesion: bool = True,
    export_lung: bool = True
) -> dict:
    """
    将病灶区域和正常肺部区域转换为 STL 3D 网格

    Args:
        lesion_mask_path: 病灶掩膜 NIfTI 文件路径
        lung_mask_path: 肺部掩膜 NIfTI 文件路径
        lesion_output_path: 病灶 STL 输出路径
        lung_output_path: 肺部 STL 输出路径
        level: 等值面阈值
        step_size: 采样步长
        export_lesion: 是否导出病灶区域
        export_lung: 是否导出正常肺部区域

    Returns:
        dict: 转换结果统计
    """
    print("=" * 60)
    print("  病灶区域 + 正常肺部 STL 3D 网格导出")
    print("=" * 60)

    results = {}

    # 1. 加载病灶掩膜
    print("\n[1/4] 加载病灶掩膜...")
    lesion_data, affine, spacing = load_nifti_with_spatial_info(lesion_mask_path)

    # 2. 加载肺部掩膜
    print("\n[2/4] 加载肺部掩膜...")
    lung_data, _, _ = load_nifti_with_spatial_info(lung_mask_path)

    # 检查形状是否一致
    if lesion_data.shape != lung_data.shape:
        print(f"  警告: 掩膜形状不一致!")
        print(f"    病灶掩膜: {lesion_data.shape}")
        print(f"    肺部掩膜: {lung_data.shape}")
        raise ValueError("病灶掩膜和肺部掩膜形状不一致，无法计算正常肺部区域")

    # 3. 导出病灶区域
    if export_lesion:
        print("\n[3/4] 生成病灶区域 STL...")
        try:
            results['lesion'] = convert_mask_to_stl(
                lesion_data, affine, spacing, lesion_output_path,
                region_name="病灶", level=level, step_size=step_size
            )
        except Exception as e:
            print(f"  ✗ 病灶区域导出失败: {e}")
            results['lesion'] = {'error': str(e)}
    else:
        print("\n[3/4] 跳过病灶区域导出")

    # 4. 计算并导出正常肺部区域
    if export_lung:
        print("\n[4/4] 生成正常肺部区域 STL...")

        # 正常肺部 = 肺部掩膜 - 病灶区域
        normal_lung_data = lung_data.copy()
        normal_lung_data[lesion_data > level] = 0  # 从肺部掩膜中移除病灶区域

        # 统计
        lung_voxels = np.sum(lung_data > level)
        lesion_voxels = np.sum(lesion_data > level)
        normal_lung_voxels = np.sum(normal_lung_data > level)
        print(f"  肺部总体素: {lung_voxels}")
        print(f"  病灶体素: {lesion_voxels}")
        print(f"  正常肺部体素: {normal_lung_voxels}")

        try:
            results['lung'] = convert_mask_to_stl(
                normal_lung_data, affine, spacing, lung_output_path,
                region_name="正常肺部", level=level, step_size=step_size
            )
        except Exception as e:
            print(f"  ✗ 正常肺部区域导出失败: {e}")
            results['lung'] = {'error': str(e)}
    else:
        print("\n[4/4] 跳过正常肺部区域导出")

    print("\n" + "=" * 60)
    print("  ✓ 导出完成!")
    print("=" * 60)

    return results


# =============================================================================
# 命令行入口
# =============================================================================

def normalize_patient_id(patient_id: str) -> str:
    """规范化患者 ID，允许输入 024 或 copd_024。"""
    patient_id = patient_id.strip()
    return patient_id if patient_id.startswith("copd_") else f"copd_{patient_id}"


def parse_patient_ids(patient_text: str) -> list:
    """解析逗号分隔的患者 ID 列表。"""
    return [normalize_patient_id(pid) for pid in patient_text.split(',') if pid.strip()]


def find_lesion_mask(patient_id: str, lesion_root: Optional[Path] = None) -> Optional[str]:
    """查找患者的病灶掩膜文件。"""
    base_dir = Path(__file__).parent.parent
    lesion_root = lesion_root or (base_dir / "data" / "03_mapped")
    patient_id = normalize_patient_id(patient_id)
    mask_path = lesion_root / patient_id / f"{patient_id}_warped_lesion.nii.gz"
    return str(mask_path) if mask_path.exists() else None


def find_lung_mask(lung_mask_path: Optional[str] = None) -> Optional[str]:
    """查找肺部掩膜文件。"""
    if lung_mask_path:
        mask_path = Path(lung_mask_path)
        return str(mask_path) if mask_path.exists() else None
    base_dir = Path(__file__).parent.parent
    mask_path = base_dir / "data" / "02_atlas" / "standard_mask.nii.gz"
    return str(mask_path) if mask_path.exists() else None


def get_all_patients(lesion_root: Optional[Path] = None) -> list:
    """获取所有可用的患者 ID。"""
    base_dir = Path(__file__).parent.parent
    mapped_dir = lesion_root or (base_dir / "data" / "03_mapped")
    patients = []
    if mapped_dir.exists():
        for subdir in mapped_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith("copd_"):
                mask_file = subdir / f"{subdir.name}_warped_lesion.nii.gz"
                if mask_file.exists():
                    patients.append(subdir.name)
    return sorted(patients)


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(
        description='病灶区域与正常肺部 STL 3D 网格导出',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 导出单个患者（病灶+肺部）
  python export_lesion_stl.py --patient copd_001

  # 批量导出指定患者
  python export_lesion_stl.py --patients copd_024,copd_025,copd_026,copd_027,copd_028,copd_029

  # 导出所有患者
  python export_lesion_stl.py --all

  # 指定输入文件（仅病灶）
  python export_lesion_stl.py --input mask.nii.gz --output lesion.stl
        """
    )

    parser.add_argument('--patient', type=str, default=None,
                        help='单个患者 ID (如 copd_024 或 024)')
    parser.add_argument('--patients', type=str, default=None,
                        help='批量患者 ID，逗号分隔，如 copd_024,copd_025')
    parser.add_argument('--all', action='store_true',
                        help='导出 lesion-root 下所有患者')
    parser.add_argument('--lesion-root', type=str, default='data/03_mapped',
                        help='病灶掩膜根目录 (默认: data/03_mapped)')
    parser.add_argument('--lung-mask', type=str, default=None,
                        help='肺部掩膜路径 (默认: data/02_atlas/standard_mask.nii.gz)')
    parser.add_argument('--input', type=str, default=None,
                        help='输入病灶掩膜文件路径（仅导出病灶）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 STL 文件路径')
    parser.add_argument('--lesion-only', action='store_true',
                        help='仅导出病灶区域')
    parser.add_argument('--lung-only', action='store_true',
                        help='仅导出正常肺部区域')
    parser.add_argument('--level', type=float, default=0.5,
                        help='等值面阈值 (默认: 0.5)')
    parser.add_argument('--step-size', type=int, default=1,
                        help='采样步长 (默认: 1, 值越大网格越粗糙)')

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    lesion_root = Path(args.lesion_root)
    if not lesion_root.is_absolute():
        lesion_root = base_dir / lesion_root

    output_dir = base_dir / "data" / "05_stl_export"
    output_dir.mkdir(parents=True, exist_ok=True)

    export_lesion = not args.lung_only
    export_lung = not args.lesion_only

    lung_mask_path = find_lung_mask(args.lung_mask)
    if export_lung and not lung_mask_path:
        print("错误: 未找到肺部掩膜文件")
        sys.exit(1)

    if args.input:
        if not Path(args.input).exists():
            print(f"错误: 输入文件不存在: {args.input}")
            sys.exit(1)
        output_path = args.output or str(output_dir / "custom_lesion.stl")
        convert_lesion_to_stl(args.input, output_path, args.level, args.step_size)
        return

    if args.patients:
        patients = parse_patient_ids(args.patients)
    elif args.all:
        patients = get_all_patients(lesion_root)
    else:
        patients = [normalize_patient_id(args.patient or "copd_001")]

    if not patients:
        print("错误: 未找到任何可处理患者")
        sys.exit(1)

    print(f"病灶掩膜根目录: {lesion_root}")
    print(f"输出目录: {output_dir}")
    print(f"待处理患者数: {len(patients)}")

    success = skipped = failed = 0
    for patient_id in patients:
        print(f"\n{'=' * 60}")
        print(f"处理患者: {patient_id}")
        lesion_path = find_lesion_mask(patient_id, lesion_root)
        expected_path = lesion_root / patient_id / f"{patient_id}_warped_lesion.nii.gz"
        if not lesion_path:
            print(f"  [SKIP] 未找到病灶掩膜文件: {expected_path}")
            skipped += 1
            continue

        lesion_output = args.output or str(output_dir / f"{patient_id}_lesion.stl")
        lung_output = str(output_dir / f"{patient_id}_lung.stl")
        try:
            convert_lesion_and_lung_to_stl(
                lesion_path, lung_mask_path, lesion_output, lung_output,
                args.level, args.step_size, export_lesion, export_lung
            )
            success += 1
        except Exception as e:
            print(f"  [FAIL] 导出失败: {e}")
            failed += 1

    print(f"\n完成: success={success}, skipped={skipped}, failed={failed}")



if __name__ == "__main__":
    main()

