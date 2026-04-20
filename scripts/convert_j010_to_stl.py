#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 J010_20170807_E_035.nii.gz 标签文件转换为 STL 文件

标签映射:
  0: 背景
  1-5: 五个肺叶 (LUL, LLL, RUL, RML, RLL)
  6-7: 气道结构（合并为一个气管树 STL）

输出:
  results/stl_models/J010/
  ├── lobe_LUL.stl
  ├── lobe_LLL.stl
  ├── lobe_RUL.stl
  ├── lobe_RML.stl
  ├── lobe_RLL.stl
  └── airway.stl

用法:
    python scripts/convert_j010_to_stl.py
    python scripts/convert_j010_to_stl.py --smooth 2
    python scripts/convert_j010_to_stl.py --no-decimate
"""

import argparse
import time
import numpy as np
import nibabel as nib
import trimesh
from pathlib import Path
from skimage import measure
from scipy import ndimage


# 肺叶标签映射
LOBE_MAP = {
    1: ('左上叶 LUL', 'lobe_LUL'),
    2: ('左下叶 LLL', 'lobe_LLL'),
    3: ('右上叶 RUL', 'lobe_RUL'),
    4: ('右中叶 RML', 'lobe_RML'),
    5: ('右下叶 RLL', 'lobe_RLL'),
}

# 气道标签（6和7合并）
AIRWAY_LABELS = [6, 7]


def mask_to_stl(mask_data, affine, output_path, smooth_sigma=1.0,
                decimate_ratio=0.5, structure_name=''):
    """
    将二值 mask 转换为 STL 文件。
    
    参数:
        mask_data:      3D numpy array, 二值 mask (0/1)
        affine:         4x4 仿射矩阵, 体素坐标 → 物理坐标
        output_path:    STL 输出路径
        smooth_sigma:   高斯平滑 sigma（0 = 不平滑）
        decimate_ratio: 减面目标比例（0.5 = 保留50%面片）
        structure_name: 结构名称（用于日志）
    """
    t0 = time.time()

    # 1. 高斯平滑
    if smooth_sigma > 0:
        volume = ndimage.gaussian_filter(mask_data.astype(np.float32), sigma=smooth_sigma)
    else:
        volume = mask_data.astype(np.float32)

    # 2. Marching Cubes 提取等值面
    verts_voxel, faces, normals, _ = measure.marching_cubes(
        volume, level=0.5, spacing=(1.0, 1.0, 1.0)
    )

    # 3. 体素坐标 → 物理坐标
    ones = np.ones((verts_voxel.shape[0], 1))
    verts_homo = np.hstack([verts_voxel, ones])
    verts_phys = (affine @ verts_homo.T).T[:, :3]

    # 4. 构建 trimesh 对象
    mesh = trimesh.Trimesh(vertices=verts_phys, faces=faces,
                           vertex_normals=normals)

    # 5. 减面（可选）
    n_faces_before = len(mesh.faces)
    if decimate_ratio and 0 < decimate_ratio < 1.0:
        target = int(n_faces_before * decimate_ratio)
        try:
            mesh = mesh.simplify_quadric_decimation(target)
        except Exception as e:
            print(f"    ⚠ 减面失败（{e}），保留原始面片数")

    # 6. 保存 STL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path), file_type='stl')

    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  ✓ {structure_name:12s} | "
          f"面片: {n_faces_before:>8,} → {len(mesh.faces):>8,} | "
          f"{size_mb:.1f} MB | {elapsed:.1f}s")

    return mesh


def main():
    parser = argparse.ArgumentParser(
        description='将 J010_20170807_E_035.nii.gz 转换为 STL',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input', default='data/02_atlas/J010_20170807_E_035.nii.gz',
                        help='输入标签文件 (default: data/02_atlas/J010_20170807_E_035.nii.gz)')
    parser.add_argument('--output', default='results/stl_models/J010',
                        help='STL 输出目录 (default: results/stl_models/J010)')
    parser.add_argument('--smooth', type=float, default=1.0,
                        help='高斯平滑 sigma (default: 1.0, 0=不平滑)')
    parser.add_argument('--decimate', type=float, default=0.5,
                        help='减面比例 (default: 0.5, 保留50%%面片)')
    parser.add_argument('--no-decimate', action='store_true',
                        help='不进行减面')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    decimate = None if args.no_decimate else args.decimate

    print("=" * 60)
    print("  J010 标签文件 → STL 转换")
    print("=" * 60)
    print(f"  输入文件:   {input_path}")
    print(f"  输出目录:   {output_dir}")
    print(f"  高斯平滑:   sigma={args.smooth}")
    print(f"  减面比例:   {'不减面' if args.no_decimate else f'{args.decimate:.0%}'}")
    print()

    # 加载标签文件
    nii = nib.load(str(input_path))
    labels_data = nii.get_fdata().astype(int)
    affine = nii.affine

    print(f"  标签文件: shape={labels_data.shape}")
    print(f"  spacing={tuple(round(x, 4) for x in nii.header.get_zooms()[:3])}")
    print(f"  标签值: {np.unique(labels_data)}")
    print()

    t_total = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理 5 个肺叶
    for label_val, (cn_name, filename) in LOBE_MAP.items():
        mask = (labels_data == label_val).astype(np.uint8)
        n_voxels = mask.sum()
        if n_voxels == 0:
            print(f"  ⚠ {cn_name}: 标签 {label_val} 无体素，跳过")
            continue
        out_path = output_dir / f"{filename}.stl"
        mask_to_stl(mask, affine, out_path, args.smooth, decimate, cn_name)

    # 处理气管（合并标签6和7）
    airway_mask = np.isin(labels_data, AIRWAY_LABELS).astype(np.uint8)
    n_airway = airway_mask.sum()
    if n_airway > 0:
        out_path = output_dir / "airway.stl"
        mask_to_stl(airway_mask, affine, out_path, args.smooth, decimate, '气管树')
    else:
        print("  ⚠ 气管标签 (6,7) 无体素，跳过")

    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  全部完成! 总耗时: {elapsed_total:.1f}s")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*60}")

    # 列出生成的文件
    for stl_file in sorted(output_dir.glob('*.stl')):
        size_mb = stl_file.stat().st_size / 1024 / 1024
        print(f"  {stl_file.name}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()

