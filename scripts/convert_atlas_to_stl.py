#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 NIfTI 肺部图谱按解剖结构转换为 STL 文件

输出：
  results/stl_models/
  ├── inspiration/  (吸气相)
  │   ├── lobe_LUL.stl, lobe_LLL.stl, lobe_RUL.stl, lobe_RML.stl, lobe_RLL.stl
  │   └── airway.stl
  └── expiration/   (呼气相)
      ├── lobe_LUL.stl, ...
      └── airway.stl

用法:
    python scripts/convert_atlas_to_stl.py
    python scripts/convert_atlas_to_stl.py --smooth 2      # 高斯平滑sigma（默认1）
    python scripts/convert_atlas_to_stl.py --decimate 0.3   # 减面比例（默认0.5）
    python scripts/convert_atlas_to_stl.py --no-decimate     # 不减面
"""

import argparse
import time
import numpy as np
import nibabel as nib
import trimesh
from pathlib import Path
from skimage import measure
from scipy import ndimage


# 肺叶标签映射: label_value -> (name, filename)
LOBE_MAP = {
    1: ('左上叶 LUL', 'lobe_LUL'),
    2: ('左下叶 LLL', 'lobe_LLL'),
    3: ('右上叶 RUL', 'lobe_RUL'),
    4: ('右中叶 RML', 'lobe_RML'),
    5: ('右下叶 RLL', 'lobe_RLL'),
}


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

    # 1. 高斯平滑（使等值面更光滑）
    if smooth_sigma > 0:
        volume = ndimage.gaussian_filter(mask_data.astype(np.float32), sigma=smooth_sigma)
    else:
        volume = mask_data.astype(np.float32)

    # 2. Marching Cubes 提取等值面
    #    level=0.5 是二值 mask 经高斯平滑后的最佳阈值
    verts_voxel, faces, normals, _ = measure.marching_cubes(
        volume, level=0.5, spacing=(1.0, 1.0, 1.0)
    )

    # 3. 体素坐标 → 物理坐标（使用 NIfTI affine 矩阵）
    #    verts_voxel 形状: (N, 3), 需要齐次坐标变换
    ones = np.ones((verts_voxel.shape[0], 1))
    verts_homo = np.hstack([verts_voxel, ones])          # (N, 4)
    verts_phys = (affine @ verts_homo.T).T[:, :3]        # (N, 3)

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


def process_phase(lobes_path, trachea_path, output_dir, phase_name,
                  smooth_sigma=1.0, decimate_ratio=0.5):
    """处理一个呼吸相（吸气/呼气）的所有结构。"""
    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    print(f"{'='*60}")

    # 加载 NIfTI
    lobes_nii = nib.load(str(lobes_path))
    lobes_data = lobes_nii.get_fdata()
    affine = lobes_nii.affine
    print(f"  肺叶标签: {lobes_path.name}, shape={lobes_data.shape}")
    print(f"  spacing={tuple(round(x, 4) for x in lobes_nii.header.get_zooms()[:3])}")

    trachea_nii = nib.load(str(trachea_path))
    trachea_data = trachea_nii.get_fdata()
    print(f"  气管mask: {trachea_path.name}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)
    meshes = {}

    # 处理 5 个肺叶
    for label_val, (cn_name, filename) in LOBE_MAP.items():
        mask = (lobes_data.astype(int) == label_val).astype(np.uint8)
        n_voxels = mask.sum()
        if n_voxels == 0:
            print(f"  ⚠ {cn_name}: 标签 {label_val} 无体素，跳过")
            continue
        out_path = output_dir / f"{filename}.stl"
        meshes[filename] = mask_to_stl(
            mask, affine, out_path, smooth_sigma, decimate_ratio, cn_name
        )

    # 处理气管
    airway_mask = (trachea_data > 0.5).astype(np.uint8)
    n_airway = airway_mask.sum()
    if n_airway > 0:
        out_path = output_dir / "airway.stl"
        meshes['airway'] = mask_to_stl(
            airway_mask, affine, out_path, smooth_sigma, decimate_ratio, '气管树'
        )
    else:
        print("  ⚠ 气管 mask 无体素，跳过")

    return meshes




def main():
    parser = argparse.ArgumentParser(
        description='将 NIfTI 肺部图谱按解剖结构转换为 STL',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--atlas-dir', default='data/02_atlas',
                        help='图谱数据目录 (default: data/02_atlas)')
    parser.add_argument('--output', default='results/stl_models',
                        help='STL 输出根目录 (default: results/stl_models)')
    parser.add_argument('--smooth', type=float, default=1.0,
                        help='高斯平滑 sigma (default: 1.0, 0=不平滑)')
    parser.add_argument('--decimate', type=float, default=0.5,
                        help='减面比例 (default: 0.5, 保留50%%面片)')
    parser.add_argument('--no-decimate', action='store_true',
                        help='不进行减面')
    args = parser.parse_args()

    atlas_dir = Path(args.atlas_dir)
    output_root = Path(args.output)
    decimate = None if args.no_decimate else args.decimate

    print("=" * 60)
    print("  NIfTI → STL 解剖结构转换")
    print("=" * 60)
    print(f"  图谱目录:   {atlas_dir}")
    print(f"  输出目录:   {output_root}")
    print(f"  高斯平滑:   sigma={args.smooth}")
    print(f"  减面比例:   {'不减面' if args.no_decimate else f'{args.decimate:.0%}'}")

    t_total = time.time()

    # ---- 吸气相 (Inspiration) ----
    insp_lobes = atlas_dir / 'standard_lung_lobes_labeled.nii.gz'
    insp_airway = atlas_dir / 'standard_trachea_mask.nii.gz'
    if insp_lobes.exists() and insp_airway.exists():
        process_phase(
            insp_lobes, insp_airway,
            output_root / 'inspiration', '吸气相 (Inspiration)',
            smooth_sigma=args.smooth, decimate_ratio=decimate
        )
    else:
        print(f"\n⚠ 吸气相文件缺失: {insp_lobes} / {insp_airway}")

    # ---- 呼气相 (Expiration) ----
    exp_lobes = atlas_dir / 'exp' / 'standard_lung_lobes_labeled.nii.gz'
    exp_airway = atlas_dir / 'exp' / 'standard_trachea_mask.nii.gz'
    if exp_lobes.exists() and exp_airway.exists():
        process_phase(
            exp_lobes, exp_airway,
            output_root / 'expiration', '呼气相 (Expiration)',
            smooth_sigma=args.smooth, decimate_ratio=decimate
        )
    else:
        print(f"\n⚠ 呼气相文件缺失: {exp_lobes} / {exp_airway}")

    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  全部完成! 总耗时: {elapsed_total:.1f}s")
    print(f"  输出目录: {output_root}")
    print(f"{'='*60}")

    # 列出生成的文件
    for stl_file in sorted(output_root.rglob('*.stl')):
        rel = stl_file.relative_to(output_root)
        size_mb = stl_file.stat().st_size / 1024 / 1024
        print(f"  {rel}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()