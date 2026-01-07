#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
气管树融合诊断脚本

分析融合是否生效，以及为什么在 3D Slicer 中可能看不到气管树
"""

import numpy as np
import nibabel as nib
from pathlib import Path


def diagnose_fusion():
    """诊断气管树融合情况"""
    
    atlas_dir = Path("data/02_atlas")
    
    print("=" * 70)
    print("气管树融合诊断分析")
    print("=" * 70)
    
    # 加载数据
    print("\n加载数据...")
    template = nib.load(atlas_dir / "standard_template.nii.gz").get_fdata()
    fused = nib.load(atlas_dir / "standard_template_with_airway.nii.gz").get_fdata()
    trachea_mask = nib.load(atlas_dir / "standard_trachea_mask.nii.gz").get_fdata() > 0
    
    print(f"\n1. 基本统计:")
    print(f"   模板形状: {template.shape}")
    print(f"   气管树体素数: {np.sum(trachea_mask):,}")
    
    print(f"\n2. 原始模板气管区域 HU 分布:")
    orig_airway = template[trachea_mask]
    print(f"   最小值: {orig_airway.min():.1f}")
    print(f"   最大值: {orig_airway.max():.1f}")
    print(f"   平均值: {orig_airway.mean():.1f}")
    print(f"   中位数: {np.median(orig_airway):.1f}")
    print(f"   标准差: {orig_airway.std():.1f}")
    
    # 分析 HU 分布
    print(f"\n3. 原始气管区域 HU 分布统计:")
    for threshold in [-1000, -995, -990, -980, -950, -900, -800, -500]:
        count = np.sum(orig_airway < threshold)
        pct = count / len(orig_airway) * 100
        print(f"   HU < {threshold}: {count:,} ({pct:.1f}%)")
    
    print(f"\n4. 融合后气管区域 HU 分布:")
    fused_airway = fused[trachea_mask]
    print(f"   最小值: {fused_airway.min():.1f}")
    print(f"   最大值: {fused_airway.max():.1f}")
    print(f"   平均值: {fused_airway.mean():.1f}")
    print(f"   中位数: {np.median(fused_airway):.1f}")
    
    print(f"\n5. 实际修改分析:")
    diff = np.abs(template - fused)
    modified = diff > 0.01
    modified_in_airway = np.sum(modified & trachea_mask)
    total_airway = np.sum(trachea_mask)
    print(f"   被修改的总体素数: {np.sum(modified):,}")
    print(f"   气管区域被修改体素数: {modified_in_airway:,}")
    print(f"   气管区域修改比例: {modified_in_airway / total_airway * 100:.1f}%")
    
    # 分析为什么没修改
    print(f"\n6. 未修改原因分析:")
    airway_hu_target = -995
    already_low = orig_airway <= airway_hu_target
    not_low = orig_airway > airway_hu_target
    print(f"   目标气道 HU 值: {airway_hu_target}")
    print(f"   原始 HU <= {airway_hu_target} 的体素数: {np.sum(already_low):,} ({np.sum(already_low)/len(orig_airway)*100:.1f}%)")
    print(f"   原始 HU >  {airway_hu_target} 的体素数: {np.sum(not_low):,} ({np.sum(not_low)/len(orig_airway)*100:.1f}%)")
    print(f"   --> 由于 preserve_existing_low_hu=True，只有后者被修改")
    
    print(f"\n7. 肺实质 HU 参考 (用于对比):")
    # 加载肺 mask
    try:
        lung_mask = nib.load(atlas_dir / "standard_mask.nii.gz").get_fdata() > 0
        lung_only = lung_mask & (~trachea_mask)  # 肺实质（排除气管）
        lung_hu = template[lung_only]
        print(f"   肺实质 HU 范围: [{lung_hu.min():.1f}, {lung_hu.max():.1f}]")
        print(f"   肺实质 HU 平均值: {lung_hu.mean():.1f}")
        print(f"   气管树 vs 肺实质 HU 差异: {lung_hu.mean() - orig_airway.mean():.1f}")
    except Exception as e:
        print(f"   无法加载肺 mask: {e}")
    
    print(f"\n8. 问题诊断:")
    print(f"   [!] 原始模板气管区域 HU 已经是 {orig_airway.mean():.0f}，接近空气 (-1000)")
    print(f"   [!] 融合后变为 {fused_airway.mean():.0f}，差异只有 {orig_airway.mean() - fused_airway.mean():.1f} HU")
    print(f"   [!] 这个差异在 3D Slicer 默认窗宽 (1500) 下几乎不可见")
    
    print(f"\n9. 解决方案:")
    print(f"   方案 A: 调整 3D Slicer 窗宽窗位")
    print(f"           - 窗宽 (W): 200")
    print(f"           - 窗位 (L): -900")
    print(f"           这样可以增强气管与肺实质的对比度")
    print(f"")
    print(f"   方案 B: 修改融合参数，使气管树更明显")
    print(f"           - 设置 preserve_existing_low_hu=False")
    print(f"           - 或降低 airway_hu 到 -1024 (CT 最低值)")
    print(f"           - 这会让气管树在肺窗下更黑更明显")
    print(f"")
    print(f"   方案 C: 将气管树设为非空气值 (如 -500 或更高)")
    print(f"           - 这会让气管树在渲染中非常明显")
    print(f"           - 但不符合真实 CT 的物理意义")
    
    print("\n" + "=" * 70)
    
    return {
        'orig_mean': orig_airway.mean(),
        'fused_mean': fused_airway.mean(),
        'modified_ratio': modified_in_airway / total_airway
    }


if __name__ == "__main__":
    diagnose_fusion()

