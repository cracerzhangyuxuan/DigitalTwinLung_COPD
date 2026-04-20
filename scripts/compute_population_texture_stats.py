#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算群体纹理统计 (B1 Sharpness, B2 Frangi Ratio)

用法:
    python scripts/compute_population_texture_stats.py
    python scripts/compute_population_texture_stats.py --skip-frangi  # 只计算 Sharpness
    python scripts/compute_population_texture_stats.py --limit 5      # 限制受试者数量（测试用）
"""

import sys
import json
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.evaluate_atlas_quality import TextureTopologyMetrics


def main():
    parser = argparse.ArgumentParser(description='计算群体纹理统计 (B1 Sharpness, B2 Frangi)')
    parser.add_argument('--atlas-dir', default='data/02_atlas', help='图谱目录')
    parser.add_argument('--normal-mapped-dir', default='data/04_normal_mapped', help='配准后正常人数据目录')
    parser.add_argument('--output', default='results/atlas_eval/atlas_quality_metrics.json', help='输出JSON文件')
    parser.add_argument('--skip-frangi', action='store_true', help='跳过Frangi计算（加速）')
    parser.add_argument('--limit', type=int, default=None, help='限制受试者数量（测试用）')
    args = parser.parse_args()

    atlas_dir = Path(args.atlas_dir)
    lung_mask_path = atlas_dir / 'standard_mask.nii.gz'
    airway_mask_path = atlas_dir / 'standard_trachea_mask.nii.gz'
    normal_mapped_dir = Path(args.normal_mapped_dir)
    output_path = Path(args.output)

    # 加载mask
    print(f"加载 mask: {lung_mask_path}")
    lung_mask = nib.load(str(lung_mask_path)).get_fdata()
    airway_mask = nib.load(str(airway_mask_path)).get_fdata()

    # 获取受试者列表
    subject_dirs = sorted(normal_mapped_dir.glob('normal_*'))
    if args.limit:
        subject_dirs = subject_dirs[:args.limit]
    
    print(f"\n找到 {len(subject_dirs)} 个受试者")
    print(f"{'='*60}")

    all_sharpness = []
    all_frangi = []

    for i, subj_dir in enumerate(subject_dirs):
        pid = subj_dir.name
        print(f"\n[{i+1}/{len(subject_dirs)}] {pid}")
        
        warped_path = subj_dir / f"{pid}_warped.nii.gz"
        if not warped_path.exists():
            print(f"  ⚠ 跳过: 未找到 {warped_path.name}")
            continue
        
        # 加载数据
        subj_data = nib.load(str(warped_path)).get_fdata()
        
        # B1. Sharpness
        print(f"  计算 Sharpness...")
        sharp = TextureTopologyMetrics.compute_sharpness(subj_data, lung_mask)
        all_sharpness.append(sharp)
        print(f"    Sharpness = {sharp['sharpness_laplacian_var']:.2f}")
        
        # B2. Frangi (可选)
        if not args.skip_frangi:
            print(f"  计算 Frangi Ratio (可能需要几分钟)...")
            frangi = TextureTopologyMetrics.compute_frangi(subj_data, airway_mask, lung_mask)
            all_frangi.append(frangi)
            print(f"    Frangi Ratio = {frangi['frangi_ratio']:.4f}")
        
        del subj_data

    # 计算统计
    print(f"\n{'='*60}")
    print("汇总统计:")
    print(f"{'='*60}")

    population_sharpness = {}
    if all_sharpness:
        sharp_vals = [s['sharpness_laplacian_var'] for s in all_sharpness]
        population_sharpness = {
            'mean': round(float(np.mean(sharp_vals)), 2),
            'std': round(float(np.std(sharp_vals)), 2),
            'n': len(sharp_vals),
            'min': round(float(np.min(sharp_vals)), 2),
            'max': round(float(np.max(sharp_vals)), 2)
        }
        print(f"B1 Sharpness: {population_sharpness['mean']:.2f} ± {population_sharpness['std']:.2f}")
        print(f"  范围: [{population_sharpness['min']:.2f}, {population_sharpness['max']:.2f}]")
        print(f"  样本数: {population_sharpness['n']}")

    population_frangi = {}
    if all_frangi:
        frangi_vals = [f['frangi_ratio'] for f in all_frangi]
        population_frangi = {
            'mean': round(float(np.mean(frangi_vals)), 4),
            'std': round(float(np.std(frangi_vals)), 4),
            'n': len(frangi_vals),
            'min': round(float(np.min(frangi_vals)), 4),
            'max': round(float(np.max(frangi_vals)), 4)
        }
        print(f"\nB2 Frangi Ratio: {population_frangi['mean']:.4f} ± {population_frangi['std']:.4f}")
        print(f"  范围: [{population_frangi['min']:.4f}, {population_frangi['max']:.4f}]")
        print(f"  样本数: {population_frangi['n']}")

    # 更新JSON文件
    if output_path.exists():
        print(f"\n更新现有JSON文件: {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['population_sharpness'] = population_sharpness
        data['population_frangi'] = population_frangi
        data['population_stats_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 已更新 {output_path}")
    else:
        print(f"\n⚠ JSON文件不存在: {output_path}")
        print("请先运行: python scripts/evaluate_atlas_quality.py")

    print(f"\n{'='*60}")
    print("完成!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

