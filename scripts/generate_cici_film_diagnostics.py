#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CICI-FiLM 结果诊断可视化脚本

功能：
    1. 对比真实 CT vs Exp-0 vs Exp-1 的切片可视化
    2. 病灶区域 HU 分布直方图
    3. EI 指标对比柱状图
    4. 生成诊断报告 JSON

使用方法：
    python scripts/generate_cici_film_diagnostics.py
    python scripts/generate_cici_film_diagnostics.py --patients copd_024 copd_026
    python scripts/generate_cici_film_diagnostics.py --output results/cici_film/diagnostics
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_nifti(path):
    return nib.load(str(path)).get_fdata().astype(np.float32)


def emphysema_index(ct, lung_mask, threshold=-950.0):
    vals = ct[lung_mask > 0]
    return float((vals < threshold).mean() * 100.0) if vals.size else 0.0


def find_center_slice(mask):
    z_indices = np.where(mask.sum(axis=(0, 1)) > 100)[0]
    return z_indices[len(z_indices) // 2] if len(z_indices) > 0 else mask.shape[2] // 2


def plot_slice_comparison(ref_ct, exp0_ct, exp1_ct, lesion_mask, z_slice, output_path, patient_id):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    ref_slice = ref_ct[:, :, z_slice]
    exp0_slice = exp0_ct[:, :, z_slice]
    exp1_slice = exp1_ct[:, :, z_slice]
    mask_slice = lesion_mask[:, :, z_slice]
    
    axes[0].imshow(ref_slice.T, cmap='gray', vmin=-1000, vmax=200, origin='lower')
    axes[0].set_title(f'{patient_id} - Ref CT (Warped)')
    axes[0].axis('off')
    
    axes[1].imshow(exp0_slice.T, cmap='gray', vmin=-1000, vmax=200, origin='lower')
    axes[1].set_title('Exp-0 (Fixed Calib)')
    axes[1].axis('off')
    
    axes[2].imshow(exp1_slice.T, cmap='gray', vmin=-1000, vmax=200, origin='lower')
    axes[2].set_title('Exp-1 (Adaptive Calib)')
    axes[2].axis('off')
    
    axes[3].imshow(mask_slice.T, cmap='Reds', alpha=0.6, origin='lower')
    axes[3].set_title('Lesion Mask')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_hu_histogram(ref_lesion, exp0_lesion, exp1_lesion, output_path, patient_id):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.arange(-1000, 0, 10)
    ax.hist(ref_lesion, bins=bins, alpha=0.5, label='Ref (Warped)', color='blue', density=True)
    ax.hist(exp0_lesion, bins=bins, alpha=0.5, label='Exp-0', color='orange', density=True)
    ax.hist(exp1_lesion, bins=bins, alpha=0.5, label='Exp-1', color='green', density=True)
    
    ax.axvline(ref_lesion.mean(), color='blue', linestyle='--', linewidth=2, label=f'Ref Mean: {ref_lesion.mean():.1f}')
    ax.axvline(exp0_lesion.mean(), color='orange', linestyle='--', linewidth=2, label=f'Exp-0 Mean: {exp0_lesion.mean():.1f}')
    ax.axvline(exp1_lesion.mean(), color='green', linestyle='--', linewidth=2, label=f'Exp-1 Mean: {exp1_lesion.mean():.1f}')
    
    ax.set_xlabel('HU Value')
    ax.set_ylabel('Density')
    ax.set_title(f'{patient_id} - Lesion Region HU Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ei_comparison(ei_data, output_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    patients = list(ei_data.keys())
    x = np.arange(len(patients))
    width = 0.25
    
    ref_vals = [ei_data[p]['ref'] for p in patients]
    exp0_vals = [ei_data[p]['exp0'] for p in patients]
    exp1_vals = [ei_data[p]['exp1'] for p in patients]
    
    ax.bar(x - width, ref_vals, width, label='Ref (Native)', color='blue', alpha=0.7)
    ax.bar(x, exp0_vals, width, label='Exp-0', color='orange', alpha=0.7)
    ax.bar(x + width, exp1_vals, width, label='Exp-1', color='green', alpha=0.7)
    
    ax.set_xlabel('Patient ID')
    ax.set_ylabel('Emphysema Index (%)')
    ax.set_title('EI Comparison: Ref vs Exp-0 vs Exp-1')
    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_diagnostics_for_patient(patient_id, exp0_dir, exp1_dir, ref_ct_dir, ref_lung_dir, mapped_dir,
                                      atlas_lung_mask, output_dir):
    print(f'  处理患者: {patient_id}')

    exp0_ct = load_nifti(exp0_dir / f'{patient_id}.nii.gz')
    exp1_ct = load_nifti(exp1_dir / f'{patient_id}.nii.gz')
    ref_native = load_nifti(ref_ct_dir / f'{patient_id}_clean.nii.gz')
    ref_lung = load_nifti(ref_lung_dir / f'{patient_id}_mask.nii.gz') > 0
    warped_ref = load_nifti(mapped_dir / patient_id / f'{patient_id}_warped.nii.gz')
    lesion_mask = load_nifti(mapped_dir / patient_id / f'{patient_id}_warped_lesion.nii.gz') > 0

    z_slice = find_center_slice(lesion_mask)

    slice_output = output_dir / f'{patient_id}_slice_comparison.png'
    plot_slice_comparison(warped_ref, exp0_ct, exp1_ct, lesion_mask, z_slice, slice_output, patient_id)

    ref_lesion = warped_ref[lesion_mask]
    exp0_lesion = exp0_ct[lesion_mask]
    exp1_lesion = exp1_ct[lesion_mask]

    hist_output = output_dir / f'{patient_id}_hu_histogram.png'
    plot_hu_histogram(ref_lesion, exp0_lesion, exp1_lesion, hist_output, patient_id)

    ei_ref = emphysema_index(ref_native, ref_lung)
    ei_exp0 = emphysema_index(exp0_ct, atlas_lung_mask > 0)
    ei_exp1 = emphysema_index(exp1_ct, atlas_lung_mask > 0)

    return {
        'ei_ref': ei_ref,
        'ei_exp0': ei_exp0,
        'ei_exp1': ei_exp1,
        'hu_mean_ref': float(ref_lesion.mean()),
        'hu_mean_exp0': float(exp0_lesion.mean()),
        'hu_mean_exp1': float(exp1_lesion.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description='生成 CICI-FiLM 诊断可视化')
    parser.add_argument('--patients', nargs='+', default=[f'copd_{i:03d}' for i in range(24, 30)])
    parser.add_argument('--exp0-dir', default='results/cici_film/exp0')
    parser.add_argument('--exp1-dir', default='results/cici_film/exp1')
    parser.add_argument('--ref-ct-dir', default='data/01_cleaned/copd_clean')
    parser.add_argument('--ref-lung-mask-dir', default='data/01_cleaned/copd_mask')
    parser.add_argument('--mapped-dir', default='data/03_mapped')
    parser.add_argument('--atlas-lung-mask', default='data/02_atlas/standard_mask.nii.gz')
    parser.add_argument('--output', default='results/cici_film/diagnostics')
    args = parser.parse_args()

    exp0_dir = PROJECT_ROOT / args.exp0_dir
    exp1_dir = PROJECT_ROOT / args.exp1_dir
    ref_ct_dir = PROJECT_ROOT / args.ref_ct_dir
    ref_lung_dir = PROJECT_ROOT / args.ref_lung_mask_dir
    mapped_dir = PROJECT_ROOT / args.mapped_dir
    output_dir = PROJECT_ROOT / args.output

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'CICI-FiLM 诊断可视化')
    print(f'输出目录: {output_dir}')
    print(f'患者列表: {args.patients}')

    atlas_lung_mask = load_nifti(PROJECT_ROOT / args.atlas_lung_mask)
    ei_data = {}

    for pid in args.patients:
        try:
            patient_metrics = generate_diagnostics_for_patient(
                pid, exp0_dir, exp1_dir, ref_ct_dir, ref_lung_dir,
                mapped_dir, atlas_lung_mask, output_dir
            )
            ei_data[pid] = {
                'ref': patient_metrics['ei_ref'],
                'exp0': patient_metrics['ei_exp0'],
                'exp1': patient_metrics['ei_exp1']
            }
            print(f'    ✓ {pid}: EI_ref={patient_metrics["ei_ref"]:.2f}%, '
                  f'EI_exp0={patient_metrics["ei_exp0"]:.2f}%, '
                  f'EI_exp1={patient_metrics["ei_exp1"]:.2f}%')
        except Exception as e:
            print(f'    ✗ {pid}: 失败 - {e}')
            continue

    if ei_data:
        print(f'\n生成汇总对比图...')
        plot_ei_comparison(ei_data, output_dir / 'ei_comparison_all.png')

        summary = {
            'patients': ei_data,
            'timestamp': str(Path(__file__).stat().st_mtime),
            'n_patients': len(ei_data)
        }
        summary_path = output_dir / 'diagnostic_summary.json'
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'✓ 诊断汇总: {summary_path}')

    print(f'\n诊断完成，输出目录: {output_dir}')


if __name__ == '__main__':
    main()


