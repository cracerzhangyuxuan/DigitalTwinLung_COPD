#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CICI-FiLM 结果审计脚本
系统性排查 EI 异常偏高（~50%）的根本原因

检查项:
  1. 生成文件全局/肺区/病灶区 HU 统计
  2. 非病灶肺区是否被意外修改（与 template 逐体素对比）
  3. EI 评估空间一致性验证
  4. Exp-2 FiLM 权重 γ/β 参数分析
"""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_nifti(path):
    return nib.load(str(path)).get_fdata().astype(np.float32)


def audit_patient(pid, exp_dirs, template, atlas_lung, lesion_mask):
    """审计单个患者"""
    non_lesion_lung = atlas_lung & (~lesion_mask)
    t_lung = template[atlas_lung]
    t_non_lesion = template[non_lesion_lung]

    result = {
        'patient_id': pid,
        'lesion_voxels': int(lesion_mask.sum()),
        'lung_voxels': int(atlas_lung.sum()),
        'lesion_ratio_pct': round(float(lesion_mask.sum() / atlas_lung.sum() * 100), 4),
        'template': {
            'lung_mean_hu': round(float(t_lung.mean()), 2),
            'lung_ei_pct': round(float((t_lung < -950).mean() * 100), 4),
            'non_lesion_mean_hu': round(float(t_non_lesion.mean()), 2),
            'non_lesion_ei_pct': round(float((t_non_lesion < -950).mean() * 100), 4),
        },
    }

    for exp_name, exp_dir in exp_dirs.items():
        fpath = exp_dir / f'{pid}.nii.gz'
        if not fpath.exists():
            result[exp_name] = 'FILE_NOT_FOUND'
            continue
        vol = load_nifti(fpath)
        lung_vals = vol[atlas_lung]
        lesion_vals = vol[lesion_mask]
        non_lesion_vals = vol[non_lesion_lung]
        diff = non_lesion_vals - t_non_lesion

        result[exp_name] = {
            'global': {'min': round(float(vol.min()), 2), 'max': round(float(vol.max()), 2),
                       'mean': round(float(vol.mean()), 2)},
            'lung_ei_pct': round(float((lung_vals < -950).mean() * 100), 4),
            'lung_mean_hu': round(float(lung_vals.mean()), 2),
            'lesion': {'mean_hu': round(float(lesion_vals.mean()), 2),
                       'std_hu': round(float(lesion_vals.std()), 2),
                       'ei_pct': round(float((lesion_vals < -950).mean() * 100), 2)},
            'non_lesion_lung': {
                'ei_pct': round(float((non_lesion_vals < -950).mean() * 100), 4),
                'mean_hu': round(float(non_lesion_vals.mean()), 2),
                'diff_mean': round(float(diff.mean()), 4),
                'diff_max_abs': round(float(np.abs(diff).max()), 2),
                'diff_nonzero_pct': round(float((np.abs(diff) > 0.01).mean() * 100), 4),
            },
        }
    return result


def audit_film_weights(ckpt_path):
    """分析 FiLM 权重 γ/β"""
    try:
        import torch
        ckpt = torch.load(str(ckpt_path), map_location='cpu')
    except Exception as e:
        return {'error': str(e)}

    sd = ckpt.get('generator_state_dict', ckpt)
    result = {}
    for k, v in sd.items():
        if 'film' in k or 'cond' in k:
            arr = v.detach().cpu().numpy()
            result[k] = {
                'shape': list(arr.shape),
                'mean': round(float(arr.mean()), 8),
                'std': round(float(arr.std()), 8),
                'min': round(float(arr.min()), 8),
                'max': round(float(arr.max()), 8),
                'abs_mean': round(float(np.abs(arr).mean()), 8),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description='CICI-FiLM 结果审计')
    parser.add_argument('--patients', nargs='+', default=[f'copd_{i:03d}' for i in range(24, 30)])
    parser.add_argument('--exp0-dir', default='results/cici_film/exp0')
    parser.add_argument('--exp1-dir', default='results/cici_film/exp1')
    parser.add_argument('--exp2-dir', default='results/cici_film/exp2')
    parser.add_argument('--template', default='data/02_atlas/standard_template.nii.gz')
    parser.add_argument('--atlas-lung-mask', default='data/02_atlas/standard_mask.nii.gz')
    parser.add_argument('--mapped-dir', default='data/03_mapped')
    parser.add_argument('--film-checkpoint', default='checkpoints/cici_film/best.pth')
    parser.add_argument('--output', default='results/cici_film/audit_report.json')
    args = parser.parse_args()

    print('=' * 60)
    print('CICI-FiLM 结果审计')
    print('=' * 60)

    template = load_nifti(PROJECT_ROOT / args.template)
    atlas_lung = load_nifti(PROJECT_ROOT / args.atlas_lung_mask) > 0
    print(f'Template shape: {template.shape}')
    print(f'Template 肺区 EI: {(template[atlas_lung] < -950).mean() * 100:.4f}%')
    print(f'Template 肺区 mean HU: {template[atlas_lung].mean():.2f}')

    exp_dirs = {}
    for name in ['exp0', 'exp1', 'exp2']:
        d = PROJECT_ROOT / getattr(args, f'{name}_dir')
        if d.exists():
            exp_dirs[name] = d
    print(f'实验目录: {list(exp_dirs.keys())}')

    report = {'template_lung_ei_pct': round(float((template[atlas_lung] < -950).mean() * 100), 4),
              'patients': []}

    for pid in args.patients:
        print(f'\n--- {pid} ---')
        lm_path = PROJECT_ROOT / args.mapped_dir / pid / f'{pid}_warped_lesion.nii.gz'
        lesion_mask = load_nifti(lm_path) > 0
        r = audit_patient(pid, exp_dirs, template, atlas_lung, lesion_mask)
        report['patients'].append(r)

        t_ei = r['template']['non_lesion_ei_pct']
        for exp in exp_dirs:
            if isinstance(r.get(exp), dict):
                ei = r[exp]['lung_ei_pct']
                nl_ei = r[exp]['non_lesion_lung']['ei_pct']
                diff = r[exp]['non_lesion_lung']['diff_mean']
                dnz = r[exp]['non_lesion_lung']['diff_nonzero_pct']
                print(f'  {exp}: 全肺EI={ei:.2f}%, 非病灶EI={nl_ei:.2f}%(模板={t_ei:.2f}%), '
                      f'HU偏移={diff:.2f}, 非零差异={dnz:.2f}%')

    # FiLM 权重
    film_ckpt = PROJECT_ROOT / args.film_checkpoint
    if film_ckpt.exists():
        print(f'\n{"=" * 60}')
        print(f'FiLM 权重分析: {film_ckpt}')
        print(f'{"=" * 60}')
        report['film_weights'] = audit_film_weights(film_ckpt)
        for k, v in report['film_weights'].items():
            if isinstance(v, dict) and 'mean' in v:
                print(f'  {k}: mean={v["mean"]:.8f}, std={v["std"]:.8f}, '
                      f'min={v["min"]:.8f}, max={v["max"]:.8f}')
    else:
        print(f'\n⚠ FiLM checkpoint 不存在: {film_ckpt}')
        report['film_weights'] = {'error': 'checkpoint not found'}

    out = Path(PROJECT_ROOT / args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\n审计报告: {out}')


if __name__ == '__main__':
    main()

