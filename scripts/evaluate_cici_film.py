#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import entropy
from skimage.metrics import structural_similarity


def load_nifti(path):
    return nib.load(str(path)).get_fdata().astype(np.float32)


def find_fused(exp_dir: Path, pid: str) -> Path:
    candidates = [exp_dir / f'{pid}.nii.gz', exp_dir / f'{pid}_fused.nii.gz', exp_dir / f'exp_{pid}.nii.gz']
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f'未找到 {pid} 的生成结果: {exp_dir}')


def emphysema_index(ct, lung_mask, threshold=-950.0):
    vals = ct[lung_mask > 0]
    return float((vals < threshold).mean() * 100.0) if vals.size else 0.0


def kl_divergence(a, b, bins):
    pa, _ = np.histogram(a, bins=bins, density=True)
    pb, _ = np.histogram(b, bins=bins, density=True)
    pa = np.clip(pa, 1e-8, None)
    pb = np.clip(pb, 1e-8, None)
    pa /= pa.sum()
    pb /= pb.sum()
    return float(entropy(pa, pb))


def compute_psnr(a, b, dynamic_range=1400.0):
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse < 1e-12 else float(20.0 * np.log10(dynamic_range / np.sqrt(mse)))


def compute_ssim_slicewise(ref_vol, pred_vol, mask):
    zs = np.where(mask.sum(axis=(0, 1)) > 50)[0]
    scores = []
    for z in zs:
        idx = mask[:, :, z] > 0
        if idx.sum() < 50:
            continue
        ys, xs = np.where(idx)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        ref = ref_vol[y0:y1, x0:x1, z]
        pred = pred_vol[y0:y1, x0:x1, z]
        min_side = min(ref.shape)
        if min_side < 3:
            continue
        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        if win_size < 3:
            continue
        try:
            scores.append(structural_similarity(ref, pred, data_range=1400.0, win_size=win_size))
        except ValueError:
            continue
    return float(np.mean(scores)) if scores else 0.0


def safe_stats_error(arr_a, arr_b, reducer):
    if arr_a.size == 0 or arr_b.size == 0:
        return None
    return abs(float(reducer(arr_a)) - float(reducer(arr_b)))


def evaluate_one(exp_dir, pid, ref_ct_dir, ref_lung_dir, mapped_dir, atlas_lung_mask):
    fused = load_nifti(find_fused(exp_dir, pid))
    ref_native = load_nifti(ref_ct_dir / f'{pid}_clean.nii.gz')
    ref_lung = load_nifti(ref_lung_dir / f'{pid}_mask.nii.gz') > 0
    warped_ref = load_nifti(mapped_dir / pid / f'{pid}_warped.nii.gz')
    lesion_mask = load_nifti(mapped_dir / pid / f'{pid}_warped_lesion.nii.gz') > 0
    atlas_lung = atlas_lung_mask > 0

    ei_fused = emphysema_index(fused, atlas_lung)
    ei_ref = emphysema_index(ref_native, ref_lung)
    fused_lesion = fused[lesion_mask]
    ref_lesion = warped_ref[lesion_mask]
    lung_ref = warped_ref[atlas_lung]
    lung_fused = fused[atlas_lung]

    return {
        'delta_ei_pct': abs(ei_fused - ei_ref) / (ei_ref + 1e-10) * 100.0,
        'ei_fused_pct': ei_fused,
        'ei_ref_pct': ei_ref,
        'hu_kl': kl_divergence(fused_lesion, ref_lesion, np.arange(-1000, 405, 5)) if fused_lesion.size and ref_lesion.size else None,
        'hu_mean_error': safe_stats_error(fused_lesion, ref_lesion, np.mean),
        'hu_std_error': safe_stats_error(fused_lesion, ref_lesion, np.std),
        'psnr': compute_psnr(lung_ref, lung_fused),
        'ssim': compute_ssim_slicewise(warped_ref, fused, atlas_lung),
    }


def summarize(patient_metrics):
    keys = list(next(iter(patient_metrics.values())).keys())
    return {
        k: {
            'mean': float(np.mean([m[k] for m in patient_metrics.values() if m[k] is not None])),
            'std': float(np.std([m[k] for m in patient_metrics.values() if m[k] is not None])),
        }
        for k in keys if any(m[k] is not None for m in patient_metrics.values())
    }


def main():
    parser = argparse.ArgumentParser(description='评估 Exp-0/Exp-1 的统计-生理等价性指标')
    parser.add_argument('--exp0-dir', required=True)
    parser.add_argument('--exp1-dir', required=True)
    parser.add_argument('--ref-ct-dir', required=True)
    parser.add_argument('--ref-lung-mask-dir', required=True)
    parser.add_argument('--mapped-dir', required=True)
    parser.add_argument('--atlas-lung-mask', default='data/02_atlas/standard_mask.nii.gz')
    parser.add_argument('--patients', nargs='+', default=[f'copd_{i:03d}' for i in range(24, 30)])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    atlas_lung_mask = load_nifti(args.atlas_lung_mask)
    report = {'exp0': {'patients': {}}, 'exp1': {'patients': {}}}
    for exp_name, exp_dir in [('exp0', Path(args.exp0_dir)), ('exp1', Path(args.exp1_dir))]:
        for pid in args.patients:
            report[exp_name]['patients'][pid] = evaluate_one(
                exp_dir, pid, Path(args.ref_ct_dir), Path(args.ref_lung_mask_dir), Path(args.mapped_dir), atlas_lung_mask
            )
        report[exp_name]['summary'] = summarize(report[exp_name]['patients'])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'评估完成: {out}')


if __name__ == '__main__':
    main()

