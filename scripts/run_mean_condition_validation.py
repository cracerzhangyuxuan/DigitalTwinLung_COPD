#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CICI-FiLM 条件向量因果验证实验
================================

用途：验证 Exp-2 / CICI-FiLM v3 的条件向量是否确实传递了患者特异性信息。

实验方案（对应 CICI_FiLM_Experiment_Report §5.5(2)）：
  将验证集每位患者的个体化条件向量替换为训练集平均条件向量 c̄_train，
  重新执行 Exp-2 推理，再与"个体条件 Exp-2"和"真实 COPD CT"对比。

  若个体条件 Exp-2 系统性优于平均条件 Exp-2，说明条件向量确实携带了
  患者特异性信息；否则改善可能来自损失函数工程而非条件驱动。

用法：
  # 1. 推理 + 评估（默认 GPU）
  bash:
    python scripts/run_mean_condition_validation.py

  # 2. 仅推理
    python scripts/run_mean_condition_validation.py --infer-only

  # 3. 仅评估（推理结果已存在时）
    python scripts/run_mean_condition_validation.py --eval-only

  # 4. 使用 CPU
    python scripts/run_mean_condition_validation.py --device cpu
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── 常量 ──────────────────────────────────────────────────────────────
TRAIN_IDS = [f'{i:03d}' for i in range(1, 24)]    # 001–023
VAL_IDS   = [f'copd_{i:03d}' for i in range(24, 30)]  # copd_024–029

FILM_CHECKPOINT = 'checkpoints/cici_film_v3/best.pth'
BACKBONE_CHECKPOINT = 'checkpoints/patchgan/best.pth'
TEMPLATE = 'data/02_atlas/standard_template.nii.gz'
FEATURES_JSON = 'data/patient_features.json'

OUT_DIR_MEAN = 'results/cici_film/exp2_v3_mean_condition'
OUT_DIR_INDIV = 'results/cici_film/exp2_v3'          # 已有的个体条件结果
EVAL_OUTPUT = 'results/cici_film/evaluation_mean_condition_vs_individual.json'


def compute_train_mean_condition(features: dict) -> dict:
    """从训练集 (001–023) 计算 5 维原始特征均值。"""
    vals = {k: [] for k in ['global_EI', 'lesion_vol_ratio',
                             'lesion_HU_mean', 'lesion_HU_std', 'GOLD']}
    for pid in TRAIN_IDS:
        if pid not in features:
            continue
        for k in vals:
            vals[k].append(float(features[pid][k]))
    mean_cond = {k: float(np.mean(v)) for k, v in vals.items() if v}
    mean_cond['GOLD'] = round(mean_cond['GOLD'])   # GOLD 取整
    return mean_cond


def run_inference(device: str):
    """对 6 例验证集患者用平均条件向量执行 Exp-2 v3 推理。"""
    import subprocess

    features_path = PROJECT_ROOT / FEATURES_JSON
    with open(features_path, 'r', encoding='utf-8') as f:
        features = json.load(f)
    mean_cond = compute_train_mean_condition(features)
    print(f'\n训练集平均条件向量 (原始特征空间):')
    for k, v in mean_cond.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

    # 临时写入一个只包含平均条件的 features JSON
    mean_features = {}
    for pid in VAL_IDS:
        short = pid.replace('copd_', '')
        mean_features[short] = {**mean_cond}
    # 复制归一化统计量
    if '_norm_stats' in features:
        mean_features['_norm_stats'] = features['_norm_stats']

    tmp_json = PROJECT_ROOT / 'results/cici_film/_tmp_mean_features.json'
    tmp_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json.write_text(json.dumps(mean_features, indent=2), encoding='utf-8')

    out_dir = PROJECT_ROOT / OUT_DIR_MEAN
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n开始推理 (平均条件向量, device={device})...')
    for pid in VAL_IDS:
        mask = PROJECT_ROOT / f'data/03_mapped/{pid}/{pid}_warped_lesion.nii.gz'
        output = out_dir / f'{pid}.nii.gz'
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'scripts/inference_cici_film.py'),
            '--mode', 'exp2',
            '--film-version', 'v3',
            '--film-checkpoint', str(PROJECT_ROOT / FILM_CHECKPOINT),
            '--template', str(PROJECT_ROOT / TEMPLATE),
            '--mask', str(mask),
            '--patient-features', str(tmp_json),
            '--patient-id', pid,
            '--output', str(output),
            '--device', device,
        ]
        print(f'  [{pid}] ...', end=' ', flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print('OK')
        else:
            print(f'FAIL\n{result.stderr[-500:]}')

    # 清理临时文件
    if tmp_json.exists():
        tmp_json.unlink()
    print(f'推理完成: {out_dir}')


def run_evaluation():
    """对比 个体条件 Exp-2 vs 平均条件 Exp-2 vs 真实 COPD。"""
    import subprocess

    eval_script = PROJECT_ROOT / 'scripts/evaluate_cici_film.py'
    out_json = PROJECT_ROOT / EVAL_OUTPUT

    # ── 分别评估两组 ──
    results = {}
    for label, exp_dir in [('individual', OUT_DIR_INDIV),
                           ('mean_condition', OUT_DIR_MEAN)]:
        print(f'\n评估 {label}...')
        tmp_out = out_json.parent / f'_tmp_eval_{label}.json'
        cmd = [
            sys.executable, str(eval_script),
            '--exp0-dir', str(PROJECT_ROOT / exp_dir),   # 复用 exp0 参数
            '--exp1-dir', str(PROJECT_ROOT / exp_dir),   # 同上
            '--ref-ct-dir', 'data/01_cleaned/copd_clean',
            '--ref-lung-mask-dir', 'data/01_cleaned/copd_mask',
            '--mapped-dir', 'data/03_mapped',
            '--atlas-lung-mask', 'data/02_atlas/standard_mask.nii.gz',
            '--patients'] + VAL_IDS + [
            '--output', str(tmp_out),
        ]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        with open(tmp_out, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        # evaluate_cici_film 输出 exp0/exp1，两者相同，取 exp0
        results[label] = raw.get('exp0', raw.get('exp1', {}))
        tmp_out.unlink(missing_ok=True)

    # ── 合并为对比报告 ──
    report = {'experiment': 'Mean-condition causal validation',
              'description': '对比个体条件向量 vs 训练集平均条件向量的 Exp-2 推理结果',
              'individual': results.get('individual', {}),
              'mean_condition': results.get('mean_condition', {})}

    # ── 逐患者差异汇总 ──
    indiv_p = results.get('individual', {}).get('patients', {})
    mean_p = results.get('mean_condition', {}).get('patients', {})
    comparison = {}
    for pid in VAL_IDS:
        if pid in indiv_p and pid in mean_p:
            diff = {}
            for k in indiv_p[pid]:
                vi = indiv_p[pid][k]
                vm = mean_p[pid][k]
                if vi is not None and vm is not None:
                    diff[k] = {'individual': vi, 'mean_cond': vm,
                               'delta': vi - vm}
            comparison[pid] = diff
    report['per_patient_comparison'] = comparison

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    print(f'\n评估完成: {out_json}')

    # ── 打印摘要表 ──
    print('\n' + '=' * 78)
    print('  个体条件 vs 平均条件 — 逐患者关键指标对比')
    print('=' * 78)
    header = f'{"Patient":<12} {"Metric":<18} {"Individual":>11} {"MeanCond":>11} {"Delta":>9} {"Winner":>8}'
    print(header)
    print('-' * 78)
    key_metrics = ['hu_kl', 'hu_std_error', 'hu_mean_error', 'delta_ei_abs_pp']
    for pid in VAL_IDS:
        if pid not in comparison:
            continue
        for km in key_metrics:
            if km not in comparison[pid]:
                continue
            d = comparison[pid][km]
            vi, vm, delta = d['individual'], d['mean_cond'], d['delta']
            # 所有关键指标均为越小越好
            winner = 'Indiv' if vi < vm else ('Mean' if vm < vi else 'Tie')
            print(f'{pid:<12} {km:<18} {vi:>11.4f} {vm:>11.4f} {delta:>+9.4f} {winner:>8}')
        print()


def main():
    parser = argparse.ArgumentParser(
        description='CICI-FiLM 条件向量因果验证：平均条件 vs 个体条件')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--infer-only', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    if not args.eval_only:
        run_inference(args.device)
    if not args.infer_only:
        run_evaluation()


if __name__ == '__main__':
    main()
