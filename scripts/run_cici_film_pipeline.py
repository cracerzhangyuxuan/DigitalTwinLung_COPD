#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CICI-FiLM 端到端流水线

功能：
    1. 提取患者特征，生成 patient_features.json
    2. 批量执行 Exp-0（无条件 baseline）推理
    3. 批量执行 Exp-1（自适应 HU 校准）推理
    4. 评估统计-生理等价性指标

使用方法：
    python scripts/run_cici_film_pipeline.py
    python scripts/run_cici_film_pipeline.py --device cpu --skip-extract
    python scripts/run_cici_film_pipeline.py --patients copd_024 copd_025
    nohup python scripts/run_cici_film_pipeline.py --device cuda:0 > logs/cici_film_pipeline.out 2>&1 &
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='运行 CICI-FiLM 完整流水线')
    parser.add_argument('--device', default='cpu', help='推理设备，如 cpu / cuda / cuda:0')
    parser.add_argument('--patients', nargs='+', default=[f'copd_{i:03d}' for i in range(24, 30)])
    parser.add_argument('--skip-extract', action='store_true', help='跳过特征提取阶段')
    parser.add_argument('--skip-exp0', action='store_true', help='跳过 Exp-0 推理阶段')
    parser.add_argument('--skip-exp1', action='store_true', help='跳过 Exp-1 推理阶段')
    parser.add_argument('--skip-eval', action='store_true', help='跳过评估阶段')
    parser.add_argument('--python-bin', default=sys.executable, help='Python 解释器路径')
    parser.add_argument('--backbone-checkpoint', default='checkpoints/patchgan/best.pth')
    parser.add_argument('--template', default='data/02_atlas/standard_template.nii.gz')
    parser.add_argument('--atlas-lung-mask', default='data/02_atlas/standard_mask.nii.gz')
    parser.add_argument('--ct-dir', default='data/01_cleaned/copd_clean')
    parser.add_argument('--lung-mask-dir', default='data/01_cleaned/copd_mask')
    parser.add_argument('--emphysema-dir', default='data/01_cleaned/copd_emphysema')
    parser.add_argument('--clinical-data', default='data/clinical_data.csv')
    parser.add_argument('--mapped-dir', default='data/03_mapped')
    parser.add_argument('--features-json', default='data/patient_features.json')
    parser.add_argument('--output-root', default='results/cici_film')
    return parser.parse_args()


def normalize_patient_id(patient_id):
    value = str(patient_id).strip()
    if value.startswith('copd_'):
        suffix = value.split('copd_', 1)[1]
        return f'copd_{int(suffix):03d}' if suffix.isdigit() else value
    return f'copd_{int(value):03d}' if value.isdigit() else value


def build_config(args):
    patients = []
    for pid in args.patients:
        pid = normalize_patient_id(pid)
        if pid not in patients:
            patients.append(pid)
    output_root = PROJECT_ROOT / args.output_root
    return {
        'python_bin': str(args.python_bin),
        'device': args.device,
        'patients': patients,
        'paths': {
            'template': PROJECT_ROOT / args.template,
            'atlas_lung_mask': PROJECT_ROOT / args.atlas_lung_mask,
            'ct_dir': PROJECT_ROOT / args.ct_dir,
            'lung_mask_dir': PROJECT_ROOT / args.lung_mask_dir,
            'emphysema_dir': PROJECT_ROOT / args.emphysema_dir,
            'clinical_data': PROJECT_ROOT / args.clinical_data,
            'mapped_dir': PROJECT_ROOT / args.mapped_dir,
            'features_json': PROJECT_ROOT / args.features_json,
            'exp0_dir': output_root / 'exp0',
            'exp1_dir': output_root / 'exp1',
            'report_json': output_root / 'evaluation_exp0_exp1.json',
        },
        'scripts': {
            'extract': PROJECT_ROOT / 'scripts/extract_patient_features.py',
            'infer': PROJECT_ROOT / 'scripts/inference_cici_film.py',
            'evaluate': PROJECT_ROOT / 'scripts/evaluate_cici_film.py',
        },
        'checkpoints': {'backbone': PROJECT_ROOT / args.backbone_checkpoint},
        'skip': {'extract': args.skip_extract, 'exp0': args.skip_exp0, 'exp1': args.skip_exp1, 'eval': args.skip_eval},
    }


def run_command(command, logger, stage):
    logger.info(f'[RUN] {stage}')
    logger.info('  ' + ' '.join(str(x) for x in command))
    start = time.time()
    subprocess.run([str(x) for x in command], cwd=PROJECT_ROOT, check=True)
    logger.info(f'[OK] {stage} | 耗时 {time.time() - start:.1f}s')


def find_result_file(exp_dir, pid):
    for name in [f'{pid}.nii.gz', f'{pid}_fused.nii.gz', f'exp_{pid}.nii.gz']:
        path = exp_dir / name
        if path.exists():
            return path
    return None


def validate_config(config, logger):
    missing = []
    for path in [config['paths']['exp0_dir'], config['paths']['exp1_dir'], config['paths']['report_json'].parent]:
        Path(path).mkdir(parents=True, exist_ok=True)
    config['paths']['features_json'].parent.mkdir(parents=True, exist_ok=True)

    if not config['skip']['extract']:
        for path in [config['scripts']['extract'], config['paths']['ct_dir'], config['paths']['lung_mask_dir'],
                     config['paths']['emphysema_dir'], config['paths']['clinical_data']]:
            if not Path(path).exists():
                missing.append(str(path))

    if not config['skip']['exp0'] or not config['skip']['exp1']:
        for path in [config['scripts']['infer'], config['paths']['template'], config['paths']['mapped_dir'],
                     config['checkpoints']['backbone']]:
            if not Path(path).exists():
                missing.append(str(path))
        if config['skip']['extract'] and not config['paths']['features_json'].exists():
            missing.append(str(config['paths']['features_json']))
        for pid in config['patients']:
            lesion_mask = config['paths']['mapped_dir'] / pid / f'{pid}_warped_lesion.nii.gz'
            if not lesion_mask.exists():
                missing.append(str(lesion_mask))

    if not config['skip']['eval']:
        for path in [config['scripts']['evaluate'], config['paths']['ct_dir'], config['paths']['lung_mask_dir'],
                     config['paths']['mapped_dir'], config['paths']['atlas_lung_mask']]:
            if not Path(path).exists():
                missing.append(str(path))

    if missing:
        for path in sorted(set(missing)):
            logger.error(f'缺失路径: {path}')
        raise FileNotFoundError('存在必需文件/目录缺失，流水线终止')


def run_feature_extraction(config, logger):
    cmd = [config['python_bin'], config['scripts']['extract'], '--ct-dir', config['paths']['ct_dir'], '--lung-mask-dir',
           config['paths']['lung_mask_dir'], '--emphysema-dir', config['paths']['emphysema_dir'], '--clinical-data',
           config['paths']['clinical_data'], '--output', config['paths']['features_json']]
    run_command(cmd, logger, 'Step 1/4 特征提取')


def run_inference_stage(config, logger, mode, step_no):
    out_dir = config['paths'][f'{mode}_dir']
    logger.info('=' * 60)
    logger.info(f'Step {step_no}/4 {mode.upper()} 推理')
    for i, pid in enumerate(config['patients'], start=1):
        mask = config['paths']['mapped_dir'] / pid / f'{pid}_warped_lesion.nii.gz'
        logger.info(f'[{i}/{len(config["patients"])}] {pid}')
        cmd = [config['python_bin'], config['scripts']['infer'], '--mode', mode, '--backbone-checkpoint',
               config['checkpoints']['backbone'], '--template', config['paths']['template'], '--mask', mask,
               '--patient-features', config['paths']['features_json'], '--patient-id', pid, '--output', out_dir / f'{pid}.nii.gz',
               '--device', config['device']]
        run_command(cmd, logger, f'{mode.upper()}::{pid}')


def run_evaluation(config, logger):
    missing = {mode: [pid for pid in config['patients'] if find_result_file(config['paths'][f'{mode}_dir'], pid) is None]
               for mode in ('exp0', 'exp1')}
    missing = {mode: pids for mode, pids in missing.items() if pids}
    if missing:
        if config['skip']['exp0'] or config['skip']['exp1']:
            logger.warning('检测到联合评估所需结果不完整，自动跳过评估阶段。')
            for mode, pids in missing.items():
                logger.warning(f'{mode} 缺失结果: {", ".join(pids)}')
            return False
        raise FileNotFoundError('评估阶段缺少推理输出，请先完成 Exp-0/Exp-1 或使用 --skip-eval')

    cmd = [config['python_bin'], config['scripts']['evaluate'], '--exp0-dir', config['paths']['exp0_dir'], '--exp1-dir',
           config['paths']['exp1_dir'], '--ref-ct-dir', config['paths']['ct_dir'], '--ref-lung-mask-dir',
           config['paths']['lung_mask_dir'], '--mapped-dir', config['paths']['mapped_dir'], '--atlas-lung-mask',
           config['paths']['atlas_lung_mask'], '--patients', *config['patients'], '--output', config['paths']['report_json']]
    run_command(cmd, logger, 'Step 4/4 评估')
    return True


def main():
    args = parse_args()
    logger = setup_logger('cici_film_pipeline', log_dir='logs', file=True)
    config = build_config(args)
    start = time.time()
    try:
        logger.info('=' * 60)
        logger.info('CICI-FiLM Pipeline 启动')
        logger.info('=' * 60)
        logger.info(f"患者列表: {', '.join(config['patients'])}")
        logger.info(f"设备: {config['device']}")
        validate_config(config, logger)
        if all(config['skip'].values()):
            logger.warning('所有阶段均已跳过，没有需要执行的任务。')
            return
        if not config['skip']['extract']:
            run_feature_extraction(config, logger)
        if not config['skip']['exp0']:
            run_inference_stage(config, logger, 'exp0', 2)
        if not config['skip']['exp1']:
            run_inference_stage(config, logger, 'exp1', 3)
        eval_done = False
        if not config['skip']['eval']:
            eval_done = run_evaluation(config, logger)
        logger.info('=' * 60)
        logger.info(f"流水线完成，总耗时 {time.time() - start:.1f}s")
        if eval_done:
            logger.info(f"评估输出: {config['paths']['report_json']}")
    except Exception as e:
        logger.exception(f'流水线失败: {e}')
        raise


if __name__ == '__main__':
    main()

