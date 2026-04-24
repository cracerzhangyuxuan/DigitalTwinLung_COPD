#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CICI-FiLM 推理脚本
==================

使用训练好的条件化生成器进行推理

使用方法：

1. 阶段① 自适应 HU 校准（零训练）：
   python scripts/inference_cici_film.py \\
     --mode exp1 \\
     --backbone-checkpoint checkpoints/patchgan/best.pth \\
     --template data/02_atlas/insp_template.nii.gz \\
     --mask data/copd_masks/mask_J010.nii.gz \\
     --patient-features data/patient_features.json \\
     --patient-id J010 \\
     --output results/exp1_J010.nii.gz

2. 阶段② FiLM 微调推理：
   python scripts/inference_cici_film.py \\
     --mode exp2 \\
     --film-checkpoint checkpoints/cici_film/best.pth \\
     --template data/02_atlas/insp_template.nii.gz \\
     --mask data/copd_masks/mask_J010.nii.gz \\
     --patient-features data/patient_features.json \\
     --patient-id J010 \\
     --output results/exp2_J010.nii.gz
"""

import argparse
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import importlib

import numpy as np
import torch
from src.utils.logger import get_logger

create_model = importlib.import_module('src.04_texture_synthesis.network').create_model
ConditionedGenerator = importlib.import_module('src.04_texture_synthesis.conditioned_model').ConditionedGenerator
fuse_lesion = importlib.import_module('src.04_texture_synthesis.inference_fuse').fuse_lesion

logger = get_logger(__name__)


def _resolve_patient_key(patient_features, patient_id):
    candidates = [patient_id]
    if patient_id.startswith('copd_'):
        candidates.append(patient_id.replace('copd_', ''))
    else:
        candidates.append(f'copd_{patient_id}')
    for key in candidates:
        if key in patient_features:
            return key
    return None


def _build_condition_tensor(patient_condition, patient_features):
    norm_stats = patient_features.get('_norm_stats', {})
    c3_mu = norm_stats.get('c3_mean', -960.0)
    c3_sigma = norm_stats.get('c3_std', 15.0)
    c4_mu = norm_stats.get('c4_mean', 45.0)
    c4_sigma = norm_stats.get('c4_std', 12.5)
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    cond_vec = [
        patient_condition['global_EI'] / 100.0,
        patient_condition['lesion_vol_ratio'] / 100.0,
        float(sigmoid((patient_condition['lesion_HU_mean'] - c3_mu) / (c3_sigma + 1e-8))),
        float(sigmoid((patient_condition['lesion_HU_std'] - c4_mu) / (c4_sigma + 1e-8))),
        float(patient_condition['GOLD']) / 4.0,
    ]
    return cond_vec, torch.tensor([cond_vec], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description='CICI-FiLM 推理')
    parser.add_argument('--mode', required=True, choices=['exp0', 'exp1', 'exp2'])
    parser.add_argument('--backbone-checkpoint', help='预训练 backbone 检查点（exp0/exp1 用）')
    parser.add_argument('--film-checkpoint', help='CICI-FiLM 检查点（exp2 用）')
    parser.add_argument('--template', required=True, help='健康 Atlas 模板')
    parser.add_argument('--mask', required=True, help='病灶 mask')
    parser.add_argument('--patient-features', required=True, help='患者特征 JSON')
    parser.add_argument('--patient-id', required=True, help='患者 ID，支持 024 或 copd_024')
    parser.add_argument('--output', required=True, help='输出路径')
    parser.add_argument('--device', default='cuda', help='设备')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    with open(args.patient_features, 'r', encoding='utf-8') as f:
        patient_features = json.load(f)
    patient_key = _resolve_patient_key(patient_features, args.patient_id)
    if patient_key is None:
        logger.error(f'未找到患者特征: {args.patient_id}')
        return
    patient_condition = patient_features[patient_key]
    logger.info(f'患者 {patient_key} 条件向量:')
    logger.info(f"  c₁ (global_EI): {patient_condition['global_EI']:.2f}%")
    logger.info(f"  c₂ (lesion_vol_ratio): {patient_condition['lesion_vol_ratio']:.2f}%")
    logger.info(f"  c₃ (lesion_HU_mean): {patient_condition['lesion_HU_mean']:.1f} HU")
    logger.info(f"  c₄ (lesion_HU_std): {patient_condition['lesion_HU_std']:.1f} HU")
    logger.info(f"  c₅ (GOLD): {patient_condition['GOLD']}")

    patient_condition_for_calibration = None
    model_condition_tensor = None
    if args.mode in ('exp0', 'exp1'):
        if not args.backbone_checkpoint:
            logger.error(f'{args.mode} 模式需要 --backbone-checkpoint')
            return
        model, _ = create_model('patchgan')
        checkpoint = torch.load(args.backbone_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['generator_state_dict'])
        logger.info(f'✓ 加载 backbone: {args.backbone_checkpoint}')
        if args.mode == 'exp1':
            logger.info('\n[Exp-1] 阶段① 自适应 HU 校准（零训练）')
            patient_condition_for_calibration = {
                'lesion_HU_mean': patient_condition['lesion_HU_mean'],
                'lesion_HU_std': patient_condition['lesion_HU_std'],
            }
        else:
            logger.info('\n[Exp-0] 无条件 Baseline（固定 HU 校准）')
    else:
        logger.info('\n[Exp-2] 阶段② FiLM 微调推理')
        if not args.film_checkpoint:
            logger.error('exp2 模式需要 --film-checkpoint')
            return
        backbone, _ = create_model('patchgan')
        model = ConditionedGenerator(backbone, cond_dim=5, cond_emb_dim=64)
        checkpoint = torch.load(args.film_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['generator_state_dict'])
        logger.info(f'✓ 加载 CICI-FiLM: {args.film_checkpoint}')
        cond_vec, model_condition_tensor = _build_condition_tensor(patient_condition, patient_features)
        logger.info(f"  归一化条件向量: {[f'{v:.4f}' for v in cond_vec]}")

    logger.info('\n开始推理...')
    output_path = fuse_lesion(
        template_path=args.template,
        lesion_mask_path=args.mask,
        model=model,
        output_path=args.output,
        device=str(device),
        model_type='patchgan',
        patient_condition=patient_condition_for_calibration,
        model_condition=model_condition_tensor,
    )
    logger.info(f'\n✓ 推理完成: {output_path}')





if __name__ == '__main__':
    main()

