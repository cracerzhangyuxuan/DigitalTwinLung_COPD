#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CICI-FiLM v3 训练脚本
====================

核心改进：
  1. SPADE-based 多尺度条件化架构
  2. Wasserstein + Soft Histogram + HU Stats 多层次分布约束
  3. 更长训练：150 epochs
  4. 更小学习率：5e-5（防止过拟合）
  5. 更强正则化：weight decay + gradient clipping
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.04_texture_synthesis.conditioned_model_v3 import ConditionedGeneratorV3
from src.04_texture_synthesis.dataset import LungPatchDataset
from src.04_texture_synthesis.train import Trainer

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='CICI-FiLM v3 训练')
    parser.add_argument('--backbone-checkpoint', required=True,
                        help='预训练 backbone 检查点')
    parser.add_argument('--mapped-dir', required=True,
                        help='配准后数据目录')
    parser.add_argument('--patient-features', required=True,
                        help='患者特征 JSON')
    parser.add_argument('--output-dir', required=True,
                        help='输出目录')
    parser.add_argument('--epochs', type=int, default=150,
                        help='训练轮数（v3 默认 150）')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='学习率（v3 默认 5e-5）')
    parser.add_argument('--patience', type=int, default=40,
                        help='Early stopping patience')
    parser.add_argument('--lambda-wasserstein', type=float, default=1.0,
                        help='Wasserstein-1 loss 权重')
    parser.add_argument('--lambda-hu-stats', type=float, default=0.5,
                        help='HU Statistics loss 权重（归一化空间）')
    parser.add_argument('--lambda-soft-histogram', type=float, default=0.3,
                        help='Soft Histogram loss 权重')
    parser.add_argument('--lambda-ei', type=float, default=0.5,
                        help='EI loss 权重')
    parser.add_argument('--device', default='cuda',
                        help='训练设备')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')

    # 加载 backbone
    logger.info(f'加载 backbone: {args.backbone_checkpoint}')
    from src.04_texture_synthesis.network import create_model
    backbone, _ = create_model('patchgan')
    checkpoint = torch.load(args.backbone_checkpoint, map_location=device, weights_only=False)
    backbone.load_state_dict(checkpoint['generator_state_dict'])
    logger.info('  ✓ Backbone 加载完成')

    # 构建 v3 模型
    logger.info('构建 CICI-FiLM v3 模型（SPADE-based）')
    model = ConditionedGeneratorV3(backbone, cond_dim=5, cond_emb_dim=512)
    model.freeze_backbone()
    model = model.to(device)

    # 准备数据集
    logger.info('准备数据集...')
    mapped_dir = Path(args.mapped_dir)
    
    # 训练集：copd_001-023
    train_ct = [mapped_dir / f'copd_{i:03d}' / f'copd_{i:03d}_warped.nii.gz'
                for i in range(1, 24)]
    train_mask = [mapped_dir / f'copd_{i:03d}' / f'copd_{i:03d}_warped_lesion.nii.gz'
                  for i in range(1, 24)]
    
    # 验证集：copd_024-029
    val_ct = [mapped_dir / f'copd_{i:03d}' / f'copd_{i:03d}_warped.nii.gz'
              for i in range(24, 30)]
    val_mask = [mapped_dir / f'copd_{i:03d}' / f'copd_{i:03d}_warped_lesion.nii.gz'
                for i in range(24, 30)]

    train_ds = LungPatchDataset(
        train_ct, train_mask,
        patch_size=(64, 64, 64),
        patches_per_volume=50,
        augment=True,
        patient_features_path=args.patient_features,
        use_condition=True
    )
    val_ds = LungPatchDataset(
        val_ct, val_mask,
        patch_size=(64, 64, 64),
        patches_per_volume=50,
        augment=False,
        patient_features_path=args.patient_features,
        use_condition=True
    )

    logger.info(f'  训练集: {len(train_ds)} patches')
    logger.info(f'  验证集: {len(val_ds)} patches')

    # 训练配置
    config = {
        'training': {
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'beta1': 0.5,
            'beta2': 0.999,
            'weight_decay': 1e-4,  # v3: 添加 weight decay
            'lr_scheduler': 'warmup_cosine',
            'warmup_epochs': 10,  # v3: 更长 warmup
            'loss_weights': {
                'reconstruction': 1.0,
                'perceptual': 0.1,
                'adversarial': 0.0,
                'hu_constraint': 0.5,
            },
            'enable_hu_constraint': True,
            'lambda_ei': args.lambda_ei,
            'lambda_contrast': 0.0,
            'ei_temperature': 10.0,
            # v3 新增
            'lambda_wasserstein': args.lambda_wasserstein,
            'lambda_hu_stats': args.lambda_hu_stats,
            'lambda_soft_histogram': args.lambda_soft_histogram,
            'use_v3_losses': True,  # 标记使用 v3 损失
            'gradient_clip_norm': 1.0,  # v3: 梯度裁剪
        }
    }

    # 创建 Trainer
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        generator=model,
        discriminator=None,
        train_config=config['training'],
        checkpoint_dir=output_dir,
        device=device,
        use_condition=True,
    )

    # 保存配置
    config_path = output_dir / 'config.json'
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    logger.info(f'配置已保存: {config_path}')

    # 开始训练
    logger.info('\n' + '=' * 60)
    logger.info('开始训练 CICI-FiLM v3')
    logger.info('=' * 60)
    logger.info(f'  Epochs: {args.epochs}')
    logger.info(f'  Learning Rate: {args.lr}')
    logger.info(f'  Patience: {args.patience}')
    logger.info(f'  λ_wasserstein: {args.lambda_wasserstein}')
    logger.info(f'  λ_hu_stats: {args.lambda_hu_stats}')
    logger.info(f'  λ_soft_histogram: {args.lambda_soft_histogram}')
    logger.info(f'  λ_ei: {args.lambda_ei}')
    logger.info('=' * 60 + '\n')

    trainer.train(
        train_loader=torch.utils.data.DataLoader(
            train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True
        ),
        val_loader=torch.utils.data.DataLoader(
            val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True
        ),
        epochs=args.epochs,
        patience=args.patience,
    )

    logger.info('\n训练完成！')


if __name__ == '__main__':
    main()

