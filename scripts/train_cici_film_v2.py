#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CICI-FiLM v2 训练脚本
======================

使用多层 FiLM + 增强条件 loss 训练条件化生成器。

改进点（相比 v1）：
  - 多层 FiLM 注入（bottleneck + last_decoder + output）
  - 更强的条件编码器（5→128→256，残差连接）
  - HU Statistics Loss + Soft EI Loss 替代失效的 Contrast Loss
  - 可训练参数 ~315K（v1 仅 2,434）

使用方法：
  python scripts/train_cici_film_v2.py \
    --backbone-checkpoint checkpoints/patchgan/best.pth \
    --mapped-dir data/03_mapped \
    --patient-features data/patient_features.json \
    --output-dir checkpoints/cici_film_v2 \
    --epochs 80 --lr 1e-4 --patience 25 --device cuda
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import importlib
from src.utils.logger import get_logger

create_model = importlib.import_module('src.04_texture_synthesis.network').create_model
ConditionedGeneratorV2 = importlib.import_module('src.04_texture_synthesis.conditioned_model_v2').ConditionedGeneratorV2
ConditionLosses = importlib.import_module('src.04_texture_synthesis.condition_losses').ConditionLosses
LungPatchDataset = importlib.import_module('src.04_texture_synthesis.dataset').LungPatchDataset
Trainer = importlib.import_module('src.04_texture_synthesis.train').Trainer
from torch.utils.data import DataLoader

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='CICI-FiLM v2 训练')
    parser.add_argument('--backbone-checkpoint', required=True)
    parser.add_argument('--mapped-dir', required=True)
    parser.add_argument('--patient-features', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--device', default='cuda')
    # 条件 loss 权重
    parser.add_argument('--lambda-hu-stats', type=float, default=1.0)
    parser.add_argument('--lambda-ei', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 1. 加载 backbone
    logger.info("=" * 60)
    logger.info("步骤 1: 加载预训练 backbone")
    backbone, _ = create_model('patchgan')
    ckpt = torch.load(args.backbone_checkpoint, map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt['generator_state_dict'])
    logger.info(f"✓ backbone: {args.backbone_checkpoint}")

    # 2. 创建 v2 条件化生成器
    logger.info("=" * 60)
    logger.info("步骤 2: 创建 CICI-FiLM v2 条件化生成器")
    model = ConditionedGeneratorV2(backbone, cond_dim=5, cond_emb_dim=256)
    model.freeze_backbone()

    # 3. 数据加载
    logger.info("=" * 60)
    logger.info("步骤 3: 创建数据加载器")
    mapped_dir = Path(args.mapped_dir)
    VAL_IDS = {"copd_024", "copd_025", "copd_026", "copd_027", "copd_028", "copd_029"}
    train_ct, train_mask, val_ct, val_mask = [], [], [], []
    for d in sorted(mapped_dir.iterdir()):
        if not d.is_dir() or d.name == 'visualizations':
            continue
        ct = d / f"{d.name}_warped.nii.gz"
        mk = d / f"{d.name}_warped_lesion.nii.gz"
        if ct.exists() and mk.exists():
            if d.name in VAL_IDS:
                val_ct.append(ct); val_mask.append(mk)
            else:
                train_ct.append(ct); train_mask.append(mk)
    if not val_ct:
        val_ct, val_mask = train_ct[-1:], train_mask[-1:]
    logger.info(f"  训练: {len(train_ct)} 例, 验证: {len(val_ct)} 例")

    train_ds = LungPatchDataset(train_ct, train_mask, (64,64,64), True, args.patient_features, True)
    val_ds = LungPatchDataset(val_ct, val_mask, (64,64,64), False, args.patient_features, True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info(f"✓ 训练: {len(train_ds)} patches, 验证: {len(val_ds)} patches")

    # 4. 训练器（复用 Trainer，通过 config 控制）
    logger.info("=" * 60)
    logger.info("步骤 4: 创建训练器")
    config = {
        'training': {
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'beta1': 0.5, 'beta2': 0.999,
            'lr_scheduler': 'warmup_cosine',
            'warmup_epochs': 5,
            'loss_weights': {
                'reconstruction': 1.0, 'perceptual': 0.1,
                'adversarial': 0.0, 'hu_constraint': 0.5,
            },
            'enable_hu_constraint': True,
            'lambda_ei': args.lambda_ei,
            'lambda_contrast': 0.0,  # v2 不用 contrast loss
            'ei_temperature': 10.0,
            # v2 新增
            'lambda_hu_stats': args.lambda_hu_stats,
        }
    }
    trainer = Trainer(model, None, config, args.device, use_condition=True)
    logger.info("✓ 训练器创建完成")

    # 5. 训练
    logger.info("=" * 60)
    logger.info("步骤 5: 开始训练")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.train(train_loader, val_loader, args.epochs, out_dir, 10, args.patience)

    logger.info("=" * 60)
    logger.info("训练完成！")
    logger.info(f"最佳验证损失: {trainer.best_loss:.4f}")
    logger.info(f"检查点: {out_dir}")


if __name__ == '__main__':
    main()

