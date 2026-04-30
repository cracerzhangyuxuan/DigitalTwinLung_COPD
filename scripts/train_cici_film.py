#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CICI-FiLM 训练脚本（阶段②）
============================

训练条件化生成器，冻结 backbone，只训练 FiLM 分支（2,434 参数）

使用方法：
  python scripts/train_cici_film.py \\
    --backbone-checkpoint checkpoints/patchgan/best.pth \\
    --mapped-dir data/03_mapped \\
    --patient-features data/patient_features.json \\
    --output-dir checkpoints/cici_film \\
    --epochs 25 --batch-size 4 \\
    --lambda-ei 0.5 --lambda-contrast 0.1 --ei-temperature 10.0
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import importlib
from src.utils.logger import get_logger

create_model = importlib.import_module('src.04_texture_synthesis.network').create_model
ConditionedGenerator = importlib.import_module('src.04_texture_synthesis.conditioned_model').ConditionedGenerator
LungPatchDataset = importlib.import_module('src.04_texture_synthesis.dataset').LungPatchDataset
Trainer = importlib.import_module('src.04_texture_synthesis.train').Trainer
from torch.utils.data import DataLoader

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='CICI-FiLM 训练（阶段②）')
    parser.add_argument('--backbone-checkpoint', required=True, help='预训练 backbone 检查点')
    parser.add_argument('--mapped-dir', required=True, help='已配准数据目录 (data/03_mapped)')
    parser.add_argument('--patient-features', required=True, help='患者特征 JSON')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率 (FiLM 分支建议 1e-5)')
    parser.add_argument('--device', default='cuda', help='设备')
    parser.add_argument('--lambda-ei', type=float, default=0.5, help='方案C: EI 感知 Loss 权重')
    parser.add_argument('--lambda-contrast', type=float, default=0.1, help='方案A: 条件响应 Loss 权重')
    parser.add_argument('--ei-temperature', type=float, default=10.0, help='方案C: soft EI sigmoid 温度')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping 耐心值')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 1. 加载预训练 backbone
    logger.info("=" * 60)
    logger.info("步骤 1: 加载预训练 backbone")
    logger.info("=" * 60)
    
    backbone, discriminator = create_model('patchgan')
    checkpoint = torch.load(args.backbone_checkpoint, map_location=device)
    backbone.load_state_dict(checkpoint['generator_state_dict'])
    logger.info(f"✓ 加载 backbone: {args.backbone_checkpoint}")
    
    # 2. 包裹为条件化生成器
    logger.info("\n" + "=" * 60)
    logger.info("步骤 2: 创建 CICI-FiLM 条件化生成器")
    logger.info("=" * 60)
    
    model = ConditionedGenerator(backbone, cond_dim=5, cond_emb_dim=64)
    model.freeze_backbone()  # 冻结主干，只训练 FiLM 分支
    # 确认可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"✓ 条件化生成器创建完成")
    logger.info(f"  可训练参数: {trainable:,} | 冻结参数: {frozen:,}")
    
    # 3. 创建数据加载器（从 mapped_dir 扫描子目录）
    logger.info("\n" + "=" * 60)
    logger.info("步骤 3: 创建数据加载器")
    logger.info("=" * 60)

    mapped_dir = Path(args.mapped_dir)
    TRAIN_CUTOFF = "copd_024"
    train_ct, train_mask, val_ct, val_mask = [], [], [], []
    for patient_dir in sorted(mapped_dir.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name == 'visualizations':
            continue
        warped_ct = patient_dir / f"{patient_dir.name}_warped.nii.gz"
        warped_mask = patient_dir / f"{patient_dir.name}_warped_lesion.nii.gz"
        if warped_ct.exists() and warped_mask.exists():
            if patient_dir.name < TRAIN_CUTOFF:
                train_ct.append(warped_ct); train_mask.append(warped_mask)
            else:
                val_ct.append(warped_ct); val_mask.append(warped_mask)
    if not val_ct:
        val_ct, val_mask = train_ct[-1:], train_mask[-1:]
    logger.info(f"  训练集: {len(train_ct)} 例, 验证集: {len(val_ct)} 例")

    train_dataset = LungPatchDataset(
        ct_paths=train_ct, mask_paths=train_mask, patch_size=(64, 64, 64),
        augment=True, patient_features_path=args.patient_features, use_condition=True
    )
    val_dataset = LungPatchDataset(
        ct_paths=val_ct, mask_paths=val_mask, patch_size=(64, 64, 64),
        augment=False, patient_features_path=args.patient_features, use_condition=True
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info(f"✓ 训练集: {len(train_dataset)} patches")
    logger.info(f"✓ 验证集: {len(val_dataset)} patches")
    
    # 4. 创建训练器
    logger.info("\n" + "=" * 60)
    logger.info("步骤 4: 创建训练器")
    logger.info("=" * 60)
    
    config = {
        'training': {
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'beta1': 0.5,
            'beta2': 0.999,
            'lr_scheduler': 'cosine',
            'warmup_epochs': 5,
            'loss_weights': {
                'reconstruction': 1.0,
                'perceptual': 0.1,
                'adversarial': 0.01,
                'hu_constraint': 0.5
            },
            'enable_hu_constraint': True,
            'lambda_ei': args.lambda_ei,
            'lambda_contrast': args.lambda_contrast,
            'ei_temperature': args.ei_temperature,
        }
    }
    
    trainer = Trainer(
        generator=model,
        discriminator=discriminator,
        config=config,
        device=args.device,
        use_condition=True  # 启用条件化训练
    )
    logger.info("✓ 训练器创建完成")
    
    # 5. 开始训练
    logger.info("\n" + "=" * 60)
    logger.info("步骤 5: 开始训练")
    logger.info("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_dir=output_dir,
        save_frequency=10,
        patience=args.patience
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)
    logger.info(f"最佳验证损失: {trainer.best_loss:.4f}")
    logger.info(f"检查点保存至: {output_dir}")


if __name__ == '__main__':
    main()

