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
create_dataloader = importlib.import_module('src.04_texture_synthesis.dataset').create_dataloader
Trainer = importlib.import_module('src.04_texture_synthesis.train').Trainer

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='CICI-FiLM 训练（阶段②）')
    parser.add_argument('--backbone-checkpoint', required=True, help='预训练 backbone 检查点')
    parser.add_argument('--ct-dir', required=True, help='COPD CT 目录')
    parser.add_argument('--mask-dir', required=True, help='病灶 mask 目录')
    parser.add_argument('--patient-features', required=True, help='患者特征 JSON')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4, help='批大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--device', default='cuda', help='设备')
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
    logger.info("✓ 条件化生成器创建完成")
    
    # 3. 创建数据加载器
    logger.info("\n" + "=" * 60)
    logger.info("步骤 3: 创建数据加载器")
    logger.info("=" * 60)
    
    train_loader, val_loader = create_dataloader(
        ct_dir=args.ct_dir,
        mask_dir=args.mask_dir,
        batch_size=args.batch_size,
        patient_features_path=args.patient_features,
        use_condition=True  # 启用条件向量
    )
    logger.info(f"✓ 训练集: {len(train_loader.dataset)} patches")
    logger.info(f"✓ 验证集: {len(val_loader.dataset)} patches")
    
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
            'enable_hu_constraint': True
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
        save_frequency=10
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)
    logger.info(f"最佳验证损失: {trainer.best_loss:.4f}")
    logger.info(f"检查点保存至: {output_dir}")


if __name__ == '__main__':
    main()

