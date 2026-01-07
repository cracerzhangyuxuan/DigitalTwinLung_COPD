#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 3B 训练脚本

训练 Inpainting 网络，用于 COPD 病灶纹理合成

使用方法:
    python run_phase3b_training.py [--epochs 100] [--batch-size 4] [--no-gan]

依赖:
    - Phase 3A 已完成（data/03_mapped/ 中有配准结果）
    - PyTorch 已安装
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import yaml

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Phase 3B: AI 纹理融合训练')

    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--model-type', type=str, default=None,
                        choices=['unet', 'partial_conv', 'patchgan'],
                        help='模型类型: unet(基线), partial_conv(进阶), patchgan(高级)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数（覆盖配置文件）')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批次大小（覆盖配置文件）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率（覆盖配置文件）')
    parser.add_argument('--no-gan', action='store_true',
                        help='不使用 GAN（等同于 --model-type unet）')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (cuda/cpu)')

    return parser.parse_args()


def check_prerequisites(config: dict, logger) -> bool:
    """检查前置条件"""
    logger.info("检查前置条件...")
    
    # 检查 PyTorch
    try:
        import torch
        logger.info(f"  ✓ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"  ✓ CUDA: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("  ⚠ CUDA 不可用，将使用 CPU 训练")
    except ImportError:
        logger.error("  ✗ PyTorch 未安装")
        return False
    
    # 检查 Phase 3A 输出
    paths = config.get('paths', {})
    mapped_dir = Path(paths.get('mapped', 'data/03_mapped'))
    
    if not mapped_dir.exists():
        logger.error(f"  ✗ 配准输出目录不存在: {mapped_dir}")
        return False
    
    # 统计已配准的数据
    patient_count = 0
    for patient_dir in mapped_dir.iterdir():
        if not patient_dir.is_dir() or patient_dir.name == 'visualizations':
            continue
        warped_ct = patient_dir / f"{patient_dir.name}_warped.nii.gz"
        warped_mask = patient_dir / f"{patient_dir.name}_warped_lesion.nii.gz"
        if warped_ct.exists() and warped_mask.exists():
            patient_count += 1
    
    if patient_count == 0:
        logger.error("  ✗ 未找到已配准的数据，请先运行 Phase 3A")
        return False
    
    logger.info(f"  ✓ 已配准数据: {patient_count} 例")
    
    return True


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'phase3b_training_{timestamp}.log'
    logger = setup_logger('phase3b', log_file)
    
    logger.info("=" * 60)
    logger.info("  Phase 3B: AI 纹理融合训练")
    logger.info("=" * 60)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 命令行参数覆盖配置
    if args.model_type:
        config['training']['model_type'] = args.model_type
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.no_gan:
        # --no-gan 等同于使用 unet 模型
        config['training']['model_type'] = 'unet'

    # 显示模型类型
    model_type = config['training'].get('model_type', 'unet')
    model_names = {
        'unet': '基线方案 (3D U-Net)',
        'partial_conv': '进阶方案 (Partial Conv)',
        'patchgan': '高级方案 (PatchGAN)'
    }
    logger.info(f"模型类型: {model_names.get(model_type, model_type)}")
    
    # 检查前置条件
    if not check_prerequisites(config, logger):
        logger.error("前置条件检查失败，退出")
        sys.exit(1)
    
    # 导入训练模块
    logger.info("")
    logger.info("初始化训练...")

    import importlib
    train_module = importlib.import_module("src.04_texture_synthesis.train")

    # 开始训练
    try:
        train_module.main(config)
        logger.info("")
        logger.info("=" * 60)
        logger.info("训练完成!")
        logger.info("=" * 60)
    except KeyboardInterrupt:
        logger.warning("训练被用户中断")
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise


if __name__ == '__main__':
    main()

