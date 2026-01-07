#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 3B 推理脚本

使用训练好的 Inpainting 模型生成融合后的 COPD 数字孪生 CT

使用方法:
    python run_phase3b_inference.py [--checkpoint checkpoints/best.pth]

依赖:
    - Phase 3A 已完成（data/03_mapped/ 中有配准结果）
    - Phase 3B 训练已完成（checkpoints/best.pth 存在）
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
    parser = argparse.ArgumentParser(description='Phase 3B: AI 纹理融合推理')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型检查点路径（默认: checkpoints/best.pth）')
    parser.add_argument('--patient', type=str, default=None,
                        help='指定患者 ID（默认处理所有）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (cuda/cpu)')
    parser.add_argument('--no-smooth', action='store_true',
                        help='禁用边界平滑')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'phase3b_inference_{timestamp}.log'
    logger = setup_logger('phase3b_inference', log_file)
    
    logger.info("=" * 60)
    logger.info("  Phase 3B: AI 纹理融合推理")
    logger.info("=" * 60)
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    paths = config.get('paths', {})
    
    # 确定路径
    template_path = Path(paths.get('atlas', 'data/02_atlas')) / 'standard_template.nii.gz'
    mapped_dir = Path(paths.get('mapped', 'data/03_mapped'))
    output_dir = Path(paths.get('final_viz', 'data/04_final_viz'))
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else \
                      Path(paths.get('checkpoints', 'checkpoints')) / 'best.pth'
    
    # 检查文件
    if not checkpoint_path.exists():
        logger.error(f"模型检查点不存在: {checkpoint_path}")
        logger.error("请先运行 Phase 3B 训练: python run_phase3b_training.py")
        sys.exit(1)
    
    if not template_path.exists():
        logger.error(f"模板文件不存在: {template_path}")
        sys.exit(1)
    
    logger.info(f"模板: {template_path}")
    logger.info(f"检查点: {checkpoint_path}")
    logger.info(f"输出目录: {output_dir}")
    
    # 导入推理模块
    import importlib
    inference_module = importlib.import_module("src.04_texture_synthesis.inference_fuse")
    
    # 加载模型
    logger.info("加载模型...")
    model = inference_module.load_model(checkpoint_path, device=args.device)
    
    # 查找待处理的患者
    if args.patient:
        patient_dirs = [mapped_dir / args.patient]
    else:
        patient_dirs = [d for d in sorted(mapped_dir.iterdir()) 
                       if d.is_dir() and d.name != 'visualizations']
    
    logger.info(f"待处理: {len(patient_dirs)} 例")
    
    # 处理每个患者
    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0
    
    for i, patient_dir in enumerate(patient_dirs):
        patient_id = patient_dir.name
        mask_path = patient_dir / f"{patient_id}_warped_lesion.nii.gz"
        
        if not mask_path.exists():
            logger.warning(f"  [{i+1}/{len(patient_dirs)}] 跳过 {patient_id}: mask 不存在")
            continue
        
        logger.info(f"  [{i+1}/{len(patient_dirs)}] 处理 {patient_id}...")
        
        try:
            output_path = output_dir / f"{patient_id}_fused.nii.gz"
            inference_module.fuse_lesion(
                template_path=template_path,
                lesion_mask_path=mask_path,
                model=model,
                output_path=output_path,
                device=args.device,
                smooth_boundary_width=0 if args.no_smooth else 3
            )
            logger.info(f"    ✓ 完成: {output_path.name}")
            success_count += 1
        except Exception as e:
            logger.error(f"    ✗ 失败: {e}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"推理完成: {success_count}/{len(patient_dirs)} 成功")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

