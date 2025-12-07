#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COPD 数字孪生肺项目 - 主流程入口脚本

一键运行完整流水线：预处理 → 底座构建 → 配准 → AI融合 → 可视化
"""

import argparse
import sys
from pathlib import Path

import yaml

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger


def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_preprocessing(config: dict, logger):
    """阶段1：数据预处理"""
    logger.info("=" * 60)
    logger.info("阶段1：数据预处理")
    logger.info("=" * 60)
    
    from src.preprocessing import run_segmentation, clean_background, extract_emphysema
    
    # TODO: 实现预处理流程
    logger.info("运行 TotalSegmentator 分割...")
    # run_segmentation.main(config)
    
    logger.info("清洗背景...")
    # clean_background.main(config)
    
    logger.info("提取病灶 Mask...")
    # extract_emphysema.main(config)
    
    logger.info("阶段1 完成!")


def run_atlas_build(config: dict, logger):
    """阶段2：标准底座构建"""
    logger.info("=" * 60)
    logger.info("阶段2：标准底座构建")
    logger.info("=" * 60)
    
    from src.atlas_build import build_template_ants
    
    # TODO: 实现底座构建
    logger.info("构建标准模板 (预计耗时数小时)...")
    # build_template_ants.main(config)
    
    logger.info("阶段2 完成!")


def run_registration(config: dict, logger):
    """阶段3上：空间配准"""
    logger.info("=" * 60)
    logger.info("阶段3：病灶空间映射")
    logger.info("=" * 60)
    
    from src.registration import register_lesions
    
    # TODO: 实现配准流程
    logger.info("批量配准 COPD 病灶...")
    # register_lesions.main(config)
    
    logger.info("阶段3上 完成!")


def run_texture_synthesis(config: dict, logger):
    """阶段3下：AI纹理融合"""
    logger.info("=" * 60)
    logger.info("阶段3下：AI纹理融合训练")
    logger.info("=" * 60)
    
    from src.texture_synthesis import train
    
    # TODO: 实现AI训练
    logger.info("训练 Inpainting 模型...")
    # train.main(config)
    
    logger.info("阶段3下 完成!")


def run_visualization(config: dict, logger):
    """阶段4：3D可视化"""
    logger.info("=" * 60)
    logger.info("阶段4：3D可视化输出")
    logger.info("=" * 60)
    
    from src.visualization import static_render, dynamic_breath
    
    # TODO: 实现可视化
    logger.info("生成静态渲染图...")
    # static_render.main(config)
    
    logger.info("生成呼吸动画...")
    # dynamic_breath.main(config)
    
    logger.info("阶段4 完成!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="COPD 数字孪生肺项目 - 主流程入口"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)"
    )
    parser.add_argument(
        "--stage", "-s",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="仅运行指定阶段 (1-4)，默认运行全部"
    )
    parser.add_argument(
        "--skip-ai",
        action="store_true",
        help="跳过AI训练阶段"
    )
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    logger = setup_logger("pipeline", config['paths']['logs'])
    logger.info("COPD 数字孪生肺项目 - 流水线启动")
    logger.info(f"配置文件: {args.config}")
    
    try:
        if args.stage is None:
            # 运行全部阶段
            run_preprocessing(config, logger)
            run_atlas_build(config, logger)
            run_registration(config, logger)
            if not args.skip_ai:
                run_texture_synthesis(config, logger)
            run_visualization(config, logger)
        else:
            # 运行指定阶段
            stage_funcs = {
                1: run_preprocessing,
                2: run_atlas_build,
                3: run_registration,
                4: run_visualization,
            }
            stage_funcs[args.stage](config, logger)
        
        logger.info("=" * 60)
        logger.info("全部流程完成!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"流程执行失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

