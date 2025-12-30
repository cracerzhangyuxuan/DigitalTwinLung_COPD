#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 3 端到端流水线（病理映射与 AI 融合）

功能：
    1. 环境检查：验证 ANTsPy、PyTorch 可用性
    2. 数据验证：检查 COPD 数据和标准底座完整性
    3. 空间映射：将 COPD 病灶配准到标准底座空间
    4. 可视化验证：生成映射结果对比图
    5. [预留] AI 纹理融合：训练 Inpainting 模型

使用方法：
    # 完整流水线
    python run_phase3_pipeline.py

    # 快速测试（仅处理 3 例）
    python run_phase3_pipeline.py --quick-test

    # 跳过配准（使用已有结果）
    python run_phase3_pipeline.py --skip-registration

    # 仅执行可视化
    python run_phase3_pipeline.py --viz-only

    # 限制处理数量
    python run_phase3_pipeline.py --limit 5

    # 后台运行
    nohup python run_phase3_pipeline.py > logs/phase3.log 2>&1 &

作者：DigitalTwinLung COPD Project
日期：2025-12-30
"""

import os
import sys
import argparse
import importlib
import time
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import logging

import yaml
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# =============================================================================
# 依赖检查
# =============================================================================

def check_antspy() -> Tuple[bool, str]:
    """检查 ANTsPy 是否可用"""
    try:
        import ants
        return True, f"ANTsPy 可用: {getattr(ants, '__version__', 'unknown')}"
    except ImportError:
        return False, "ANTsPy 未安装"


def check_pytorch() -> Tuple[bool, str]:
    """检查 PyTorch 是否可用"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            return True, f"PyTorch 可用: {torch.__version__}, GPU: {gpu_name}"
        else:
            return True, f"PyTorch 可用: {torch.__version__} (CPU only)"
    except ImportError:
        return False, "PyTorch 未安装"


# =============================================================================
# 日志配置
# =============================================================================

def setup_logging(log_dir: Path = None) -> logging.Logger:
    """配置日志"""
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"phase3_pipeline_{timestamp}.log"

    # 创建 logger
    logger = logging.getLogger("Phase3Pipeline")
    logger.setLevel(logging.DEBUG)

    # 清除已有的处理器
    if logger.handlers:
        logger.handlers.clear()

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"日志文件: {log_file}")

    return logger


# =============================================================================
# 流水线步骤
# =============================================================================

def run_environment_check(logger: logging.Logger) -> bool:
    """检查运行环境"""
    logger.info("")
    logger.info("[Step 1] 环境检查")
    logger.info("-" * 40)

    all_ok = True

    # ANTsPy
    ants_ok, ants_msg = check_antspy()
    status = "✓" if ants_ok else "✗"
    logger.info(f"  {status} {ants_msg}")
    if not ants_ok:
        all_ok = False

    # PyTorch
    torch_ok, torch_msg = check_pytorch()
    status = "✓" if torch_ok else "⚠"
    logger.info(f"  {status} {torch_msg}")

    return all_ok


def run_data_validation(
    config: dict,
    logger: logging.Logger,
    quick_test: bool = False
) -> Tuple[bool, Dict]:
    """验证数据完整性"""
    logger.info("")
    logger.info("[Step 2] 数据验证")
    logger.info("-" * 40)

    atlas_dir = Path(config['paths']['atlas'])
    cleaned_dir = Path(config['paths']['cleaned_data'])
    copd_ct_dir = cleaned_dir / 'copd_clean'
    copd_lesion_dir = cleaned_dir / 'copd_emphysema'

    stats = {
        'template_exists': False,
        'mask_exists': False,
        'copd_ct_count': 0,
        'copd_lesion_count': 0
    }

    # 检查标准底座
    template_file = atlas_dir / 'standard_template.nii.gz'
    mask_file = atlas_dir / 'standard_mask.nii.gz'

    stats['template_exists'] = template_file.exists()
    stats['mask_exists'] = mask_file.exists()

    status = "✓" if stats['template_exists'] else "✗"
    logger.info(f"  {status} 标准底座: {template_file}")
    status = "✓" if stats['mask_exists'] else "✗"
    logger.info(f"  {status} 底座 Mask: {mask_file}")

    if not stats['template_exists']:
        logger.error("  ❌ 标准底座不存在！请先运行 Phase 2")
        return False, stats

    # 检查 COPD 数据
    if copd_ct_dir.exists():
        copd_cts = list(copd_ct_dir.glob("*.nii.gz"))
        stats['copd_ct_count'] = len(copd_cts)
    if copd_lesion_dir.exists():
        copd_lesions = list(copd_lesion_dir.glob("*.nii.gz"))
        stats['copd_lesion_count'] = len(copd_lesions)

    logger.info(f"  ℹ COPD CT: {stats['copd_ct_count']} 例 ({copd_ct_dir})")
    logger.info(f"  ℹ 病灶 Mask: {stats['copd_lesion_count']} 例 ({copd_lesion_dir})")

    if stats['copd_ct_count'] == 0:
        logger.warning("  ⚠ COPD 数据为空！请先准备数据")
        logger.warning(f"    将 COPD CT 放入: {copd_ct_dir}")
        logger.warning(f"    将病灶 Mask 放入: {copd_lesion_dir}")
        return False, stats

    return True, stats


def run_spatial_mapping(
    config: dict,
    logger: logging.Logger,
    quick_test: bool = False,
    limit: int = None
) -> Tuple[bool, Dict]:
    """执行空间映射（配准）"""
    logger.info("")
    logger.info("[Step 3] 空间映射 (Spatial Mapping)")
    logger.info("-" * 40)

    atlas_dir = Path(config['paths']['atlas'])
    cleaned_dir = Path(config['paths']['cleaned_data'])
    output_dir = Path(config['paths'].get('mapped', 'data/03_mapped'))

    template_path = atlas_dir / 'standard_template.nii.gz'
    template_mask_path = atlas_dir / 'standard_mask.nii.gz'
    copd_ct_dir = cleaned_dir / 'copd_clean'
    copd_lesion_dir = cleaned_dir / 'copd_emphysema'

    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取 COPD 文件列表
    copd_cts = sorted(copd_ct_dir.glob("*.nii.gz"))

    if limit:
        copd_cts = copd_cts[:limit]
    if quick_test:
        copd_cts = copd_cts[:3]

    logger.info(f"  待处理: {len(copd_cts)} 例")

    if len(copd_cts) == 0:
        logger.warning("  没有找到 COPD CT 文件")
        return False, {'processed': 0, 'failed': 0}

    # 导入配准模块
    try:
        register_lesions = importlib.import_module("src.03_registration.register_lesions")
    except ImportError as e:
        logger.error(f"  导入配准模块失败: {e}")
        return False, {'processed': 0, 'failed': 0}

    # 获取配准配置
    reg_config = config.get('registration', {}).get('lesion_registration', {})

    results = {'processed': 0, 'failed': 0, 'details': []}

    for i, ct_path in enumerate(copd_cts):
        patient_id = ct_path.stem.replace('.nii', '').replace('_clean', '')
        patient_output_dir = output_dir / patient_id

        # 查找对应的病灶 mask
        lesion_patterns = [
            copd_lesion_dir / f"{patient_id}_emphysema.nii.gz",
            copd_lesion_dir / f"{patient_id}_lesion.nii.gz",
            copd_lesion_dir / f"{patient_id}.nii.gz",
        ]
        lesion_path = None
        for p in lesion_patterns:
            if p.exists():
                lesion_path = p
                break

        if lesion_path is None:
            logger.warning(f"  [{i+1}/{len(copd_cts)}] 未找到病灶 Mask: {patient_id}")
            results['failed'] += 1
            continue

        logger.info(f"  [{i+1}/{len(copd_cts)}] 配准: {patient_id}")
        start_time = time.time()

        try:
            # 配准 CT
            reg_outputs = register_lesions.register_to_template(
                moving_image_path=ct_path,
                template_path=template_path,
                output_dir=patient_output_dir,
                type_of_transform=reg_config.get('type_of_transform', 'SyNRA'),
                reg_iterations=tuple(reg_config.get('reg_iterations', [20, 10, 0])),
            )

            # 扭曲病灶 mask
            transform_paths = [
                reg_outputs.get(f'transform_{j}')
                for j in range(2)
                if reg_outputs.get(f'transform_{j}') is not None
            ]

            warped_mask_path = patient_output_dir / f"{patient_id}_warped_lesion.nii.gz"
            register_lesions.warp_mask(
                mask_path=lesion_path,
                template_path=template_path,
                transform_paths=transform_paths,
                output_path=warped_mask_path,
                template_mask_path=template_mask_path if template_mask_path.exists() else None
            )

            elapsed = time.time() - start_time
            logger.info(f"    ✓ 完成 (耗时 {elapsed/60:.1f} 分钟)")
            results['processed'] += 1
            results['details'].append({
                'patient_id': patient_id,
                'warped_ct': str(reg_outputs['warped_image']),
                'warped_lesion': str(warped_mask_path),
                'elapsed_minutes': elapsed / 60
            })

        except Exception as e:
            logger.error(f"    ✗ 失败: {e}")
            results['failed'] += 1

    logger.info("")
    logger.info(f"  配准完成: {results['processed']}/{len(copd_cts)} 成功")
    if results['failed'] > 0:
        logger.warning(f"  失败: {results['failed']} 例")

    return results['processed'] > 0, results


def run_visualization(
    config: dict,
    logger: logging.Logger,
    limit: int = None
) -> bool:
    """生成映射结果可视化"""
    logger.info("")
    logger.info("[Step 4] 可视化验证")
    logger.info("-" * 40)

    output_dir = Path(config['paths'].get('mapped', 'data/03_mapped'))
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 查找已处理的患者
    patient_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name != 'visualizations']

    if limit:
        patient_dirs = patient_dirs[:limit]

    if len(patient_dirs) == 0:
        logger.warning("  没有找到映射结果")
        return False

    logger.info(f"  待可视化: {len(patient_dirs)} 例")

    # 尝试导入可视化模块
    try:
        viz_module = importlib.import_module("src.05_visualization.static_render")
        has_viz = True
    except ImportError:
        logger.warning("  无法导入可视化模块，跳过可视化")
        has_viz = False

    if has_viz:
        # 简单的统计输出
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            warped_lesion = patient_dir / f"{patient_id}_warped_lesion.nii.gz"
            if warped_lesion.exists():
                logger.info(f"  ✓ {patient_id}: {warped_lesion.name}")
            else:
                logger.warning(f"  ⚠ {patient_id}: 缺少 warped_lesion")

    logger.info("")
    logger.info(f"  可视化目录: {viz_dir}")

    return True


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 流水线: 病理映射与 AI 融合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整流水线
  python run_phase3_pipeline.py

  # 快速测试（仅处理 3 例）
  python run_phase3_pipeline.py --quick-test

  # 跳过配准（使用已有结果）
  python run_phase3_pipeline.py --skip-registration

  # 仅执行可视化
  python run_phase3_pipeline.py --viz-only

  # 限制处理数量
  python run_phase3_pipeline.py --limit 5

  # 后台运行
  nohup python run_phase3_pipeline.py > logs/phase3.log 2>&1 &
        """
    )

    # 步骤控制参数
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument(
        '--skip-registration', action='store_true',
        help='跳过配准步骤（使用已有结果）'
    )
    step_group.add_argument(
        '--viz-only', action='store_true',
        help='仅执行可视化（需要已有映射结果）'
    )

    # 其他参数
    parser.add_argument(
        '--quick-test', action='store_true',
        help='快速测试模式（仅处理 3 例）'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='限制处理数量'
    )
    parser.add_argument(
        '--input-dir', type=str, default=None,
        help='COPD CT 目录（默认: data/01_cleaned/copd_clean/）'
    )
    parser.add_argument(
        '--lesion-dir', type=str, default=None,
        help='病灶 Mask 目录（默认: data/01_cleaned/copd_emphysema/）'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='输出目录（默认: data/03_mapped/）'
    )
    parser.add_argument(
        '--atlas-template', type=str, default=None,
        help='标准底座路径（默认: data/02_atlas/standard_template.nii.gz）'
    )
    parser.add_argument(
        '--check-only', action='store_true',
        help='仅检查环境和数据，不执行流水线'
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='配置文件路径（默认: config.yaml）'
    )

    args = parser.parse_args()

    # 设置日志
    logger = setup_logging()

    # 加载配置
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"无法加载配置文件: {e}")
        sys.exit(1)

    # 覆盖配置（如果命令行指定）
    if args.output_dir:
        config['paths']['mapped'] = args.output_dir

    # 记录开始时间
    pipeline_start = time.time()

    # =========================================================================
    # 打印 Banner
    # =========================================================================
    logger.info("=" * 60)
    logger.info("  Phase 3: 病理映射与 AI 融合流水线")
    logger.info("=" * 60)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"配置文件: {args.config}")

    atlas_dir = Path(config['paths']['atlas'])
    output_dir = Path(config['paths'].get('mapped', 'data/03_mapped'))

    logger.info(f"标准底座: {atlas_dir / 'standard_template.nii.gz'}")
    logger.info(f"输出目录: {output_dir}")

    if args.quick_test:
        logger.info("模式: 快速测试 (3 例)")
    elif args.limit:
        logger.info(f"处理限制: {args.limit} 例")

    # =========================================================================
    # --viz-only 模式
    # =========================================================================
    if args.viz_only:
        logger.info("")
        logger.info("模式: 仅可视化 (--viz-only)")
        run_visualization(config, logger, limit=args.limit)

        elapsed = time.time() - pipeline_start
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"可视化完成，耗时: {elapsed:.1f} 秒")
        logger.info("=" * 60)
        sys.exit(0)

    # =========================================================================
    # 正常流水线
    # =========================================================================

    # Step 1: 环境检查
    env_ok = run_environment_check(logger)

    if args.check_only:
        # Step 2: 数据验证
        run_data_validation(config, logger, args.quick_test)
        logger.info("")
        logger.info("仅检查模式，退出")
        sys.exit(0 if env_ok else 1)

    # Step 2: 数据验证
    data_ok, data_stats = run_data_validation(config, logger, args.quick_test)
    if not data_ok:
        logger.error("")
        logger.error("数据验证失败，退出")
        logger.info("")
        logger.info("请准备 COPD 数据后重新运行:")
        logger.info(f"  1. 将 COPD CT 放入: data/01_cleaned/copd_clean/")
        logger.info(f"  2. 将病灶 Mask 放入: data/01_cleaned/copd_emphysema/")
        logger.info(f"  3. 重新运行: python run_phase3_pipeline.py")
        sys.exit(1)

    # Step 3: 空间映射
    if not args.skip_registration:
        mapping_ok, mapping_results = run_spatial_mapping(
            config, logger,
            quick_test=args.quick_test,
            limit=args.limit
        )
        if not mapping_ok:
            logger.error("空间映射失败")
            sys.exit(1)
    else:
        logger.info("")
        logger.info("[Step 3] 跳过空间映射 (--skip-registration)")

    # Step 4: 可视化
    run_visualization(config, logger, limit=args.limit)

    # =========================================================================
    # 总结
    # =========================================================================
    pipeline_elapsed = time.time() - pipeline_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 3A 流水线执行完成")
    logger.info("=" * 60)
    logger.info(f"总耗时: {pipeline_elapsed/60:.1f} 分钟")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 输出结果摘要
    logger.info("")
    logger.info("输出目录:")
    logger.info(f"  {output_dir}")

    # 统计结果
    patient_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name != 'visualizations']
    logger.info(f"  已处理患者: {len(patient_dirs)} 例")

    logger.info("")
    logger.info("下一步:")
    logger.info("  1. 检查 data/03_mapped/ 中的映射结果")
    logger.info("  2. 使用 3D Slicer 验证病灶位置是否正确")
    logger.info("  3. 准备 Phase 3B: AI 纹理融合训练")
    logger.info("     - 编写 src/04_texture_synthesis/dataset.py")
    logger.info("     - 编写 src/04_texture_synthesis/train.py")


if __name__ == "__main__":
    main()

