#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2 端到端流水线（数字孪生底座构建）

功能：
    1. 环境检查：验证 GPU、TotalSegmentator、ANTsPy 可用性
    2. 数据验证：检查原始数据完整性
    3. 肺部分割：使用 TotalSegmentator GPU 加速分割（含气管树和5肺叶标签）
    4. 质量检查：验证分割结果
    5. Atlas 构建：构建标准肺模板（含气管树模板）
    6. 结果验证：检查 Atlas 质量

使用方法：
    # 完整流水线
    python run_phase2_pipeline.py

    # 快速测试（仅处理 3 例）
    python run_phase2_pipeline.py --quick-test

    # 跳过分割（使用已有结果）
    python run_phase2_pipeline.py --skip-segmentation

    # 仅运行分割（Step 1）
    python run_phase2_pipeline.py --step1-only --device gpu

    # 仅生成气管树模板（Step 2，需要已有模板）
    python run_phase2_pipeline.py --step2-only

    # 限制处理数量（用于测试）
    python run_phase2_pipeline.py --limit 3

    # 指定 GPU
    python run_phase2_pipeline.py --device gpu:0

    # 后台运行（Linux/服务器）
    nohup python run_phase2_pipeline.py > logs/phase2_pipeline.log 2>&1 &

作者：DigitalTwinLung COPD Project
日期：2025-12-09
更新：2025-12-22 - 整合气管树和5肺叶标签分割功能
"""

import os
import sys
import argparse
import importlib
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import logging

import yaml
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入公共模块
run_seg_module = importlib.import_module("src.01_preprocessing.run_segmentation")
check_gpu = run_seg_module.check_gpu_available
check_totalsegmentator = run_seg_module.check_totalsegmentator_available

# ANTsPy 检查函数
def check_antspy():
    """检查 ANTsPy 是否可用"""
    try:
        import ants
        return True, f"ANTsPy 可用: {getattr(ants, '__version__', 'unknown')}"
    except ImportError:
        return False, "ANTsPy 未安装"

# =============================================================================
# 日志配置
# =============================================================================

def setup_logging(log_dir: Path = None) -> logging.Logger:
    """配置日志"""
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"phase2_pipeline_{timestamp}.log"

    # 创建 logger
    logger = logging.getLogger("Phase2Pipeline")
    logger.setLevel(logging.DEBUG)

    # 清除已有的处理器（避免重复添加）
    if logger.handlers:
        logger.handlers.clear()

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"日志文件: {log_file}")
    return logger


# =============================================================================
# 环境检查（使用公共模块中的函数）
# =============================================================================

def run_environment_check(logger: logging.Logger) -> bool:
    """运行环境检查"""
    logger.info("=" * 60)
    logger.info("步骤 1: 环境检查")
    logger.info("=" * 60)
    
    checks = [
        ("GPU", check_gpu),
        ("TotalSegmentator", check_totalsegmentator),
        ("ANTsPy", check_antspy),
    ]
    
    all_ok = True
    for name, check_func in checks:
        ok, msg = check_func()
        status = "✓" if ok else "✗"
        logger.info(f"  {status} {name}: {msg}")
        if not ok and name in ["GPU", "TotalSegmentator"]:
            all_ok = False
    
    if all_ok:
        logger.info("环境检查通过")
    else:
        logger.warning("环境检查未完全通过，将使用备选方案")
    
    return all_ok


# =============================================================================
# 数据验证
# =============================================================================

def run_data_validation(config: dict, logger: logging.Logger, 
                        quick_test: bool = False) -> Tuple[bool, int]:
    """验证输入数据"""
    logger.info("=" * 60)
    logger.info("步骤 2: 数据验证")
    logger.info("=" * 60)
    
    raw_dir = Path(config['paths']['raw_data'])
    normal_dir = raw_dir / 'normal'
    
    if not normal_dir.exists():
        logger.error(f"数据目录不存在: {normal_dir}")
        return False, 0
    
    nifti_files = sorted(list(normal_dir.glob("*.nii.gz")))
    file_count = len(nifti_files)
    
    logger.info(f"  数据目录: {normal_dir}")
    logger.info(f"  NIfTI 文件数: {file_count}")
    
    if file_count == 0:
        logger.error("未找到 NIfTI 文件")
        return False, 0
    
    if quick_test:
        logger.info(f"  快速测试模式: 仅处理前 3 例")
        file_count = min(3, file_count)
    
    # 检查文件大小
    total_size = sum(f.stat().st_size for f in nifti_files[:file_count])
    logger.info(f"  总数据大小: {total_size / 1e9:.2f} GB")
    
    logger.info("数据验证通过")
    return True, file_count


# =============================================================================
# 肺部分割（含气管树和5肺叶标签）
# =============================================================================

def run_segmentation(config: dict, logger: logging.Logger,
                     device: str = "gpu", quick_test: bool = False,
                     force: bool = False, limit: int = None) -> bool:
    """
    运行肺部分割（使用 TotalSegmentator）

    输出文件：
        - *_mask.nii.gz: 肺部二值 mask
        - *_clean.nii.gz: 背景清理后的 CT
        - *_trachea_mask.nii.gz: 气管树 mask
        - *_lung_lobes_labeled.nii.gz: 5肺叶标签 mask (值 1-5)

    Args:
        config: 配置字典
        logger: 日志记录器
        device: 计算设备 ("gpu", "cpu", "cuda:0" 等)
        quick_test: 是否快速测试模式（处理 3 例）
        force: 是否强制覆盖已有文件
        limit: 限制处理数量

    Returns:
        bool: 是否全部成功
    """
    logger.info("=" * 60)
    logger.info("步骤 3: 肺部分割（含气管树和5肺叶标签）")
    logger.info("=" * 60)

    # 导入分割模块
    run_seg_module = importlib.import_module("src.01_preprocessing.run_segmentation")
    run_totalsegmentator_batch = run_seg_module.run_totalsegmentator_batch

    raw_dir = Path(config['paths']['raw_data'])
    cleaned_dir = Path(config['paths']['cleaned_data'])

    normal_input = raw_dir / 'normal'
    mask_output = cleaned_dir / 'normal_mask'
    clean_output = cleaned_dir / 'normal_clean'

    # 创建输出目录
    mask_output.mkdir(parents=True, exist_ok=True)
    clean_output.mkdir(parents=True, exist_ok=True)

    # 获取输入文件数量
    nifti_files = sorted(list(normal_input.glob("*.nii.gz")))
    total_count = len(nifti_files)

    # 确定处理数量
    process_limit = None
    if quick_test:
        process_limit = 3
    elif limit:
        process_limit = limit

    logger.info(f"  输入目录: {normal_input}")
    logger.info(f"  Mask 输出: {mask_output}")
    logger.info(f"  Clean 输出: {clean_output}")
    logger.info(f"  可用文件: {total_count}")
    logger.info(f"  处理限制: {process_limit if process_limit else '全部'}")
    logger.info(f"  设备: {device}")
    logger.info(f"  覆盖模式: {'是' if force else '否'}")

    if force:
        logger.warning("  ⚠️ 覆盖模式已启用，将重新处理所有文件")

    # 批量分割
    start_time = time.time()

    try:
        results = run_totalsegmentator_batch(
            input_dir=normal_input,
            mask_output_dir=mask_output,
            clean_output_dir=clean_output,
            device=device,
            fast=False,
            skip_existing=not force,  # force=True 时不跳过
            limit=process_limit,
            extract_trachea=True,      # 启用气管树分割
            create_labeled_lobes=True  # 启用5肺叶标签
        )

        elapsed = time.time() - start_time

        success_count = len(results.get('success', []))
        fail_count = len(results.get('failed', []))
        skip_count = len(results.get('skipped', []))

        logger.info("")
        logger.info("=" * 50)
        logger.info("分割完成:")
        logger.info(f"  成功: {success_count}")
        logger.info(f"  失败: {fail_count}")
        logger.info(f"  跳过: {skip_count}")
        logger.info(f"  耗时: {elapsed/60:.1f} 分钟")
        logger.info("=" * 50)

        if results.get('failed'):
            logger.warning(f"失败的文件: {results['failed']}")

        return fail_count == 0

    except Exception as e:
        logger.error(f"分割失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# =============================================================================
# 质量检查
# =============================================================================

def run_quality_check(config: dict, logger: logging.Logger) -> Tuple[bool, int]:
    """检查分割质量"""
    logger.info("=" * 60)
    logger.info("步骤 4: 质量检查")
    logger.info("=" * 60)
    
    cleaned_dir = Path(config['paths']['cleaned_data'])
    clean_dir = cleaned_dir / 'normal_clean'
    mask_dir = cleaned_dir / 'normal_mask'
    
    clean_files = sorted(list(clean_dir.glob("*.nii.gz")))
    mask_files = sorted(list(mask_dir.glob("*.nii.gz")))
    
    logger.info(f"  Clean 文件: {len(clean_files)}")
    logger.info(f"  Mask 文件: {len(mask_files)}")
    
    if len(clean_files) == 0:
        logger.error("未找到分割结果")
        return False, 0
    
    # 检查配对
    if len(clean_files) != len(mask_files):
        logger.warning(f"  文件数不匹配: clean={len(clean_files)}, mask={len(mask_files)}")
    
    # 检查文件大小
    valid_count = 0
    for clean_file in clean_files:
        size_mb = clean_file.stat().st_size / 1e6
        if size_mb > 10:  # 至少 10MB
            valid_count += 1
        else:
            logger.warning(f"  文件过小: {clean_file.name} ({size_mb:.1f} MB)")
    
    logger.info(f"  有效文件: {valid_count}/{len(clean_files)}")
    logger.info("质量检查完成")

    return valid_count > 0, valid_count


# =============================================================================
# Atlas 构建（含气管树模板）
# =============================================================================

def run_atlas_construction(config: dict, logger: logging.Logger,
                           quick_test: bool = False,
                           skip_template_build: bool = False) -> bool:
    """
    构建标准肺模板（含气管树模板）

    输出文件：
        - standard_template.nii.gz: 标准肺部模板
        - standard_mask.nii.gz: 模板肺部 mask
        - standard_trachea_mask.nii.gz: 模板气管树 mask
        - atlas_evaluation_report.json: 质量评估报告

    Args:
        config: 配置字典
        logger: 日志记录器
        quick_test: 是否快速测试模式
        skip_template_build: 是否跳过模板构建（仅生成气管树模板）

    Returns:
        bool: 是否成功
    """
    if skip_template_build:
        logger.info("=" * 60)
        logger.info("步骤 5: 气管树模板生成（跳过模板构建）")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("步骤 5: Atlas 构建（含气管树模板）")
        logger.info("=" * 60)

    # 导入模板构建模块
    try:
        build_module = importlib.import_module("src.02_atlas_build.build_template_ants")
    except ImportError as e:
        logger.error(f"无法导入模板构建模块: {e}")
        return False

    cleaned_dir = Path(config['paths']['cleaned_data'])
    atlas_dir = Path(config['paths']['atlas'])

    input_dir = cleaned_dir / 'normal_clean'
    mask_dir = cleaned_dir / 'normal_mask'
    atlas_dir.mkdir(parents=True, exist_ok=True)

    # 检查模板文件（如果跳过构建）
    template_file = atlas_dir / 'standard_template.nii.gz'
    if skip_template_build:
        if not template_file.exists():
            logger.error(f"无法跳过模板构建: 模板文件不存在 {template_file}")
            return False
        logger.info(f"  ✓ 使用已有模板: {template_file}")
        size_mb = template_file.stat().st_size / 1e6
        logger.info(f"    文件大小: {size_mb:.1f} MB")

    # 获取输入文件
    nifti_files = sorted(list(input_dir.glob("*_clean.nii.gz")))
    trachea_files = sorted(list(mask_dir.glob("*_trachea_mask.nii.gz")))

    if not skip_template_build and len(nifti_files) < 5:
        logger.error(f"文件数不足: 需要至少 5 例，当前 {len(nifti_files)} 例")
        return False

    # 配置参数
    reg_config = config.get('registration', {})
    template_config = reg_config.get('template_build', {})

    max_subjects = template_config.get('max_subjects', 20)

    if quick_test:
        max_subjects = min(5, len(nifti_files))
        logger.info(f"  快速测试模式: {max_subjects} 例")

    logger.info(f"  输入目录: {input_dir}")
    logger.info(f"  输出目录: {atlas_dir}")
    logger.info(f"  可用 CT 文件: {len(nifti_files)}")
    logger.info(f"  可用气管树 mask: {len(trachea_files)}")
    logger.info(f"  使用数量: {min(max_subjects, len(nifti_files))}")
    logger.info(f"  跳过模板构建: {'是' if skip_template_build else '否'}")

    # 运行模板构建
    start_time = time.time()

    try:
        result = build_module.main(
            config=config,
            num_images=max_subjects,
            skip_evaluation=False,
            quick_test=quick_test,
            skip_template_build=skip_template_build  # 传递跳过参数
        )

        elapsed = time.time() - start_time
        logger.info("")
        logger.info("=" * 50)
        logger.info("Atlas 构建完成:")
        logger.info(f"  耗时: {elapsed/60:.1f} 分钟")

        # 检查输出
        if result.get('success'):
            if result.get('template_path'):
                tpl = Path(result['template_path'])
                if tpl.exists():
                    logger.info(f"  模板: {tpl.name} ({tpl.stat().st_size/1e6:.1f} MB)")

            if result.get('mask_path'):
                msk = Path(result['mask_path'])
                if msk.exists():
                    logger.info(f"  肺部 mask: {msk.name} ({msk.stat().st_size/1e6:.1f} MB)")

            if result.get('trachea_path'):
                trc = Path(result['trachea_path'])
                if trc.exists():
                    logger.info(f"  气管树 mask: {trc.name} ({trc.stat().st_size/1e6:.1f} MB)")
            elif result.get('has_trachea') is False:
                logger.warning("  气管树 mask: 未生成（输入不足）")

            logger.info("=" * 50)
            return True
        else:
            logger.error(f"Atlas 构建失败: {result.get('error', '未知错误')}")
            return False

    except Exception as e:
        logger.error(f"Atlas 构建失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# =============================================================================
# 结果验证
# =============================================================================

def run_result_validation(config: dict, logger: logging.Logger) -> bool:
    """验证 Atlas 结果"""
    logger.info("=" * 60)
    logger.info("步骤 6: 结果验证")
    logger.info("=" * 60)

    atlas_dir = Path(config['paths']['atlas'])

    # 检查必要文件
    required_files = [
        "standard_template.nii.gz",
        "standard_mask.nii.gz",  # 修正：实际生成的文件名是 standard_mask.nii.gz
    ]

    optional_files = [
        "atlas_evaluation_report.json",
        "template_metadata.json",
    ]

    all_ok = True
    for fname in required_files:
        fpath = atlas_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1e6
            logger.info(f"  ✓ {fname} ({size_mb:.1f} MB)")
        else:
            logger.error(f"  ✗ {fname} (缺失)")
            all_ok = False

    for fname in optional_files:
        fpath = atlas_dir / fname
        if fpath.exists():
            logger.info(f"  ✓ {fname}")
        else:
            logger.info(f"  - {fname} (可选)")

    # 读取评估报告
    report_file = atlas_dir / "atlas_evaluation_report.json"
    if report_file.exists():
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)

            mean_dice = report.get('mean_dice', 0)
            logger.info(f"  平均 Dice 系数: {mean_dice:.4f}")

            if mean_dice >= 0.85:
                logger.info(f"  ✓ Dice 系数达标 (>= 0.85)")
            else:
                logger.warning(f"  ⚠ Dice 系数未达标 (< 0.85)")

        except Exception as e:
            logger.warning(f"  无法读取评估报告: {e}")

    if all_ok:
        logger.info("结果验证通过")
    else:
        logger.error("结果验证失败")

    return all_ok


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Phase 2 端到端流水线（数字孪生底座构建）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流水线
  python run_phase2_pipeline.py

  # 快速测试（仅处理 3 例）
  python run_phase2_pipeline.py --quick-test

  # 仅执行分割步骤（Step 1）
  python run_phase2_pipeline.py --step1-only

  # 仅生成气管树模板（Step 2，需要已有模板）
  python run_phase2_pipeline.py --step2-only

  # 限制处理数量
  python run_phase2_pipeline.py --limit 5

  # 强制覆盖（重新分割所有文件）
  python run_phase2_pipeline.py --force

  # 指定 GPU
  python run_phase2_pipeline.py --device cuda:0

  # 后台运行
  nohup python run_phase2_pipeline.py > logs/phase2.log 2>&1 &
        """
    )

    # 步骤控制参数
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument(
        '--step1-only', action='store_true',
        help='仅执行分割步骤（生成气管树和5肺叶标签 mask）'
    )
    step_group.add_argument(
        '--step2-only', action='store_true',
        help='仅生成气管树模板（需要已有 standard_template.nii.gz）'
    )
    step_group.add_argument(
        '--skip-segmentation', action='store_true',
        help='跳过分割步骤（使用已有的分割结果）'
    )
    step_group.add_argument(
        '--skip-atlas', action='store_true',
        help='跳过 Atlas 构建（仅运行分割，等同于 --step1-only）'
    )

    # 其他参数
    parser.add_argument(
        '--quick-test', action='store_true',
        help='快速测试模式（仅处理 3 例数据，2 次迭代）'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='限制处理数量（用于测试）'
    )
    parser.add_argument(
        '--device', type=str, default='gpu',
        choices=['gpu', 'cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
        help='计算设备（默认: gpu）'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='强制重新处理（覆盖已有结果）'
    )
    parser.add_argument(
        '--check-only', action='store_true',
        help='仅检查环境，不执行流水线'
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='配置文件路径（默认: config.yaml）'
    )

    args = parser.parse_args()

    # 设置日志
    logger = setup_logging()

    # 确定执行模式
    run_segmentation_step = True
    run_atlas_step = True
    skip_template_build = False

    if args.step1_only or args.skip_atlas:
        run_atlas_step = False
    elif args.step2_only:
        run_segmentation_step = False
        skip_template_build = True
    elif args.skip_segmentation:
        run_segmentation_step = False

    logger.info("=" * 60)
    logger.info("Phase 2 流水线: 数字孪生底座构建")
    logger.info("=" * 60)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"执行分割: {'是' if run_segmentation_step else '否'}")
    logger.info(f"执行Atlas: {'是' if run_atlas_step else '否'}")
    if skip_template_build:
        logger.info(f"跳过模板构建: 是（仅生成气管树模板）")
    logger.info(f"设备: {args.device}")
    if args.force:
        logger.info(f"覆盖模式: 是")
    if args.limit:
        logger.info(f"处理限制: {args.limit}")
    if args.quick_test:
        logger.info(f"快速测试: 是")

    # 加载配置
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"无法加载配置文件: {e}")
        sys.exit(1)

    # 记录开始时间
    pipeline_start = time.time()

    # Step 1: 环境检查
    env_ok = run_environment_check(logger)

    if args.check_only:
        logger.info("仅检查环境模式，退出")
        sys.exit(0 if env_ok else 1)

    # Step 2: 数据验证
    data_ok, _ = run_data_validation(config, logger, args.quick_test)
    if not data_ok:
        logger.error("数据验证失败，退出")
        sys.exit(1)

    # Step 3: 肺部分割
    if run_segmentation_step:
        seg_ok = run_segmentation(
            config, logger,
            device=args.device,
            quick_test=args.quick_test,
            force=args.force,
            limit=args.limit
        )
        if not seg_ok:
            logger.warning("分割过程有错误，继续执行")
    else:
        if args.step2_only:
            logger.info("跳过分割步骤（--step2-only 模式）")
        else:
            logger.info("跳过分割步骤（--skip-segmentation）")

    # Step 4: 质量检查
    quality_ok, _ = run_quality_check(config, logger)
    if not quality_ok:
        logger.error("质量检查失败，退出")
        sys.exit(1)

    # Step 5: Atlas 构建
    if run_atlas_step:
        atlas_ok = run_atlas_construction(
            config, logger,
            quick_test=args.quick_test,
            skip_template_build=skip_template_build
        )
        if not atlas_ok:
            logger.error("Atlas 构建失败")
            sys.exit(1)

        # Step 6: 结果验证
        result_ok = run_result_validation(config, logger)
        if not result_ok:
            logger.warning("结果验证未完全通过")
    else:
        if args.step1_only:
            logger.info("跳过 Atlas 构建（--step1-only 模式）")
        else:
            logger.info("跳过 Atlas 构建（--skip-atlas）")

    # 总结
    pipeline_elapsed = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info("流水线执行完成")
    logger.info("=" * 60)
    logger.info(f"总耗时: {pipeline_elapsed/60:.1f} 分钟")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 输出结果摘要
    atlas_dir = Path(config['paths']['atlas'])
    template_file = atlas_dir / "standard_template.nii.gz"
    trachea_file = atlas_dir / "standard_trachea_mask.nii.gz"

    logger.info("")
    if template_file.exists():
        logger.info("✓ 模板文件已生成")
        logger.info(f"  - {template_file}")
    if trachea_file.exists():
        logger.info("✓ 气管树模板已生成")
        logger.info(f"  - {trachea_file}")

    if template_file.exists() or trachea_file.exists():
        logger.info("")
        logger.info("✓ 数字孪生底座构建成功！")
        logger.info("")

    logger.info("下一步:")
    logger.info("  1. 检查 data/02_atlas/ 中的模板文件")
    logger.info("  2. 使用 3D Slicer 可视化检查模板质量")
    logger.info("  3. 运行 Phase 3: COPD 数据配准和 AI 模型训练")


if __name__ == "__main__":
    main()

