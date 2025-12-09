#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2 端到端流水线（数字孪生底座构建）

功能：
    1. 环境检查：验证 GPU、TotalSegmentator、ANTsPy 可用性
    2. 数据验证：检查原始数据完整性
    3. 肺部分割：使用 TotalSegmentator GPU 加速分割
    4. 质量检查：验证分割结果
    5. Atlas 构建：构建标准肺模板
    6. 结果验证：检查 Atlas 质量

使用方法：
    # 完整流水线
    python run_phase2_pipeline.py
    
    # 快速测试（仅处理 3 例）
    python run_phase2_pipeline.py --quick-test
    
    # 跳过分割（使用已有结果）
    python run_phase2_pipeline.py --skip-segmentation
    
    # 仅运行分割
    python run_phase2_pipeline.py --skip-atlas
    
    # 指定 GPU
    python run_phase2_pipeline.py --device cuda:0

作者：DigitalTwinLung COPD Project
日期：2025-12-09
"""

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional

import yaml
import numpy as np

# =============================================================================
# 日志配置
# =============================================================================
import logging

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
# 环境检查
# =============================================================================

def check_gpu() -> Tuple[bool, str]:
    """检查 GPU 可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            return True, f"{gpu_count} GPU(s): {gpu_name}, {memory:.1f} GB"
        return False, "CUDA 不可用"
    except ImportError:
        return False, "PyTorch 未安装"


def check_totalsegmentator() -> Tuple[bool, str]:
    """检查 TotalSegmentator 可用性"""
    try:
        result = subprocess.run(
            ["TotalSegmentator", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            return True, f"版本 {version}"
        return False, "命令执行失败"
    except FileNotFoundError:
        return False, "未安装"
    except subprocess.TimeoutExpired:
        return False, "响应超时"


def check_antspy() -> Tuple[bool, str]:
    """检查 ANTsPy 可用性"""
    try:
        import ants
        return True, f"版本 {ants.__version__}"
    except ImportError:
        return False, "未安装"


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
# 肺部分割
# =============================================================================

def run_segmentation(config: dict, logger: logging.Logger,
                     device: str = "gpu", quick_test: bool = False,
                     force: bool = False) -> bool:
    """运行肺部分割"""
    logger.info("=" * 60)
    logger.info("步骤 3: 肺部分割")
    logger.info("=" * 60)
    
    # 导入分割模块
    try:
        from src.preprocessing import simple_lung_segment
    except ImportError:
        # 尝试动态导入
        import importlib
        simple_lung_segment = importlib.import_module("src.01_preprocessing.simple_lung_segment")
    
    raw_dir = Path(config['paths']['raw_data'])
    cleaned_dir = Path(config['paths']['cleaned_data'])
    
    normal_input = raw_dir / 'normal'
    mask_output = cleaned_dir / 'normal_mask'
    clean_output = cleaned_dir / 'normal_clean'
    
    # 创建输出目录
    mask_output.mkdir(parents=True, exist_ok=True)
    clean_output.mkdir(parents=True, exist_ok=True)
    
    # 获取输入文件
    nifti_files = sorted(list(normal_input.glob("*.nii.gz")))
    if quick_test:
        nifti_files = nifti_files[:3]
    
    logger.info(f"  输入目录: {normal_input}")
    logger.info(f"  输出目录: {clean_output}")
    logger.info(f"  待处理文件: {len(nifti_files)}")
    
    # 检查已存在的文件
    if not force:
        existing = [f for f in nifti_files 
                   if (clean_output / f"{f.stem.replace('.nii', '')}_clean.nii.gz").exists()]
        if existing:
            logger.info(f"  跳过已存在: {len(existing)} 个文件")
            nifti_files = [f for f in nifti_files if f not in existing]
    
    if not nifti_files:
        logger.info("  所有文件已处理，跳过分割")
        return True
    
    # 确定分割方法
    method = simple_lung_segment.get_default_method()
    logger.info(f"  分割方法: {method}")
    logger.info(f"  设备: {device}")
    
    # 批量分割
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    for i, nifti_path in enumerate(nifti_files, 1):
        try:
            logger.info(f"  [{i}/{len(nifti_files)}] 处理: {nifti_path.name}")
            
            stats = simple_lung_segment.segment_lung_from_file(
                input_path=nifti_path,
                mask_output_dir=mask_output,
                clean_output_dir=clean_output,
                method=method,
                device=device
            )
            
            lung_ratio = stats.get('lung_ratio', 0) * 100
            logger.info(f"      肺占比: {lung_ratio:.1f}%")
            success_count += 1
            
        except Exception as e:
            logger.error(f"      失败: {e}")
            fail_count += 1
    
    elapsed = time.time() - start_time
    logger.info(f"  分割完成: {success_count} 成功, {fail_count} 失败")
    logger.info(f"  耗时: {elapsed/60:.1f} 分钟")
    
    return fail_count == 0


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
# Atlas 构建
# =============================================================================

def run_atlas_construction(config: dict, logger: logging.Logger,
                           quick_test: bool = False) -> bool:
    """构建标准肺模板"""
    logger.info("=" * 60)
    logger.info("步骤 5: Atlas 构建")
    logger.info("=" * 60)

    # 导入模板构建模块
    try:
        import importlib
        build_module = importlib.import_module("src.02_atlas_build.build_template_ants")
    except ImportError as e:
        logger.error(f"无法导入模板构建模块: {e}")
        return False

    cleaned_dir = Path(config['paths']['cleaned_data'])
    atlas_dir = Path(config['paths']['atlas'])

    input_dir = cleaned_dir / 'normal_clean'
    atlas_dir.mkdir(parents=True, exist_ok=True)

    # 获取输入文件
    nifti_files = sorted(list(input_dir.glob("*_clean.nii.gz")))

    if len(nifti_files) < 5:
        logger.error(f"文件数不足: 需要至少 5 例，当前 {len(nifti_files)} 例")
        return False

    # 配置参数
    reg_config = config.get('registration', {})
    template_config = reg_config.get('template_build', {})

    max_subjects = template_config.get('max_subjects', 20)
    n_iterations = template_config.get('n_iterations', 5)

    if quick_test:
        max_subjects = min(5, len(nifti_files))
        n_iterations = 2
        logger.info(f"  快速测试模式: {max_subjects} 例, {n_iterations} 次迭代")

    logger.info(f"  输入目录: {input_dir}")
    logger.info(f"  输出目录: {atlas_dir}")
    logger.info(f"  可用文件: {len(nifti_files)}")
    logger.info(f"  使用数量: {min(max_subjects, len(nifti_files))}")
    logger.info(f"  迭代次数: {n_iterations}")

    # 运行模板构建
    start_time = time.time()

    try:
        result = build_module.build_template(
            input_dir=input_dir,
            output_dir=atlas_dir,
            max_subjects=max_subjects,
            n_iterations=n_iterations
        )

        elapsed = time.time() - start_time
        logger.info(f"  Atlas 构建完成")
        logger.info(f"  耗时: {elapsed/60:.1f} 分钟")

        # 检查输出
        template_file = atlas_dir / "standard_template.nii.gz"
        if template_file.exists():
            size_mb = template_file.stat().st_size / 1e6
            logger.info(f"  模板文件: {template_file.name} ({size_mb:.1f} MB)")
            return True
        else:
            logger.error("模板文件未生成")
            return False

    except Exception as e:
        logger.error(f"Atlas 构建失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
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
        "template_mask.nii.gz",
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

  # 跳过分割步骤
  python run_phase2_pipeline.py --skip-segmentation

  # 仅运行分割
  python run_phase2_pipeline.py --skip-atlas

  # 指定 GPU
  python run_phase2_pipeline.py --device cuda:0
        """
    )

    parser.add_argument(
        '--skip-segmentation', action='store_true',
        help='跳过分割步骤（使用已有的分割结果）'
    )
    parser.add_argument(
        '--skip-atlas', action='store_true',
        help='跳过 Atlas 构建（仅运行分割）'
    )
    parser.add_argument(
        '--quick-test', action='store_true',
        help='快速测试模式（仅处理 3 例数据，2 次迭代）'
    )
    parser.add_argument(
        '--device', type=str, default='gpu',
        help='GPU 设备（默认: gpu）'
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

    logger.info("=" * 60)
    logger.info("Phase 2 流水线: 数字孪生底座构建")
    logger.info("=" * 60)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"参数: skip_seg={args.skip_segmentation}, skip_atlas={args.skip_atlas}")
    logger.info(f"       quick_test={args.quick_test}, device={args.device}")

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
    data_ok, file_count = run_data_validation(config, logger, args.quick_test)
    if not data_ok:
        logger.error("数据验证失败，退出")
        sys.exit(1)

    # Step 3: 肺部分割
    if not args.skip_segmentation:
        seg_ok = run_segmentation(
            config, logger,
            device=args.device,
            quick_test=args.quick_test,
            force=args.force
        )
        if not seg_ok:
            logger.warning("分割过程有错误，继续执行")
    else:
        logger.info("跳过分割步骤（--skip-segmentation）")

    # Step 4: 质量检查
    quality_ok, valid_count = run_quality_check(config, logger)
    if not quality_ok:
        logger.error("质量检查失败，退出")
        sys.exit(1)

    # Step 5: Atlas 构建
    if not args.skip_atlas:
        atlas_ok = run_atlas_construction(config, logger, args.quick_test)
        if not atlas_ok:
            logger.error("Atlas 构建失败")
            sys.exit(1)

        # Step 6: 结果验证
        result_ok = run_result_validation(config, logger)
        if not result_ok:
            logger.warning("结果验证未完全通过")
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

    if template_file.exists():
        logger.info("")
        logger.info("✓ 数字孪生底座构建成功！")
        logger.info(f"  模板文件: {template_file}")
        logger.info("")

    logger.info("下一步:")
    logger.info("  1. 检查 data/02_atlas/ 中的模板文件")
    logger.info("  2. 使用 3D Slicer 可视化检查模板质量")
    logger.info("  3. 运行 Phase 3: COPD 数据配准和 AI 模型训练")


if __name__ == "__main__":
    main()

