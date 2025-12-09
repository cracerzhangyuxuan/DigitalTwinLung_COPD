#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU 加速肺部分割入口脚本 (Phase 2 预处理)

支持两种分割模式：
1. TotalSegmentator (GPU 加速，高精度) - 推荐用于生产环境
2. 阈值分割 (CPU，轻量) - 用于无 GPU 环境或快速测试

使用方法：
    # 使用 TotalSegmentator (GPU)
    python run_segmentation_gpu.py --method totalsegmentator --device cuda:0
    
    # 使用阈值方法 (CPU)
    python run_segmentation_gpu.py --method threshold
    
    # 仅处理正常肺数据
    python run_segmentation_gpu.py --type normal
    
    # 快速测试 (处理前 2 个文件)
    python run_segmentation_gpu.py --test-run 2
    
    # 后台运行 (服务器)
    nohup python run_segmentation_gpu.py > logs/segmentation.log 2>&1 &

输入数据：
    - data/00_raw/normal/*.nii.gz (正常肺)
    - data/00_raw/copd/*.nii.gz (COPD 患者)

输出数据：
    - data/01_cleaned/normal_mask/*.nii.gz
    - data/01_cleaned/normal_clean/*.nii.gz
    - data/01_cleaned/copd_mask/*.nii.gz
    - data/01_cleaned/copd_clean/*.nii.gz

作者: DigitalTwinLung_COPD Team
日期: 2025-12-09
"""

import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_gpu_available() -> Tuple[bool, str]:
    """检查 GPU 是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return True, f"GPU 可用: {gpu_name}"
        else:
            return False, "CUDA 不可用"
    except ImportError:
        return False, "PyTorch 未安装"


def check_totalsegmentator_available() -> Tuple[bool, str]:
    """检查 TotalSegmentator 是否可用"""
    try:
        result = subprocess.run(
            ["TotalSegmentator", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return True, f"TotalSegmentator 可用"
        return False, "TotalSegmentator 命令执行失败"
    except FileNotFoundError:
        return False, "TotalSegmentator 未安装"
    except subprocess.TimeoutExpired:
        return False, "TotalSegmentator 响应超时"
    except Exception as e:
        return False, f"检查失败: {e}"


def check_prerequisites(method: str, device: str) -> bool:
    """检查运行前提条件"""
    print("=" * 70)
    print("环境检查")
    print("=" * 70)
    
    errors = []
    warnings = []
    
    # 检查 GPU
    gpu_ok, gpu_msg = check_gpu_available()
    if gpu_ok:
        print(f"✅ {gpu_msg}")
    else:
        if device.startswith("cuda"):
            errors.append(f"❌ {gpu_msg}")
        else:
            warnings.append(f"⚠️ {gpu_msg} (使用 CPU 模式)")
    
    # 检查 TotalSegmentator
    if method == "totalsegmentator":
        ts_ok, ts_msg = check_totalsegmentator_available()
        if ts_ok:
            print(f"✅ {ts_msg}")
        else:
            errors.append(f"❌ {ts_msg}")
            errors.append("   安装命令: pip install TotalSegmentator")
    
    # 检查 nibabel
    try:
        import nibabel
        print(f"✅ nibabel 版本: {nibabel.__version__}")
    except ImportError:
        errors.append("❌ nibabel 未安装")
    
    # 检查 scipy
    try:
        import scipy
        print(f"✅ scipy 版本: {scipy.__version__}")
    except ImportError:
        if method == "threshold":
            errors.append("❌ scipy 未安装 (阈值方法需要)")
        else:
            warnings.append("⚠️ scipy 未安装 (仅阈值方法需要)")
    
    # 检查输入数据
    raw_dir = project_root / "data" / "00_raw"
    normal_count = len(list((raw_dir / "normal").glob("*.nii.gz"))) if (raw_dir / "normal").exists() else 0
    copd_count = len(list((raw_dir / "copd").glob("*.nii.gz"))) if (raw_dir / "copd").exists() else 0
    
    if normal_count > 0 or copd_count > 0:
        print(f"✅ 输入数据: {normal_count} 正常肺, {copd_count} COPD")
    else:
        errors.append(f"❌ 未找到输入数据: {raw_dir}")
    
    # 输出警告
    for w in warnings:
        print(w)
    
    # 输出错误
    if errors:
        print()
        for e in errors:
            print(e)
        return False
    
    print()
    return True


def run_totalsegmentator_batch(
    input_dir: Path,
    mask_output_dir: Path,
    clean_output_dir: Path,
    device: str = "gpu",
    fast: bool = False,
    skip_existing: bool = True,
    limit: Optional[int] = None,
    background_hu: float = -1000
) -> dict:
    """
    使用 TotalSegmentator 批量分割
    
    Args:
        input_dir: 输入目录
        mask_output_dir: mask 输出目录
        clean_output_dir: 清洗后 CT 输出目录
        device: 设备 ("gpu", "cpu", "cuda:0", etc.)
        fast: 是否使用快速模式
        skip_existing: 是否跳过已处理的文件
        limit: 限制处理数量 (用于测试)
        background_hu: 背景 HU 值
    
    Returns:
        results: 处理结果
    """
    from src.utils.io import load_nifti, save_nifti
    
    input_dir = Path(input_dir)
    mask_output_dir = Path(mask_output_dir)
    clean_output_dir = Path(clean_output_dir)
    
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 临时目录用于 TotalSegmentator 输出
    temp_dir = project_root / "data" / ".temp_segmentation"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    nifti_files = sorted(list(input_dir.glob("*.nii.gz")))
    if limit:
        nifti_files = nifti_files[:limit]
    
    print(f"找到 {len(nifti_files)} 个文件待处理")
    
    results = {"success": [], "failed": [], "skipped": []}
    
    for i, nifti_path in enumerate(nifti_files, 1):
        stem = nifti_path.name.replace('.nii.gz', '').replace('.nii', '')
        mask_path = mask_output_dir / f"{stem}_mask.nii.gz"
        clean_path = clean_output_dir / f"{stem}_clean.nii.gz"
        
        # 检查是否已处理
        if skip_existing and mask_path.exists() and clean_path.exists():
            print(f"[{i}/{len(nifti_files)}] 跳过已处理: {stem}")
            results["skipped"].append(stem)
            continue
        
        print(f"[{i}/{len(nifti_files)}] 处理: {stem}")
        
        try:
            # 运行 TotalSegmentator
            seg_output = temp_dir / f"{stem}_seg"
            
            cmd = ["TotalSegmentator", "-i", str(nifti_path), "-o", str(seg_output)]
            
            # 仅分割肺部 (使用 lung_vessels 任务包含肺叶)
            cmd.extend(["--roi_subset", 
                       "lung_upper_lobe_left", "lung_lower_lobe_left",
                       "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"])
            
            if fast:
                cmd.append("--fast")
            
            if device == "cpu":
                cmd.extend(["--device", "cpu"])
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # 合并肺叶 mask
            lung_parts = [
                "lung_upper_lobe_left.nii.gz",
                "lung_lower_lobe_left.nii.gz",
                "lung_upper_lobe_right.nii.gz",
                "lung_middle_lobe_right.nii.gz",
                "lung_lower_lobe_right.nii.gz",
            ]
            
            combined_mask = None
            affine = None
            
            for part in lung_parts:
                part_path = seg_output / part
                if part_path.exists():
                    import nibabel as nib
                    nii = nib.load(str(part_path))
                    mask = np.asanyarray(nii.dataobj) > 0
                    if combined_mask is None:
                        combined_mask = mask
                        affine = nii.affine
                    else:
                        combined_mask = combined_mask | mask
            
            if combined_mask is None:
                raise ValueError("未找到肺部分割结果")
            
            combined_mask = combined_mask.astype(np.uint8)
            
            # 保存 mask
            save_nifti(combined_mask, mask_path, affine=affine, dtype='uint8')
            
            # 加载原始 CT 并创建清洗后版本
            ct_data, ct_affine = load_nifti(nifti_path, return_affine=True)
            ct_clean = ct_data.copy()
            ct_clean[combined_mask == 0] = background_hu
            save_nifti(ct_clean, clean_path, affine=ct_affine)
            
            # 清理临时文件
            if seg_output.exists():
                shutil.rmtree(seg_output)
            
            lung_ratio = np.sum(combined_mask) / combined_mask.size * 100
            print(f"    ✅ 完成 - 肺占比: {lung_ratio:.1f}%")
            results["success"].append(stem)
            
        except Exception as e:
            print(f"    ❌ 失败: {e}")
            results["failed"].append((stem, str(e)))
    
    # 清理临时目录
    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()
    
    return results


def run_threshold_batch(
    input_dir: Path,
    mask_output_dir: Path,
    clean_output_dir: Path,
    skip_existing: bool = True,
    limit: Optional[int] = None
) -> dict:
    """使用阈值方法批量分割"""
    # 动态导入（模块名以数字开头）
    import importlib
    simple_lung_segment = importlib.import_module("src.01_preprocessing.simple_lung_segment")
    segment_lung_from_file = simple_lung_segment.segment_lung_from_file

    # 收集需要处理的文件
    input_dir = Path(input_dir)
    mask_output_dir = Path(mask_output_dir)
    clean_output_dir = Path(clean_output_dir)

    mask_output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(list(input_dir.glob("*.nii.gz")))

    if limit:
        nifti_files = nifti_files[:limit]

    results = {"success": [], "failed": [], "skipped": []}

    for i, nifti_path in enumerate(nifti_files, 1):
        stem = nifti_path.name.replace('.nii.gz', '').replace('.nii', '')
        mask_path = mask_output_dir / f"{stem}_mask.nii.gz"
        clean_path = clean_output_dir / f"{stem}_clean.nii.gz"

        # 检查是否已处理
        if skip_existing and mask_path.exists() and clean_path.exists():
            print(f"[{i}/{len(nifti_files)}] 跳过已处理: {stem}")
            results["skipped"].append(stem)
            continue

        print(f"[{i}/{len(nifti_files)}] 处理: {stem}")

        try:
            result = segment_lung_from_file(
                nifti_path,
                mask_output_dir=mask_output_dir,
                clean_output_dir=clean_output_dir
            )
            if result.get('status') == 'success':
                lung_ratio = result.get('lung_ratio', 0) * 100
                print(f"    ✅ 完成 - 肺占比: {lung_ratio:.1f}%")
                results["success"].append(stem)
            else:
                print(f"    ❌ 失败: {result.get('error', '未知错误')}")
                results["failed"].append((stem, result.get('error', '未知错误')))
        except Exception as e:
            print(f"    ❌ 异常: {e}")
            results["failed"].append((stem, str(e)))

    return results


def main():
    parser = argparse.ArgumentParser(
        description='GPU 加速肺部分割',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python run_segmentation_gpu.py --method totalsegmentator --device cuda:0
  python run_segmentation_gpu.py --method threshold
  python run_segmentation_gpu.py --type normal --test-run 2
        """
    )

    parser.add_argument(
        '--method', choices=['totalsegmentator', 'threshold', 'auto'],
        default='auto',
        help='分割方法 (default: auto - 自动选择)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='设备 (cuda:0, cpu, auto)'
    )
    parser.add_argument(
        '--type', choices=['all', 'normal', 'copd'],
        default='all',
        help='处理的数据类型'
    )
    parser.add_argument(
        '--test-run', type=int, default=None,
        help='测试运行：仅处理指定数量的文件'
    )
    parser.add_argument(
        '--fast', action='store_true',
        help='使用快速模式 (TotalSegmentator)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='强制重新处理已存在的文件'
    )
    parser.add_argument(
        '--check-only', action='store_true',
        help='仅检查环境，不执行分割'
    )

    args = parser.parse_args()

    # 自动选择设备
    if args.device == 'auto':
        gpu_ok, _ = check_gpu_available()
        args.device = 'cuda:0' if gpu_ok else 'cpu'

    # 自动选择方法
    if args.method == 'auto':
        ts_ok, _ = check_totalsegmentator_available()
        if ts_ok:
            args.method = 'totalsegmentator'
        else:
            print("TotalSegmentator 不可用，使用阈值方法")
            args.method = 'threshold'

    # 环境检查
    if not check_prerequisites(args.method, args.device):
        sys.exit(1)

    if args.check_only:
        print("环境检查完成，未执行分割。")
        sys.exit(0)

    # 加载配置
    import yaml
    config_path = project_root / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config['paths']['raw_data'])
    cleaned_dir = Path(config['paths']['cleaned_data'])

    # 创建日志目录
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    # 确定要处理的数据类型
    data_types = []
    if args.type in ['all', 'normal']:
        data_types.append(('normal', raw_dir / 'normal',
                          cleaned_dir / 'normal_mask', cleaned_dir / 'normal_clean'))
    if args.type in ['all', 'copd']:
        data_types.append(('copd', raw_dir / 'copd',
                          cleaned_dir / 'copd_mask', cleaned_dir / 'copd_clean'))

    # 记录开始时间
    start_time = datetime.now()
    print("=" * 70)
    print(f"开始分割 - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"方法: {args.method}")
    print(f"设备: {args.device}")
    print(f"数据类型: {args.type}")
    if args.test_run:
        print(f"测试模式: 仅处理 {args.test_run} 个文件")
    print("=" * 70)

    total_results = {"success": 0, "failed": 0, "skipped": 0}

    for data_name, input_dir, mask_dir, clean_dir in data_types:
        if not input_dir.exists():
            print(f"\n⚠️ 跳过 {data_name}: 目录不存在 {input_dir}")
            continue

        print(f"\n{'=' * 70}")
        print(f"处理 {data_name} 数据...")
        print(f"输入: {input_dir}")
        print(f"输出 Mask: {mask_dir}")
        print(f"输出 Clean: {clean_dir}")
        print('=' * 70)

        if args.method == 'totalsegmentator':
            results = run_totalsegmentator_batch(
                input_dir=input_dir,
                mask_output_dir=mask_dir,
                clean_output_dir=clean_dir,
                device='cpu' if args.device == 'cpu' else 'gpu',
                fast=args.fast,
                skip_existing=not args.force,
                limit=args.test_run
            )
        else:
            results = run_threshold_batch(
                input_dir=input_dir,
                mask_output_dir=mask_dir,
                clean_output_dir=clean_dir,
                skip_existing=not args.force,
                limit=args.test_run
            )

        # 统计结果
        if isinstance(results, dict):
            if 'success' in results:
                total_results["success"] += len(results.get("success", []))
                total_results["failed"] += len(results.get("failed", []))
                total_results["skipped"] += len(results.get("skipped", []))
            else:
                # batch_segment_lungs 返回格式不同
                for v in results.values():
                    if isinstance(v, dict):
                        if v.get('status') == 'success':
                            total_results["success"] += 1
                        else:
                            total_results["failed"] += 1

    # 输出总结
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 70)
    print("分割完成!")
    print("=" * 70)
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration}")
    print(f"成功: {total_results['success']}")
    print(f"失败: {total_results['failed']}")
    print(f"跳过: {total_results['skipped']}")

    if total_results['failed'] > 0:
        print("\n⚠️ 有文件处理失败，请检查日志")
        sys.exit(1)
    else:
        print("\n✅ 全部处理完成!")
        sys.exit(0)


if __name__ == "__main__":
    main()

