#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2: Atlas Construction - 标准底座构建入口脚本

使用 ANTsPy 从多例正常肺 CT 构建标准模板。

使用方法：
    # 默认运行（使用所有可用的正常肺数据）
    python run_phase2_atlas.py
    
    # 指定使用 20 例数据
    python run_phase2_atlas.py --num-images 20
    
    # 快速测试模式（使用 3 例数据，2 次迭代）
    python run_phase2_atlas.py --quick-test
    
    # 跳过质量评估
    python run_phase2_atlas.py --skip-eval
    
    # 服务器后台运行
    nohup python run_phase2_atlas.py > logs/phase2_atlas.log 2>&1 &

输入数据要求：
    - 数据位置：data/01_cleaned/normal_clean/*.nii.gz
    - 数据格式：NIfTI (.nii.gz)
    - 推荐数量：15-40 例正常肺 CT
    - 对应 mask：data/01_cleaned/normal_mask/*.nii.gz（可选，用于 Dice 评估）

输出文件：
    - data/02_atlas/standard_template.nii.gz  - 标准模板
    - data/02_atlas/standard_mask.nii.gz      - 模板肺部 mask
    - data/02_atlas/atlas_evaluation_report.json - 质量评估报告

验收标准：
    - 模板文件大小 > 10MB
    - 与任一输入肺的 Dice >= 0.85
    - 血管/气管结构可辨识（在 3D Slicer 中目视确认）

预计运行时间：
    - 15-20 例数据：4-8 小时
    - 40 例数据：8-16 小时
    - 快速测试模式：10-30 分钟

作者: DigitalTwinLung_COPD Team
日期: 2025-12-09
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_prerequisites():
    """检查运行前提条件"""
    print("=" * 70)
    print("Phase 2: Atlas Construction - 环境检查")
    print("=" * 70)
    
    errors = []
    warnings = []
    
    # 检查 ANTsPy
    try:
        import ants
        print(f"✅ ANTsPy 版本: {ants.__version__ if hasattr(ants, '__version__') else '已安装'}")
    except ImportError:
        errors.append("❌ ANTsPy 未安装。请运行: pip install antspyx")
    
    # 检查 nibabel
    try:
        import nibabel
        print(f"✅ nibabel 版本: {nibabel.__version__}")
    except ImportError:
        errors.append("❌ nibabel 未安装。请运行: pip install nibabel")
    
    # 检查 numpy
    try:
        import numpy
        print(f"✅ numpy 版本: {numpy.__version__}")
    except ImportError:
        errors.append("❌ numpy 未安装。请运行: pip install numpy")
    
    # 检查 scipy
    try:
        import scipy
        print(f"✅ scipy 版本: {scipy.__version__}")
    except ImportError:
        warnings.append("⚠️ scipy 未安装，形态学操作将被跳过")
    
    # 检查输入数据
    input_dir = project_root / "data" / "01_cleaned" / "normal_clean"
    if input_dir.exists():
        files = list(input_dir.glob("*.nii.gz"))
        if len(files) >= 2:
            print(f"✅ 输入数据: {len(files)} 个 NIfTI 文件")
        else:
            errors.append(f"❌ 输入数据不足: 需要至少 2 个文件，当前 {len(files)} 个")
    else:
        errors.append(f"❌ 输入目录不存在: {input_dir}")
    
    # 检查输出目录
    output_dir = project_root / "data" / "02_atlas"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ 输出目录: {output_dir}")
    
    # 检查 config.yaml
    config_path = project_root / "config.yaml"
    if config_path.exists():
        print(f"✅ 配置文件: {config_path}")
    else:
        warnings.append(f"⚠️ 配置文件不存在，将使用默认参数")
    
    print()
    
    # 输出警告
    for w in warnings:
        print(w)
    
    # 输出错误
    if errors:
        print()
        for e in errors:
            print(e)
        print()
        print("请修复上述错误后重新运行。")
        return False
    
    print()
    return True


def estimate_runtime(num_images: int, quick_test: bool = False) -> str:
    """估算运行时间"""
    if quick_test:
        return "10-30 分钟"
    
    # 基于经验估算：每个图像大约需要 10-20 分钟配准
    # 迭代次数默认为 5
    min_hours = num_images * 0.15  # 每个图像 9 分钟
    max_hours = num_images * 0.30  # 每个图像 18 分钟
    
    if min_hours < 1:
        return f"{int(min_hours * 60)}-{int(max_hours * 60)} 分钟"
    else:
        return f"{min_hours:.1f}-{max_hours:.1f} 小时"


def main():
    parser = argparse.ArgumentParser(
        description='Phase 2: Atlas Construction - 构建标准底座',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python run_phase2_atlas.py                   # 使用所有数据运行
  python run_phase2_atlas.py --num-images 20   # 使用 20 例数据
  python run_phase2_atlas.py --quick-test      # 快速测试模式
  python run_phase2_atlas.py --skip-eval       # 跳过质量评估
        """
    )
    
    parser.add_argument(
        '--num-images', type=int, default=None,
        help='使用的图像数量（默认使用所有可用图像）'
    )
    parser.add_argument(
        '--quick-test', action='store_true',
        help='快速测试模式（3 例数据，2 次迭代）'
    )
    parser.add_argument(
        '--skip-eval', action='store_true',
        help='跳过质量评估步骤'
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='配置文件路径（默认: config.yaml）'
    )
    parser.add_argument(
        '--check-only', action='store_true',
        help='仅检查环境，不执行构建'
    )
    
    args = parser.parse_args()
    
    # 检查环境
    if not check_prerequisites():
        sys.exit(1)
    
    if args.check_only:
        print("环境检查完成，未执行构建。")
        sys.exit(0)
    
    # 估算运行时间
    input_dir = project_root / "data" / "01_cleaned" / "normal_clean"
    num_files = len(list(input_dir.glob("*.nii.gz")))
    num_to_use = args.num_images if args.num_images else num_files
    num_to_use = min(num_to_use, num_files)
    
    if args.quick_test:
        num_to_use = min(3, num_files)
    
    estimated_time = estimate_runtime(num_to_use, args.quick_test)
    
    print("=" * 70)
    print("运行配置:")
    print(f"  使用图像数量: {num_to_use}")
    print(f"  快速测试模式: {'是' if args.quick_test else '否'}")
    print(f"  跳过质量评估: {'是' if args.skip_eval else '否'}")
    print(f"  预计运行时间: {estimated_time}")
    print("=" * 70)
    print()
    
    # 确认运行
    if not args.quick_test:
        print("⚠️  Atlas 构建是一个长时间任务，建议在服务器上使用后台运行。")
        print("    示例: nohup python run_phase2_atlas.py > logs/phase2.log 2>&1 &")
        print()
        try:
            confirm = input("是否继续? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("已取消。")
                sys.exit(0)
        except EOFError:
            # 在后台运行时自动继续
            pass
    
    # 加载配置
    import yaml
    config = None
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # 创建日志目录
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"\n开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 运行 Atlas 构建
    try:
        import importlib
        build_module = importlib.import_module("src.02_atlas_build.build_template_ants")
        build_main = build_module.main

        result = build_main(
            config=config,
            num_images=args.num_images,
            skip_evaluation=args.skip_eval,
            quick_test=args.quick_test
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"实际耗时: {duration}")
        
        if result.get('success'):
            print("\n✅ Phase 2 成功完成!")
            print(f"   模板: {result.get('template_path')}")
            print(f"   Mask: {result.get('mask_path')}")
            sys.exit(0)
        else:
            print(f"\n❌ Phase 2 失败: {result.get('error', '未知错误')}")
            sys.exit(1)
            
    except Exception as e:
        import traceback
        print(f"\n❌ 运行失败: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

