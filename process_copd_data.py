#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COPD 数据预处理脚本

功能：
    1. 肺部分割：使用 LungMask 对 COPD CT 进行肺部分割
    2. 背景清洗：使用肺部 mask 清除 CT 背景（骨骼、软组织等）
    3. 病灶提取：使用 LAA-950 算法（HU < -950）提取肺气肿病灶区域

输入：
    - data/00_raw/copd/*.nii.gz  (29 例 COPD CT)

输出：
    - data/01_cleaned/copd_mask/{patient_id}_mask.nii.gz      # 肺部二值 mask
    - data/01_cleaned/copd_clean/{patient_id}_clean.nii.gz    # 清洗后的 CT
    - data/01_cleaned/copd_emphysema/{patient_id}_emphysema.nii.gz  # 病灶 mask

使用方法：
    # 完整处理
    python process_copd_data.py

    # 快速测试（仅处理 3 例）
    python process_copd_data.py --quick-test

    # 限制处理数量
    python process_copd_data.py --limit 5

    # 强制覆盖已有文件
    python process_copd_data.py --force

    # 后台运行
    nohup python process_copd_data.py > logs/copd_preprocess.log 2>&1 &

作者：DigitalTwinLung COPD Project
日期：2025-12-30
"""

import sys
import argparse
import importlib
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List

import yaml
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# =============================================================================
# 日志配置
# =============================================================================

def setup_logging(log_dir: Path = None) -> logging.Logger:
    """配置日志"""
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"process_copd_{timestamp}.log"

    # 创建 logger
    logger = logging.getLogger("COPDPreprocess")
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
# 环境检查
# =============================================================================

def check_dependencies(logger: logging.Logger) -> bool:
    """检查必要的依赖"""
    all_ok = True

    # 检查 LungMask
    try:
        import lungmask
        logger.info("  ✓ LungMask 可用")
    except ImportError:
        logger.error("  ✗ LungMask 未安装 (pip install lungmask)")
        all_ok = False

    # 检查 SimpleITK
    try:
        import SimpleITK
        logger.info("  ✓ SimpleITK 可用")
    except ImportError:
        logger.error("  ✗ SimpleITK 未安装 (pip install SimpleITK)")
        all_ok = False

    # 检查 scipy
    try:
        from scipy import ndimage
        logger.info("  ✓ scipy 可用")
    except ImportError:
        logger.error("  ✗ scipy 未安装 (pip install scipy)")
        all_ok = False

    return all_ok


# =============================================================================
# 核心处理函数
# =============================================================================

def process_single_copd(
    ct_path: Path,
    mask_output_dir: Path,
    clean_output_dir: Path,
    lesion_output_dir: Path,
    logger: logging.Logger,
    force: bool = False
) -> Dict:
    """
    处理单个 COPD CT 文件

    Args:
        ct_path: CT 文件路径
        mask_output_dir: 肺部 mask 输出目录
        clean_output_dir: 清洗后 CT 输出目录
        lesion_output_dir: 病灶 mask 输出目录
        logger: 日志记录器
        force: 是否强制覆盖

    Returns:
        result: 处理结果字典
    """
    # 导入必要模块
    import nibabel as nib
    from scipy import ndimage

    # 生成输出路径
    stem = ct_path.name.replace('.nii.gz', '').replace('.nii', '')
    mask_path = mask_output_dir / f"{stem}_mask.nii.gz"
    clean_path = clean_output_dir / f"{stem}_clean.nii.gz"
    lesion_path = lesion_output_dir / f"{stem}_emphysema.nii.gz"

    result = {
        'patient_id': stem,
        'mask_path': str(mask_path),
        'clean_path': str(clean_path),
        'lesion_path': str(lesion_path),
        'success': False,
        'error': None,
        'stats': {}
    }

    # 检查是否已处理
    all_exist = mask_path.exists() and clean_path.exists() and lesion_path.exists()
    if all_exist and not force:
        result['success'] = True
        result['skipped'] = True
        return result

    try:
        start_time = time.time()

        # Step 1: 读取 CT
        logger.debug(f"  读取 CT: {ct_path}")
        ct_nii = nib.load(str(ct_path))
        ct_data = ct_nii.get_fdata().astype(np.float32)
        affine = ct_nii.affine
        header = ct_nii.header

        # Step 2: 肺部分割 (使用 LungMask)
        logger.debug("  运行 LungMask 分割...")
        import SimpleITK as sitk
        from lungmask import LMInferer

        # 读取为 SimpleITK 格式
        sitk_image = sitk.ReadImage(str(ct_path))

        # 创建推理器并运行
        # 可用模型: 'R231', 'LTRCLobes', 'R231CovidWeb'
        inferer = LMInferer(modelname='LTRCLobes')  # 使用肺叶模型
        segmentation = inferer.apply(sitk_image)

        # LungMask 输出形状为 (Z, Y, X)，需要转置为 nibabel 的 (X, Y, Z) 顺序
        segmentation = np.transpose(segmentation, (2, 1, 0))
        logger.debug(f"  分割结果形状（转置后）: {segmentation.shape}")
        logger.debug(f"  原始 CT 形状: {ct_data.shape}")

        # 合并肺叶为二值 mask (值 1-5 表示不同肺叶)
        lung_mask = (segmentation > 0).astype(np.uint8)

        # 统计
        lung_voxels = np.sum(lung_mask > 0)
        result['stats']['lung_voxels'] = int(lung_voxels)
        logger.debug(f"  肺部体素数: {lung_voxels:,}")

        # Step 3: 背景清洗
        logger.debug("  清洗背景...")
        background_hu = -1000
        clean_ct = ct_data.copy()
        clean_ct[lung_mask == 0] = background_hu

        # Step 4: 病灶提取 (LAA-950)
        logger.debug("  提取 LAA-950 病灶...")

        # LAA-950: HU < -950 的区域
        threshold = -950
        emphysema_candidate = (ct_data < threshold) & (lung_mask > 0)

        # 形态学清理
        from scipy.ndimage import binary_opening, binary_closing

        # Opening 去除噪点
        structure = ndimage.generate_binary_structure(3, 1)
        emphysema_clean = binary_opening(emphysema_candidate, structure=structure, iterations=1)

        # Closing 填充小孔
        emphysema_clean = binary_closing(emphysema_clean, structure=structure, iterations=2)

        # 连通域过滤 (去除小于 50 体素的区域)
        labeled, num_features = ndimage.label(emphysema_clean)
        min_voxels = 50
        for i in range(1, num_features + 1):
            component_size = np.sum(labeled == i)
            if component_size < min_voxels:
                emphysema_clean[labeled == i] = 0

        emphysema_mask = emphysema_clean.astype(np.uint8)

        # 确保病灶在肺内
        emphysema_mask = emphysema_mask & lung_mask

        lesion_voxels = np.sum(emphysema_mask > 0)
        laa_percentage = (lesion_voxels / lung_voxels * 100) if lung_voxels > 0 else 0
        result['stats']['lesion_voxels'] = int(lesion_voxels)
        result['stats']['laa_percentage'] = round(laa_percentage, 2)
        logger.debug(f"  病灶体素数: {lesion_voxels:,} (LAA: {laa_percentage:.2f}%)")

        # Step 5: 保存结果
        logger.debug("  保存结果...")

        # 保存肺部 mask
        mask_nii = nib.Nifti1Image(lung_mask, affine, header)
        nib.save(mask_nii, str(mask_path))

        # 保存清洗后 CT
        clean_nii = nib.Nifti1Image(clean_ct, affine, header)
        nib.save(clean_nii, str(clean_path))

        # 保存病灶 mask
        lesion_nii = nib.Nifti1Image(emphysema_mask, affine, header)
        nib.save(lesion_nii, str(lesion_path))

        elapsed = time.time() - start_time
        result['success'] = True
        result['elapsed_seconds'] = round(elapsed, 1)
        result['skipped'] = False

    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        import traceback
        logger.debug(traceback.format_exc())

    return result


def process_copd_batch(
    input_dir: Path,
    mask_output_dir: Path,
    clean_output_dir: Path,
    lesion_output_dir: Path,
    logger: logging.Logger,
    limit: int = None,
    force: bool = False
) -> Dict:
    """批量处理 COPD 数据"""
    # 创建输出目录
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir.mkdir(parents=True, exist_ok=True)
    lesion_output_dir.mkdir(parents=True, exist_ok=True)

    # 获取输入文件
    nifti_files = sorted(list(input_dir.glob("*.nii.gz")))
    if limit:
        nifti_files = nifti_files[:limit]

    total = len(nifti_files)
    logger.info(f"待处理: {total} 例")

    results = {
        'success': [],
        'failed': [],
        'skipped': [],
        'total_time': 0
    }

    batch_start = time.time()

    for i, ct_path in enumerate(nifti_files, 1):
        logger.info(f"[{i}/{total}] 处理: {ct_path.name}")

        result = process_single_copd(
            ct_path=ct_path,
            mask_output_dir=mask_output_dir,
            clean_output_dir=clean_output_dir,
            lesion_output_dir=lesion_output_dir,
            logger=logger,
            force=force
        )

        if result.get('skipped', False):
            logger.info(f"  ⏭ 跳过（已存在）")
            results['skipped'].append(result)
        elif result['success']:
            stats = result.get('stats', {})
            logger.info(f"  ✓ 完成 ({result.get('elapsed_seconds', 0)}s)")
            logger.info(f"    肺: {stats.get('lung_voxels', 0):,} 体素")
            logger.info(f"    病灶: {stats.get('lesion_voxels', 0):,} 体素 (LAA: {stats.get('laa_percentage', 0):.1f}%)")
            results['success'].append(result)
        else:
            logger.error(f"  ✗ 失败: {result.get('error', 'Unknown error')}")
            results['failed'].append(result)

    results['total_time'] = time.time() - batch_start

    return results


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="COPD 数据预处理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整处理
  python process_copd_data.py

  # 快速测试（仅处理 3 例）
  python process_copd_data.py --quick-test

  # 限制处理数量
  python process_copd_data.py --limit 5

  # 强制覆盖已有文件
  python process_copd_data.py --force

  # 后台运行
  nohup python process_copd_data.py > logs/copd_preprocess.log 2>&1 &
        """
    )

    parser.add_argument(
        '--input-dir', type=str, default=None,
        help='COPD 原始数据目录（默认: data/00_raw/copd/）'
    )
    parser.add_argument(
        '--output-clean', type=str, default=None,
        help='清洗后 CT 输出目录（默认: data/01_cleaned/copd_clean/）'
    )
    parser.add_argument(
        '--output-mask', type=str, default=None,
        help='肺部 mask 输出目录（默认: data/01_cleaned/copd_mask/）'
    )
    parser.add_argument(
        '--output-lesion', type=str, default=None,
        help='病灶 mask 输出目录（默认: data/01_cleaned/copd_emphysema/）'
    )
    parser.add_argument(
        '--quick-test', action='store_true',
        help='快速测试模式（仅处理 3 例）'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='限制处理数量'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='强制覆盖已有文件'
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
        logger.warning(f"无法加载配置文件: {e}，使用默认路径")
        config = {'paths': {}}

    # 确定路径
    data_root = Path(config.get('paths', {}).get('raw_data', 'data/00_raw'))
    cleaned_root = Path(config.get('paths', {}).get('cleaned_data', 'data/01_cleaned'))

    input_dir = Path(args.input_dir) if args.input_dir else data_root / 'copd'
    mask_output_dir = Path(args.output_mask) if args.output_mask else cleaned_root / 'copd_mask'
    clean_output_dir = Path(args.output_clean) if args.output_clean else cleaned_root / 'copd_clean'
    lesion_output_dir = Path(args.output_lesion) if args.output_lesion else cleaned_root / 'copd_emphysema'

    # 确定处理数量
    limit = args.limit
    if args.quick_test:
        limit = 3

    # =========================================================================
    # 打印 Banner
    # =========================================================================
    logger.info("=" * 60)
    logger.info("  COPD 数据预处理")
    logger.info("=" * 60)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录:")
    logger.info(f"  - 肺部 mask: {mask_output_dir}")
    logger.info(f"  - 清洗 CT: {clean_output_dir}")
    logger.info(f"  - 病灶 mask: {lesion_output_dir}")
    if limit:
        logger.info(f"处理限制: {limit} 例")
    if args.force:
        logger.info("覆盖模式: 是")

    # 检查输入目录
    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        sys.exit(1)

    nifti_count = len(list(input_dir.glob("*.nii.gz")))
    if nifti_count == 0:
        logger.error(f"输入目录为空: {input_dir}")
        sys.exit(1)

    logger.info(f"找到 {nifti_count} 个 COPD CT 文件")

    # 检查依赖
    logger.info("")
    logger.info("[Step 1] 检查依赖")
    logger.info("-" * 40)
    if not check_dependencies(logger):
        logger.error("依赖检查失败，退出")
        sys.exit(1)

    # 执行处理
    logger.info("")
    logger.info("[Step 2] 批量处理")
    logger.info("-" * 40)

    results = process_copd_batch(
        input_dir=input_dir,
        mask_output_dir=mask_output_dir,
        clean_output_dir=clean_output_dir,
        lesion_output_dir=lesion_output_dir,
        logger=logger,
        limit=limit,
        force=args.force
    )

    # =========================================================================
    # 总结
    # =========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("处理完成统计")
    logger.info("=" * 60)
    logger.info(f"总耗时: {results['total_time']/60:.1f} 分钟")
    logger.info(f"成功: {len(results['success'])} 例")
    logger.info(f"跳过: {len(results['skipped'])} 例")
    logger.info(f"失败: {len(results['failed'])} 例")

    logger.info("")
    logger.info("输出目录:")
    logger.info(f"  - {mask_output_dir} ({len(list(mask_output_dir.glob('*_mask.nii.gz')))} 个文件)")
    logger.info(f"  - {clean_output_dir} ({len(list(clean_output_dir.glob('*_clean.nii.gz')))} 个文件)")
    logger.info(f"  - {lesion_output_dir} ({len(list(lesion_output_dir.glob('*_emphysema.nii.gz')))} 个文件)")

    # 验证结果
    if results['success']:
        logger.info("")
        logger.info("病灶统计摘要:")
        lesion_stats = [r['stats'].get('lesion_voxels', 0) for r in results['success']]
        laa_stats = [r['stats'].get('laa_percentage', 0) for r in results['success']]
        logger.info(f"  病灶体素: min={min(lesion_stats):,}, max={max(lesion_stats):,}, avg={sum(lesion_stats)/len(lesion_stats):,.0f}")
        logger.info(f"  LAA%: min={min(laa_stats):.1f}%, max={max(laa_stats):.1f}%, avg={sum(laa_stats)/len(laa_stats):.1f}%")

    if results['failed']:
        logger.warning("")
        logger.warning("失败的文件:")
        for r in results['failed']:
            logger.warning(f"  - {r['patient_id']}: {r.get('error', 'Unknown')}")

    logger.info("")
    logger.info("下一步:")
    logger.info("  1. 运行 Phase 3 流水线: python run_phase3_pipeline.py --check-only")
    logger.info("  2. 执行空间映射: python run_phase3_pipeline.py --quick-test")


if __name__ == "__main__":
    main()

