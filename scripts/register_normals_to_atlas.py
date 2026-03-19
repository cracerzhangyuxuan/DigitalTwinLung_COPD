#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤一：将健康人 CT 配准到标准图谱模板

输入:
  data/01_cleaned/normal_clean/normal_*_clean.nii.gz    -- 健康人原始 CT
  data/01_cleaned/normal_mask/normal_*_mask.nii.gz      -- 全肺二值 mask
  data/01_cleaned/normal_mask/normal_*_trachea_mask.nii.gz  -- 气管树 mask
  data/01_cleaned/normal_mask/normal_*_lung_lobes_labeled.nii.gz -- 肺叶标签 (1-5)
  data/02_atlas/standard_template_with_airway.nii.gz    -- 标准模板 (固定像)

输出 (data/04_normal_mapped/normal_XXX/):
  normal_XXX_warped.nii.gz          -- 配准后 CT (用于 Wasserstein 对比)
  normal_XXX_transform_0.nii.gz     -- ANTs SyN 非线性形变场 (用于 Jacobian)
  normal_XXX_transform_1.mat        -- ANTs 仿射变换矩阵
  normal_XXX_warped_lung_mask.nii.gz    -- 配准后全肺 mask (最近邻插值)
  normal_XXX_warped_trachea.nii.gz      -- 配准后气管树 mask (最近邻插值)
  normal_XXX_warped_lobes.nii.gz        -- 配准后肺叶标签 (最近邻插值, 保持 1-5 整数)

使用方法:
    python scripts/register_normals_to_atlas.py
    python scripts/register_normals_to_atlas.py --limit 2   # 先用少量测试
"""

import sys
import gc
import time
import argparse
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logger():
    logger = logging.getLogger('reg_normals')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(h)
    return logger


def register_one_subject(ct_path, pid, template_path, mask_dir, output_dir, logger,
                         transform_type='SyNRA', reg_iterations=(20, 10, 0)):
    """配准单个健康人受试者并保存所有输出文件"""
    try:
        import ants
    except ImportError:
        logger.error("请安装 antspyx: pip install antspyx")
        return False

    subj_output = output_dir / pid
    subj_output.mkdir(parents=True, exist_ok=True)
    warped_ct_path = subj_output / f'{pid}_warped.nii.gz'

    # 已完成则跳过（支持断点续传）
    if warped_ct_path.exists():
        logger.info(f"  [{pid}] 已存在，跳过")
        return True

    t0 = time.time()
    logger.info(f"  [{pid}] 开始配准...")

    moving = ants.image_read(str(ct_path))
    fixed = ants.image_read(str(template_path))
    out_prefix = str(subj_output / f'{pid}_transform_')

    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=transform_type,
        outprefix=out_prefix,
        reg_iterations=reg_iterations,
        write_composite_transform=False,
    )

    ants.image_write(reg['warpedmovout'], str(warped_ct_path))

    # 对 mask/label 文件应用相同变换（最近邻插值保持标签整数值）
    fwd_transforms = reg['fwdtransforms']
    mask_files = {
        'lung_mask':     (f'{pid}_mask.nii.gz',               f'{pid}_warped_lung_mask.nii.gz'),
        'trachea_mask':  (f'{pid}_trachea_mask.nii.gz',       f'{pid}_warped_trachea.nii.gz'),
        'lobes_labeled': (f'{pid}_lung_lobes_labeled.nii.gz', f'{pid}_warped_lobes.nii.gz'),
    }
    for key, (src_name, dst_name) in mask_files.items():
        src_path = mask_dir / src_name
        dst_path = subj_output / dst_name
        if not src_path.exists():
            logger.warning(f"    [{pid}] 缺少 {src_name}，跳过")
            continue
        moving_mask = ants.image_read(str(src_path))
        warped_mask = ants.apply_transforms(
            fixed=fixed,
            moving=moving_mask,
            transformlist=fwd_transforms,
            interpolator='nearestNeighbor',  # 保持标签整数，不使用线性插值
        )
        ants.image_write(warped_mask, str(dst_path))

    elapsed = time.time() - t0
    logger.info(f"  [{pid}] done ({elapsed/60:.1f} min)  => {subj_output.name}/")
    gc.collect()
    return True


def main():
    parser = argparse.ArgumentParser(
        description='将健康人 CT 配准到标准图谱模板',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--normal-ct-dir', default='data/01_cleaned/normal_clean')
    parser.add_argument('--normal-mask-dir', default='data/01_cleaned/normal_mask')
    parser.add_argument('--atlas-dir', default='data/02_atlas')
    parser.add_argument('--output-dir', default='data/04_normal_mapped')
    parser.add_argument('--limit', type=int, default=None, help='限制处理数量（测试用）')
    parser.add_argument('--transform-type', default='SyNRA')
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("健康人 -> 模板配准 (步骤一)")
    logger.info("=" * 60)

    normal_ct_dir = Path(args.normal_ct_dir)
    normal_mask_dir = Path(args.normal_mask_dir)
    template_path = Path(args.atlas_dir) / 'standard_template_with_airway.nii.gz'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not template_path.exists():
        logger.error(f"模板不存在: {template_path}")
        sys.exit(1)

    ct_files = sorted(normal_ct_dir.glob('normal_*_clean.nii.gz'))
    if args.limit:
        ct_files = ct_files[:args.limit]

    logger.info(f"  模板: {template_path.name}")
    logger.info(f"  待处理: {len(ct_files)} 例")
    logger.info(f"  输出目录: {output_dir}")
    logger.info("")

    success_count, fail_count = 0, 0
    for ct_path in ct_files:
        # 正确处理双扩展名 .nii.gz：
        # ct_path.name = "normal_001_clean.nii.gz"
        # ct_path.stem = "normal_001_clean.nii"  ← 只去掉最后一个 .gz，不够
        # 需要手动去掉 .nii.gz 后再去掉 _clean
        name_no_gz = ct_path.name  # "normal_001_clean.nii.gz"
        if name_no_gz.endswith('.nii.gz'):
            name_no_gz = name_no_gz[:-7]          # "normal_001_clean"
        pid = name_no_gz.replace('_clean', '')     # "normal_001"
        ok = register_one_subject(
            ct_path=ct_path, pid=pid,
            template_path=template_path,
            mask_dir=normal_mask_dir,
            output_dir=output_dir,
            logger=logger,
            transform_type=args.transform_type,
        )
        (success_count if ok else fail_count).__class__  # noqa
        if ok:
            success_count += 1
        else:
            fail_count += 1

    logger.info("")
    logger.info(f"{'='*60}")
    logger.info(f"完成: {success_count} 成功 / {fail_count} 失败")
    logger.info(f"结果目录: {output_dir}")
    logger.info("下一步:")
    logger.info("  python scripts/evaluate_atlas_quality.py "
                "--normal-mapped-dir data/04_normal_mapped")


if __name__ == '__main__':
    main()

