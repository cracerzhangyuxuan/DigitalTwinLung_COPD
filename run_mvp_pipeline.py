#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MVP 流水线 - 第一阶段

完整流程：DICOM -> NIfTI -> 肺分割 -> 背景清洗 -> 病灶提取 -> 配准 -> 可视化

使用说明：
    python run_mvp_pipeline.py --all          # 运行全部步骤
    python run_mvp_pipeline.py --step 1       # 只运行步骤1
    python run_mvp_pipeline.py --step 1,2,3   # 运行步骤1-3
"""

import argparse
import sys
from pathlib import Path
import shutil
import importlib

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from src.utils.logger import setup_logger

logger = setup_logger("mvp_pipeline")


def import_module_by_path(module_path: str):
    """动态导入模块（支持以数字开头的模块名）"""
    return importlib.import_module(module_path)


def load_config():
    """加载配置"""
    with open("config.yaml", 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def step1_dicom_to_nifti(config: dict):
    """步骤1: DICOM 转 NIfTI"""
    logger.info("=" * 60)
    logger.info("步骤 1: DICOM 转 NIfTI")
    logger.info("=" * 60)

    from src.utils.io import load_dicom_series, save_nifti
    import numpy as np

    suffix = '_exp' if config.get('_use_expiration', False) else ''
    raw_dir = Path(config['paths']['raw_data'])
    cleaned_dir = Path(config['paths']['cleaned_data'])

    # 处理正常肺（目录固定，文件名通过 suffix 区分相位）
    normal_input = raw_dir / 'normal'
    normal_output = cleaned_dir / 'normal_nifti'
    normal_output.mkdir(parents=True, exist_ok=True)

    if normal_input.exists():
        subdirs = [d for d in normal_input.iterdir() if d.is_dir()]
        logger.info(f"找到 {len(subdirs)} 个正常肺 DICOM 目录")

        for i, subdir in enumerate(subdirs, start=1):
            output_path = normal_output / f"normal_{i:03d}{suffix}.nii.gz"
            try:
                volume, metadata = load_dicom_series(subdir)
                # 创建简单的仿射矩阵
                spacing = metadata.get('PixelSpacing', [1.0, 1.0])
                slice_thickness = metadata.get('SliceThickness', 1.0)
                affine = np.diag([spacing[0], spacing[1], slice_thickness, 1.0])
                save_nifti(volume, output_path, affine=affine)
                logger.info(f"[{i}/{len(subdirs)}] {subdir.name} -> {output_path.name}")
            except Exception as e:
                logger.error(f"转换失败 {subdir.name}: {e}")

    # 处理 COPD（目录固定，文件名通过 suffix 区分相位）
    copd_input = raw_dir / 'copd'
    copd_output = cleaned_dir / 'copd_nifti'
    copd_output.mkdir(parents=True, exist_ok=True)

    if copd_input.exists():
        subdirs = [d for d in copd_input.iterdir() if d.is_dir()]
        logger.info(f"找到 {len(subdirs)} 个 COPD DICOM 目录")

        for i, subdir in enumerate(subdirs, start=1):
            output_path = copd_output / f"copd_{i:03d}{suffix}.nii.gz"
            try:
                volume, metadata = load_dicom_series(subdir)
                spacing = metadata.get('PixelSpacing', [1.0, 1.0])
                slice_thickness = metadata.get('SliceThickness', 1.0)
                affine = np.diag([spacing[0], spacing[1], slice_thickness, 1.0])
                save_nifti(volume, output_path, affine=affine)
                logger.info(f"[{i}/{len(subdirs)}] {subdir.name} -> {output_path.name}")
            except Exception as e:
                logger.error(f"转换失败 {subdir.name}: {e}")

    logger.info("步骤 1 完成!")


def step2_lung_segmentation(config: dict):
    """步骤2: 肺部分割"""
    logger.info("=" * 60)
    logger.info("步骤 2: 肺部分割（阈值方法）")
    logger.info("=" * 60)

    # 使用 importlib 导入以数字开头的模块
    simple_lung_segment = importlib.import_module("src.01_preprocessing.simple_lung_segment")
    batch_segment_lungs = simple_lung_segment.batch_segment_lungs

    cleaned_dir = Path(config['paths']['cleaned_data'])

    # 分割正常肺（目录固定，吸气相/呼气相文件共存于同一目录）
    normal_nifti = cleaned_dir / 'normal_nifti'
    if normal_nifti.exists():
        logger.info("分割正常肺...")
        batch_segment_lungs(
            normal_nifti,
            mask_output_dir=cleaned_dir / 'normal_mask',
            clean_output_dir=cleaned_dir / 'normal_clean'
        )

    # 分割 COPD
    copd_nifti = cleaned_dir / 'copd_nifti'
    if copd_nifti.exists():
        logger.info("分割 COPD...")
        batch_segment_lungs(
            copd_nifti,
            mask_output_dir=cleaned_dir / 'copd_mask',
            clean_output_dir=cleaned_dir / 'copd_clean'
        )

    logger.info("步骤 2 完成!")


def step3_extract_emphysema(config: dict):
    """步骤3: 提取肺气肿病灶"""
    logger.info("=" * 60)
    logger.info("步骤 3: 提取肺气肿病灶 (LAA-950)")
    logger.info("=" * 60)

    # 使用 importlib 导入以数字开头的模块
    extract_emphysema = importlib.import_module("src.01_preprocessing.extract_emphysema")
    compute_laa950 = extract_emphysema.compute_laa950
    remove_small_components = extract_emphysema.remove_small_components

    from src.utils.io import load_nifti, save_nifti
    import numpy as np

    cleaned_dir = Path(config['paths']['cleaned_data'])
    threshold = config.get('preprocessing', {}).get('laa_threshold', -950)
    min_size = config.get('preprocessing', {}).get('min_lesion_size', 100)

    copd_clean_dir = cleaned_dir / 'copd_clean'
    copd_mask_dir  = cleaned_dir / 'copd_mask'
    emphysema_dir  = cleaned_dir / 'copd_emphysema'
    emphysema_dir.mkdir(parents=True, exist_ok=True)

    if copd_clean_dir.exists():
        # 按文件名筛选对应相位
        all_ct = list(copd_clean_dir.glob("*.nii.gz"))
        use_exp = config.get('_use_expiration', False)
        if use_exp:
            ct_files = [f for f in all_ct if '_exp' in f.stem]
        else:
            ct_files = [f for f in all_ct if '_exp' not in f.stem]
        logger.info(f"找到 {len(ct_files)} 个 COPD CT 文件")

        for ct_path in ct_files:
            stem = ct_path.name.replace('_clean.nii.gz', '').replace('.nii.gz', '')
            mask_path = copd_mask_dir / f"{stem}_mask.nii.gz"

            if not mask_path.exists():
                # 尝试其他命名
                mask_candidates = list(copd_mask_dir.glob(f"*{stem}*.nii.gz"))
                if mask_candidates:
                    mask_path = mask_candidates[0]

            try:
                ct_data, affine = load_nifti(ct_path, return_affine=True)

                if mask_path.exists():
                    lung_mask = load_nifti(mask_path)
                else:
                    logger.warning(f"未找到 mask: {mask_path}, 使用全图阈值")
                    lung_mask = np.ones_like(ct_data, dtype=np.uint8)

                # 计算 LAA-950 (返回 mask 和百分比)
                emphysema_mask, laa_percentage = compute_laa950(ct_data, lung_mask, threshold=threshold)

                # 去除小连通分量
                emphysema_mask = remove_small_components(emphysema_mask, min_volume_mm3=min_size, voxel_spacing=(1.0, 1.0, 1.0))

                # 保存
                output_path = emphysema_dir / f"{stem}_emphysema.nii.gz"
                save_nifti(emphysema_mask, output_path, affine=affine, dtype='uint8')

                logger.info(f"{stem}: 肺气肿占比 {laa_percentage:.2f}%")

            except Exception as e:
                logger.error(f"处理失败 {ct_path.name}: {e}")
                import traceback
                traceback.print_exc()

    logger.info("步骤 3 完成!")


def step4_create_template(config: dict):
    """步骤4: 创建临时模板（使用第一个正常肺）"""
    logger.info("=" * 60)
    logger.info("步骤 4: 创建临时模板")
    logger.info("=" * 60)
    
    cleaned_dir = Path(config['paths']['cleaned_data'])
    atlas_dir = Path(config['paths']['atlas'])
    atlas_dir.mkdir(parents=True, exist_ok=True)

    template_path = atlas_dir / 'temp_template.nii.gz'
    template_mask_path = atlas_dir / 'temp_template_mask.nii.gz'

    normal_clean_dir = cleaned_dir / 'normal_clean'
    normal_mask_dir  = cleaned_dir / 'normal_mask'

    if normal_clean_dir.exists():
        # 按文件名筛选对应相位
        all_ct = sorted(normal_clean_dir.glob("*.nii.gz"))
        use_exp = config.get('_use_expiration', False)
        if use_exp:
            ct_files = [f for f in all_ct if '_exp' in f.stem]
        else:
            ct_files = [f for f in all_ct if '_exp' not in f.stem]
        if ct_files:
            # 使用第一个正常肺作为模板
            src_ct = ct_files[0]
            shutil.copy(src_ct, template_path)
            logger.info(f"模板 CT: {src_ct.name} -> {template_path.name}")
            
            # 复制对应的 mask
            stem = src_ct.name.replace('_clean.nii.gz', '').replace('.nii.gz', '')
            mask_candidates = list(normal_mask_dir.glob(f"*{stem}*.nii.gz"))
            if mask_candidates:
                shutil.copy(mask_candidates[0], template_mask_path)
                logger.info(f"模板 Mask: {mask_candidates[0].name}")
        else:
            logger.error("未找到正常肺 CT 文件!")
    
    logger.info("步骤 4 完成!")


def step5_register_copd(config: dict):
    """步骤5: 将 COPD 配准到模板"""
    logger.info("=" * 60)
    logger.info("步骤 5: COPD 配准到模板")
    logger.info("=" * 60)

    # 使用 importlib 导入以数字开头的模块
    register_sitk = importlib.import_module("src.03_registration.register_sitk")
    register_copd_to_template = register_sitk.register_copd_to_template

    cleaned_dir = Path(config['paths']['cleaned_data'])
    atlas_dir   = Path(config['paths']['atlas'])
    mapped_dir  = Path(config['paths']['mapped'])
    mapped_dir.mkdir(parents=True, exist_ok=True)

    template_path = atlas_dir / 'temp_template.nii.gz'

    if not template_path.exists():
        logger.error(f"模板不存在: {template_path}")
        return

    copd_clean_dir = cleaned_dir / 'copd_clean'
    emphysema_dir  = cleaned_dir / 'copd_emphysema'

    if copd_clean_dir.exists():
        # 按文件名筛选对应相位
        all_ct = list(copd_clean_dir.glob("*.nii.gz"))
        use_exp = config.get('_use_expiration', False)
        if use_exp:
            ct_files = [f for f in all_ct if '_exp' in f.stem]
        else:
            ct_files = [f for f in all_ct if '_exp' not in f.stem]
        logger.info(f"找到 {len(ct_files)} 个 COPD CT 文件")
        
        for ct_path in ct_files:
            stem = ct_path.name.replace('_clean.nii.gz', '').replace('.nii.gz', '')
            
            # 查找病灶 mask
            lesion_path = emphysema_dir / f"{stem}_emphysema.nii.gz"
            if not lesion_path.exists():
                lesion_candidates = list(emphysema_dir.glob(f"*{stem}*.nii.gz"))
                if lesion_candidates:
                    lesion_path = lesion_candidates[0]
            
            if not lesion_path.exists():
                logger.warning(f"未找到病灶 mask: {stem}")
                continue
            
            try:
                patient_output = mapped_dir / stem
                register_copd_to_template(
                    template_path=template_path,
                    copd_ct_path=ct_path,
                    copd_lesion_path=lesion_path,
                    output_dir=patient_output,
                    transform_type="affine"
                )
                logger.info(f"配准完成: {stem}")
            except Exception as e:
                logger.error(f"配准失败 {stem}: {e}")
    
    logger.info("步骤 5 完成!")


def step6_visualize(config: dict):
    """步骤6: 可视化验证"""
    logger.info("=" * 60)
    logger.info("步骤 6: 可视化验证")
    logger.info("=" * 60)

    # 使用 importlib 导入以数字开头的模块
    static_render = importlib.import_module("src.05_visualization.static_render")
    render_static = static_render.render_static

    atlas_dir = Path(config['paths']['atlas'])
    mapped_dir = Path(config['paths']['mapped'])
    final_viz_dir = Path(config['paths']['final_viz'])
    final_viz_dir.mkdir(parents=True, exist_ok=True)
    
    template_path = atlas_dir / 'temp_template.nii.gz'
    
    # 渲染每个配准后的 COPD
    if mapped_dir.exists():
        patient_dirs = [d for d in mapped_dir.iterdir() if d.is_dir()]
        logger.info(f"找到 {len(patient_dirs)} 个配准结果")
        
        for patient_dir in patient_dirs:
            # 查找配准后的病灶
            lesion_files = list(patient_dir.glob("*_emphysema_warped.nii.gz"))
            if not lesion_files:
                lesion_files = list(patient_dir.glob("*warped*.nii.gz"))
            
            if lesion_files and template_path.exists():
                output_path = final_viz_dir / f"{patient_dir.name}_render.png"
                try:
                    render_static(
                        ct_path=template_path,
                        lesion_mask_path=lesion_files[0],
                        output_path=output_path,
                        show=False
                    )
                    logger.info(f"渲染完成: {output_path.name}")
                except Exception as e:
                    logger.error(f"渲染失败 {patient_dir.name}: {e}")
    
    logger.info("步骤 6 完成!")
    logger.info("=" * 60)
    logger.info("🎉 MVP 流水线全部完成!")
    logger.info(f"请查看渲染结果: {final_viz_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="MVP 流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用方法:
  python run_mvp_pipeline.py --all          # 运行全部步骤（吸气相）
  python run_mvp_pipeline.py --step 1       # 只运行步骤1
  python run_mvp_pipeline.py --step 1,2,3   # 运行步骤1-3
  python run_mvp_pipeline.py --all --expiration  # 使用呼气相数据流
        """
    )
    parser.add_argument('--all', action='store_true', help='运行全部步骤')
    parser.add_argument('--step', type=str, help='运行指定步骤 (如: 1 或 1,2,3)')
    parser.add_argument(
        '--expiration',
        action='store_true',
        default=False,
        help='使用呼气相数据流（默认使用吸气相数据流）。\n'
             '文件名自动插入 _exp 中缀（如 normal_001_exp.nii.gz），\n'
             '与吸气相数据共享同一目录，不会覆盖任何吸气相文件。'
    )
    args = parser.parse_args()

    config = load_config()

    # 呼气相模式：不修改任何目录路径，仅在各函数中通过文件名筛选区分相位
    if args.expiration:
        config['_use_expiration'] = True
        logger.info("[呼气相模式] 文件名中缀: '_exp'，目录路径不变")
    else:
        config['_use_expiration'] = False

    steps = {
        1: ("DICOM 转 NIfTI", step1_dicom_to_nifti),
        2: ("肺部分割", step2_lung_segmentation),
        3: ("提取肺气肿", step3_extract_emphysema),
        4: ("创建模板", step4_create_template),
        5: ("COPD 配准", step5_register_copd),
        6: ("可视化验证", step6_visualize),
    }

    if args.all:
        for step_num, (name, func) in steps.items():
            func(config)
    elif args.step:
        step_nums = [int(s.strip()) for s in args.step.split(',')]
        for step_num in step_nums:
            if step_num in steps:
                name, func = steps[step_num]
                func(config)
            else:
                logger.error(f"未知步骤: {step_num}")
    else:
        print("使用方法:")
        print("  python run_mvp_pipeline.py --all          # 运行全部步骤")
        print("  python run_mvp_pipeline.py --step 1       # 只运行步骤1")
        print("  python run_mvp_pipeline.py --step 1,2,3   # 运行步骤1-3")
        print("  python run_mvp_pipeline.py --all --expiration  # 呼气相模式")
        print("\n可用步骤:")
        for num, (name, _) in steps.items():
            print(f"  {num}: {name}")


if __name__ == "__main__":
    main()

