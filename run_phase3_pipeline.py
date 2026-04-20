#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 3 端到端流水线（病理映射与 AI 融合）

功能：
    1. 环境检查：验证 ANTsPy、PyTorch 可用性
    2. 数据验证：检查 COPD 数据和标准底座完整性
    3. 空间映射：将 COPD 病灶配准到标准底座空间
    4. 可视化验证：生成映射结果对比图
    5. AI 纹理融合训练：训练 Inpainting 模型 (Phase 3B)
    6. AI 纹理融合推理：生成融合后的数字孪生 CT

使用方法：
    # 完整流水线 (3A + 3B 训练 + 推理)
    python run_phase3_pipeline.py --full

    # 仅 Phase 3A（默认，空间映射 + 可视化）
    python run_phase3_pipeline.py

    # 快速测试（仅处理 3 例）
    python run_phase3_pipeline.py --quick-test

    # 跳过配准（使用已有结果）
    python run_phase3_pipeline.py --skip-registration

    # 仅执行可视化
    python run_phase3_pipeline.py --viz-only

    # Phase 3B 训练
    python run_phase3_pipeline.py --phase3b --model-type unet --epochs 50

    # Phase 3B 推理
    python run_phase3_pipeline.py --inference --checkpoint checkpoints/best.pth

    # 限制处理数量
    python run_phase3_pipeline.py --limit 3

    # ============================================================
    # 推理命令（3条）
    # ============================================================

    # UNet 模型推理
    python run_phase3_pipeline.py --inference --model-type unet --limit 3

    # Partial Conv 模型推理
    python run_phase3_pipeline.py --inference --model-type partial_conv

    # PatchGAN 模型推理
    python run_phase3_pipeline.py --inference --model-type patchgan


    # ============================================================
    # 评估命令（3条）
    # ============================================================

    # UNet 模型评估
    python run_phase3_pipeline.py --evaluate --model-type unet --limit 3

    # Partial Conv 模型评估
    python run_phase3_pipeline.py --evaluate --model-type partial_conv --limit 3

    # PatchGAN 模型评估
    python run_phase3_pipeline.py --evaluate --model-type patchgan --limit 3


    # ============================================================
    # 可视化命令（3条）
    # ============================================================

    # UNet 结果可视化
    python run_phase3_pipeline.py --visualize --model-type unet

    # Partial Conv 结果可视化
    python run_phase3_pipeline.py --visualize --model-type partial_conv

    # PatchGAN 结果可视化 #vertical——2行4列  horizontal——4行2列
    python run_phase3_pipeline.py --visualize --model-type patchgan --limit 3


    # ============================================================
    # 验证集评估命令（使用 --start-patient-id 跳过训练集）
    # 训练集: copd_001~023 | 验证集: copd_024~029
    # ============================================================

    # 示例：在验证集 copd_024~026 上评估 PatchGAN
    python run_phase3_pipeline.py --inference  --model-type patchgan --start-patient-id copd_024 --limit 3
    python run_phase3_pipeline.py --evaluate   --model-type patchgan --start-patient-id copd_024 --limit 3
    python run_phase3_pipeline.py --visualize  --model-type patchgan --start-patient-id copd_024 --limit 3


    # ============================================================
    # ★ 新模型完整工作流命令 ★
    # AttGAN / MAE-PatchGAN (两阶段) / DDPM
    # ============================================================

    # ---- [1] AttGAN: 注意力增强 GAN (AttentionUNet + PatchDiscriminator) ----
    # 使用 GPU:1 训练/推理（两种方式二选一）:
    #   A. 直接指定设备: --device cuda:1
    #   B. 或先设置环境变量: set CUDA_VISIBLE_DEVICES=1  然后使用 --device cuda
    # 训练 (推荐 150 epochs，约 2~3 小时)
    python run_phase3_pipeline.py --phase3b --model-type attgan --epochs 150 --device cuda:1
    # 推理 + 评估 + 可视化 (验证集 copd_024~026)
    python run_phase3_pipeline.py --inference  --model-type attgan --device cuda:1 --start-patient-id copd_024 --limit 3
    python run_phase3_pipeline.py --evaluate   --model-type attgan --device cuda:1 --start-patient-id copd_024 --limit 3
    python run_phase3_pipeline.py --visualize  --model-type attgan --device cuda:1 --start-patient-id copd_024 --limit 3

    # ---- [2] MAE-PatchGAN: 自监督预训练 + PatchGAN 微调 (两阶段) ----
    # 阶段 1: MAE 自监督预训练 (推荐 100 epochs，约 1~2 小时)
    #   → 输出: checkpoints/mae_pretrain/encoder_weights.pth
    python run_phase3_pipeline.py --mae-pretrain --epochs 100 --device cuda:1
    # 阶段 2: PatchGAN 微调 (加载预训练 encoder，推荐 80 epochs)
    #   → 自动从 checkpoints/mae_pretrain/encoder_weights.pth 加载
    python run_phase3_pipeline.py --phase3b --model-type mae_patchgan --epochs 80 --device cuda:1
    # 推理 + 评估 + 可视化 (验证集 copd_024~026)
    python run_phase3_pipeline.py --inference  --model-type mae_patchgan --device cuda:1 --start-patient-id copd_024 --limit 3
    python run_phase3_pipeline.py --evaluate   --model-type mae_patchgan --device cuda:1 --start-patient-id copd_024 --limit 3
    python run_phase3_pipeline.py --visualize  --model-type mae_patchgan --device cuda:1 --start-patient-id copd_024 --limit 3

    # ---- [3] DDPM: 去噪扩散概率模型 (DiffusionUNet) ----
    # 训练 (推荐 200 epochs，约 4~6 小时; 使用 EMA + 梯度裁剪)
    python run_phase3_pipeline.py --phase3b --model-type ddpm --epochs 200 --device cuda:1
    # 推理 + 评估 + 可视化 (验证集 copd_024~026)
    python run_phase3_pipeline.py --inference  --model-type ddpm --device cuda:1 --start-patient-id copd_024 --limit 3
    python run_phase3_pipeline.py --evaluate   --model-type ddpm --device cuda:1 --start-patient-id copd_024 --limit 3
    python run_phase3_pipeline.py --visualize  --model-type ddpm --device cuda:1 --start-patient-id copd_024 --limit 3




    # ---- [4] 验证集 L1-L4 跨模型评估 (训练完成后) ----
    python scripts/evaluate_validation_l1_l4.py
    python scripts/generate_validation_charts.py


作者：DigitalTwinLung COPD Project
日期：2025-12-30
更新：2026-01-07 (添加 Phase 3B 支持)
更新：2026-02-XX (添加 AttGAN/MAE-PatchGAN/DDPM 新模型、MAE 预训练入口、数据集硬编码划分)
"""

import sys
import argparse
import importlib
import time
import gc
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict
import logging

import yaml

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 高级纹理评估指标
from src.utils.metrics_advanced import compute_sharpness, compute_boundary_continuity, compute_glcm_features


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
    """配置日志（强制 UTF-8，避免 Windows 终端乱码）"""
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"phase3_pipeline_{timestamp}.log"

    # 强制 stdout 使用 UTF-8（Python 3.7+）
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

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

    # 控制台处理器（显式指定 sys.stdout + UTF-8）
    console_stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8',
                          errors='replace', closefd=False) \
        if hasattr(sys.stdout, 'fileno') else sys.stdout
    console_handler = logging.StreamHandler(console_stream)
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
    quick_test: bool = False  # noqa: ARG001 - 保留用于未来扩展
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

    # 选择模板：优先使用包含气管树的完整模板
    reg_config = config.get('registration', {}).get('lesion_registration', {})
    airway_fusion_config = config.get('registration', {}).get('airway_fusion', {})

    use_airway_template = reg_config.get('use_airway_template', True)
    airway_template_filename = airway_fusion_config.get(
        'output_filename', 'standard_template_with_airway.nii.gz'
    )
    airway_template_path = atlas_dir / airway_template_filename
    original_template_path = atlas_dir / 'standard_template.nii.gz'

    # 如果启用气管树模板且文件存在，则使用它
    if use_airway_template and airway_template_path.exists():
        template_path = airway_template_path
        logger.info(f"  使用完整模板（含气管树）: {template_path.name}")
    else:
        template_path = original_template_path
        if use_airway_template and not airway_template_path.exists():
            logger.warning(f"  气管树模板不存在: {airway_template_path}")
            logger.warning("  回退到原始模板。请先运行:")
            logger.warning("    python -m src.02_atlas_build.fuse_airway_to_template")
        logger.info(f"  使用原始模板: {template_path.name}")

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

    # 注意: reg_config 已在上方获取，无需重复获取

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

        # 【关键修复】显式释放内存，防止内存累积导致后续配准失败
        gc.collect()
        logger.debug(f"    内存已清理")

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
    """
    生成映射结果可视化

    为每个患者生成三视图渲染图（X/Y/Z 三个轴向）：
    - 显示变形后的病灶 mask 叠加在标准模板上
    - 输出 PNG 图片到 data/03_mapped/visualizations/ 目录

    Args:
        config: 配置字典
        logger: 日志记录器
        limit: 限制可视化数量（默认无限制）

    Returns:
        bool: 是否成功生成可视化
    """
    logger.info("")
    logger.info("[Step 4] 可视化验证")
    logger.info("-" * 40)

    output_dir = Path(config['paths'].get('mapped', 'data/03_mapped'))
    atlas_dir = Path(config['paths'].get('atlas', 'data/02_atlas'))
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 标准模板路径
    template_path = atlas_dir / 'standard_template.nii.gz'
    template_mask_path = atlas_dir / 'standard_mask.nii.gz'

    # 数字肺底座文件（融合标签）
    digital_lung_labels_path = atlas_dir / 'digital_lung_labels.nii.gz'
    digital_lung_meta_path = atlas_dir / 'digital_lung_base.json'

    # 分散文件（向后兼容）
    template_lobes_path = atlas_dir / 'standard_lung_lobes_labeled.nii.gz'
    template_trachea_path = atlas_dir / 'standard_trachea_mask.nii.gz'

    if not template_path.exists():
        logger.warning(f"  模板文件不存在: {template_path}")
        return False

    # 检查数字肺底座是否存在，如果不存在则尝试构建
    use_digital_base = False
    if digital_lung_labels_path.exists() and digital_lung_meta_path.exists():
        use_digital_base = True
        logger.info(f"  数字肺底座: {digital_lung_labels_path.name} (融合标签)")
    else:
        # 尝试构建数字肺底座
        try:
            build_module = importlib.import_module("src.02_atlas_build.build_digital_lung_base")
            logger.info("  数字肺底座不存在，尝试构建...")
            success, info = build_module.build_digital_lung_base(atlas_dir)
            if success:
                use_digital_base = True
                logger.info(f"  数字肺底座: 构建成功")
            else:
                logger.warning(f"  数字肺底座构建失败: {info}")
        except Exception as e:
            logger.warning(f"  无法构建数字肺底座: {e}")

    # 检查肺叶和气管树（从数字肺底座或分散文件）
    if use_digital_base:
        has_lobes = True
        has_trachea = True
        # 从融合标签中提取（在渲染时处理）
        lobes_to_render = str(digital_lung_labels_path)
        trachea_to_render = str(digital_lung_labels_path)
        logger.info(f"  肺叶标签: 从数字肺底座提取 (5 肺叶着色)")
        logger.info(f"  气管树: 从数字肺底座提取 (橙色渲染)")
    else:
        # 向后兼容：使用分散文件
        has_lobes = template_lobes_path.exists()
        has_trachea = template_trachea_path.exists()
        lobes_to_render = str(template_lobes_path) if has_lobes else None
        trachea_to_render = str(template_trachea_path) if has_trachea else None

        if has_lobes:
            logger.info(f"  肺叶标签: {template_lobes_path.name} (5 肺叶着色)")
        else:
            logger.info(f"  肺叶标签: 无 (使用二值 mask)")

        if has_trachea:
            logger.info(f"  气管树: {template_trachea_path.name} (橙色渲染)")
        else:
            logger.info(f"  气管树: 无")

    # 查找已处理的患者
    patient_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name != 'visualizations']
    patient_dirs = sorted(patient_dirs)

    if limit:
        patient_dirs = patient_dirs[:limit]

    if len(patient_dirs) == 0:
        logger.warning("  没有找到映射结果")
        return False

    logger.info(f"  待可视化: {len(patient_dirs)} 例")
    logger.info(f"  标准模板: {template_path}")
    logger.info(f"  输出目录: {viz_dir}")

    # 尝试导入可视化模块
    try:
        viz_module = importlib.import_module("src.05_visualization.static_render")
        has_viz = True
    except ImportError as e:
        logger.warning(f"  无法导入可视化模块: {e}")
        logger.warning("  跳过可视化（请安装: pip install pyvista）")
        has_viz = False

    if not has_viz:
        # 只打印统计信息
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            warped_lesion = patient_dir / f"{patient_id}_warped_lesion.nii.gz"
            if warped_lesion.exists():
                logger.info(f"  ✓ {patient_id}: {warped_lesion.name}")
            else:
                logger.warning(f"  ⚠ {patient_id}: 缺少 warped_lesion")
        return True

    # 执行可视化渲染
    success_count = 0
    fail_count = 0

    for i, patient_dir in enumerate(patient_dirs):
        patient_id = patient_dir.name
        warped_lesion = patient_dir / f"{patient_id}_warped_lesion.nii.gz"

        logger.info(f"  [{i+1}/{len(patient_dirs)}] 渲染 {patient_id}...")

        if not warped_lesion.exists():
            logger.warning(f"    ⚠ 缺少 warped_lesion，跳过")
            fail_count += 1
            continue

        try:
            # 使用 render_multiview 生成三视图渲染
            # 使用标准模板作为背景（因为 warped_ct 可能缺少气管树）
            ct_to_render = str(template_path)  # 使用模板作为背景
            lesion_to_render = str(warped_lesion)
            mask_to_render = str(template_mask_path) if template_mask_path.exists() else None
            # lobes_to_render 和 trachea_to_render 已在前面根据数字肺底座或分散文件设置

            result = viz_module.render_multiview(
                ct_path=ct_to_render,
                lesion_mask_path=lesion_to_render,
                lung_mask_path=mask_to_render,
                output_prefix=patient_id,
                output_dir=str(viz_dir),
                lung_opacity=0.25,  # 肺叶透明度
                lesion_opacity=0.9,  # 病灶不透明
                window_size=(800, 800),
                use_mask_surface=True,
                auto_threshold=True,
                lobes_mask_path=lobes_to_render,  # 5 肺叶着色
                trachea_mask_path=trachea_to_render,  # 气管树
                trachea_color=(0.8, 0.4, 0.2),  # 橙色
                trachea_opacity=0.9,  # 不透明
                use_digital_base=use_digital_base  # 是否使用数字肺底座
            )

            if result:
                logger.info(f"    ✓ 生成: {patient_id}_view_*.png")
                success_count += 1
            else:
                logger.warning(f"    ⚠ 渲染失败")
                fail_count += 1

        except Exception as e:
            logger.error(f"    ✗ 渲染异常: {e}")
            fail_count += 1

    logger.info("")
    logger.info(f"  可视化完成: {success_count}/{len(patient_dirs)} 成功")
    if fail_count > 0:
        logger.warning(f"  失败: {fail_count} 例")
    logger.info(f"  输出目录: {viz_dir}")

    return success_count > 0


# =============================================================================
# Phase 3B: AI 纹理融合
# =============================================================================


def run_mae_pretrain(
    config: dict,
    logger: logging.Logger,
    epochs: int = None,
) -> Tuple[bool, Dict]:
    """
    执行 MAE 自监督预训练（MAE-PatchGAN 的第一阶段）

    使用 InpaintingUNet 在无标签 CT patches 上进行自监督预训练，
    训练完成后导出 encoder 权重到 checkpoints/mae_pretrain/encoder_weights.pth。

    Args:
        config: 配置字典
        logger: 日志记录器
        epochs: 预训练轮数（覆盖配置文件，默认 100）

    Returns:
        Tuple[bool, Dict]: (是否成功, 结果信息)
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("  MAE 自监督预训练 (MAE-PatchGAN 第一阶段)")
    logger.info("=" * 60)

    paths = config.get('paths', {})
    mapped_dir = Path(paths.get('mapped', 'data/03_mapped'))

    # MAE 预训练的检查点目录（固定路径，不随 model_type 变化）
    checkpoint_dir = Path('checkpoints') / 'mae_pretrain'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 覆盖 epochs
    train_config = config.get('training', {})
    mae_config = train_config.get('mae', {})
    if epochs:
        mae_config['pretrain_epochs'] = epochs
    else:
        mae_config.setdefault('pretrain_epochs', 100)
    train_config['mae'] = mae_config
    config['training'] = train_config

    pretrain_epochs = mae_config['pretrain_epochs']
    logger.info(f"  预训练轮数: {pretrain_epochs}")
    logger.info(f"  检查点目录: {checkpoint_dir}")

    try:
        # 构建数据集（使用与训练相同的数据划分逻辑）
        ct_files, mask_files = [], []
        for patient_dir in sorted(mapped_dir.iterdir()):
            if not patient_dir.is_dir() or patient_dir.name == 'visualizations':
                continue
            warped_ct = patient_dir / f"{patient_dir.name}_warped.nii.gz"
            warped_mask = patient_dir / f"{patient_dir.name}_warped_lesion.nii.gz"
            if warped_ct.exists() and warped_mask.exists():
                ct_files.append(warped_ct)
                mask_files.append(warped_mask)

        if not ct_files:
            logger.error("  ✗ 未找到已配准数据，请先运行 Phase 3A")
            return False, {}

        # 硬编码数据集划分（与 train.py 一致）
        TRAIN_CUTOFF = "copd_024"
        train_ct = [f for f in ct_files if f.parent.name < TRAIN_CUTOFF]
        train_mask = [f for f in mask_files if f.parent.name < TRAIN_CUTOFF]
        val_ct = [f for f in ct_files if f.parent.name >= TRAIN_CUTOFF]
        val_mask = [f for f in mask_files if f.parent.name >= TRAIN_CUTOFF]

        if not val_ct:
            val_ct, val_mask = train_ct[-1:], train_mask[-1:]

        logger.info(f"  训练集: {len(train_ct)} 例, 验证集: {len(val_ct)} 例")

        # 创建 DataLoader
        dataset_mod = importlib.import_module("src.04_texture_synthesis.dataset")
        LungPatchDataset = dataset_mod.LungPatchDataset
        from torch.utils.data import DataLoader

        patch_size = tuple(train_config.get('patch_size', [64, 64, 64]))
        batch_size = train_config.get('batch_size', 4)
        num_workers = train_config.get('num_workers', 0)

        train_dataset = LungPatchDataset(
            ct_paths=train_ct, mask_paths=train_mask,
            patch_size=patch_size, augment=True
        )
        val_dataset = LungPatchDataset(
            ct_paths=val_ct, mask_paths=val_mask,
            patch_size=patch_size, augment=False
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers
        )

        # 创建模型和预训练器
        net_mod = importlib.import_module("src.04_texture_synthesis.network")
        mae_mod = importlib.import_module("src.04_texture_synthesis.mae_pretrain")

        model = net_mod.create_model('unet')  # MAE 预训练使用 InpaintingUNet
        pretrainer = mae_mod.MAEPretrainer(model, config)

        # 执行预训练
        pretrainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=pretrain_epochs,
            checkpoint_dir=str(checkpoint_dir),
            save_frequency=20,
        )

        encoder_path = checkpoint_dir / 'encoder_weights.pth'
        logger.info("")
        logger.info("  ✓ MAE 预训练完成!")
        logger.info(f"  Encoder 权重: {encoder_path}")

        return True, {
            'checkpoint_dir': str(checkpoint_dir),
            'encoder_weights': str(encoder_path),
        }

    except KeyboardInterrupt:
        logger.warning("  MAE 预训练被用户中断")
        return False, {}
    except Exception as e:
        logger.error(f"  ✗ MAE 预训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False, {'error': str(e)}


def run_texture_training(
    config: dict,
    logger: logging.Logger,
    model_type: str = 'partial_conv',
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None
) -> Tuple[bool, Dict]:
    """
    执行 Phase 3B AI 纹理融合训练

    Args:
        config: 配置字典
        logger: 日志记录器
        model_type: 模型类型 ('unet', 'partial_conv', 'patchgan')
        epochs: 训练轮数（覆盖配置）
        batch_size: 批次大小（覆盖配置）
        learning_rate: 学习率（覆盖配置）

    Returns:
        Tuple[bool, Dict]: (是否成功, 训练结果)
    """
    logger.info("")
    logger.info("[Step 5] AI 纹理融合训练 (Phase 3B)")
    logger.info("-" * 40)

    # 检查 PyTorch
    torch_ok, torch_msg = check_pytorch()
    if not torch_ok:
        logger.error(f"  ✗ {torch_msg}")
        logger.error("  请安装 PyTorch: pip install torch")
        return False, {}
    logger.info(f"  ✓ {torch_msg}")

    # 检查 Phase 3A 输出
    mapped_dir = Path(config['paths'].get('mapped', 'data/03_mapped'))
    if not mapped_dir.exists():
        logger.error(f"  ✗ 配准输出目录不存在: {mapped_dir}")
        logger.error("  请先运行 Phase 3A")
        return False, {}

    # 统计已配准的数据
    patient_count = 0
    for patient_dir in mapped_dir.iterdir():
        if not patient_dir.is_dir() or patient_dir.name == 'visualizations':
            continue
        warped_ct = patient_dir / f"{patient_dir.name}_warped.nii.gz"
        warped_mask = patient_dir / f"{patient_dir.name}_warped_lesion.nii.gz"
        if warped_ct.exists() and warped_mask.exists():
            patient_count += 1

    if patient_count == 0:
        logger.error("  ✗ 未找到已配准的数据，请先运行 Phase 3A")
        return False, {}

    logger.info(f"  ✓ 已配准数据: {patient_count} 例")

    # 更新配置
    train_config = config.get('training', {})
    if model_type:
        train_config['model_type'] = model_type
    if epochs:
        train_config['epochs'] = epochs
    if batch_size:
        train_config['batch_size'] = batch_size
    if learning_rate:
        train_config['learning_rate'] = learning_rate
    config['training'] = train_config

    # 根据模型类型创建独立的检查点目录
    checkpoint_base = Path(config['paths'].get('checkpoints', 'checkpoints'))
    checkpoint_dir = checkpoint_base / model_type  # 添加模型类型子目录
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 更新配置中的检查点路径
    config['paths']['checkpoints'] = str(checkpoint_dir)
    logger.info(f"  检查点目录: {checkpoint_dir}")

    # 显示模型类型
    model_names = {
        'unet': '基线方案 (3D U-Net)',
        'partial_conv': '进阶方案 (Partial Conv)',
        'patchgan': '高级方案 (PatchGAN)',
        'attgan': '注意力增强方案 (AttGAN)',
        'mae_patchgan': '自监督预训练方案 (MAE-PatchGAN)',
        'ddpm': '扩散模型方案 (DDPM)'
    }
    logger.info(f"  模型类型: {model_names.get(model_type, model_type)}")
    logger.info(f"  训练轮数: {train_config.get('epochs', 100)}")
    logger.info(f"  批次大小: {train_config.get('batch_size', 4)}")

    # 导入训练模块
    logger.info("")
    logger.info("  初始化训练...")

    try:
        train_module = importlib.import_module("src.04_texture_synthesis.train")
        train_module.main(config)

        logger.info("")
        logger.info("  ✓ 训练完成!")

        checkpoint_dir = Path(config['paths'].get('checkpoints', 'checkpoints'))
        return True, {
            'checkpoint_dir': str(checkpoint_dir),
            'best_model': str(checkpoint_dir / 'best.pth'),
            'model_type': model_type
        }

    except KeyboardInterrupt:
        logger.warning("  训练被用户中断")
        return False, {}
    except Exception as e:
        logger.error(f"  ✗ 训练失败: {e}")
        return False, {'error': str(e)}


def run_texture_inference(
    config: dict,
    logger: logging.Logger,
    checkpoint_path: str = None,
    patient_id: str = None,
    device: str = 'cuda',
    smooth_boundary: bool = True,
    model_type: str = 'partial_conv',
    limit: int = None,
    start_patient_id: str = None
) -> Tuple[bool, Dict]:
    """
    执行 Phase 3B AI 纹理融合推理

    Args:
        config: 配置字典
        logger: 日志记录器
        checkpoint_path: 模型检查点路径（如果指定，优先使用）
        patient_id: 指定患者 ID（默认处理所有）
        device: 计算设备
        smooth_boundary: 是否平滑边界
        model_type: 模型类型（用于查找对应的检查点目录）
        limit: 限制处理的患者数量（默认处理所有）
        start_patient_id: 起始患者 ID（如 copd_024，从该患者开始处理）

    Returns:
        Tuple[bool, Dict]: (是否成功, 推理结果)
    """
    logger.info("")
    logger.info("[Step 6] AI 纹理融合推理 (Phase 3B)")
    logger.info("-" * 40)

    paths = config.get('paths', {})

    # 确定路径
    template_path = Path(paths.get('atlas', 'data/02_atlas')) / 'standard_template.nii.gz'
    mapped_dir = Path(paths.get('mapped', 'data/03_mapped'))

    # 输出目录：添加模型类型子目录
    base_output_dir = Path(paths.get('final_viz', 'data/04_final_viz'))
    output_dir = base_output_dir / model_type  # 按模型类型分离输出
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定检查点路径（按优先级查找）
    if checkpoint_path:
        # 1. 用户指定的路径
        checkpoint = Path(checkpoint_path)
    else:
        checkpoint_base = Path(paths.get('checkpoints', 'checkpoints'))

        # 2. 模型类型对应的子目录
        model_checkpoint = checkpoint_base / model_type / 'best.pth'

        # 3. 向后兼容：旧的检查点位置
        legacy_checkpoint = checkpoint_base / 'best.pth'

        if model_checkpoint.exists():
            checkpoint = model_checkpoint
            logger.info(f"  使用模型类型 '{model_type}' 的检查点")
        elif legacy_checkpoint.exists():
            checkpoint = legacy_checkpoint
            logger.info(f"  使用旧版检查点位置（向后兼容）")
        else:
            checkpoint = model_checkpoint  # 使用新路径报错

    # 检查文件
    if not checkpoint.exists():
        logger.error(f"  ✗ 模型检查点不存在: {checkpoint}")
        logger.error("  请先运行 Phase 3B 训练: --phase3b")
        logger.error(f"  提示: 检查点应位于 checkpoints/{model_type}/best.pth")
        return False, {}

    if not template_path.exists():
        logger.error(f"  ✗ 模板文件不存在: {template_path}")
        return False, {}

    logger.info(f"  模板: {template_path}")
    logger.info(f"  检查点: {checkpoint}")
    logger.info(f"  输出目录: {output_dir}")

    # 导入推理模块
    try:
        inference_module = importlib.import_module("src.04_texture_synthesis.inference_fuse")
    except ImportError as e:
        logger.error(f"  ✗ 导入推理模块失败: {e}")
        return False, {}

    # 加载模型
    logger.info("  加载模型...")
    try:
        model = inference_module.load_model(checkpoint, device=device, model_type=model_type)
    except Exception as e:
        logger.error(f"  ✗ 加载模型失败: {e}")
        return False, {}

    # 查找待处理的患者
    if patient_id:
        patient_dirs = [mapped_dir / patient_id]
    else:
        patient_dirs = [d for d in sorted(mapped_dir.iterdir())
                       if d.is_dir() and d.name != 'visualizations']

    # 应用 start_patient_id 过滤（从指定患者开始）
    if start_patient_id:
        patient_dirs = [d for d in patient_dirs if d.name >= start_patient_id]
        logger.info(f"  从 {start_patient_id} 开始处理")

    # 应用 limit 参数
    if limit and limit > 0:
        patient_dirs = patient_dirs[:limit]

    logger.info(f"  待处理: {len(patient_dirs)} 例")

    # 处理每个患者
    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0
    results = {'outputs': [], 'failed': []}

    for i, patient_dir in enumerate(patient_dirs):
        pid = patient_dir.name
        mask_path = patient_dir / f"{pid}_warped_lesion.nii.gz"

        if not mask_path.exists():
            logger.warning(f"  [{i+1}/{len(patient_dirs)}] 跳过 {pid}: mask 不存在")
            results['failed'].append(pid)
            continue

        logger.info(f"  [{i+1}/{len(patient_dirs)}] 处理 {pid}...")

        try:
            output_path = output_dir / f"{pid}_fused.nii.gz"
            inference_module.fuse_lesion(
                template_path=template_path,
                lesion_mask_path=mask_path,
                model=model,
                output_path=output_path,
                device=device,
                smooth_boundary_width=3 if smooth_boundary else 0,
                model_type=model_type
            )
            logger.info(f"    ✓ 完成: {output_path.name}")
            success_count += 1
            results['outputs'].append(str(output_path))
        except Exception as e:
            logger.error(f"    ✗ 失败: {e}")
            results['failed'].append(pid)

    logger.info("")
    logger.info(f"  推理完成: {success_count}/{len(patient_dirs)} 成功")
    logger.info(f"  输出目录: {output_dir}")

    return success_count > 0, results


def _generate_patient_report(
    patient_output_dir: Path,
    patient_id: str,
    metrics: dict,
    model_type: str
) -> None:
    """
    生成单个患者的评估报告（Markdown 格式）

    Args:
        patient_output_dir: 患者输出目录
        patient_id: 患者 ID
        metrics: 患者指标字典
        model_type: 模型类型
    """
    from datetime import datetime

    # ====== 新逻辑：AI vs Real COPD + AI vs Embed 双轨对比 ======
    # Warp (Healthy Atlas) 是≈类指标的参照基准 (Ref)
    # Real COPD 替代原先 Warp 的位置，作为 AI 的主要对比对象
    m = metrics  # 简写

    # ---------- 辅助函数：统一计算 winner 和 improvement ----------
    def _calc_mono_up(ai_val, cmp_val, label):
        """单调↑指标：越大越好"""
        win = '✓ AI' if ai_val > cmp_val else f'✗ {label}'
        imp = ((ai_val - cmp_val) / (cmp_val + 1e-10)) * 100
        return win, imp

    def _calc_mono_down(ai_val, cmp_val, label):
        """单调↓指标：越小越好"""
        win = '✓ AI' if ai_val < cmp_val else f'✗ {label}'
        imp = ((cmp_val - ai_val) / (cmp_val + 1e-10)) * 100
        return win, imp

    def _calc_approx(ai_val, cmp_val, ref_val, label):
        """≈类指标：越接近 Ref(Warp) 越好"""
        dist_ai = abs(ai_val - ref_val)
        dist_cmp = abs(cmp_val - ref_val)
        win = '✓ AI' if dist_ai < dist_cmp else f'✗ {label}'
        imp = ((dist_cmp - dist_ai) / (dist_cmp + 1e-10)) * 100
        return win, imp

    # --- 清晰度 (↑): AI vs Real COPD ---
    sharp_r_win, sharp_r_imp = _calc_mono_up(m['sharpness_ai'], m['sharpness_real_copd'], 'Real')
    sharp_e_win, sharp_e_imp = _calc_mono_up(m['sharpness_ai'], m['sharpness_real'], 'Embed')

    # --- 边界梯度 (↓): AI vs Real COPD ---
    bound_r_win, bound_r_imp = _calc_mono_down(m['boundary_grad_ai'], m['boundary_grad_real_copd'], 'Real')
    bound_e_win, bound_e_imp = _calc_mono_down(m['boundary_grad_ai'], m['boundary_grad_real'], 'Embed')

    # --- GLCM 对比度 (↑): AI vs Real COPD ---
    g_con_r_win, g_con_r_imp = _calc_mono_up(m['glcm_contrast_ai'], m['glcm_contrast_real_copd'], 'Real')
    g_con_e_win, g_con_e_imp = _calc_mono_up(m['glcm_contrast_ai'], m['glcm_contrast_real'], 'Embed')

    # --- GLCM 熵 (↑): AI vs Real COPD ---
    g_ent_r_win, g_ent_r_imp = _calc_mono_up(m['glcm_entropy_ai'], m['glcm_entropy_real_copd'], 'Real')
    g_ent_e_win, g_ent_e_imp = _calc_mono_up(m['glcm_entropy_ai'], m['glcm_entropy_real'], 'Embed')

    # --- GLCM 能量 (≈W): ref = Warp ---
    g_ene_r_win, g_ene_r_imp = _calc_approx(m['glcm_energy_ai'], m['glcm_energy_real_copd'], m['glcm_energy_warp'], 'Real')
    g_ene_e_win, g_ene_e_imp = _calc_approx(m['glcm_energy_ai'], m['glcm_energy_real'], m['glcm_energy_warp'], 'Embed')

    # --- GLCM 相关性 (≈W): ref = Warp ---
    g_cor_r_win, g_cor_r_imp = _calc_approx(m['glcm_correlation_ai'], m['glcm_correlation_real_copd'], m['glcm_correlation_warp'], 'Real')
    g_cor_e_win, g_cor_e_imp = _calc_approx(m['glcm_correlation_ai'], m['glcm_correlation_real'], m['glcm_correlation_warp'], 'Embed')

    # --- GLCM 同质性 (≈W): ref = Warp ---
    g_hom_r_win, g_hom_r_imp = _calc_approx(m['glcm_homogeneity_ai'], m['glcm_homogeneity_real_copd'], m['glcm_homogeneity_warp'], 'Real')
    g_hom_e_win, g_hom_e_imp = _calc_approx(m['glcm_homogeneity_ai'], m['glcm_homogeneity_real'], m['glcm_homogeneity_warp'], 'Embed')

    # 生成报告
    report_path = patient_output_dir / f"{patient_id}_evaluation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 患者评估报告 - {patient_id}\n\n")
        f.write(f"模型类型: {model_type}\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 基础指标\n\n")
        f.write("| 指标 | 值 |\n|------|----|\n")
        f.write(f"| PSNR | {m['psnr']:.2f} dB |\n")
        f.write(f"| SSIM | {m['ssim']:.4f} |\n")
        f.write(f"| 真实肺气肿比例 | {m['real_emphysema_ratio']:.1%} |\n")
        f.write(f"| AI肺气肿比例 | {m['fused_emphysema_ratio']:.1%} |\n")
        f.write(f"| 病灶体素数 | {m['voxel_count']} |\n\n")

        f.write(f"""## 纹理质量分析

> **评估目标**：对比 AI 相较于 Embedded Template 的改进和优化。
> ↑/↓ 指标按箭头方向直接比较数值；≈ 类指标以 Real COPD 为参照，越接近越好。

| 指标 | Healthy Atlas | Embed | AI | Real COPD | AI vs Real COPD | AI vs Real COPD改进 | AI vs Embed | AI vs Embed改进 |
|------|-----------|-------|------|-----------|-----------------|-------------------|-------------|----------------|
| 清晰度 (↑) | {m['sharpness_warp']:.1f} | {m['sharpness_real']:.1f} | {m['sharpness_ai']:.1f} | {m['sharpness_real_copd']:.1f} | {sharp_r_win} | {sharp_r_imp:+.1f}% | {sharp_e_win} | {sharp_e_imp:+.1f}% |
| 边界梯度 (↓) | {m['boundary_grad_warp']:.2f} | {m['boundary_grad_real']:.2f} | {m['boundary_grad_ai']:.2f} | {m['boundary_grad_real_copd']:.2f} | {bound_r_win} | {bound_r_imp:+.1f}% | {bound_e_win} | {bound_e_imp:+.1f}% |
| GLCM 对比度 (↑) | {m['glcm_contrast_warp']:.4f} | {m['glcm_contrast_real']:.4f} | {m['glcm_contrast_ai']:.4f} | {m['glcm_contrast_real_copd']:.4f} | {g_con_r_win} | {g_con_r_imp:+.1f}% | {g_con_e_win} | {g_con_e_imp:+.1f}% |
| GLCM 能量 (≈) | {m['glcm_energy_warp']:.4f} | {m['glcm_energy_real']:.4f} | {m['glcm_energy_ai']:.4f} | {m['glcm_energy_real_copd']:.4f} | {g_ene_r_win} | {g_ene_r_imp:+.1f}% | {g_ene_e_win} | {g_ene_e_imp:+.1f}% |
| GLCM 熵 (↑) | {m['glcm_entropy_warp']:.4f} | {m['glcm_entropy_real']:.4f} | {m['glcm_entropy_ai']:.4f} | {m['glcm_entropy_real_copd']:.4f} | {g_ent_r_win} | {g_ent_r_imp:+.1f}% | {g_ent_e_win} | {g_ent_e_imp:+.1f}% |
| GLCM 相关性 (≈) | {m['glcm_correlation_warp']:.4f} | {m['glcm_correlation_real']:.4f} | {m['glcm_correlation_ai']:.4f} | {m['glcm_correlation_real_copd']:.4f} | {g_cor_r_win} | {g_cor_r_imp:+.1f}% | {g_cor_e_win} | {g_cor_e_imp:+.1f}% |
| GLCM 同质性 (≈) | {m['glcm_homogeneity_warp']:.4f} | {m['glcm_homogeneity_real']:.4f} | {m['glcm_homogeneity_ai']:.4f} | {m['glcm_homogeneity_real_copd']:.4f} | {g_hom_r_win} | {g_hom_r_imp:+.1f}% | {g_hom_e_win} | {g_hom_e_imp:+.1f}% |

![纹理质量雷达图]({patient_id}_texture_radar.png)
""")


def _generate_texture_radar_chart(
    output_dir: Path,
    contrast_real: float, contrast_ai: float, contrast_warp: float,
    energy_real: float, energy_ai: float, energy_warp: float,
    entropy_real: float, entropy_ai: float, entropy_warp: float,
    correlation_real: float, correlation_ai: float, correlation_warp: float,
    homogeneity_real: float, homogeneity_ai: float, homogeneity_warp: float,
    logger: logging.Logger,
    filename: str = 'texture_quality_radar.png',
    sharpness_real: float = None, sharpness_ai: float = None, sharpness_warp: float = None,
    boundary_real: float = None, boundary_ai: float = None, boundary_warp: float = None
) -> None:
    """
    生成综合质量雷达图（蜘蛛图）

    展示三条曲线：Real COPD、AI Fused、Healthy Atlas。
    每个维度的节点位置直接对齐 Markdown 表格中的真实数值（通过 min-max
    归一化映射到 [0.1, 1.0] 区间），并在顶点处标注原始数值。

    Args:
        output_dir: 输出目录
        contrast_*, energy_*, entropy_*, correlation_*, homogeneity_*:
            各方法的 GLCM 特征值（*_real = Real COPD）
        logger: 日志记录器
        filename: 输出文件名
        sharpness_*, boundary_*: 清晰度和边界梯度值（可选）
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # ---------- 辅助：中值锚定 + 范围扩展 归一化 ----------
    # 核心思路：当三个值非常接近时，纯 min-max 会把微小差异拉伸到全范围，
    # 视觉上严重夸大。采用"中值锚定"策略：
    #   1) 以三个值的中位数(median)为中心；
    #   2) 向外扩展 max(实际半径 * expand_ratio, median * min_pct) 作为归一化范围；
    #   3) 这样值差微小时，三个点聚拢在中心附近，不会极端分散。
    center_target = 0.68  # 归一化后的中心锚定位置（上移至0.68，使曲线更靠近外圈）
    spread_half   = 0.38  # 可用的半幅度（上下各0.38 → 映射到[0.30, 1.06]，充分利用外圈）
    expand_ratio  = 1.8   # 扩展系数：实际数据范围 × 1.8 作为归一化范围
    min_pct       = 0.10  # 最小范围保底：中位数 × 10%（防止三值几乎相等时退化）

    def _norm_centered(val, median, half_range):
        """将 val 映射到以 center_target 为中心的归一化坐标"""
        if half_range < 1e-12:
            return center_target
        return center_target + spread_half * (val - median) / half_range

    def _norm_centered_inv(val, median, half_range):
        """越低越好的指标：取反映射"""
        if half_range < 1e-12:
            return center_target
        return center_target - spread_half * (val - median) / half_range

    # ---------- 构建维度 ----------
    include_sharpness_boundary = (sharpness_real is not None and boundary_real is not None)

    if include_sharpness_boundary:
        categories = ['Sharpness\n(↑)', 'Boundary\n(↓)',
                      'GLCM\nContrast(↑)', 'GLCM\nEnergy(≈Real)',
                      'GLCM\nEntropy(↑)', 'GLCM\nCorrelation(≈Real)',
                      'GLCM\nHomogeneity(≈Real)']
        # 三组原始值：(real, ai, warp) × 7 维度
        raw_real = [sharpness_real, boundary_real,
                    contrast_real, energy_real, entropy_real,
                    correlation_real, homogeneity_real]
        raw_ai   = [sharpness_ai, boundary_ai,
                    contrast_ai, energy_ai, entropy_ai,
                    correlation_ai, homogeneity_ai]
        raw_warp = [sharpness_warp, boundary_warp,
                    contrast_warp, energy_warp, entropy_warp,
                    correlation_warp, homogeneity_warp]
        # 每个维度的归一化方向：True = 越大越好(正序), False = 越小越好(反序)
        ascending = [True, False, True, True, True, True, True]
    else:
        categories = ['Contrast', 'Energy', 'Entropy', 'Correlation', 'Homogeneity']
        raw_real = [contrast_real, energy_real, entropy_real, correlation_real, homogeneity_real]
        raw_ai   = [contrast_ai, energy_ai, entropy_ai, correlation_ai, homogeneity_ai]
        raw_warp = [contrast_warp, energy_warp, entropy_warp, correlation_warp, homogeneity_warp]
        ascending = [True, True, True, True, True]

    N = len(categories)

    # ---------- 逐维度：中值锚定归一化 ----------
    real_scores, ai_scores, warp_scores = [], [], []
    for i in range(N):
        vals = [raw_real[i], raw_ai[i], raw_warp[i]]
        median = sorted(vals)[1]  # 三个值的中位数
        actual_half = (max(vals) - min(vals)) / 2.0
        # 归一化半径：取 (实际半径 × expand_ratio) 和 (median × min_pct) 中的较大者
        half_range = max(actual_half * expand_ratio,
                         abs(median) * min_pct,
                         1e-6)  # 绝对保底

        if ascending[i]:
            real_scores.append(_norm_centered(raw_real[i], median, half_range))
            ai_scores.append(_norm_centered(raw_ai[i], median, half_range))
            warp_scores.append(_norm_centered(raw_warp[i], median, half_range))
        else:
            real_scores.append(_norm_centered_inv(raw_real[i], median, half_range))
            ai_scores.append(_norm_centered_inv(raw_ai[i], median, half_range))
            warp_scores.append(_norm_centered_inv(raw_warp[i], median, half_range))

    # 安全裁剪到 [0.10, 1.10]，防止极端值溢出绘图范围
    real_scores = [max(0.10, min(1.10, s)) for s in real_scores]
    ai_scores   = [max(0.10, min(1.10, s)) for s in ai_scores]
    warp_scores = [max(0.10, min(1.10, s)) for s in warp_scores]

    # ---------- 原始值的显示格式 ----------
    def _fmt(v, idx):
        """根据维度选择合适的显示格式"""
        if include_sharpness_boundary:
            if idx == 0:   return f'{v:.1f}'    # Sharpness
            if idx == 1:   return f'{v:.2f}'    # Boundary
            return f'{v:.4f}'                   # GLCM
        return f'{v:.4f}'

    # ---------- 闭合多边形 ----------
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    real_scores += real_scores[:1]
    ai_scores   += ai_scores[:1]
    warp_scores += warp_scores[:1]

    # ---------- 绘图 ----------
    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw=dict(polar=True))

    # 三条曲线（颜色与图例保持不变）
    ax.plot(angles, real_scores, 'o-', linewidth=2.5, label='Real COPD',
            color='#2ecc71', markersize=8)
    ax.fill(angles, real_scores, alpha=0.10, color='#2ecc71')

    ax.plot(angles, ai_scores, 's-', linewidth=2.5, label='AI Fused',
            color='#3498db', markersize=8)
    ax.fill(angles, ai_scores, alpha=0.20, color='#3498db')

    ax.plot(angles, warp_scores, '^-', linewidth=2.5, label='Healthy Atlas',
            color='#e74c3c', markersize=8)
    ax.fill(angles, warp_scores, alpha=0.20, color='#e74c3c')

    # ---------- 数值标签 ----------
    label_offset = 0.09   # 径向偏移量（略减以确保标签不超出 ylim=1.20）
    for i in range(N):
        angle = angles[i]
        # 三条曲线各自标注，添加角度偏移避免重叠
        items = [
            (real_scores[i], raw_real[i], '#2ecc71', -0.08),
            (ai_scores[i],   raw_ai[i],   '#3498db',  0.00),
            (warp_scores[i], raw_warp[i],  '#e74c3c',  0.08),
        ]
        for score, raw_val, color, angle_shift in items:
            ax.text(angle + angle_shift, score + label_offset,
                    _fmt(raw_val, i),
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    color=color, alpha=0.95,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.6))

    # 轴标签和网格
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=13, weight='bold')
    ax.set_ylim(0, 1.20)  # 缩小上限：曲线最高达~1.10，ylim=1.20 留一点边距即可
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=11)
    ax.grid(True, linestyle='--', alpha=0.7)

    # 标题和图例
    if include_sharpness_boundary:
        title = ('Comprehensive Quality Radar Chart\n'
                 '(Median-anchored normalization; labels = raw values)')
    else:
        title = ('GLCM Texture Quality Radar Chart\n'
                 '(Median-anchored normalization; labels = raw values)')
    ax.set_title(title, size=17, weight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=14,
              framealpha=0.9)

    # 保存
    radar_path = output_dir / filename
    plt.tight_layout()
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  ✓ 雷达图已保存: {radar_path}")


def run_model_evaluation(
    config: dict,
    logger: logging.Logger,
    model_type: str = 'partial_conv',
    num_patients: int = 10,
    start_patient_id: str = None
) -> bool:
    """
    执行模型评估（与真实 COPD CT 对比）

    Args:
        config: 配置字典
        logger: 日志记录器
        model_type: 模型类型
        num_patients: 评估的患者数量
        start_patient_id: 起始患者 ID（如 copd_024，仅评估从该患者开始的数据）

    Returns:
        bool: 是否成功
    """
    logger.info("")
    logger.info("[评估] 模型质量评估")
    logger.info("-" * 40)

    try:
        import json
        import nibabel as nib
        import numpy as np
    except ImportError as e:
        logger.error(f"  ✗ 缺少依赖: {e}")
        return False

    paths = config.get('paths', {})
    mapped_dir = Path(paths.get('mapped', 'data/03_mapped'))
    fused_dir = Path(paths.get('final_viz', 'data/04_final_viz')) / model_type
    output_dir = Path(f'results/{model_type}')  # 合并后的输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载标准模板作为 Healthy Atlas 基线
    # Healthy Atlas = 健康肺模板在病灶区域的原始 HU 值（未经 AI 合成）
    atlas_dir = Path(paths.get('atlas', 'data/02_atlas'))
    airway_fusion_config = config.get('registration', {}).get('airway_fusion', {})
    airway_template_filename = airway_fusion_config.get(
        'output_filename', 'standard_template_with_airway.nii.gz'
    )
    template_path = atlas_dir / airway_template_filename
    if not template_path.exists():
        template_path = atlas_dir / 'standard_template.nii.gz'
    if template_path.exists():
        template_data_for_warp = nib.load(str(template_path)).get_fdata()
        logger.info(f"  Healthy Atlas 基线: {template_path.name}")
    else:
        template_data_for_warp = None
        logger.warning(f"  ⚠ 模板不存在，Healthy Atlas 将回退为 Real COPD")

    # 检查融合结果目录
    if not fused_dir.exists():
        logger.error(f"  ✗ 融合结果目录不存在: {fused_dir}")
        logger.error(f"  请先运行推理: --inference --model-type {model_type}")
        return False

    fused_files = sorted(fused_dir.glob("*_fused.nii.gz"))

    # 应用 start_patient_id 过滤
    if start_patient_id:
        fused_files = [f for f in fused_files
                       if f.name.replace('_fused.nii.gz', '') >= start_patient_id]
        logger.info(f"  从 {start_patient_id} 开始评估")

    fused_files = fused_files[:num_patients]
    if not fused_files:
        logger.error(f"  ✗ 未找到融合结果文件")
        return False

    logger.info(f"  融合结果目录: {fused_dir}")
    logger.info(f"  评估患者数: {len(fused_files)}")
    logger.info(f"  输出目录: {output_dir}")

    # 评估每个患者
    all_metrics = []
    for fused_path in fused_files:
        patient_id = fused_path.name.replace('_fused.nii.gz', '')
        warped_path = mapped_dir / patient_id / f"{patient_id}_warped.nii.gz"
        mask_path = mapped_dir / patient_id / f"{patient_id}_warped_lesion.nii.gz"

        if not warped_path.exists() or not mask_path.exists():
            logger.warning(f"  ⚠ 跳过 {patient_id}: 缺少配准数据")
            continue

        try:
            # 创建患者子目录
            patient_output_dir = output_dir / patient_id
            patient_output_dir.mkdir(parents=True, exist_ok=True)

            # 加载数据
            fused_data = nib.load(str(fused_path)).get_fdata()
            real_data = nib.load(str(warped_path)).get_fdata()
            mask_data = nib.load(str(mask_path)).get_fdata()

            mask_bool = mask_data > 0
            if mask_bool.sum() < 100:
                logger.warning(f"  ⚠ 跳过 {patient_id}: 病灶区域太小")
                continue

            # 计算病灶区域指标
            real_lesion = real_data[mask_bool]
            fused_lesion = fused_data[mask_bool]

            # PSNR
            mse = np.mean((real_lesion - fused_lesion) ** 2)
            psnr = 20 * np.log10(1400 / np.sqrt(mse)) if mse > 0 else float('inf')

            # SSIM (简化版)
            mu1, mu2 = np.mean(real_lesion), np.mean(fused_lesion)
            s1, s2 = np.var(real_lesion), np.var(fused_lesion)
            s12 = np.mean((real_lesion - mu1) * (fused_lesion - mu2))
            C1, C2 = 0.01**2, 0.03**2
            ssim = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))

            # HU 分析
            real_emph = (real_lesion < -950).sum() / len(real_lesion)
            fused_emph = (fused_lesion < -950).sum() / len(fused_lesion)

            # ========== 高级纹理质量指标 ==========
            # Healthy Atlas 基线：健康模板在病灶区域的原始 HU 值
            # 概念：如果不用 AI，健康模板在这个病灶位置的纹理特征如何？
            if template_data_for_warp is not None:
                warp_data = template_data_for_warp
            else:
                warp_data = real_data  # 回退：模板不可用时用 real_data

            # 找到病灶面积最大的切片用于 2D 指标计算（向量化操作，无 Python 循环）
            slice_areas = np.sum(mask_data > 0, axis=(0, 1))
            best_slice_idx = int(np.argmax(slice_areas))

            # 提取 2D 切片
            fused_slice = fused_data[:, :, best_slice_idx]
            warp_slice = warp_data[:, :, best_slice_idx]
            mask_slice = mask_data[:, :, best_slice_idx]

            # ====== 构建 Embedded Lesion 评估基准（仅 2D 切片级别操作）======
            # 将 real_data 中的病灶区域嵌入到 template (warp_data) 的最佳切片中
            # 以此作为清晰度和边界梯度的计算基准 (Embedded Template)
            # 注意：仅拷贝单个 2D 切片，避免拷贝整个 3D 体积（~500MB+）
            embedded_slice = warp_slice.copy()
            mask_2d = mask_slice > 0
            real_slice_2d = real_data[:, :, best_slice_idx]
            embedded_slice[mask_2d] = real_slice_2d[mask_2d]

            # 计算清晰度 (值越高越好)
            sharpness_real = compute_sharpness(embedded_slice, mask_slice)      # Embedded Template
            sharpness_real_copd = compute_sharpness(real_slice_2d, mask_slice)  # Real COPD
            sharpness_ai = compute_sharpness(fused_slice, mask_slice)
            sharpness_warp = compute_sharpness(warp_slice, mask_slice)

            # 计算边界连续性 (值越低融合越平滑)
            boundary_real = compute_boundary_continuity(embedded_slice, mask_slice)      # Embedded Template
            boundary_real_copd = compute_boundary_continuity(real_slice_2d, mask_slice)  # Real COPD
            boundary_ai = compute_boundary_continuity(fused_slice, mask_slice)
            boundary_warp = compute_boundary_continuity(warp_slice, mask_slice)

            # 计算 GLCM 纹理特征
            # glcm_real: 基准为 Embedded Template（用于对比度/熵等单调指标的参照）
            # glcm_real_copd: 基准为真实 COPD warped 切片（用于能量/相关性/同质性等
            #   "≈越近越好"指标的参照，因 GLCM 使用 bounding-box 裁切，包含 mask 外
            #   背景像素；Embedded 的背景为健康模板（HU ≈ -836），Real 为真实 COPD
            #   组织（HU ≈ -977），使用 Real 作为参照更能反映真实纹理分布）
            glcm_real = compute_glcm_features(embedded_slice, mask_slice)
            glcm_real_copd = compute_glcm_features(real_slice_2d, mask_slice)
            glcm_ai = compute_glcm_features(fused_slice, mask_slice)
            glcm_warp = compute_glcm_features(warp_slice, mask_slice)

            metrics = {
                'patient_id': patient_id,
                # 原有指标
                'psnr': float(psnr),
                'ssim': float(ssim),
                'real_emphysema_ratio': float(real_emph),
                'fused_emphysema_ratio': float(fused_emph),
                'voxel_count': int(mask_bool.sum()),
                # 纹理质量指标
                # *_real = Embedded Template, *_real_copd = Real COPD（配准后真实数据）
                'sharpness_real': sharpness_real,
                'sharpness_real_copd': sharpness_real_copd,
                'sharpness_ai': sharpness_ai,
                'sharpness_warp': sharpness_warp,
                'boundary_grad_real': boundary_real,
                'boundary_grad_real_copd': boundary_real_copd,
                'boundary_grad_ai': boundary_ai,
                'boundary_grad_warp': boundary_warp,
                # GLCM: *_real = Embedded Template, *_real_copd = 真实 COPD
                'glcm_contrast_real': glcm_real['glcm_contrast'],
                'glcm_contrast_real_copd': glcm_real_copd['glcm_contrast'],
                'glcm_contrast_ai': glcm_ai['glcm_contrast'],
                'glcm_contrast_warp': glcm_warp['glcm_contrast'],
                'glcm_energy_real': glcm_real['glcm_energy'],
                'glcm_energy_real_copd': glcm_real_copd['glcm_energy'],
                'glcm_energy_ai': glcm_ai['glcm_energy'],
                'glcm_energy_warp': glcm_warp['glcm_energy'],
                'glcm_entropy_real': glcm_real['glcm_entropy'],
                'glcm_entropy_real_copd': glcm_real_copd['glcm_entropy'],
                'glcm_entropy_ai': glcm_ai['glcm_entropy'],
                'glcm_entropy_warp': glcm_warp['glcm_entropy'],
                'glcm_correlation_real': glcm_real['glcm_correlation'],
                'glcm_correlation_real_copd': glcm_real_copd['glcm_correlation'],
                'glcm_correlation_ai': glcm_ai['glcm_correlation'],
                'glcm_correlation_warp': glcm_warp['glcm_correlation'],
                'glcm_homogeneity_real': glcm_real['glcm_homogeneity'],
                'glcm_homogeneity_real_copd': glcm_real_copd['glcm_homogeneity'],
                'glcm_homogeneity_ai': glcm_ai['glcm_homogeneity'],
                'glcm_homogeneity_warp': glcm_warp['glcm_homogeneity'],
            }
            all_metrics.append(metrics)

            # 保存患者单独的 JSON 评估报告
            patient_report_path = patient_output_dir / f"{patient_id}_evaluation_report.json"
            with open(patient_report_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            # 生成患者单独的 Markdown 报告
            try:
                _generate_patient_report(patient_output_dir, patient_id, metrics, model_type)
            except Exception as e:
                logger.warning(f"  ⚠ {patient_id} Markdown 报告生成失败: {e}")

            # 生成患者单独的雷达图（包含 Sharpness 和 Boundary）
            # 传入 Real COPD 数据（glcm_real_copd）替代 Embedded Template（glcm_real）
            try:
                _generate_texture_radar_chart(
                    patient_output_dir,
                    glcm_real_copd['glcm_contrast'], glcm_ai['glcm_contrast'], glcm_warp['glcm_contrast'],
                    glcm_real_copd['glcm_energy'], glcm_ai['glcm_energy'], glcm_warp['glcm_energy'],
                    glcm_real_copd['glcm_entropy'], glcm_ai['glcm_entropy'], glcm_warp['glcm_entropy'],
                    glcm_real_copd['glcm_correlation'], glcm_ai['glcm_correlation'], glcm_warp['glcm_correlation'],
                    glcm_real_copd['glcm_homogeneity'], glcm_ai['glcm_homogeneity'], glcm_warp['glcm_homogeneity'],
                    logger,
                    filename=f"{patient_id}_texture_radar.png",
                    sharpness_real=sharpness_real_copd,
                    sharpness_ai=sharpness_ai,
                    sharpness_warp=sharpness_warp,
                    boundary_real=boundary_real_copd,
                    boundary_ai=boundary_ai,
                    boundary_warp=boundary_warp
                )
            except Exception as e:
                logger.warning(f"  ⚠ {patient_id} 雷达图生成失败: {e}")

            logger.info(f"  ✓ {patient_id}: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}, "
                       f"肺气肿: Real={real_emph:.1%} AI={fused_emph:.1%}")
            logger.info(f"    纹理质量: Sharpness(AI/Warp)={sharpness_ai:.1f}/{sharpness_warp:.1f}, "
                       f"Boundary(AI/Warp)={boundary_ai:.2f}/{boundary_warp:.2f}")

        except Exception as e:
            logger.error(f"  ✗ {patient_id} 评估失败: {e}")

    if not all_metrics:
        logger.error("  ✗ 没有成功评估的患者")
        return False

    # 计算平均指标
    avg_psnr = np.mean([m['psnr'] for m in all_metrics])
    avg_ssim = np.mean([m['ssim'] for m in all_metrics])
    avg_real_emph = np.mean([m['real_emphysema_ratio'] for m in all_metrics])
    avg_fused_emph = np.mean([m['fused_emphysema_ratio'] for m in all_metrics])

    # 计算平均纹理质量指标
    avg_sharpness_real = np.mean([m['sharpness_real'] for m in all_metrics])
    avg_sharpness_real_copd = np.mean([m['sharpness_real_copd'] for m in all_metrics])
    avg_sharpness_ai = np.mean([m['sharpness_ai'] for m in all_metrics])
    avg_sharpness_warp = np.mean([m['sharpness_warp'] for m in all_metrics])

    avg_boundary_real = np.mean([m['boundary_grad_real'] for m in all_metrics])
    avg_boundary_real_copd = np.mean([m['boundary_grad_real_copd'] for m in all_metrics])
    avg_boundary_ai = np.mean([m['boundary_grad_ai'] for m in all_metrics])
    avg_boundary_warp = np.mean([m['boundary_grad_warp'] for m in all_metrics])

    # 扩展 GLCM 特征平均值
    avg_glcm_contrast_real = np.mean([m['glcm_contrast_real'] for m in all_metrics])
    avg_glcm_contrast_real_copd = np.mean([m['glcm_contrast_real_copd'] for m in all_metrics])
    avg_glcm_contrast_ai = np.mean([m['glcm_contrast_ai'] for m in all_metrics])
    avg_glcm_contrast_warp = np.mean([m['glcm_contrast_warp'] for m in all_metrics])

    avg_glcm_energy_real = np.mean([m['glcm_energy_real'] for m in all_metrics])
    avg_glcm_energy_real_copd = np.mean([m['glcm_energy_real_copd'] for m in all_metrics])
    avg_glcm_energy_ai = np.mean([m['glcm_energy_ai'] for m in all_metrics])
    avg_glcm_energy_warp = np.mean([m['glcm_energy_warp'] for m in all_metrics])

    avg_glcm_entropy_real = np.mean([m['glcm_entropy_real'] for m in all_metrics])
    avg_glcm_entropy_real_copd = np.mean([m['glcm_entropy_real_copd'] for m in all_metrics])
    avg_glcm_entropy_ai = np.mean([m['glcm_entropy_ai'] for m in all_metrics])
    avg_glcm_entropy_warp = np.mean([m['glcm_entropy_warp'] for m in all_metrics])

    avg_glcm_correlation_real = np.mean([m['glcm_correlation_real'] for m in all_metrics])
    avg_glcm_correlation_real_copd = np.mean([m['glcm_correlation_real_copd'] for m in all_metrics])
    avg_glcm_correlation_ai = np.mean([m['glcm_correlation_ai'] for m in all_metrics])
    avg_glcm_correlation_warp = np.mean([m['glcm_correlation_warp'] for m in all_metrics])

    avg_glcm_homogeneity_real = np.mean([m['glcm_homogeneity_real'] for m in all_metrics])
    avg_glcm_homogeneity_real_copd = np.mean([m['glcm_homogeneity_real_copd'] for m in all_metrics])
    avg_glcm_homogeneity_ai = np.mean([m['glcm_homogeneity_ai'] for m in all_metrics])
    avg_glcm_homogeneity_warp = np.mean([m['glcm_homogeneity_warp'] for m in all_metrics])

    # ====== 辅助函数：统一计算 winner 和 improvement ======
    def _calc_mono_up(ai_v, cmp_v, label):
        win = '✓ AI' if ai_v > cmp_v else f'✗ {label}'
        imp = ((ai_v - cmp_v) / (cmp_v + 1e-10)) * 100
        return win, imp

    def _calc_mono_down(ai_v, cmp_v, label):
        win = '✓ AI' if ai_v < cmp_v else f'✗ {label}'
        imp = ((cmp_v - ai_v) / (cmp_v + 1e-10)) * 100
        return win, imp

    def _calc_approx(ai_v, cmp_v, ref_v, label):
        """≈类指标：越接近 Ref(Warp) 越好"""
        dist_ai = abs(ai_v - ref_v)
        dist_cmp = abs(cmp_v - ref_v)
        win = '✓ AI' if dist_ai < dist_cmp else f'✗ {label}'
        imp = ((dist_cmp - dist_ai) / (dist_cmp + 1e-10)) * 100
        return win, imp

    # --- 清晰度 (↑): AI vs Real COPD ---
    s_r_win, s_r_imp = _calc_mono_up(avg_sharpness_ai, avg_sharpness_real_copd, 'Real')
    s_e_win, s_e_imp = _calc_mono_up(avg_sharpness_ai, avg_sharpness_real, 'Embed')
    # --- 边界梯度 (↓): AI vs Real COPD ---
    b_r_win, b_r_imp = _calc_mono_down(avg_boundary_ai, avg_boundary_real_copd, 'Real')
    b_e_win, b_e_imp = _calc_mono_down(avg_boundary_ai, avg_boundary_real, 'Embed')
    # --- GLCM 对比度 (↑): AI vs Real COPD ---
    gc_r_win, gc_r_imp = _calc_mono_up(avg_glcm_contrast_ai, avg_glcm_contrast_real_copd, 'Real')
    gc_e_win, gc_e_imp = _calc_mono_up(avg_glcm_contrast_ai, avg_glcm_contrast_real, 'Embed')
    # --- GLCM 熵 (↑): AI vs Real COPD ---
    gn_r_win, gn_r_imp = _calc_mono_up(avg_glcm_entropy_ai, avg_glcm_entropy_real_copd, 'Real')
    gn_e_win, gn_e_imp = _calc_mono_up(avg_glcm_entropy_ai, avg_glcm_entropy_real, 'Embed')
    # --- GLCM 能量 (≈W): ref = Warp ---
    ge_r_win, ge_r_imp = _calc_approx(avg_glcm_energy_ai, avg_glcm_energy_real_copd, avg_glcm_energy_warp, 'Real')
    ge_e_win, ge_e_imp = _calc_approx(avg_glcm_energy_ai, avg_glcm_energy_real, avg_glcm_energy_warp, 'Embed')
    # --- GLCM 相关性 (≈W): ref = Warp ---
    gr_r_win, gr_r_imp = _calc_approx(avg_glcm_correlation_ai, avg_glcm_correlation_real_copd, avg_glcm_correlation_warp, 'Real')
    gr_e_win, gr_e_imp = _calc_approx(avg_glcm_correlation_ai, avg_glcm_correlation_real, avg_glcm_correlation_warp, 'Embed')
    # --- GLCM 同质性 (≈W): ref = Warp ---
    gh_r_win, gh_r_imp = _calc_approx(avg_glcm_homogeneity_ai, avg_glcm_homogeneity_real_copd, avg_glcm_homogeneity_warp, 'Real')
    gh_e_win, gh_e_imp = _calc_approx(avg_glcm_homogeneity_ai, avg_glcm_homogeneity_real, avg_glcm_homogeneity_warp, 'Embed')

    logger.info("")
    logger.info("  === 评估汇总 ===")
    logger.info(f"  平均 PSNR: {avg_psnr:.2f} dB")
    logger.info(f"  平均 SSIM: {avg_ssim:.4f}")
    logger.info(f"  真实 COPD 平均肺气肿比例: {avg_real_emph:.1%}")
    logger.info(f"  AI 融合 平均肺气肿比例: {avg_fused_emph:.1%}")
    logger.info("")
    logger.info("  === 纹理质量分析 ===")
    logger.info(f"  清晰度 (↑): Warp={avg_sharpness_warp:.2f}, Embed={avg_sharpness_real:.2f}, AI={avg_sharpness_ai:.2f}, RealCOPD={avg_sharpness_real_copd:.2f} | vs Real: {s_r_imp:+.1f}% | vs Embed: {s_e_imp:+.1f}%")
    logger.info(f"  边界梯度 (↓): Warp={avg_boundary_warp:.2f}, Embed={avg_boundary_real:.2f}, AI={avg_boundary_ai:.2f}, RealCOPD={avg_boundary_real_copd:.2f} | vs Real: {b_r_imp:+.1f}% | vs Embed: {b_e_imp:+.1f}%")
    logger.info(f"  GLCM对比度 (↑): Warp={avg_glcm_contrast_warp:.2f}, Embed={avg_glcm_contrast_real:.2f}, AI={avg_glcm_contrast_ai:.2f}, RealCOPD={avg_glcm_contrast_real_copd:.2f}")
    logger.info(f"  GLCM熵 (↑): Warp={avg_glcm_entropy_warp:.2f}, Embed={avg_glcm_entropy_real:.2f}, AI={avg_glcm_entropy_ai:.2f}, RealCOPD={avg_glcm_entropy_real_copd:.2f}")

    # 生成雷达图（包含 Sharpness 和 Boundary）
    # 传入 Real COPD 数据（avg_*_real_copd）替代 Embedded Template（avg_*_real）
    try:
        _generate_texture_radar_chart(
            output_dir,
            avg_glcm_contrast_real_copd, avg_glcm_contrast_ai, avg_glcm_contrast_warp,
            avg_glcm_energy_real_copd, avg_glcm_energy_ai, avg_glcm_energy_warp,
            avg_glcm_entropy_real_copd, avg_glcm_entropy_ai, avg_glcm_entropy_warp,
            avg_glcm_correlation_real_copd, avg_glcm_correlation_ai, avg_glcm_correlation_warp,
            avg_glcm_homogeneity_real_copd, avg_glcm_homogeneity_ai, avg_glcm_homogeneity_warp,
            logger,
            filename='texture_quality_radar.png',
            sharpness_real=avg_sharpness_real_copd,
            sharpness_ai=avg_sharpness_ai,
            sharpness_warp=avg_sharpness_warp,
            boundary_real=avg_boundary_real_copd,
            boundary_ai=avg_boundary_ai,
            boundary_warp=avg_boundary_warp
        )
    except Exception as e:
        logger.warning(f"  ⚠ 雷达图生成失败: {e}")

    # 保存报告
    report_path = output_dir / 'evaluation_report.md'

    json_report_path = output_dir / 'evaluation_report.json'
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 模型评估报告 - {model_type}\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 汇总结果\n\n")
        f.write("| 指标 | 值 |\n|------|----|\n")
        f.write(f"| 平均 PSNR | {avg_psnr:.2f} dB |\n")
        f.write(f"| 平均 SSIM | {avg_ssim:.4f} |\n")
        f.write(f"| 真实肺气肿比例 | {avg_real_emph:.1%} |\n")
        f.write(f"| AI肺气肿比例 | {avg_fused_emph:.1%} |\n")

        f.write(f"""
## 纹理质量分析

> **评估目标**：对比 AI 相较于 Embedded Template 的改进和优化。
> ↑/↓ 指标按箭头方向直接比较数值；≈ 类指标以 Real COPD 为参照，越接近越好。

| 指标 | Healthy Atlas | Embed | AI | Real COPD | AI vs Real COPD | AI vs Real COPD改进 | AI vs Embed | AI vs Embed改进 |
|------|-----------|-------|------|-----------|-----------------|-------------------|-------------|----------------|
| 清晰度 (↑) | {avg_sharpness_warp:.1f} | {avg_sharpness_real:.1f} | {avg_sharpness_ai:.1f} | {avg_sharpness_real_copd:.1f} | {s_r_win} | {s_r_imp:+.1f}% | {s_e_win} | {s_e_imp:+.1f}% |
| 边界梯度 (↓) | {avg_boundary_warp:.2f} | {avg_boundary_real:.2f} | {avg_boundary_ai:.2f} | {avg_boundary_real_copd:.2f} | {b_r_win} | {b_r_imp:+.1f}% | {b_e_win} | {b_e_imp:+.1f}% |
| GLCM 对比度 (↑) | {avg_glcm_contrast_warp:.4f} | {avg_glcm_contrast_real:.4f} | {avg_glcm_contrast_ai:.4f} | {avg_glcm_contrast_real_copd:.4f} | {gc_r_win} | {gc_r_imp:+.1f}% | {gc_e_win} | {gc_e_imp:+.1f}% |
| GLCM 能量 (≈) | {avg_glcm_energy_warp:.4f} | {avg_glcm_energy_real:.4f} | {avg_glcm_energy_ai:.4f} | {avg_glcm_energy_real_copd:.4f} | {ge_r_win} | {ge_r_imp:+.1f}% | {ge_e_win} | {ge_e_imp:+.1f}% |
| GLCM 熵 (↑) | {avg_glcm_entropy_warp:.4f} | {avg_glcm_entropy_real:.4f} | {avg_glcm_entropy_ai:.4f} | {avg_glcm_entropy_real_copd:.4f} | {gn_r_win} | {gn_r_imp:+.1f}% | {gn_e_win} | {gn_e_imp:+.1f}% |
| GLCM 相关性 (≈) | {avg_glcm_correlation_warp:.4f} | {avg_glcm_correlation_real:.4f} | {avg_glcm_correlation_ai:.4f} | {avg_glcm_correlation_real_copd:.4f} | {gr_r_win} | {gr_r_imp:+.1f}% | {gr_e_win} | {gr_e_imp:+.1f}% |
| GLCM 同质性 (≈) | {avg_glcm_homogeneity_warp:.4f} | {avg_glcm_homogeneity_real:.4f} | {avg_glcm_homogeneity_ai:.4f} | {avg_glcm_homogeneity_real_copd:.4f} | {gh_r_win} | {gh_r_imp:+.1f}% | {gh_e_win} | {gh_e_imp:+.1f}% |

### 列说明：
- **Healthy Atlas**: 健康肺模板基线，提供健康纹理的参照对比
- **Embed**: Embedded Template（病灶嵌入模板），AI 需要改进和超越的目标
- **AI vs Real COPD**: AI 融合纹理与真实 COPD 纹理的接近程度（所有指标均以 Real COPD 为目标）
- **AI vs Real COPD改进**: AI 相对 Real COPD 的改进幅度百分比
- **AI vs Embed**: AI 融合是否优于 Embedded Template（核心评估维度）
- **AI vs Embed改进**: AI 相对 Embed 的改进幅度百分比

### 指标解释：
- **清晰度 (↑)**: 拉普拉斯方差，越高 = 纹理越清晰
- **边界梯度 (↓)**: 越低 = 病灶边界融合越平滑
- **GLCM 对比度 (↑)**: 局部强度变化，越高 = 纹理细节越丰富
- **GLCM 能量 (≈)**: 纹理均匀性，越接近 Warp 越真实
- **GLCM 熵 (↑)**: 纹理复杂度，越高 = 纹理越丰富
- **GLCM 相关性 (≈)**: 灰度级线性依赖性，越接近 Warp 越真实
- **GLCM 同质性 (≈)**: 分布集中度，越接近 Warp 越真实

**符号**: ↑ = 越大越好, ↓ = 越小越好, ≈ = 越接近 Warp 越好

![纹理质量雷达图](texture_quality_radar.png)
""")

    logger.info(f"  ✓ 报告已保存: {report_path}")
    return True


# =============================================================================
# 可视化辅助函数
# =============================================================================

def apply_lung_window(img_data, level=-600, width=1500):
    """
    应用肺窗显示

    Args:
        img_data: CT 图像数据 (HU 值)
        level: 窗位 (默认 -600)
        width: 窗宽 (默认 1500)

    Returns:
        归一化到 [0, 1] 的图像
    """
    import numpy as np
    lower = level - width / 2
    upper = level + width / 2
    img_windowed = np.clip(img_data, lower, upper)
    img_windowed = (img_windowed - lower) / (upper - lower)
    return img_windowed


def get_roi_slice(mask_3d, context_size=32, view='axial'):
    """
    自动查找最佳切片和 ROI 区域

    Args:
        mask_3d: 3D mask 数据
        context_size: ROI 半径
        view: 视图类型 ('axial', 'coronal', 'sagittal')

    Returns:
        slice_idx: 最佳切片索引
        roi_coords: (y_start, y_end, x_start, x_end)
    """
    import numpy as np

    # 根据视图类型选择切片轴
    if view == 'axial':
        # Axial: 沿 Z 轴切片，显示 X-Y 平面
        axis_sum = (0, 1)  # 对 X, Y 求和
        slice_axis = 2
    elif view == 'coronal':
        # Coronal: 沿 Y 轴切片，显示 X-Z 平面
        axis_sum = (0, 2)  # 对 X, Z 求和
        slice_axis = 1
    else:  # sagittal
        # Sagittal: 沿 X 轴切片，显示 Y-Z 平面
        axis_sum = (1, 2)  # 对 Y, Z 求和
        slice_axis = 0

    # 找到病灶面积最大的切片
    sums = np.sum(mask_3d, axis=axis_sum)
    slice_idx = int(np.argmax(sums))
    if sums[slice_idx] == 0:
        slice_idx = mask_3d.shape[slice_axis] // 2

    # 提取 2D 切片
    if view == 'axial':
        mask_slice = mask_3d[:, :, slice_idx]
    elif view == 'coronal':
        mask_slice = mask_3d[:, slice_idx, :]
    else:  # sagittal
        mask_slice = mask_3d[slice_idx, :, :]

    # 找到病灶中心
    coords = np.argwhere(mask_slice > 0)
    if len(coords) > 0:
        y_c, x_c = coords.mean(axis=0).astype(int)
    else:
        y_c, x_c = mask_slice.shape[0] // 2, mask_slice.shape[1] // 2

    # 定义 ROI 边界
    y_start = max(0, y_c - context_size)
    y_end = min(mask_slice.shape[0], y_c + context_size)
    x_start = max(0, x_c - context_size)
    x_end = min(mask_slice.shape[1], x_c + context_size)

    return slice_idx, (y_start, y_end, x_start, x_end)


def plot_comparison(img_real, img_fused, mask, roi_coords, save_path, suptitle,
                    view='axial', layout='vertical', img_native=None):
    """
    绘制5列对比图（全局视图 + ROI 放大视图），列布局如下：
      第1列：COPD 原始 CT（配准前，Native 空间；若无则显示占位符）
      第2列：Real COPD（配准到模板空间的 COPD 患者 CT）
      第3列：AI 融合（模型生成结果）
      第4列：差值图（|AI 融合 - Real COPD|，jet 热力叠加）
      第5列：病变 Mask 叠加（叠加在 AI 融合图上）

    Args:
        img_real    : 第2列，配准后 Real COPD，2D 切片
        img_fused   : 第3列，AI 融合结果，2D 切片
        mask        : 病变二值 Mask，2D 切片
        roi_coords  : ROI 坐标 (y1, y2, x1, x2)
        save_path   : 输出文件路径
        suptitle    : 整图标题
        view        : 视图类型 ('axial', 'coronal', 'sagittal')
        layout      : 布局方式（'vertical' = 2行5列；'horizontal' = 5行2列）
        img_native  : 第1列，COPD 原始 CT 切片（可为 None，为 None 时显示占位符）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # 注入全局字体配置，确保中文和负号显示正常
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    y1, y2, x1, x2 = roi_coords

    # ---- 各列 ROI 切片 ----
    roi_native  = img_native[y1:y2, x1:x2] if img_native is not None else None
    roi_real    = img_real[y1:y2, x1:x2]    if img_real is not None else None
    roi_fused   = img_fused[y1:y2, x1:x2]
    roi_mask    = mask[y1:y2, x1:x2]

    # ---- 第4列：差异图 |AI 融合 - Real COPD|（仅病灶区域内）----
    mask_bool = mask > 0
    if img_real is not None:
        raw_diff = np.abs(img_fused.astype(float) - img_real.astype(float))
        # 仅保留病灶区域内的差异，非病灶区域置零
        diff = raw_diff * mask_bool.astype(float)
    else:
        diff = np.zeros_like(img_fused, dtype=float)
    roi_diff = diff[y1:y2, x1:x2]
    # 动态计算色条上限：仅基于病灶内像素的分位数
    _diff_lesion = diff[mask_bool]
    _diff_lesion_nonzero = _diff_lesion[_diff_lesion > 0] if _diff_lesion.size > 0 else np.array([])
    if _diff_lesion_nonzero.size > 0:
        # 使用病灶内第 97.5 百分位 × 1.5，聚焦病灶差异分布
        diff_vmax = max(float(np.percentile(_diff_lesion_nonzero, 97.5)) * 1.5, 10.0)
    else:
        diff_vmax = 50.0

    view_label = view.capitalize()

    # ---- 布局参数：5列 ----
    N_COLS = 5
    if layout == 'horizontal':
        fig = plt.figure(figsize=(15, 30), dpi=300)
        gs  = fig.add_gridspec(N_COLS, 2, wspace=0.12, hspace=0.15)
        pos_global = [(r, 0) for r in range(N_COLS)]
        pos_roi    = [(r, 1) for r in range(N_COLS)]
    else:
        fig = plt.figure(figsize=(30, 12), dpi=300)
        gs  = fig.add_gridspec(2, N_COLS, wspace=0.12, hspace=0.22)
        pos_global = [(0, c) for c in range(N_COLS)]
        pos_roi    = [(1, c) for c in range(N_COLS)]

    # ========================================================
    # 辅助函数
    # ========================================================
    def _add_roi_rect(ax):
        rect = mpatches.Rectangle(
            (y1, x1), y2 - y1, x2 - x1,
            linewidth=2, edgecolor='yellow', facecolor='none'
        )
        ax.add_patch(rect)

    def _roi_border(ax, color='yellow'):
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)

    # ========================================================
    # ===== 第1行（全局视图）=====
    # ========================================================

    # 列1：COPD 原始 CT（配准前）
    ax1 = fig.add_subplot(gs[pos_global[0]])
    if img_native is not None:
        ax1.imshow(apply_lung_window(img_native).T, cmap='gray', origin='lower')
        _add_roi_rect(ax1)
    else:
        ax1.text(0.5, 0.5, 'Native CT\nN/A', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=16, color='gray')
    ax1.set_title("COPD 原始 CT", fontsize=20, fontweight='bold')
    ax1.axis('off')

    # 列2：Real COPD（配准后）
    ax2 = fig.add_subplot(gs[pos_global[1]])
    if img_real is not None:
        ax2.imshow(apply_lung_window(img_real).T, cmap='gray', origin='lower')
        _add_roi_rect(ax2)
    else:
        ax2.text(0.5, 0.5, 'N/A', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=16, color='gray')
    ax2.set_title("Real COPD（配准后）", fontsize=20, fontweight='bold')
    ax2.axis('off')

    # 列3：AI 融合
    ax3 = fig.add_subplot(gs[pos_global[2]])
    ax3.imshow(apply_lung_window(img_fused).T, cmap='gray', origin='lower')
    ax3.set_title("AI 融合", fontsize=20, fontweight='bold')
    _add_roi_rect(ax3)
    ax3.axis('off')

    # 列4：差异图（|AI - Real COPD| 仅病灶区域内）
    ax4 = fig.add_subplot(gs[pos_global[3]])
    if img_real is not None:
        ax4.imshow(apply_lung_window(img_real).T, cmap='gray', alpha=0.5, origin='lower')
    im4 = ax4.imshow(diff.T, cmap='jet', alpha=0.7, origin='lower', vmin=0, vmax=diff_vmax)
    ax4.set_title("差异图 (病灶内 AI-Real)", fontsize=20, fontweight='bold')
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    cbar4 = plt.colorbar(im4, cax=cax4)
    cbar4.ax.tick_params(labelsize=11)
    _add_roi_rect(ax4)
    ax4.axis('off')

    # 列5：病变 Mask 叠加
    ax5 = fig.add_subplot(gs[pos_global[4]])
    ax5.imshow(apply_lung_window(img_fused).T, cmap='gray', origin='lower')
    ax5.imshow(mask.T, cmap='Reds', alpha=0.45, origin='lower')
    ax5.set_title("病变 Mask 叠加", fontsize=20, fontweight='bold')
    _add_roi_rect(ax5)
    ax5.axis('off')

    # ========================================================
    # ===== 第2行（ROI 放大视图）=====
    # ========================================================

    # ROI列1：COPD 原始 CT
    ax6 = fig.add_subplot(gs[pos_roi[0]])
    if roi_native is not None:
        ax6.imshow(apply_lung_window(roi_native).T, cmap='gray', origin='lower')
    else:
        ax6.text(0.5, 0.5, 'Native CT\nN/A', ha='center', va='center',
                 transform=ax6.transAxes, fontsize=14, color='gray')
    ax6.set_title("ROI：COPD 原始 CT", fontsize=19, fontweight='bold')
    _roi_border(ax6)
    ax6.axis('off')

    # ROI列2：Real COPD（配准后）
    ax7 = fig.add_subplot(gs[pos_roi[1]])
    if roi_real is not None:
        ax7.imshow(apply_lung_window(roi_real).T, cmap='gray', origin='lower')
    else:
        ax7.text(0.5, 0.5, 'N/A', ha='center', va='center',
                 transform=ax7.transAxes, fontsize=14, color='gray')
    ax7.set_title("ROI：Real COPD", fontsize=19, fontweight='bold')
    _roi_border(ax7)
    ax7.axis('off')

    # ROI列3：AI 融合
    ax8 = fig.add_subplot(gs[pos_roi[2]])
    ax8.imshow(apply_lung_window(roi_fused).T, cmap='gray', origin='lower')
    ax8.set_title("ROI：AI 融合", fontsize=19, fontweight='bold')
    _roi_border(ax8)
    ax8.axis('off')

    # ROI列4：差异图（|AI - Real COPD| 仅病灶区域内）
    ax9 = fig.add_subplot(gs[pos_roi[3]])
    if roi_real is not None:
        ax9.imshow(apply_lung_window(roi_real).T, cmap='gray', alpha=0.5, origin='lower')
    im9 = ax9.imshow(roi_diff.T, cmap='jet', alpha=0.7, origin='lower', vmin=0, vmax=diff_vmax)
    ax9.set_title("ROI: 差异图 (病灶内)", fontsize=19, fontweight='bold')
    divider9 = make_axes_locatable(ax9)
    cax9 = divider9.append_axes("right", size="5%", pad=0.05)
    cbar9 = plt.colorbar(im9, cax=cax9)
    cbar9.ax.tick_params(labelsize=11)
    _roi_border(ax9)
    ax9.axis('off')

    # ROI列5：病变 Mask 叠加
    ax10 = fig.add_subplot(gs[pos_roi[4]])
    ax10.imshow(apply_lung_window(roi_fused).T, cmap='gray', origin='lower')
    ax10.imshow(roi_mask.T, cmap='Reds', alpha=0.45, origin='lower')
    ax10.set_title("ROI：病变 Mask 叠加", fontsize=19, fontweight='bold')
    _roi_border(ax10)
    ax10.axis('off')

    plt.suptitle(f"{suptitle} [{view_label}]", fontsize=20, fontweight='bold', y=0.995)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)


def plot_histogram(data_dict, save_path, title):
    """
    绘制专业的 HU 分布直方图（论文发表质量）

    参考图片格式：左侧直方图 + 右侧详细统计信息文本框

    Args:
        data_dict: {'Label': numpy_array_of_values, ...}
        save_path: 保存路径
        title: 图表标题
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # 注入全局字体配置，确保中文和负号显示正常
    # plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
    # plt.rcParams['axes.unicode_minus'] = False

    # 设置专业风格（兼容不同版本的 matplotlib）
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            pass  # 使用默认样式

    # 创建图形：左侧直方图 + 右侧文本框
    fig = plt.figure(figsize=(16, 8), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])

    # 颜色方案（专业配色）
    colors = {
        'Healthy Template': '#3498db',      # 蓝色
        'AI Fused': '#e74c3c',      # 红色
        'Real COPD': '#27ae60'      # 绿色
    }

    # 存储统计数据
    stats_data = {}

    # 绘制直方图
    for label, data in data_dict.items():
        if data is None or len(data) == 0:
            continue

        data_flat = data.flatten()
        color = colors.get(label, '#7f8c8d')

        # 绘制直方图（填充 + 边框）- 使用 density=True 进行归一化
        ax_hist.hist(data_flat, bins=80, range=(-1024, 0),
                     alpha=0.35, label=None, color=color, density=True, histtype='stepfilled')
        ax_hist.hist(data_flat, bins=80, range=(-1024, 0),
                     alpha=1.0, color=color, density=True, histtype='step', linewidth=2, label=label)

        # 计算统计信息
        stats_data[label] = {
            'mean': np.mean(data_flat),
            'std': np.std(data_flat),
            'min': np.min(data_flat),
            'max': np.max(data_flat),
            'emphysema_ratio': np.sum(data_flat < -950) / len(data_flat) * 100
        }

    # 添加肺气肿阈值参考线
    ax_hist.axvline(x=-950, color='#2c3e50', linestyle='--', linewidth=2, alpha=0.8, label='Emphysema Threshold (-950)')

    # 设置直方图坐标轴
    ax_hist.set_xlim(-1024, 0)
    ax_hist.set_xlabel("HU Value", fontsize=16, fontweight='bold')
    ax_hist.set_ylabel("Density", fontsize=16, fontweight='bold')
    ax_hist.set_title("HU Distribution in 3D Lesion", fontsize=20, fontweight='bold')
    ax_hist.legend(loc='upper right', fontsize=14, framealpha=0.9)
    ax_hist.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_hist.set_axisbelow(True)
    ax_hist.tick_params(axis='both', which='major', labelsize=14)

    # 构建右侧统计信息文本框（参考上传图片格式）
    text_lines = ["HU Statistics (3D Lesion Volume)", "=" * 42, ""]

    # Real COPD 统计
    if 'Real COPD' in stats_data:
        s = stats_data['Real COPD']
        text_lines.extend([
            "Real COPD:",
            f"  Mean HU: {s['mean']:.1f}",
            f"  Std HU: {s['std']:.1f}",
            f"  Min HU: {s['min']:.1f}",
            f"  Max HU: {s['max']:.1f}",
            f"  Emphysema (HU<-950): {s['emphysema_ratio']:.1f}%",
            ""
        ])

    # AI Fused CT 统计
    if 'AI Fused' in stats_data:
        s = stats_data['AI Fused']
        text_lines.extend([
            "AI Fused CT:",
            f"  Mean HU: {s['mean']:.1f}",
            f"  Std HU: {s['std']:.1f}",
            f"  Min HU: {s['min']:.1f}",
            f"  Max HU: {s['max']:.1f}",
            f"  Emphysema (HU<-950): {s['emphysema_ratio']:.1f}%",
            ""
        ])

    # Template 统计（如果存在）
    if 'Healthy Template' in stats_data:
        s = stats_data['Healthy Template']
        text_lines.extend([
            "Healthy Template:",
            f"  Mean HU: {s['mean']:.1f}",
            f"  Std HU: {s['std']:.1f}",
            f"  Min HU: {s['min']:.1f}",
            f"  Max HU: {s['max']:.1f}",
            f"  Emphysema (HU<-950): {s['emphysema_ratio']:.1f}%",
            ""
        ])

    # 计算差异（如果 Real COPD 和 AI Fused 都存在）
    if 'Real COPD' in stats_data and 'AI Fused' in stats_data:
        mean_diff = stats_data['AI Fused']['mean'] - stats_data['Real COPD']['mean']
        emph_diff = stats_data['AI Fused']['emphysema_ratio'] - stats_data['Real COPD']['emphysema_ratio']
        text_lines.extend([
            "Difference (AI vs Real COPD):",
            f"  Mean HU Diff: {mean_diff:+.1f}",
            f"  Emphysema Diff: {emph_diff:+.1f}%"
        ])

    # 在右侧子图显示文本（无背景填充，纯文字）
    ax_text.axis('off')
    text_str = '\n'.join(text_lines)
    ax_text.text(0.05, 0.95, text_str, transform=ax_text.transAxes,
                 fontsize=14, verticalalignment='top', fontfamily='monospace')

    # 设置总标题
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)

    # 保存
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
    plt.close(fig)
    plt.style.use('default')  # 恢复默认样式


def run_result_visualization(
    config: dict,
    logger: logging.Logger,
    model_type: str = 'partial_conv',
    num_patients: int = 5,
    start_patient_id: str = None
) -> bool:
    """
    生成可视化结果（三图输出策略）

    输出文件：
    - {patient_id}_viz_1_generation.png: 生成效果分析（Template vs AI Fused）
    - {patient_id}_viz_2_realism.png: 真实性分析（Real COPD vs AI Fused）
    - {patient_id}_viz_3_histogram.png: HU 分布直方图

    Args:
        config: 配置字典
        logger: 日志记录器
        model_type: 模型类型
        num_patients: 可视化的患者数量
        start_patient_id: 起始患者 ID（如 copd_024，仅可视化从该患者开始的数据）

    Returns:
        bool: 是否成功
    """
    logger.info("")
    logger.info("[可视化] 生成结果可视化（三图输出）")
    logger.info("-" * 40)

    try:
        import nibabel as nib
        import numpy as np
    except ImportError as e:
        logger.error(f"  ✗ 缺少依赖: {e}")
        return False

    paths = config.get('paths', {})
    mapped_dir = Path(paths.get('mapped', 'data/03_mapped'))
    atlas_dir = Path(paths.get('atlas', 'data/02_atlas'))
    cleaned_dir = Path(paths.get('cleaned_data', 'data/01_cleaned'))
    fused_dir = Path(paths.get('final_viz', 'data/04_final_viz')) / model_type
    output_dir = Path(f'results/{model_type}')  # 合并后的输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 原始 COPD CT（配准前 Native CT）目录
    native_ct_dir = cleaned_dir / 'copd_clean'

    template_path = atlas_dir / 'standard_template.nii.gz'

    # 检查文件
    if not fused_dir.exists():
        logger.error(f"  ✗ 融合结果目录不存在: {fused_dir}")
        logger.error(f"  请先运行推理: --inference --model-type {model_type}")
        return False

    if not template_path.exists():
        logger.error(f"  ✗ 模板不存在: {template_path}")
        return False

    fused_files = sorted(fused_dir.glob("*_fused.nii.gz"))

    # 应用 start_patient_id 过滤
    if start_patient_id:
        fused_files = [f for f in fused_files
                       if f.name.replace('_fused.nii.gz', '') >= start_patient_id]
        logger.info(f"  从 {start_patient_id} 开始可视化")

    fused_files = fused_files[:num_patients]
    if not fused_files:
        logger.error(f"  ✗ 未找到融合结果文件")
        return False

    logger.info(f"  融合结果目录: {fused_dir}")
    logger.info(f"  可视化患者数: {len(fused_files)}")
    logger.info(f"  输出目录: {output_dir}")

    # 加载模板
    template_data = nib.load(str(template_path)).get_fdata()

    # model_names = {
    #     'unet': '3D U-Net',
    #     'partial_conv': 'Partial Conv',
    #     'patchgan': 'PatchGAN'
    # }
    # model_name = model_names.get(model_type, model_type)

    success_count = 0
    for fused_path in fused_files:
        patient_id = fused_path.name.replace('_fused.nii.gz', '')
        mask_path = mapped_dir / patient_id / f"{patient_id}_warped_lesion.nii.gz"
        real_path = mapped_dir / patient_id / f"{patient_id}_warped.nii.gz"

        # 原始 COPD CT（配准前 Native CT）
        native_path = native_ct_dir / f"{patient_id}_clean.nii.gz"

        if not mask_path.exists():
            logger.warning(f"  ⚠ 跳过 {patient_id}: mask 不存在")
            continue

        try:
            # 创建患者子目录
            patient_output_dir = output_dir / patient_id
            patient_output_dir.mkdir(parents=True, exist_ok=True)

            # 1. 加载数据
            fused_data = nib.load(str(fused_path)).get_fdata()
            mask_data = nib.load(str(mask_path)).get_fdata()
            real_data = nib.load(str(real_path)).get_fdata() if real_path.exists() else None

            # 加载原始 COPD CT（配准前）
            native_data = None
            if native_path.exists():
                native_data = nib.load(str(native_path)).get_fdata()

            files_generated = 0

            # 2. 为每个视图生成可视化
            for view in ['axial', 'coronal', 'sagittal']:
                # 获取该视图的最佳切片和 ROI
                slice_idx, roi_coords = get_roi_slice(mask_data, context_size=48, view=view)

                # 提取 2D 切片（模板空间：fused / mask / real）
                if view == 'axial':
                    fused_slice    = fused_data[:, :, slice_idx]
                    mask_slice     = mask_data[:, :, slice_idx]
                    real_slice     = real_data[:, :, slice_idx] if real_data is not None else None
                elif view == 'coronal':
                    fused_slice    = fused_data[:, slice_idx, :]
                    mask_slice     = mask_data[:, slice_idx, :]
                    real_slice     = real_data[:, slice_idx, :] if real_data is not None else None
                else:  # sagittal
                    fused_slice    = fused_data[slice_idx, :, :]
                    mask_slice     = mask_data[slice_idx, :, :]
                    real_slice     = real_data[slice_idx, :, :] if real_data is not None else None

                # 提取原始 COPD CT 切片（配准前 Native 空间）
                # 由于 Native CT 和模板空间维度可能不同，按比例映射切片索引
                native_slice = None
                if native_data is not None:
                    if view == 'axial':
                        native_z = native_data.shape[2]
                        template_z = template_data.shape[2]
                        native_idx = int(slice_idx * native_z / template_z)
                        native_idx = min(native_idx, native_z - 1)
                        native_slice = native_data[:, :, native_idx]
                    elif view == 'coronal':
                        native_z = native_data.shape[1]
                        template_z = template_data.shape[1]
                        native_idx = int(slice_idx * native_z / template_z)
                        native_idx = min(native_idx, native_z - 1)
                        native_slice = native_data[:, native_idx, :]
                    else:  # sagittal
                        native_z = native_data.shape[0]
                        template_z = template_data.shape[0]
                        native_idx = int(slice_idx * native_z / template_z)
                        native_idx = min(native_idx, native_z - 1)
                        native_slice = native_data[native_idx, :, :]

                # 生成图像 1：5 列全对比图（viz_1_generation）
                # 新布局：原始CT | Real COPD | AI 融合 | 差异图(AI-Real) | Mask叠加
                plot_comparison(
                    real_slice, fused_slice,
                    mask_slice, roi_coords,
                    patient_output_dir / f"{patient_id}_viz_1_generation_{view}.png",
                    f"{patient_id} - 生成效果五列对比",
                    view=view,
                    img_native=native_slice
                )
                files_generated += 1

                # 生成图像 2：Realism（Real vs Fused）
                # 复用 plot_comparison 接口，不再传 img_native
                if real_slice is not None:
                    plot_comparison(
                        real_slice, fused_slice,
                        mask_slice, roi_coords,
                        patient_output_dir / f"{patient_id}_viz_2_realism_{view}.png",
                        f"{patient_id} - 真实度分析",
                        view=view
                    )
                    files_generated += 1

            # 3. 生成 Histogram（使用完整 3D 体积的病灶区域，与评估报告保持一致）
            # Use full 3D lesion volume for histogram, consistent with evaluation metrics
            mask_bool = mask_data > 0  # 完整 3D 病灶 mask

            hist_data = {
                'Healthy Template': template_data[mask_bool],
                'AI Fused': fused_data[mask_bool],
            }
            if real_data is not None:
                # Real COPD 在 mask 区域内的 HU 值就是 real_data 本身
                # （因为 embedded = template.copy(); embedded[mask] = real[mask]
                #   → embedded[mask] ≡ real[mask]）
                # 无需拷贝整个 3D 体积，直接取 real_data[mask_bool] 即可
                hist_data['Real COPD'] = real_data[mask_bool]

            plot_histogram(
                hist_data,
                patient_output_dir / f"{patient_id}_viz_3_histogram.png",
                f"{patient_id} - HU Distribution in 3D Lesion"
            )
            files_generated += 1

            logger.info(f"  ✓ {patient_id}: 生成 {files_generated} 张可视化图像")
            success_count += 1

        except Exception as e:
            logger.error(f"  ✗ {patient_id} 可视化失败: {e}")

    logger.info("")
    logger.info(f"  可视化完成: {success_count}/{len(fused_files)} 成功")
    return success_count > 0


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 流水线: 病理映射与 AI 融合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整流水线 (3A + 3B)
  python run_phase3_pipeline.py --full

  # 仅 Phase 3A（默认）
  python run_phase3_pipeline.py

  # 快速测试（仅处理 3 例）
  python run_phase3_pipeline.py --quick-test

  # Phase 3B 训练（三种模型）
  python run_phase3_pipeline.py --phase3b --model-type unet
  python run_phase3_pipeline.py --phase3b --model-type partial_conv
  python run_phase3_pipeline.py --phase3b --model-type patchgan

  # Phase 3B 推理
  python run_phase3_pipeline.py --inference --model-type unet

  # 模型评估（与真实 COPD CT 对比）
  python run_phase3_pipeline.py --evaluate --model-type unet

  # 生成可视化结果
  python run_phase3_pipeline.py --visualize --model-type unet

  # 使用呼气相数据流（文件名自动插入 _exp 中缀，目录不变）
  python run_phase3_pipeline.py --expiration

  # 跳过配准（使用已有结果）
  python run_phase3_pipeline.py --skip-registration

  # 仅执行可视化
  python run_phase3_pipeline.py --viz-only
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
    step_group.add_argument(
        '--mae-pretrain', action='store_true',
        help='执行 MAE 自监督预训练（MAE-PatchGAN 的第一阶段，输出 encoder 权重到 checkpoints/mae_pretrain/）'
    )
    step_group.add_argument(
        '--phase3b', '--train', action='store_true',
        help='执行 Phase 3B 训练'
    )
    step_group.add_argument(
        '--inference', action='store_true',
        help='执行 Phase 3B 推理'
    )
    step_group.add_argument(
        '--evaluate', '-e', action='store_true',
        help='执行模型评估（与真实 COPD CT 对比）'
    )
    step_group.add_argument(
        '--visualize', '-v', action='store_true',
        help='生成可视化结果（多视图对比图）'
    )
    step_group.add_argument(
        '--full', action='store_true',
        help='完整流水线 (3A + 3B 训练 + 推理)'
    )

    # Phase 3B 参数
    parser.add_argument(
        '--model-type', type=str, default='partial_conv',
        choices=['unet', 'partial_conv', 'patchgan', 'attgan', 'mae_patchgan', 'ddpm'],
        help='模型类型: unet(基线), partial_conv(进阶), patchgan(高级), attgan(注意力增强), mae_patchgan(MAE预训练), ddpm(扩散模型)'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='训练轮数（覆盖配置文件）'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='批次大小（覆盖配置文件）'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='学习率（覆盖配置文件）'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='推理使用的模型检查点路径'
    )
    parser.add_argument(
        '--patient', type=str, default=None,
        help='指定患者 ID（推理时使用）'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='计算设备 (cuda/cpu)'
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
        '--start-patient-id', type=str, default=None,
        help='起始患者 ID（如 copd_024），从该患者开始处理/评估/可视化。'
             '用于在验证集（未参与训练的患者）上评估模型泛化能力。'
             '示例: --start-patient-id copd_024 --limit 3'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='输出目录（默认: data/03_mapped/）'
    )
    parser.add_argument(
        '--check-only', action='store_true',
        help='仅检查环境和数据，不执行流水线'
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='配置文件路径（默认: config.yaml）'
    )
    parser.add_argument(
        '--expiration',
        action='store_true',
        default=False,
        help='使用呼气相数据流（默认使用吸气相数据流）。\n'
             '文件名自动插入 _exp 中缀（如 copd_001_exp_warped.nii.gz），\n'
             '与吸气相数据共享同一目录，不会覆盖任何吸气相文件。'
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

    # 呼气相模式：不修改任何目录路径，仅在各函数中通过文件名筛选区分相位
    if args.expiration:
        logger.info("[呼气相模式] 文件名中缀: '_exp'，目录路径不变")

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
    logger.info(f"数据相位: {'呼气相 (--expiration)' if args.expiration else '吸气相（默认）'}")

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
    # --mae-pretrain 模式 (MAE 自监督预训练)
    # =========================================================================
    if args.mae_pretrain:
        logger.info("")
        logger.info("模式: MAE 自监督预训练 (--mae-pretrain)")

        pretrain_ok, pretrain_results = run_mae_pretrain(
            config, logger,
            epochs=args.epochs,
        )

        elapsed = time.time() - pipeline_start
        logger.info("")
        logger.info("=" * 60)
        if pretrain_ok:
            logger.info(f"MAE 预训练完成，耗时: {elapsed/60:.1f} 分钟")
            logger.info(f"Encoder 权重: {pretrain_results.get('encoder_weights', '')}")
            logger.info("下一步: python run_phase3_pipeline.py --phase3b --model-type mae_patchgan --epochs 50")
        else:
            logger.error("MAE 预训练失败")
        logger.info("=" * 60)
        sys.exit(0 if pretrain_ok else 1)

    # =========================================================================
    # --phase3b 模式 (仅训练)
    # =========================================================================
    if args.phase3b:
        logger.info("")
        logger.info("模式: Phase 3B 训练 (--phase3b)")

        train_ok, train_results = run_texture_training(
            config, logger,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )

        elapsed = time.time() - pipeline_start
        logger.info("")
        logger.info("=" * 60)
        if train_ok:
            logger.info(f"Phase 3B 训练完成，耗时: {elapsed/60:.1f} 分钟")
            logger.info(f"模型保存: {train_results.get('best_model', 'checkpoints/best.pth')}")
        else:
            logger.error("Phase 3B 训练失败")
        logger.info("=" * 60)
        sys.exit(0 if train_ok else 1)

    # =========================================================================
    # --inference 模式 (仅推理)
    # =========================================================================
    if args.inference:
        logger.info("")
        logger.info("模式: Phase 3B 推理 (--inference)")

        infer_ok, _ = run_texture_inference(
            config, logger,
            checkpoint_path=args.checkpoint,
            patient_id=args.patient,
            device=args.device,
            model_type=args.model_type,
            limit=args.limit,  # 传递 limit 参数
            start_patient_id=args.start_patient_id
        )

        elapsed = time.time() - pipeline_start
        logger.info("")
        logger.info("=" * 60)
        if infer_ok:
            logger.info(f"Phase 3B 推理完成，耗时: {elapsed:.1f} 秒")
        else:
            logger.error("Phase 3B 推理失败")
        logger.info("=" * 60)
        sys.exit(0 if infer_ok else 1)

    # =========================================================================
    # --evaluate 模式 (模型评估)
    # =========================================================================
    if args.evaluate:
        logger.info("")
        logger.info("模式: 模型评估 (--evaluate)")
        logger.info(f"模型类型: {args.model_type}")

        eval_ok = run_model_evaluation(
            config, logger,
            model_type=args.model_type,
            num_patients=args.limit or 10,
            start_patient_id=args.start_patient_id
        )

        elapsed = time.time() - pipeline_start
        logger.info("")
        logger.info("=" * 60)
        if eval_ok:
            logger.info(f"模型评估完成，耗时: {elapsed:.1f} 秒")
            logger.info(f"结果保存: results/{args.model_type}/")
        else:
            logger.error("模型评估失败")
        logger.info("=" * 60)
        sys.exit(0 if eval_ok else 1)

    # =========================================================================
    # --visualize 模式 (生成可视化)
    # =========================================================================
    if args.visualize:
        logger.info("")
        logger.info("模式: 生成可视化 (--visualize)")
        logger.info(f"模型类型: {args.model_type}")

        viz_ok = run_result_visualization(
            config, logger,
            model_type=args.model_type,
            num_patients=args.limit or 5,
            start_patient_id=args.start_patient_id
        )

        elapsed = time.time() - pipeline_start
        logger.info("")
        logger.info("=" * 60)
        if viz_ok:
            logger.info(f"可视化生成完成，耗时: {elapsed:.1f} 秒")
            logger.info(f"结果保存: results/{args.model_type}/")
        else:
            logger.error("可视化生成失败")
        logger.info("=" * 60)
        sys.exit(0 if viz_ok else 1)

    # =========================================================================
    # 正常流水线 (Phase 3A) 或完整流水线 (--full)
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
    data_ok, _ = run_data_validation(config, logger, args.quick_test)
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
        mapping_ok, _ = run_spatial_mapping(
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
    # Phase 3B (如果 --full 模式)
    # =========================================================================
    if args.full:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  继续执行 Phase 3B: AI 纹理融合")
        logger.info("=" * 60)

        # Step 5: 训练
        train_ok, _ = run_texture_training(
            config, logger,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )

        if train_ok:
            # Step 6: 推理
            run_texture_inference(
                config, logger,
                checkpoint_path=args.checkpoint,
                device=args.device,
                model_type=args.model_type,
                limit=args.limit  # 传递 limit 参数
            )

    # =========================================================================
    # 总结
    # =========================================================================
    pipeline_elapsed = time.time() - pipeline_start
    logger.info("")
    logger.info("=" * 60)
    if args.full:
        logger.info("Phase 3 完整流水线执行完成")
    else:
        logger.info("Phase 3A 流水线执行完成")
    logger.info("=" * 60)
    logger.info(f"总耗时: {pipeline_elapsed/60:.1f} 分钟")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 输出结果摘要
    logger.info("")
    logger.info("输出目录:")
    logger.info(f"  映射结果: {output_dir}")

    final_viz_dir = Path(config['paths'].get('final_viz', 'data/04_final_viz'))
    if final_viz_dir.exists() and any(final_viz_dir.glob("*.nii.gz")):
        logger.info(f"  融合结果: {final_viz_dir}")

    # 统计结果
    patient_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name != 'visualizations']
    logger.info(f"  已处理患者: {len(patient_dirs)} 例")

    if not args.full:
        logger.info("")
        logger.info("下一步:")
        logger.info("  1. 检查 data/03_mapped/ 中的映射结果")
        logger.info("  2. 运行 Phase 3B 训练:")
        logger.info("     python run_phase3_pipeline.py --phase3b --model-type unet")
        logger.info("  3. 运行 Phase 3B 推理:")
        logger.info("     python run_phase3_pipeline.py --inference")


if __name__ == "__main__":
    main()

