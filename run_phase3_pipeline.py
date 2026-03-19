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
    python run_phase3_pipeline.py --inference --model-type unet

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
    python run_phase3_pipeline.py --visualize --model-type patchgan



作者：DigitalTwinLung COPD Project
日期：2025-12-30
更新：2026-01-07 (添加 Phase 3B 支持)
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
        'patchgan': '高级方案 (PatchGAN)'
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
    limit: int = None
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
                smooth_boundary_width=3 if smooth_boundary else 0
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

    # 计算改进百分比
    sharpness_improve = ((metrics['sharpness_ai'] - metrics['sharpness_warp']) /
                         (metrics['sharpness_warp'] + 1e-10)) * 100
    boundary_improve = ((metrics['boundary_grad_warp'] - metrics['boundary_grad_ai']) /
                        (metrics['boundary_grad_warp'] + 1e-10)) * 100

    # 判断胜负
    sharpness_winner = '✓ AI wins' if metrics['sharpness_ai'] > metrics['sharpness_warp'] else '✗ Warp wins'
    boundary_winner = '✓ AI wins' if metrics['boundary_grad_ai'] < metrics['boundary_grad_warp'] else '✗ Warp wins'

    # GLCM 特征比较
    contrast_dist_ai = abs(metrics['glcm_contrast_ai'] - metrics['glcm_contrast_real'])
    contrast_dist_warp = abs(metrics['glcm_contrast_warp'] - metrics['glcm_contrast_real'])
    contrast_winner = '✓ AI closer' if contrast_dist_ai < contrast_dist_warp else '✗ Warp closer'

    energy_dist_ai = abs(metrics['glcm_energy_ai'] - metrics['glcm_energy_real'])
    energy_dist_warp = abs(metrics['glcm_energy_warp'] - metrics['glcm_energy_real'])
    energy_winner = '✓ AI closer' if energy_dist_ai < energy_dist_warp else '✗ Warp closer'

    entropy_dist_ai = abs(metrics['glcm_entropy_ai'] - metrics['glcm_entropy_real'])
    entropy_dist_warp = abs(metrics['glcm_entropy_warp'] - metrics['glcm_entropy_real'])
    entropy_winner = '✓ AI closer' if entropy_dist_ai < entropy_dist_warp else '✗ Warp closer'

    correlation_dist_ai = abs(metrics['glcm_correlation_ai'] - metrics['glcm_correlation_real'])
    correlation_dist_warp = abs(metrics['glcm_correlation_warp'] - metrics['glcm_correlation_real'])
    correlation_winner = '✓ AI closer' if correlation_dist_ai < correlation_dist_warp else '✗ Warp closer'

    homogeneity_dist_ai = abs(metrics['glcm_homogeneity_ai'] - metrics['glcm_homogeneity_real'])
    homogeneity_dist_warp = abs(metrics['glcm_homogeneity_warp'] - metrics['glcm_homogeneity_real'])
    homogeneity_winner = '✓ AI closer' if homogeneity_dist_ai < homogeneity_dist_warp else '✗ Warp closer'

    # 生成报告
    report_path = patient_output_dir / f"{patient_id}_evaluation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 患者评估报告 - {patient_id}\n\n")
        f.write(f"模型类型: {model_type}\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 基础指标\n\n")
        f.write("| 指标 | 值 |\n|------|----|\n")
        f.write(f"| PSNR | {metrics['psnr']:.2f} dB |\n")
        f.write(f"| SSIM | {metrics['ssim']:.4f} |\n")
        f.write(f"| 真实肺气肿比例 | {metrics['real_emphysema_ratio']:.1%} |\n")
        f.write(f"| AI肺气肿比例 | {metrics['fused_emphysema_ratio']:.1%} |\n")
        f.write(f"| 病灶体素数 | {metrics['voxel_count']} |\n\n")

        f.write(f"""## 纹理质量分析

| 指标 | Real COPD | AI Fused | Direct Warp | AI vs Warp | 改进幅度 |
|------|-----------|----------|-------------|------------|----------|
| 清晰度 (↑ 越高越好) | {metrics['sharpness_real']:.2f} | {metrics['sharpness_ai']:.2f} | {metrics['sharpness_warp']:.2f} | {sharpness_winner} | {sharpness_improve:+.1f}% |
| 边界梯度 (↓ 越低越好) | {metrics['boundary_grad_real']:.2f} | {metrics['boundary_grad_ai']:.2f} | {metrics['boundary_grad_warp']:.2f} | {boundary_winner} | {boundary_improve:+.1f}% |
| GLCM 对比度 (↑ 越高越好) | {metrics['glcm_contrast_real']:.2f} | {metrics['glcm_contrast_ai']:.2f} | {metrics['glcm_contrast_warp']:.2f} | {contrast_winner} | Δ: AI={contrast_dist_ai:.2f}, Warp={contrast_dist_warp:.2f} |
| GLCM 能量 (≈ Real 越近越好) | {metrics['glcm_energy_real']:.4f} | {metrics['glcm_energy_ai']:.4f} | {metrics['glcm_energy_warp']:.4f} | {energy_winner} | Δ: AI={energy_dist_ai:.4f}, Warp={energy_dist_warp:.4f} |
| GLCM 熵 (↑ 越高越好) | {metrics['glcm_entropy_real']:.2f} | {metrics['glcm_entropy_ai']:.2f} | {metrics['glcm_entropy_warp']:.2f} | {entropy_winner} | Δ: AI={entropy_dist_ai:.2f}, Warp={entropy_dist_warp:.2f} |
| GLCM 相关性 (≈ Real 越近越好) | {metrics['glcm_correlation_real']:.4f} | {metrics['glcm_correlation_ai']:.4f} | {metrics['glcm_correlation_warp']:.4f} | {correlation_winner} | Δ: AI={correlation_dist_ai:.4f}, Warp={correlation_dist_warp:.4f} |
| GLCM 同质性 (≈ Real 越近越好) | {metrics['glcm_homogeneity_real']:.4f} | {metrics['glcm_homogeneity_ai']:.4f} | {metrics['glcm_homogeneity_warp']:.4f} | {homogeneity_winner} | Δ: AI={homogeneity_dist_ai:.4f}, Warp={homogeneity_dist_warp:.4f} |

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

    使用混合评价维度：
    - Sharpness & Boundary: 归一化后的绝对值（展示 AI 的优势）
    - GLCM 特征: 与 Real COPD 的相似度（展示纹理保真度）

    Args:
        output_dir: 输出目录
        contrast_*, energy_*, entropy_*, correlation_*, homogeneity_*: 各方法的 GLCM 特征值
        logger: 日志记录器
        filename: 输出文件名（默认: 'texture_quality_radar.png'）
        sharpness_*, boundary_*: 清晰度和边界梯度值（可选）
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 判断是否包含 Sharpness 和 Boundary 数据
    include_sharpness_boundary = (sharpness_real is not None and boundary_real is not None)

    if include_sharpness_boundary:
        # 完整版雷达图：包含 Sharpness, Boundary 和 GLCM 特征
        categories = ['Sharpness\n(↑)', 'Boundary\n(↓)', 'GLCM\nContrast', 'GLCM\nEnergy',
                      'GLCM\nEntropy', 'GLCM\nCorrelation', 'GLCM\nHomogeneity']
        N = len(categories)

        # === 1. Sharpness: 归一化到 0-1（越高越好）===
        # 使用带 padding 的归一化：防止当只有 2 个不同值时退化为 0/1 极端
        all_sharpness = [sharpness_real, sharpness_ai, sharpness_warp]
        min_sharp, max_sharp = min(all_sharpness), max(all_sharpness)
        sharp_range = max_sharp - min_sharp if max_sharp > min_sharp else 1.0
        # 添加 10% padding，使最小值归一化为 0.1 而非 0.0
        padding = 0.1

        sharp_real_norm = padding + (1 - 2 * padding) * (sharpness_real - min_sharp) / sharp_range
        sharp_ai_norm = padding + (1 - 2 * padding) * (sharpness_ai - min_sharp) / sharp_range
        sharp_warp_norm = padding + (1 - 2 * padding) * (sharpness_warp - min_sharp) / sharp_range

        # === 2. Boundary: 归一化到 0-1 并取反（越低越好 → 雷达图上越高越好）===
        all_boundary = [boundary_real, boundary_ai, boundary_warp]
        min_bound, max_bound = min(all_boundary), max(all_boundary)
        bound_range = max_bound - min_bound if max_bound > min_bound else 1.0

        # 取反 + padding：低 boundary 值映射到高分
        bound_real_norm = padding + (1 - 2 * padding) * (1.0 - (boundary_real - min_bound) / bound_range)
        bound_ai_norm = padding + (1 - 2 * padding) * (1.0 - (boundary_ai - min_bound) / bound_range)
        bound_warp_norm = padding + (1 - 2 * padding) * (1.0 - (boundary_warp - min_bound) / bound_range)

        # === 3. GLCM 特征: 使用相似度（越接近 Real 越好）===
        glcm_real_values = [contrast_real, energy_real, entropy_real, correlation_real, homogeneity_real]
        glcm_ai_values = [contrast_ai, energy_ai, entropy_ai, correlation_ai, homogeneity_ai]
        glcm_warp_values = [contrast_warp, energy_warp, entropy_warp, correlation_warp, homogeneity_warp]

        # 计算距离
        glcm_ai_distances = [abs(ai - real) for ai, real in zip(glcm_ai_values, glcm_real_values)]
        glcm_warp_distances = [abs(warp - real) for warp, real in zip(glcm_warp_values, glcm_real_values)]

        # 归一化距离
        all_glcm_distances = glcm_ai_distances + glcm_warp_distances
        max_glcm_dist = max(all_glcm_distances) if max(all_glcm_distances) > 0 else 1.0

        # 转换为相似度
        glcm_ai_similarity = [1.0 - (d / max_glcm_dist) for d in glcm_ai_distances]
        glcm_warp_similarity = [1.0 - (d / max_glcm_dist) for d in glcm_warp_distances]
        glcm_real_similarity = [1.0] * 5  # Real 与自己相似度为 1

        # 组合所有维度
        real_scores = [sharp_real_norm, bound_real_norm] + glcm_real_similarity
        ai_scores = [sharp_ai_norm, bound_ai_norm] + glcm_ai_similarity
        warp_scores = [sharp_warp_norm, bound_warp_norm] + glcm_warp_similarity

    else:
        # 简化版雷达图：只包含 GLCM 特征
        categories = ['Contrast', 'Energy', 'Entropy', 'Correlation', 'Homogeneity']
        N = len(categories)

        real_values = [contrast_real, energy_real, entropy_real, correlation_real, homogeneity_real]
        ai_values = [contrast_ai, energy_ai, entropy_ai, correlation_ai, homogeneity_ai]
        warp_values = [contrast_warp, energy_warp, entropy_warp, correlation_warp, homogeneity_warp]

        ai_distances = [abs(ai - real) for ai, real in zip(ai_values, real_values)]
        warp_distances = [abs(warp - real) for warp, real in zip(warp_values, real_values)]

        all_distances = ai_distances + warp_distances
        max_dist = max(all_distances) if max(all_distances) > 0 else 1.0

        real_scores = [1.0] * N
        ai_scores = [1.0 - (d / max_dist) for d in ai_distances]
        warp_scores = [1.0 - (d / max_dist) for d in warp_distances]

    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合多边形

    # 闭合数据
    real_scores += real_scores[:1]
    ai_scores += ai_scores[:1]
    warp_scores += warp_scores[:1]

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # 绘制三个多边形
    ax.plot(angles, real_scores, 'o-', linewidth=2.5, label='Real COPD (Reference)', color='#2ecc71', markersize=8)
    ax.fill(angles, real_scores, alpha=0.15, color='#2ecc71')

    ax.plot(angles, ai_scores, 's-', linewidth=2.5, label='AI Fused', color='#3498db', markersize=8)
    ax.fill(angles, ai_scores, alpha=0.25, color='#3498db')

    ax.plot(angles, warp_scores, '^-', linewidth=2.5, label='Direct Warp', color='#e74c3c', markersize=8)
    ax.fill(angles, warp_scores, alpha=0.25, color='#e74c3c')

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, weight='bold')

    # 设置 y 轴范围和网格
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
    ax.grid(True, linestyle='--', alpha=0.7)

    # 设置标题和图例
    if include_sharpness_boundary:
        title = 'Comprehensive Quality Radar Chart\n(Larger area = Better overall quality)'
    else:
        title = 'GLCM Texture Similarity to Real COPD\n(Larger area = More similar to Real COPD)'

    ax.set_title(title, size=14, weight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, framealpha=0.9)

    # 保存图形
    radar_path = output_dir / filename
    plt.tight_layout()
    plt.savefig(radar_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  ✓ 雷达图已保存: {radar_path}")


def run_model_evaluation(
    config: dict,
    logger: logging.Logger,
    model_type: str = 'partial_conv',
    num_patients: int = 10
) -> bool:
    """
    执行模型评估（与真实 COPD CT 对比）

    Args:
        config: 配置字典
        logger: 日志记录器
        model_type: 模型类型
        num_patients: 评估的患者数量

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

    # 加载标准模板作为 Direct Warp 基线
    # Direct Warp = 模板 + 病灶区域未经 AI 合成的原始 HU 值
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
        logger.info(f"  Direct Warp 基线: {template_path.name}")
    else:
        template_data_for_warp = None
        logger.warning(f"  ⚠ 模板不存在，Direct Warp 将回退为 Real COPD")

    # 检查融合结果目录
    if not fused_dir.exists():
        logger.error(f"  ✗ 融合结果目录不存在: {fused_dir}")
        logger.error(f"  请先运行推理: --inference --model-type {model_type}")
        return False

    fused_files = sorted(fused_dir.glob("*_fused.nii.gz"))[:num_patients]
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
            # Direct Warp 基线：模板在病灶区域的原始 HU 值
            # 概念：如果不用 AI，直接把病灶 mask 叠加到模板上，质量如何？
            if template_data_for_warp is not None:
                warp_data = template_data_for_warp
            else:
                warp_data = real_data  # 回退：模板不可用时用 real_data

            # 找到病灶面积最大的切片用于 2D 指标计算
            slice_areas = [np.sum(mask_data[:, :, z] > 0) for z in range(mask_data.shape[2])]
            best_slice_idx = int(np.argmax(slice_areas))

            # 提取 2D 切片
            real_slice = real_data[:, :, best_slice_idx]
            fused_slice = fused_data[:, :, best_slice_idx]
            warp_slice = warp_data[:, :, best_slice_idx]
            mask_slice = mask_data[:, :, best_slice_idx]

            # 计算清晰度 (值越高越好)
            sharpness_real = compute_sharpness(real_slice, mask_slice)
            sharpness_ai = compute_sharpness(fused_slice, mask_slice)
            sharpness_warp = compute_sharpness(warp_slice, mask_slice)

            # 计算边界连续性 (值越低融合越平滑)
            boundary_real = compute_boundary_continuity(real_slice, mask_slice)
            boundary_ai = compute_boundary_continuity(fused_slice, mask_slice)
            boundary_warp = compute_boundary_continuity(warp_slice, mask_slice)

            # 计算 GLCM 纹理特征
            glcm_real = compute_glcm_features(real_slice, mask_slice)
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
                # 新增纹理质量指标
                'sharpness_real': sharpness_real,
                'sharpness_ai': sharpness_ai,
                'sharpness_warp': sharpness_warp,
                'boundary_grad_real': boundary_real,
                'boundary_grad_ai': boundary_ai,
                'boundary_grad_warp': boundary_warp,
                # 扩展 GLCM 特征
                'glcm_contrast_real': glcm_real['glcm_contrast'],
                'glcm_contrast_ai': glcm_ai['glcm_contrast'],
                'glcm_contrast_warp': glcm_warp['glcm_contrast'],
                'glcm_energy_real': glcm_real['glcm_energy'],
                'glcm_energy_ai': glcm_ai['glcm_energy'],
                'glcm_energy_warp': glcm_warp['glcm_energy'],
                'glcm_entropy_real': glcm_real['glcm_entropy'],
                'glcm_entropy_ai': glcm_ai['glcm_entropy'],
                'glcm_entropy_warp': glcm_warp['glcm_entropy'],
                'glcm_correlation_real': glcm_real['glcm_correlation'],
                'glcm_correlation_ai': glcm_ai['glcm_correlation'],
                'glcm_correlation_warp': glcm_warp['glcm_correlation'],
                'glcm_homogeneity_real': glcm_real['glcm_homogeneity'],
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
            try:
                _generate_texture_radar_chart(
                    patient_output_dir,
                    glcm_real['glcm_contrast'], glcm_ai['glcm_contrast'], glcm_warp['glcm_contrast'],
                    glcm_real['glcm_energy'], glcm_ai['glcm_energy'], glcm_warp['glcm_energy'],
                    glcm_real['glcm_entropy'], glcm_ai['glcm_entropy'], glcm_warp['glcm_entropy'],
                    glcm_real['glcm_correlation'], glcm_ai['glcm_correlation'], glcm_warp['glcm_correlation'],
                    glcm_real['glcm_homogeneity'], glcm_ai['glcm_homogeneity'], glcm_warp['glcm_homogeneity'],
                    logger,
                    filename=f"{patient_id}_texture_radar.png",
                    sharpness_real=sharpness_real,
                    sharpness_ai=sharpness_ai,
                    sharpness_warp=sharpness_warp,
                    boundary_real=boundary_real,
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
    avg_sharpness_ai = np.mean([m['sharpness_ai'] for m in all_metrics])
    avg_sharpness_warp = np.mean([m['sharpness_warp'] for m in all_metrics])

    avg_boundary_real = np.mean([m['boundary_grad_real'] for m in all_metrics])
    avg_boundary_ai = np.mean([m['boundary_grad_ai'] for m in all_metrics])
    avg_boundary_warp = np.mean([m['boundary_grad_warp'] for m in all_metrics])

    # 扩展 GLCM 特征平均值
    avg_glcm_contrast_real = np.mean([m['glcm_contrast_real'] for m in all_metrics])
    avg_glcm_contrast_ai = np.mean([m['glcm_contrast_ai'] for m in all_metrics])
    avg_glcm_contrast_warp = np.mean([m['glcm_contrast_warp'] for m in all_metrics])

    avg_glcm_energy_real = np.mean([m['glcm_energy_real'] for m in all_metrics])
    avg_glcm_energy_ai = np.mean([m['glcm_energy_ai'] for m in all_metrics])
    avg_glcm_energy_warp = np.mean([m['glcm_energy_warp'] for m in all_metrics])

    avg_glcm_entropy_real = np.mean([m['glcm_entropy_real'] for m in all_metrics])
    avg_glcm_entropy_ai = np.mean([m['glcm_entropy_ai'] for m in all_metrics])
    avg_glcm_entropy_warp = np.mean([m['glcm_entropy_warp'] for m in all_metrics])

    avg_glcm_correlation_real = np.mean([m['glcm_correlation_real'] for m in all_metrics])
    avg_glcm_correlation_ai = np.mean([m['glcm_correlation_ai'] for m in all_metrics])
    avg_glcm_correlation_warp = np.mean([m['glcm_correlation_warp'] for m in all_metrics])

    avg_glcm_homogeneity_real = np.mean([m['glcm_homogeneity_real'] for m in all_metrics])
    avg_glcm_homogeneity_ai = np.mean([m['glcm_homogeneity_ai'] for m in all_metrics])
    avg_glcm_homogeneity_warp = np.mean([m['glcm_homogeneity_warp'] for m in all_metrics])

    # 计算改进百分比
    # Sharpness: 越高越好，improvement = (AI - Warp) / Warp * 100%
    sharpness_improve = ((avg_sharpness_ai - avg_sharpness_warp) / (avg_sharpness_warp + 1e-10)) * 100
    # Boundary: 越低越好，improvement = (Warp - AI) / Warp * 100%
    boundary_improve = ((avg_boundary_warp - avg_boundary_ai) / (avg_boundary_warp + 1e-10)) * 100

    logger.info("")
    logger.info("  === 评估汇总 ===")
    logger.info(f"  平均 PSNR: {avg_psnr:.2f} dB")
    logger.info(f"  平均 SSIM: {avg_ssim:.4f}")
    logger.info(f"  真实 COPD 平均肺气肿比例: {avg_real_emph:.1%}")
    logger.info(f"  AI 融合 平均肺气肿比例: {avg_fused_emph:.1%}")
    logger.info("")
    logger.info("  === 纹理质量分析 ===")
    logger.info(f"  清晰度 (↑更好): Real={avg_sharpness_real:.2f}, AI={avg_sharpness_ai:.2f}, Warp={avg_sharpness_warp:.2f} | AI改进: {sharpness_improve:+.1f}%")
    logger.info(f"  边界梯度 (↓更好): Real={avg_boundary_real:.2f}, AI={avg_boundary_ai:.2f}, Warp={avg_boundary_warp:.2f} | AI改进: {boundary_improve:+.1f}%")
    logger.info(f"  GLCM对比度: Real={avg_glcm_contrast_real:.2f}, AI={avg_glcm_contrast_ai:.2f}, Warp={avg_glcm_contrast_warp:.2f}")
    logger.info(f"  GLCM熵: Real={avg_glcm_entropy_real:.2f}, AI={avg_glcm_entropy_ai:.2f}, Warp={avg_glcm_entropy_warp:.2f}")

    # 生成雷达图（包含 Sharpness 和 Boundary）
    try:
        _generate_texture_radar_chart(
            output_dir,
            avg_glcm_contrast_real, avg_glcm_contrast_ai, avg_glcm_contrast_warp,
            avg_glcm_energy_real, avg_glcm_energy_ai, avg_glcm_energy_warp,
            avg_glcm_entropy_real, avg_glcm_entropy_ai, avg_glcm_entropy_warp,
            avg_glcm_correlation_real, avg_glcm_correlation_ai, avg_glcm_correlation_warp,
            avg_glcm_homogeneity_real, avg_glcm_homogeneity_ai, avg_glcm_homogeneity_warp,
            logger,
            filename='texture_quality_radar.png',
            sharpness_real=avg_sharpness_real,
            sharpness_ai=avg_sharpness_ai,
            sharpness_warp=avg_sharpness_warp,
            boundary_real=avg_boundary_real,
            boundary_ai=avg_boundary_ai,
            boundary_warp=avg_boundary_warp
        )
    except Exception as e:
        logger.warning(f"  ⚠ 雷达图生成失败: {e}")

    # 保存报告
    report_path = output_dir / 'evaluation_report.md'
    sharpness_winner = '✓ AI wins' if avg_sharpness_ai > avg_sharpness_warp else '✗ Warp wins'
    boundary_winner = '✓ AI wins' if avg_boundary_ai < avg_boundary_warp else '✗ Warp wins'

    # GLCM 特征比较：计算与 Real COPD 的距离，距离越小越好
    contrast_dist_ai = abs(avg_glcm_contrast_ai - avg_glcm_contrast_real)
    contrast_dist_warp = abs(avg_glcm_contrast_warp - avg_glcm_contrast_real)
    contrast_winner = '✓ AI closer' if contrast_dist_ai < contrast_dist_warp else '✗ Warp closer'

    energy_dist_ai = abs(avg_glcm_energy_ai - avg_glcm_energy_real)
    energy_dist_warp = abs(avg_glcm_energy_warp - avg_glcm_energy_real)
    energy_winner = '✓ AI closer' if energy_dist_ai < energy_dist_warp else '✗ Warp closer'

    entropy_dist_ai = abs(avg_glcm_entropy_ai - avg_glcm_entropy_real)
    entropy_dist_warp = abs(avg_glcm_entropy_warp - avg_glcm_entropy_real)
    entropy_winner = '✓ AI closer' if entropy_dist_ai < entropy_dist_warp else '✗ Warp closer'

    correlation_dist_ai = abs(avg_glcm_correlation_ai - avg_glcm_correlation_real)
    correlation_dist_warp = abs(avg_glcm_correlation_warp - avg_glcm_correlation_real)
    correlation_winner = '✓ AI closer' if correlation_dist_ai < correlation_dist_warp else '✗ Warp closer'

    homogeneity_dist_ai = abs(avg_glcm_homogeneity_ai - avg_glcm_homogeneity_real)
    homogeneity_dist_warp = abs(avg_glcm_homogeneity_warp - avg_glcm_homogeneity_real)
    homogeneity_winner = '✓ AI closer' if homogeneity_dist_ai < homogeneity_dist_warp else '✗ Warp closer'

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

| 指标 | Real COPD | AI Fused | Direct Warp | AI vs Warp | 改进幅度 |
|------|-----------|----------|-------------|------------|----------|
| 清晰度 (↑ 越高越好) | {avg_sharpness_real:.2f} | {avg_sharpness_ai:.2f} | {avg_sharpness_warp:.2f} | {sharpness_winner} | {sharpness_improve:+.1f}% |
| 边界梯度 (↓ 越低越好) | {avg_boundary_real:.2f} | {avg_boundary_ai:.2f} | {avg_boundary_warp:.2f} | {boundary_winner} | {boundary_improve:+.1f}% |
| GLCM 对比度 (↑ 越高越好) | {avg_glcm_contrast_real:.2f} | {avg_glcm_contrast_ai:.2f} | {avg_glcm_contrast_warp:.2f} | {contrast_winner} | Δ: AI={contrast_dist_ai:.2f}, Warp={contrast_dist_warp:.2f} |
| GLCM 能量 (≈ Real 越近越好) | {avg_glcm_energy_real:.4f} | {avg_glcm_energy_ai:.4f} | {avg_glcm_energy_warp:.4f} | {energy_winner} | Δ: AI={energy_dist_ai:.4f}, Warp={energy_dist_warp:.4f} |
| GLCM 熵 (↑ 越高越好) | {avg_glcm_entropy_real:.2f} | {avg_glcm_entropy_ai:.2f} | {avg_glcm_entropy_warp:.2f} | {entropy_winner} | Δ: AI={entropy_dist_ai:.2f}, Warp={entropy_dist_warp:.2f} |
| GLCM 相关性 (≈ Real 越近越好) | {avg_glcm_correlation_real:.4f} | {avg_glcm_correlation_ai:.4f} | {avg_glcm_correlation_warp:.4f} | {correlation_winner} | Δ: AI={correlation_dist_ai:.4f}, Warp={correlation_dist_warp:.4f} |
| GLCM 同质性 (≈ Real 越近越好) | {avg_glcm_homogeneity_real:.4f} | {avg_glcm_homogeneity_ai:.4f} | {avg_glcm_homogeneity_warp:.4f} | {homogeneity_winner} | Δ: AI={homogeneity_dist_ai:.4f}, Warp={homogeneity_dist_warp:.4f} |

### 指标解释：
- **清晰度 (↑ 越高越好)**: 拉普拉斯方差越高表示纹理越清晰、细节越丰富。模糊会降低清晰度。
- **边界梯度 (↓ 越低越好)**: 值越低表示病灶边界融合越平滑，无可见接缝。
- **GLCM 对比度 (↑ 越高越好)**: 测量局部强度变化。对比度越高表示纹理细节越丰富；模糊会降低对比度。
- **GLCM 能量 (≈ Real 越近越好)**: 测量纹理均匀性。越接近 Real COPD 表示纹理分布越真实。
- **GLCM 熵 (↑ 越高越好)**: 测量纹理随机性/复杂度。熵越高表示纹理越丰富；模糊会降低熵。
- **GLCM 相关性 (≈ Real 越近越好)**: 灰度级线性依赖性。越接近 Real COPD 表示空间模式越真实。
- **GLCM 同质性 (≈ Real 越近越好)**: 分布接近对角线的程度。越接近 Real COPD 表示局部平滑度越真实。

**图例**: ↑ 越高越好 = 值越大越好, ↓ 越低越好 = 值越小越好, ≈ Real 越近越好 = 越接近 Real COPD 越好

> **注意**: Direct Warp 是以标准模板为基线（不经过 AI 纹理合成），反映了模板在病灶区域的原始纹理质量。AI Fused 的优势体现在模板基础上经 AI 合成后纹理更接近真实 COPD CT。

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


def plot_comparison(img_a, title_a, img_b, title_b, mask, roi_coords, save_path, suptitle, view='axial', layout='vertical'):
    """
    绘制对比图（支持水平/垂直两种布局）
    Plot comparison images with support for horizontal/vertical layouts

    Args:
        img_a: 图像 A (2D 切片) / Image A (2D slice)
        title_a: 图像 A 标题 / Title for image A
        img_b: 图像 B (2D 切片) / Image B (2D slice)
        title_b: 图像 B 标题 / Title for image B
        mask: mask 切片 / Mask slice
        roi_coords: ROI 坐标 (y1, y2, x1, x2) / ROI coordinates
        save_path: 保存路径 / Save path
        suptitle: 总标题 / Main title
        view: 视图类型 ('axial', 'coronal', 'sagittal') / View type
        layout: 布局方式 / Layout mode
            - 'horizontal': 左右布局，1行2列 (default) / Side-by-side, 1 row × 2 cols
            - 'vertical': 上下布局，2行1列 / Stacked, 2 rows × 1 col
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    y1, y2, x1, x2 = roi_coords
    roi_a = img_a[y1:y2, x1:x2]
    roi_b = img_b[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]

    # 计算差异 / Calculate difference
    diff = np.abs(img_a - img_b)
    roi_diff = diff[y1:y2, x1:x2]

    view_label = view.capitalize()

    # 根据布局方式设置图形尺寸和网格
    # Set figure size and grid based on layout mode
    if layout == 'horizontal':
        # 水平布局（默认）：4行2列，全局视图在左列，ROI在右列
        # Horizontal layout (default): 4 rows × 2 cols
        # Left column: Global views (Image A, Image B, Diff, Mask)
        # Right column: ROI zoomed views
        fig = plt.figure(figsize=(10, 20), dpi=150)
        gs = fig.add_gridspec(4, 2, wspace=0.15, hspace=0.15)
        pos_global = [(0, 0), (1, 0), (2, 0), (3, 0)]  # 左列：全局视图
        pos_roi = [(0, 1), (1, 1), (2, 1), (3, 1)]      # 右列：ROI 放大
    else:
        # 垂直布局：2行4列，全局视图在上行，ROI在下行
        # Vertical layout: 2 rows × 4 cols
        # Top row: Global views (Image A, Image B, Diff, Mask)
        # Bottom row: ROI zoomed views
        fig = plt.figure(figsize=(20, 10), dpi=150)
        gs = fig.add_gridspec(2, 4, wspace=0.15, hspace=0.2)
        pos_global = [(0, 0), (0, 1), (0, 2), (0, 3)]  # 上行：全局视图
        pos_roi = [(1, 0), (1, 1), (1, 2), (1, 3)]      # 下行：ROI 放大

    # ========== Row 1: Global View ==========
    # Col 1: Image A
    ax1 = fig.add_subplot(gs[pos_global[0]])
    ax1.imshow(apply_lung_window(img_a).T, cmap='gray', origin='lower')
    ax1.set_title(title_a, fontsize=11, fontweight='bold')
    rect = mpatches.Rectangle((y1, x1), y2-y1, x2-x1, linewidth=2, edgecolor='yellow', facecolor='none')
    ax1.add_patch(rect)
    ax1.axis('off')

    # Col 2: Image B
    ax2 = fig.add_subplot(gs[pos_global[1]])
    ax2.imshow(apply_lung_window(img_b).T, cmap='gray', origin='lower')
    ax2.set_title(title_b, fontsize=11, fontweight='bold')
    ax2.axis('off')

    # Col 3: Difference Heatmap
    ax3 = fig.add_subplot(gs[pos_global[2]])
    bg = apply_lung_window(img_a).T
    ax3.imshow(bg, cmap='gray', alpha=0.5, origin='lower')
    im = ax3.imshow(diff.T, cmap='jet', alpha=0.6, origin='lower', vmin=0, vmax=500)
    ax3.set_title("Difference Map", fontsize=11)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    ax3.axis('off')

    # Col 4: Mask Overlay
    ax4 = fig.add_subplot(gs[pos_global[3]])
    ax4.imshow(apply_lung_window(img_b).T, cmap='gray', origin='lower')
    ax4.imshow(mask.T, cmap='Reds', alpha=0.4, origin='lower')
    ax4.set_title("Lesion Mask Overlay", fontsize=11)
    ax4.axis('off')

    # ========== Row 2: ROI Zoom ==========
    # Col 1: ROI Image A
    ax5 = fig.add_subplot(gs[pos_roi[0]])
    ax5.imshow(apply_lung_window(roi_a).T, cmap='gray', origin='lower')
    ax5.set_title(f"ROI: {title_a}", fontsize=10)
    for spine in ax5.spines.values():
        spine.set_edgecolor('yellow')
        spine.set_linewidth(2)
    ax5.axis('off')

    # Col 2: ROI Image B
    ax6 = fig.add_subplot(gs[pos_roi[1]])
    ax6.imshow(apply_lung_window(roi_b).T, cmap='gray', origin='lower')
    ax6.set_title(f"ROI: {title_b}", fontsize=10)
    ax6.axis('off')

    # Col 3: ROI Difference
    ax7 = fig.add_subplot(gs[pos_roi[2]])
    roi_bg = apply_lung_window(roi_a).T
    ax7.imshow(roi_bg, cmap='gray', alpha=0.5, origin='lower')
    ax7.imshow(roi_diff.T, cmap='jet', alpha=0.6, origin='lower', vmin=0, vmax=500)
    ax7.set_title("ROI: Difference", fontsize=10)
    ax7.axis('off')

    # Col 4: ROI Mask Overlay
    ax8 = fig.add_subplot(gs[pos_roi[3]])
    ax8.imshow(apply_lung_window(roi_b).T, cmap='gray', origin='lower')
    ax8.imshow(roi_mask.T, cmap='Reds', alpha=0.4, origin='lower')
    ax8.set_title("ROI: Mask Overlay", fontsize=10)
    ax8.axis('off')

    plt.suptitle(f"{suptitle} [{view_label}]", fontsize=14, y=0.98)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


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

    # 设置专业风格（兼容不同版本的 matplotlib）
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            pass  # 使用默认样式

    # 创建图形：左侧直方图 + 右侧文本框
    fig = plt.figure(figsize=(14, 6), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])

    # 颜色方案（专业配色）
    colors = {
        'Template': '#3498db',      # 蓝色
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
    ax_hist.axvline(x=-950, color='#2c3e50', linestyle='--', linewidth=2, alpha=0.8, label='Emphysema threshold (-950)')

    # 设置直方图坐标轴
    ax_hist.set_xlim(-1024, 0)
    ax_hist.set_xlabel("HU Value", fontsize=12, fontweight='bold')
    ax_hist.set_ylabel("Density", fontsize=12, fontweight='bold')
    ax_hist.set_title("HU Distribution in 3D Lesion Volume", fontsize=13, fontweight='bold')
    ax_hist.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_hist.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_hist.set_axisbelow(True)

    # 构建右侧统计信息文本框（参考上传图片格式）
    text_lines = ["HU Value Statistics (3D Lesion Volume)", "=" * 42, ""]

    # Real COPD CT 统计
    if 'Real COPD' in stats_data:
        s = stats_data['Real COPD']
        text_lines.extend([
            "Real COPD CT:",
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
    if 'Template' in stats_data:
        s = stats_data['Template']
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
            "Difference:",
            f"  Mean HU Diff: {mean_diff:.1f}",
            f"  Emphysema Diff: {emph_diff:.1f}%"
        ])

    # 在右侧子图显示文本（无背景填充，纯文字）
    ax_text.axis('off')
    text_str = '\n'.join(text_lines)
    ax_text.text(0.05, 0.95, text_str, transform=ax_text.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace')

    # 设置总标题
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    # 保存
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    plt.style.use('default')  # 恢复默认样式


def run_result_visualization(
    config: dict,
    logger: logging.Logger,
    model_type: str = 'partial_conv',
    num_patients: int = 5
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

    Returns:
        bool: 是否成功
    """
    logger.info("")
    logger.info("[可视化] 生成结果可视化（三图输出）")
    logger.info("-" * 40)

    try:
        import nibabel as nib
        import numpy as np  # noqa: F401 - 用于检查依赖
    except ImportError as e:
        logger.error(f"  ✗ 缺少依赖: {e}")
        return False

    paths = config.get('paths', {})
    mapped_dir = Path(paths.get('mapped', 'data/03_mapped'))
    atlas_dir = Path(paths.get('atlas', 'data/02_atlas'))
    fused_dir = Path(paths.get('final_viz', 'data/04_final_viz')) / model_type
    output_dir = Path(f'results/{model_type}')  # 合并后的输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    template_path = atlas_dir / 'standard_template.nii.gz'

    # 检查文件
    if not fused_dir.exists():
        logger.error(f"  ✗ 融合结果目录不存在: {fused_dir}")
        logger.error(f"  请先运行推理: --inference --model-type {model_type}")
        return False

    if not template_path.exists():
        logger.error(f"  ✗ 模板不存在: {template_path}")
        return False

    fused_files = sorted(fused_dir.glob("*_fused.nii.gz"))[:num_patients]
    if not fused_files:
        logger.error(f"  ✗ 未找到融合结果文件")
        return False

    logger.info(f"  融合结果目录: {fused_dir}")
    logger.info(f"  可视化患者数: {len(fused_files)}")
    logger.info(f"  输出目录: {output_dir}")

    # 加载模板
    template_data = nib.load(str(template_path)).get_fdata()

    model_names = {
        'unet': '3D U-Net',
        'partial_conv': 'Partial Conv',
        'patchgan': 'PatchGAN'
    }
    model_name = model_names.get(model_type, model_type)

    success_count = 0
    for fused_path in fused_files:
        patient_id = fused_path.name.replace('_fused.nii.gz', '')
        mask_path = mapped_dir / patient_id / f"{patient_id}_warped_lesion.nii.gz"
        real_path = mapped_dir / patient_id / f"{patient_id}_warped.nii.gz"

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

            files_generated = 0

            # 2. 为每个视图生成可视化
            for view in ['axial', 'coronal', 'sagittal']:
                # 获取该视图的最佳切片和 ROI
                slice_idx, roi_coords = get_roi_slice(mask_data, context_size=48, view=view)

                # 提取 2D 切片
                if view == 'axial':
                    template_slice = template_data[:, :, slice_idx]
                    fused_slice = fused_data[:, :, slice_idx]
                    mask_slice = mask_data[:, :, slice_idx]
                    real_slice = real_data[:, :, slice_idx] if real_data is not None else None
                elif view == 'coronal':
                    template_slice = template_data[:, slice_idx, :]
                    fused_slice = fused_data[:, slice_idx, :]
                    mask_slice = mask_data[:, slice_idx, :]
                    real_slice = real_data[:, slice_idx, :] if real_data is not None else None
                else:  # sagittal
                    template_slice = template_data[slice_idx, :, :]
                    fused_slice = fused_data[slice_idx, :, :]
                    mask_slice = mask_data[slice_idx, :, :]
                    real_slice = real_data[slice_idx, :, :] if real_data is not None else None

                # 生成图像 1：Generation（Template vs Fused）
                plot_comparison(
                    template_slice, "Healthy Template",
                    fused_slice, f"AI Fused ({model_name})",
                    mask_slice, roi_coords,
                    patient_output_dir / f"{patient_id}_viz_1_generation_{view}.png",
                    f"{patient_id} - Generation Effectiveness",
                    view=view
                )
                files_generated += 1

                # 生成图像 2：Realism（Real vs Fused）- 仅当真实数据存在时
                if real_slice is not None:
                    plot_comparison(
                        real_slice, "Real COPD",
                        fused_slice, f"AI Fused ({model_name})",
                        mask_slice, roi_coords,
                        patient_output_dir / f"{patient_id}_viz_2_realism_{view}.png",
                        f"{patient_id} - Fidelity Analysis",
                        view=view
                    )
                    files_generated += 1

            # 3. 生成 Histogram（使用完整 3D 体积的病灶区域，与评估报告保持一致）
            # Use full 3D lesion volume for histogram, consistent with evaluation metrics
            mask_bool = mask_data > 0  # 完整 3D 病灶 mask

            hist_data = {
                'Template': template_data[mask_bool],
                'AI Fused': fused_data[mask_bool],
            }
            if real_data is not None:
                hist_data['Real COPD'] = real_data[mask_bool]

            plot_histogram(
                hist_data,
                patient_output_dir / f"{patient_id}_viz_3_histogram.png",
                f"{patient_id} - HU Distribution in 3D Lesion Volume"
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
        choices=['unet', 'partial_conv', 'patchgan'],
        help='模型类型: unet(基线), partial_conv(进阶), patchgan(高级)'
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
            limit=args.limit  # 传递 limit 参数
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
            num_patients=args.limit or 10
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
            num_patients=args.limit or 5
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

