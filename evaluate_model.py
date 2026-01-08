#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 3B 模型质量评估脚本

功能：
1. 训练日志分析（loss 曲线、收敛性）
2. 定量评估（PSNR、SSIM、MAE、MSE）
3. 定性评估（可视化对比图）
4. 推理测试

使用方法：
    python evaluate_model.py [--checkpoint checkpoints/best.pth]
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Phase 3B 模型质量评估')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth',
                        help='模型检查点路径')
    parser.add_argument('--log-file', type=str, default='checkpoints/training_log.json',
                        help='训练日志文件')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='评估结果输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--num-patients', type=int, default=5,
                        help='生成可视化的患者数量')
    return parser.parse_args()


# =============================================================================
# 1. 训练日志分析
# =============================================================================

def analyze_training_log(log_file: str, output_dir: Path) -> dict:
    """分析训练日志"""
    print("\n" + "=" * 60)
    print("  1. 训练日志分析")
    print("=" * 60)
    
    with open(log_file, 'r') as f:
        history = json.load(f)
    
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    
    if not train_loss:
        print("  ⚠ 训练日志为空")
        return {}
    
    epochs = len(train_loss)
    
    # 基本统计
    analysis = {
        'epochs': epochs,
        'final_train_loss': train_loss[-1],
        'final_val_loss': val_loss[-1] if val_loss else None,
        'best_train_loss': min(train_loss),
        'best_val_loss': min(val_loss) if val_loss else None,
        'best_epoch': val_loss.index(min(val_loss)) + 1 if val_loss else train_loss.index(min(train_loss)) + 1,
    }
    
    # 收敛性分析
    if epochs >= 10:
        early_loss = np.mean(train_loss[:5])
        late_loss = np.mean(train_loss[-5:])
        analysis['convergence_ratio'] = late_loss / early_loss
        analysis['converged'] = analysis['convergence_ratio'] < 0.5
    
    # 过拟合分析
    if val_loss:
        gap = np.array(val_loss) - np.array(train_loss)
        analysis['avg_gap'] = float(np.mean(gap))
        analysis['final_gap'] = val_loss[-1] - train_loss[-1]
        analysis['overfitting'] = analysis['final_gap'] > 0.02  # 阈值
    
    # 打印结果
    print(f"  训练轮数: {epochs}")
    print(f"  最终训练 Loss: {analysis['final_train_loss']:.6f}")
    print(f"  最终验证 Loss: {analysis['final_val_loss']:.6f}")
    print(f"  最佳验证 Loss: {analysis['best_val_loss']:.6f} (Epoch {analysis['best_epoch']})")
    print(f"  收敛比率: {analysis.get('convergence_ratio', 'N/A'):.4f}" if 'convergence_ratio' in analysis else "")
    print(f"  过拟合间隙: {analysis.get('final_gap', 'N/A'):.6f}")
    
    if analysis.get('converged', False):
        print("  ✓ 模型已收敛")
    else:
        print("  ⚠ 模型可能未完全收敛，建议增加训练轮数")
    
    if analysis.get('overfitting', False):
        print("  ⚠ 检测到过拟合迹象，建议增加数据增强或正则化")
    else:
        print("  ✓ 未检测到明显过拟合")
    
    # 绘制 loss 曲线
    plot_loss_curves(train_loss, val_loss, output_dir)

    return analysis


def plot_loss_curves(train_loss: list, val_loss: list, output_dir: Path):
    """绘制训练曲线"""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_loss) + 1)

    # Loss 曲线
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    if val_loss:
        ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss 差异（过拟合指标）
    ax2 = axes[1]
    if val_loss:
        gap = np.array(val_loss) - np.array(train_loss)
        ax2.plot(epochs, gap, 'g-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.fill_between(epochs, 0, gap, where=(gap > 0), alpha=0.3, color='red', label='Overfitting')
        ax2.fill_between(epochs, 0, gap, where=(gap <= 0), alpha=0.3, color='green', label='Underfitting')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Val Loss - Train Loss', fontsize=12)
    ax2.set_title('Generalization Gap', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Loss 曲线已保存: {output_dir / 'loss_curves.png'}")


# =============================================================================
# 2. 定量评估指标
# =============================================================================

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """计算 PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算 SSIM（简化版本）"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim)


def evaluate_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray = None
) -> dict:
    """评估重建质量（优化版本，减少内存使用）"""
    # 归一化到 [0, 1]
    orig_min, orig_max = original.min(), original.max()
    recon_min, recon_max = reconstructed.min(), reconstructed.max()

    orig_norm = (original - orig_min) / (orig_max - orig_min + 1e-8)
    recon_norm = (reconstructed - recon_min) / (recon_max - recon_min + 1e-8)

    metrics = {}

    # 全局指标
    metrics['global'] = {
        'psnr': calculate_psnr(orig_norm, recon_norm),
        'ssim': calculate_ssim(orig_norm.flatten(), recon_norm.flatten()),
        'mae': float(np.mean(np.abs(original - reconstructed))),
    }

    # 病灶区域指标（仅当 mask 存在且有效时）
    if mask is not None:
        mask_bool = mask > 0
        mask_sum = mask_bool.sum()

        if mask_sum > 100:  # 确保有足够的体素
            orig_lesion = orig_norm[mask_bool]
            recon_lesion = recon_norm[mask_bool]

            metrics['lesion'] = {
                'psnr': calculate_psnr(orig_lesion, recon_lesion),
                'ssim': calculate_ssim(orig_lesion, recon_lesion),
                'mae': float(np.mean(np.abs(original[mask_bool] - reconstructed[mask_bool]))),
                'voxel_count': int(mask_sum)
            }

    return metrics


def run_quantitative_evaluation(checkpoint_path: str, config: dict, device: str) -> dict:
    """运行定量评估"""
    print("\n" + "=" * 60)
    print("  2. 定量评估")
    print("=" * 60)

    if not HAS_TORCH:
        print("  ⚠ PyTorch 未安装，跳过定量评估")
        return {}

    if not HAS_NIBABEL:
        print("  ⚠ nibabel 未安装，跳过定量评估")
        return {}

    # 自动选择设备
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print(f"  ⚠ CUDA 不可用，使用 CPU")

    # 获取路径
    paths = config.get('paths', {})
    mapped_dir = Path(paths.get('mapped', 'data/03_mapped'))
    final_viz_dir = Path(paths.get('final_viz', 'data/04_final_viz'))
    atlas_dir = Path(paths.get('atlas', 'data/02_atlas'))
    template_path = atlas_dir / 'standard_template.nii.gz'

    # 检查 fused 文件是否存在
    fused_files = list(final_viz_dir.glob("*_fused.nii.gz"))

    if not fused_files:
        print("  ⚠ 未找到融合结果文件 (04_final_viz/*_fused.nii.gz)")
        print("  请先运行推理: python run_phase3_pipeline.py --inference")
        return {}

    print(f"  找到 {len(fused_files)} 个融合结果文件")

    # 加载模板
    if not template_path.exists():
        print(f"  ⚠ 模板不存在: {template_path}")
        return {}

    template_data = nib.load(str(template_path)).get_fdata()
    print(f"  ✓ 模板加载成功: {template_path.name}")

    # 评估每个融合结果
    all_metrics = []

    for fused_path in fused_files[:5]:  # 限制评估数量
        # 从文件名提取 patient_id (copd_001_fused.nii.gz -> copd_001)
        filename = fused_path.name  # copd_001_fused.nii.gz
        patient_id = filename.replace('_fused.nii.gz', '')  # copd_001

        # 查找对应的 mask
        mask_path = mapped_dir / patient_id / f"{patient_id}_warped_lesion.nii.gz"

        if not mask_path.exists():
            print(f"    ⚠ 跳过 {patient_id}: mask 不存在")
            continue

        # 加载数据
        fused_data = nib.load(str(fused_path)).get_fdata()
        mask_data = nib.load(str(mask_path)).get_fdata()

        # 计算指标
        metrics = evaluate_reconstruction(template_data, fused_data, mask_data)
        metrics['patient_id'] = patient_id
        all_metrics.append(metrics)

        # 打印结果
        print(f"    ✓ {patient_id}:")
        print(f"      全局 - PSNR: {metrics['global']['psnr']:.2f} dB, "
              f"SSIM: {metrics['global']['ssim']:.4f}")
        if 'lesion' in metrics:
            print(f"      病灶 - PSNR: {metrics['lesion']['psnr']:.2f} dB, "
                  f"SSIM: {metrics['lesion']['ssim']:.4f}")

    # 计算平均指标
    if all_metrics:
        avg_psnr = np.mean([m['global']['psnr'] for m in all_metrics])
        avg_ssim = np.mean([m['global']['ssim'] for m in all_metrics])
        print(f"\n  平均指标:")
        print(f"    PSNR: {avg_psnr:.2f} dB")
        print(f"    SSIM: {avg_ssim:.4f}")

    return {'samples': all_metrics}


# =============================================================================
# 3. 定性评估（可视化）
# =============================================================================

def create_comparison_visualization(output_dir: Path, config: dict, num_patients: int = 5):
    """创建对比可视化（改进版）"""
    print("\n" + "=" * 60)
    print("  3. 定性评估（可视化）")
    print("=" * 60)

    if not HAS_NIBABEL:
        print("  ⚠ nibabel 未安装，跳过可视化")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取路径
    paths = config.get('paths', {})
    mapped_dir = Path(paths.get('mapped', 'data/03_mapped'))
    final_viz_dir = Path(paths.get('final_viz', 'data/04_final_viz'))
    atlas_dir = Path(paths.get('atlas', 'data/02_atlas'))
    template_path = atlas_dir / 'standard_template.nii.gz'

    if not template_path.exists():
        print(f"  ⚠ 模板不存在: {template_path}")
        return

    # 加载模板
    template_data = nib.load(str(template_path)).get_fdata()

    # 获取融合结果文件
    fused_files = sorted(final_viz_dir.glob("*_fused.nii.gz"))

    if not fused_files:
        print("  ⚠ 未找到融合结果文件")
        print("  请先运行推理: python run_phase3_pipeline.py --inference")
        return

    # 限制处理数量
    fused_files = fused_files[:num_patients]
    print(f"  生成 {len(fused_files)} 个患者的可视化对比图")

    # 为每个患者生成可视化
    for idx, fused_path in enumerate(fused_files):
        patient_id = fused_path.name.replace('_fused.nii.gz', '')

        # 查找对应的 mask
        mask_path = mapped_dir / patient_id / f"{patient_id}_warped_lesion.nii.gz"

        if not mask_path.exists():
            print(f"    ⚠ 跳过 {patient_id}: mask 不存在")
            continue

        # 加载数据
        fused_data = nib.load(str(fused_path)).get_fdata()
        mask_data = nib.load(str(mask_path)).get_fdata()

        # 创建对比图
        _create_patient_comparison(
            template_data, fused_data, mask_data,
            patient_id, output_dir
        )

        print(f"    ✓ {patient_id}: {output_dir / f'{patient_id}_comparison.png'}")


def _create_patient_comparison(
    template: np.ndarray,
    fused: np.ndarray,
    mask: np.ndarray,
    patient_id: str,
    output_dir: Path
):
    """为单个患者创建对比图"""

    # 创建 4 行 3 列的子图
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))

    # 获取中心切片
    center = [s // 2 for s in template.shape]

    # 找到病灶中心（如果存在）
    if mask.sum() > 0:
        lesion_coords = np.where(mask > 0)
        center = [int(np.mean(coords)) for coords in lesion_coords]

    # 定义切片索引
    slices = {
        'axial': (slice(None), slice(None), center[2]),
        'coronal': (slice(None), center[1], slice(None)),
        'sagittal': (center[0], slice(None), slice(None))
    }

    view_names = ['Axial', 'Coronal', 'Sagittal']
    slice_keys = ['axial', 'coronal', 'sagittal']

    # 第一行：健康模板
    for col, (view_name, slice_key) in enumerate(zip(view_names, slice_keys)):
        ax = axes[0, col]
        img_slice = template[slices[slice_key]].T
        ax.imshow(img_slice, cmap='gray', origin='lower', vmin=-1000, vmax=400)
        ax.set_title(f'Template - {view_name}', fontsize=12, fontweight='bold')
        ax.axis('off')

    # 第二行：AI 融合结果
    for col, (view_name, slice_key) in enumerate(zip(view_names, slice_keys)):
        ax = axes[1, col]
        img_slice = fused[slices[slice_key]].T
        ax.imshow(img_slice, cmap='gray', origin='lower', vmin=-1000, vmax=400)
        ax.set_title(f'AI Fused - {view_name}', fontsize=12, fontweight='bold')
        ax.axis('off')

    # 第三行：差异图（Fused - Template）
    for col, (view_name, slice_key) in enumerate(zip(view_names, slice_keys)):
        ax = axes[2, col]
        template_slice = template[slices[slice_key]].T
        fused_slice = fused[slices[slice_key]].T
        diff = fused_slice - template_slice

        im = ax.imshow(diff, cmap='RdBu_r', origin='lower', vmin=-200, vmax=200)
        ax.set_title(f'Difference - {view_name}', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 第四行：病灶区域叠加
    for col, (view_name, slice_key) in enumerate(zip(view_names, slice_keys)):
        ax = axes[3, col]
        fused_slice = fused[slices[slice_key]].T
        mask_slice = mask[slices[slice_key]].T

        ax.imshow(fused_slice, cmap='gray', origin='lower', vmin=-1000, vmax=400)
        ax.imshow(mask_slice, cmap='Reds', alpha=0.5, origin='lower')
        ax.set_title(f'Fused + Lesion Mask - {view_name}', fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle(f'Patient: {patient_id}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = output_dir / f'{patient_id}_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# 4. 推理测试
# =============================================================================

def run_inference_test(checkpoint_path: str, config: dict, device: str, output_dir: Path) -> bool:
    """运行推理测试"""
    print("\n" + "=" * 60)
    print("  4. 推理测试")
    print("=" * 60)

    if not Path(checkpoint_path).exists():
        print(f"  ⚠ 检查点不存在: {checkpoint_path}")
        return False

    # 检查模型文件
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"  ✓ 检查点加载成功")
    print(f"    - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"    - Best Loss: {checkpoint.get('best_loss', 'N/A'):.6f}" if checkpoint.get('best_loss') else "")

    # 检查模型结构
    state_dict = checkpoint.get('generator_state_dict', {})
    print(f"    - 参数数量: {len(state_dict)} 层")

    # 计算参数总量
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"    - 总参数量: {total_params:,}")

    return True


# =============================================================================
# 5. 生成评估报告
# =============================================================================

def generate_report(analysis: dict, output_dir: Path):
    """生成评估报告"""
    print("\n" + "=" * 60)
    print("  5. 评估报告")
    print("=" * 60)

    report = []
    report.append("# Phase 3B 模型质量评估报告")
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## 1. 训练概况")
    report.append(f"- 训练轮数: {analysis.get('epochs', 'N/A')}")
    report.append(f"- 最终训练 Loss: {analysis.get('final_train_loss', 'N/A'):.6f}")
    report.append(f"- 最终验证 Loss: {analysis.get('final_val_loss', 'N/A'):.6f}")
    report.append(f"- 最佳验证 Loss: {analysis.get('best_val_loss', 'N/A'):.6f}")

    report.append("\n## 2. 收敛性分析")
    if analysis.get('converged'):
        report.append("- ✅ 模型已收敛")
    else:
        report.append("- ⚠️ 模型可能未完全收敛")

    report.append("\n## 3. 过拟合分析")
    if analysis.get('overfitting'):
        report.append("- ⚠️ 检测到过拟合迹象")
    else:
        report.append("- ✅ 未检测到明显过拟合")

    report.append("\n## 4. 指标参考范围")
    report.append("| 指标 | 良好 | 一般 | 较差 |")
    report.append("|------|------|------|------|")
    report.append("| PSNR | >30 dB | 25-30 dB | <25 dB |")
    report.append("| SSIM | >0.9 | 0.8-0.9 | <0.8 |")
    report.append("| MAE | <0.02 | 0.02-0.05 | >0.05 |")

    report.append("\n## 5. 改进建议")
    if not analysis.get('converged'):
        report.append("- 增加训练轮数（建议 100-200 epochs）")
    if analysis.get('overfitting'):
        report.append("- 增加数据增强强度")
        report.append("- 添加 Dropout 或权重衰减")
        report.append("- 收集更多训练数据")

    # 保存报告
    report_path = output_dir / 'evaluation_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"  ✓ 报告已保存: {report_path}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Phase 3B 模型质量评估")
    print("=" * 60)
    print(f"检查点: {args.checkpoint}")
    print(f"输出目录: {output_dir}")
    print(f"可视化患者数: {args.num_patients}")

    # 加载配置
    import yaml
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 1. 训练日志分析
    analysis = analyze_training_log(args.log_file, output_dir)

    # 2. 定量评估（如果有 PyTorch）
    if HAS_TORCH and Path(args.checkpoint).exists():
        run_quantitative_evaluation(args.checkpoint, config, args.device)

    # 3. 定性评估
    create_comparison_visualization(output_dir, config, args.num_patients)

    # 4. 推理测试
    if HAS_TORCH:
        run_inference_test(args.checkpoint, config, args.device, output_dir)

    # 5. 生成报告
    generate_report(analysis, output_dir)

    print("\n" + "=" * 60)
    print("  评估完成!")
    print("=" * 60)
    print(f"结果保存在: {output_dir}")


if __name__ == '__main__':
    main()

