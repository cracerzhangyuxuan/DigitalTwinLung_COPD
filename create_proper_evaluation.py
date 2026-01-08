#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
正确的模型评估脚本

功能：
将 AI 融合结果与真实 COPD CT 进行对比（而非与健康模板对比）

评估内容：
1. 在病灶区域内计算 PSNR/SSIM（真实 COPD vs AI 融合）
2. HU 值分布分析（是否符合肺气肿标准 HU < -950）
3. 生成对比可视化（真实 COPD vs AI 融合）

使用方法：
    python create_proper_evaluation.py [--num-patients 5]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='正确的模型评估（与真实 COPD CT 对比）')
    parser.add_argument('--num-patients', type=int, default=5,
                        help='评估的患者数量')
    parser.add_argument('--output-dir', type=str, 
                        default='evaluation_results/proper_evaluation',
                        help='输出目录')
    return parser.parse_args()


# =============================================================================
# 辅助函数：PSNR/SSIM 计算
# =============================================================================

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """计算 PSNR (Peak Signal-to-Noise Ratio)"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算 SSIM (Structural Similarity Index)"""
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


def normalize_hu(data: np.ndarray, hu_min: float = -1000, hu_max: float = 400) -> np.ndarray:
    """将 HU 值归一化到 [0, 1]"""
    data_clipped = np.clip(data, hu_min, hu_max)
    return (data_clipped - hu_min) / (hu_max - hu_min)


# =============================================================================
# 数据加载函数
# =============================================================================

def find_patient_data(patient_id: str) -> dict:
    """
    查找患者的所有相关数据文件
    
    返回包含以下路径的字典：
    - real_copd_warped: 配准后的真实 COPD CT (03_mapped)
    - fused: AI 融合结果 (04_final_viz)
    - mask: 病灶 mask (03_mapped)
    """
    paths = {}
    
    # 配准后的真实 COPD CT
    warped_path = Path(f'data/03_mapped/{patient_id}/{patient_id}_warped.nii.gz')
    if warped_path.exists():
        paths['real_copd_warped'] = warped_path
    
    # AI 融合结果
    fused_path = Path(f'data/04_final_viz/{patient_id}_fused.nii.gz')
    if fused_path.exists():
        paths['fused'] = fused_path
    
    # 病灶 mask
    mask_path = Path(f'data/03_mapped/{patient_id}/{patient_id}_warped_lesion.nii.gz')
    if mask_path.exists():
        paths['mask'] = mask_path
    
    return paths


def load_nifti(path: Path) -> np.ndarray:
    """加载 NIfTI 文件"""
    return nib.load(str(path)).get_fdata()


# =============================================================================
# HU 值分布分析
# =============================================================================

def analyze_hu_distribution(real_data: np.ndarray, fused_data: np.ndarray, 
                            mask: np.ndarray) -> dict:
    """
    分析病灶区域的 HU 值分布
    
    Args:
        real_data: 真实 COPD CT 数据
        fused_data: AI 融合结果数据
        mask: 病灶 mask
    
    Returns:
        包含 HU 值统计信息的字典
    """
    mask_bool = mask > 0
    
    if mask_bool.sum() < 100:
        return {'error': '病灶区域太小'}
    
    # 提取病灶区域的 HU 值
    real_lesion_hu = real_data[mask_bool]
    fused_lesion_hu = fused_data[mask_bool]
    
    # 肺气肿标准：HU < -950
    emphysema_threshold = -950
    
    analysis = {
        'real': {
            'mean': float(np.mean(real_lesion_hu)),
            'std': float(np.std(real_lesion_hu)),
            'min': float(np.min(real_lesion_hu)),
            'max': float(np.max(real_lesion_hu)),
            'emphysema_ratio': float((real_lesion_hu < emphysema_threshold).sum() / len(real_lesion_hu)),
        },
        'fused': {
            'mean': float(np.mean(fused_lesion_hu)),
            'std': float(np.std(fused_lesion_hu)),
            'min': float(np.min(fused_lesion_hu)),
            'max': float(np.max(fused_lesion_hu)),
            'emphysema_ratio': float((fused_lesion_hu < emphysema_threshold).sum() / len(fused_lesion_hu)),
        },
        'voxel_count': int(mask_bool.sum()),
    }
    
    # 计算差异
    analysis['diff'] = {
        'mean_diff': analysis['fused']['mean'] - analysis['real']['mean'],
        'emphysema_ratio_diff': analysis['fused']['emphysema_ratio'] - analysis['real']['emphysema_ratio'],
    }
    
    return analysis


# =============================================================================
# 定量评估函数
# =============================================================================

def evaluate_patient(patient_id: str) -> dict:
    """
    评估单个患者的 AI 融合结果

    将 AI 融合结果与配准后的真实 COPD CT 进行对比
    """
    print(f"  评估 {patient_id}...")

    # 查找数据文件
    paths = find_patient_data(patient_id)

    required_keys = ['real_copd_warped', 'fused', 'mask']
    missing = [k for k in required_keys if k not in paths]

    if missing:
        print(f"    ⚠ 缺少文件: {missing}")
        return {'patient_id': patient_id, 'error': f'缺少文件: {missing}'}

    # 加载数据
    real_copd = load_nifti(paths['real_copd_warped'])
    fused = load_nifti(paths['fused'])
    mask = load_nifti(paths['mask'])

    # 检查尺寸是否匹配
    if real_copd.shape != fused.shape:
        print(f"    ⚠ 尺寸不匹配: real={real_copd.shape}, fused={fused.shape}")
        return {'patient_id': patient_id, 'error': '尺寸不匹配'}

    mask_bool = mask > 0
    voxel_count = mask_bool.sum()

    if voxel_count < 100:
        print(f"    ⚠ 病灶区域太小: {voxel_count} 体素")
        return {'patient_id': patient_id, 'error': '病灶区域太小'}

    # 归一化用于 PSNR/SSIM 计算
    real_norm = normalize_hu(real_copd)
    fused_norm = normalize_hu(fused)

    # 计算病灶区域的指标
    real_lesion = real_norm[mask_bool]
    fused_lesion = fused_norm[mask_bool]

    metrics = {
        'patient_id': patient_id,
        'lesion': {
            'psnr': calculate_psnr(real_lesion, fused_lesion),
            'ssim': calculate_ssim(real_lesion, fused_lesion),
            'mae': float(np.mean(np.abs(real_copd[mask_bool] - fused[mask_bool]))),
        },
        'voxel_count': int(voxel_count),
    }

    # HU 值分布分析
    metrics['hu_analysis'] = analyze_hu_distribution(real_copd, fused, mask)

    # 打印结果
    print(f"    ✓ 病灶区域 PSNR: {metrics['lesion']['psnr']:.2f} dB")
    print(f"    ✓ 病灶区域 SSIM: {metrics['lesion']['ssim']:.4f}")
    print(f"    ✓ 真实 COPD 肺气肿比例: {metrics['hu_analysis']['real']['emphysema_ratio']:.2%}")
    print(f"    ✓ AI 融合 肺气肿比例: {metrics['hu_analysis']['fused']['emphysema_ratio']:.2%}")

    return metrics


# =============================================================================
# 可视化函数
# =============================================================================

def create_comparison_figure(patient_id: str, output_dir: Path):
    """
    创建真实 COPD vs AI 融合的对比图
    """
    if not HAS_MATPLOTLIB:
        print(f"    ⚠ matplotlib 未安装，跳过可视化")
        return

    paths = find_patient_data(patient_id)
    if not all(k in paths for k in ['real_copd_warped', 'fused', 'mask']):
        return

    # 加载数据
    real_copd = load_nifti(paths['real_copd_warped'])
    fused = load_nifti(paths['fused'])
    mask = load_nifti(paths['mask'])

    # 找到病灶中心
    if mask.sum() > 0:
        lesion_coords = np.where(mask > 0)
        center = [int(np.mean(coords)) for coords in lesion_coords]
    else:
        center = [s // 2 for s in real_copd.shape]

    # 创建 3x3 子图
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    slices = {
        'axial': (slice(None), slice(None), center[2]),
        'coronal': (slice(None), center[1], slice(None)),
        'sagittal': (center[0], slice(None), slice(None))
    }
    view_names = ['Axial', 'Coronal', 'Sagittal']
    slice_keys = ['axial', 'coronal', 'sagittal']

    # 第一行：真实 COPD CT
    for col, (view_name, slice_key) in enumerate(zip(view_names, slice_keys)):
        ax = axes[0, col]
        img_slice = real_copd[slices[slice_key]].T
        ax.imshow(img_slice, cmap='gray', origin='lower', vmin=-1000, vmax=400)
        ax.set_title(f'Real COPD - {view_name}', fontsize=12, fontweight='bold')
        ax.axis('off')

    # 第二行：AI 融合结果
    for col, (view_name, slice_key) in enumerate(zip(view_names, slice_keys)):
        ax = axes[1, col]
        img_slice = fused[slices[slice_key]].T
        ax.imshow(img_slice, cmap='gray', origin='lower', vmin=-1000, vmax=400)
        ax.set_title(f'AI Fused - {view_name}', fontsize=12, fontweight='bold')
        ax.axis('off')

    # 第三行：差异图
    for col, (view_name, slice_key) in enumerate(zip(view_names, slice_keys)):
        ax = axes[2, col]
        real_slice = real_copd[slices[slice_key]].T
        fused_slice = fused[slices[slice_key]].T
        diff = fused_slice - real_slice
        im = ax.imshow(diff, cmap='RdBu_r', origin='lower', vmin=-200, vmax=200)
        ax.set_title(f'Difference (AI - Real) - {view_name}', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f'Patient: {patient_id} - Real COPD vs AI Fused', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'{patient_id}_proper_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def create_hu_histogram(patient_id: str, output_dir: Path):
    """创建 HU 值直方图对比"""
    if not HAS_MATPLOTLIB:
        return

    paths = find_patient_data(patient_id)
    if not all(k in paths for k in ['real_copd_warped', 'fused', 'mask']):
        return

    # 加载数据
    real_copd = load_nifti(paths['real_copd_warped'])
    fused = load_nifti(paths['fused'])
    mask = load_nifti(paths['mask'])

    mask_bool = mask > 0
    real_lesion_hu = real_copd[mask_bool]
    fused_lesion_hu = fused[mask_bool]

    # 创建直方图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：HU 值分布对比
    ax1 = axes[0]
    bins = np.linspace(-1100, 0, 100)
    ax1.hist(real_lesion_hu, bins=bins, alpha=0.6, label='Real COPD', color='blue')
    ax1.hist(fused_lesion_hu, bins=bins, alpha=0.6, label='AI Fused', color='red')
    ax1.axvline(x=-950, color='green', linestyle='--', linewidth=2, label='Emphysema threshold (-950)')
    ax1.set_xlabel('HU Value', fontsize=12)
    ax1.set_ylabel('Voxel Count', fontsize=12)
    ax1.set_title('HU Distribution in Lesion Region', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图：统计信息
    ax2 = axes[1]
    ax2.axis('off')

    real_mean = np.mean(real_lesion_hu)
    fused_mean = np.mean(fused_lesion_hu)
    real_emph = (real_lesion_hu < -950).sum() / len(real_lesion_hu) * 100
    fused_emph = (fused_lesion_hu < -950).sum() / len(fused_lesion_hu) * 100

    stats_text = f"""
    HU Value Statistics (Lesion Region)
    ====================================

    Real COPD CT:
      Mean HU: {real_mean:.1f}
      Std HU: {np.std(real_lesion_hu):.1f}
      Min HU: {np.min(real_lesion_hu):.1f}
      Max HU: {np.max(real_lesion_hu):.1f}
      Emphysema (HU<-950): {real_emph:.1f}%

    AI Fused CT:
      Mean HU: {fused_mean:.1f}
      Std HU: {np.std(fused_lesion_hu):.1f}
      Min HU: {np.min(fused_lesion_hu):.1f}
      Max HU: {np.max(fused_lesion_hu):.1f}
      Emphysema (HU<-950): {fused_emph:.1f}%

    Difference:
      Mean HU Diff: {fused_mean - real_mean:.1f}
      Emphysema Diff: {fused_emph - real_emph:.1f}%
    """
    ax2.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax2.transAxes)

    plt.suptitle(f'Patient: {patient_id} - HU Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'{patient_id}_hu_histogram.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  正确的模型评估（与真实 COPD CT 对比）")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"评估患者数: {args.num_patients}")

    if not HAS_NIBABEL:
        print("❌ 错误: nibabel 未安装")
        return

    # 查找所有可用的融合结果
    fused_dir = Path('data/04_final_viz')
    fused_files = sorted(fused_dir.glob('*_fused.nii.gz'))

    if not fused_files:
        print("❌ 未找到融合结果文件")
        return

    # 限制评估数量
    fused_files = fused_files[:args.num_patients]
    print(f"\n找到 {len(fused_files)} 个融合结果文件")

    # 评估每个患者
    all_metrics = []

    print("\n" + "=" * 60)
    print("  开始评估")
    print("=" * 60)

    for fused_path in fused_files:
        patient_id = fused_path.name.replace('_fused.nii.gz', '')
        metrics = evaluate_patient(patient_id)

        if 'error' not in metrics:
            all_metrics.append(metrics)
            # 生成可视化
            create_comparison_figure(patient_id, output_dir)
            create_hu_histogram(patient_id, output_dir)

    # 汇总结果
    if all_metrics:
        print("\n" + "=" * 60)
        print("  评估汇总")
        print("=" * 60)

        avg_psnr = np.mean([m['lesion']['psnr'] for m in all_metrics])
        avg_ssim = np.mean([m['lesion']['ssim'] for m in all_metrics])
        avg_real_emph = np.mean([m['hu_analysis']['real']['emphysema_ratio'] for m in all_metrics])
        avg_fused_emph = np.mean([m['hu_analysis']['fused']['emphysema_ratio'] for m in all_metrics])

        print(f"\n平均指标（病灶区域，真实 COPD vs AI 融合）:")
        print(f"  PSNR: {avg_psnr:.2f} dB")
        print(f"  SSIM: {avg_ssim:.4f}")
        print(f"\nHU 值分析:")
        print(f"  真实 COPD 平均肺气肿比例: {avg_real_emph:.2%}")
        print(f"  AI 融合 平均肺气肿比例: {avg_fused_emph:.2%}")
        print(f"  差异: {avg_fused_emph - avg_real_emph:.2%}")

        # 保存报告
        report_path = output_dir / 'proper_evaluation_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 正确的模型评估报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 评估方法\n")
            f.write("将 AI 融合结果与**真实 COPD CT**（配准后）进行对比\n\n")
            f.write("## 汇总结果\n\n")
            f.write(f"| 指标 | 值 |\n")
            f.write(f"|------|----|\n")
            f.write(f"| 平均 PSNR | {avg_psnr:.2f} dB |\n")
            f.write(f"| 平均 SSIM | {avg_ssim:.4f} |\n")
            f.write(f"| 真实 COPD 肺气肿比例 | {avg_real_emph:.2%} |\n")
            f.write(f"| AI 融合 肺气肿比例 | {avg_fused_emph:.2%} |\n")
            f.write(f"\n## 各患者详情\n\n")
            for m in all_metrics:
                f.write(f"### {m['patient_id']}\n")
                f.write(f"- PSNR: {m['lesion']['psnr']:.2f} dB\n")
                f.write(f"- SSIM: {m['lesion']['ssim']:.4f}\n")
                f.write(f"- 真实肺气肿: {m['hu_analysis']['real']['emphysema_ratio']:.2%}\n")
                f.write(f"- AI肺气肿: {m['hu_analysis']['fused']['emphysema_ratio']:.2%}\n\n")

        print(f"\n✓ 报告已保存: {report_path}")

    print("\n" + "=" * 60)
    print("  评估完成!")
    print("=" * 60)
    print(f"结果保存在: {output_dir}")


if __name__ == '__main__':
    main()

