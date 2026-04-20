#!/usr/bin/env python3
"""
生成"校准前 vs 校准后 vs Real COPD"三重直方图叠加对比图。

支持两种运行模式：
  1. 批量模式（默认）：对 3 模型 × 3 患者 = 9 组逐一生成直方图
  2. 单例模式：仅生成 PatchGAN × copd_001 的汇总图

输出目录结构：
  results/{model}/copd_{pid}/chart_histogram_precal_vs_postcal.png  (9张)
  results/chart_histogram_precal_vs_postcal.png  (汇总图, patchgan×copd_001 副本)
"""
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ─── 学术字体配置: Times New Roman (英文/数字) + SimSun (中文) ───
def _setup_academic_fonts():
    """配置学术出版级字体，含缺失回退机制"""
    available = {f.name for f in fm.fontManager.ttflist}
    serif_font = 'Times New Roman' if 'Times New Roman' in available else 'DejaVu Serif'
    cjk_font = 'SimSun' if 'SimSun' in available else (
        'Microsoft YaHei' if 'Microsoft YaHei' in available else 'DejaVu Sans')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [serif_font, cjk_font, 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False,
    })
    return serif_font, cjk_font

_setup_academic_fonts()

# ─── 全局配置 ───
BASE = os.path.join(os.path.dirname(__file__), "..")
MODELS = ["unet", "partial_conv", "patchgan"]
PATIENTS = ["copd_001", "copd_002", "copd_003"]
PATCHGAN_EXTRA_PATIENTS = [
    "copd_024", "copd_025", "copd_026", "copd_027", "copd_028", "copd_029"
]
MODEL_LABELS = {"unet": "U-Net", "partial_conv": "PartialConv", "patchgan": "PatchGAN"}
LEGEND_FONT_SIZE = 20

# 校准参数（与 inference_fuse.py 完全一致）
CAL_TARGET_MEAN = -965.0
CAL_TARGET_STD  = 45.0
RAW_SRC_MEAN = -918.0
RAW_SRC_STD  = 28.0

def load_nifti_data(path):
    """加载 NIfTI 并返回 numpy 数组"""
    import nibabel as nib
    return nib.load(path).get_fdata().astype(np.float32)

def reverse_calibration(calibrated_pixels):
    """
    反向还原校准前的 AI 原生 HU 分布。

    正向校准流程 (inference_fuse.py L237-269):
      scale = clip(ref_std / src_std, 0.5, 2.0)
      matched = (source - src_mean) * scale + ref_mean
      mask_gray = (matched > -960) & (matched < -800)  →  matched[mask_gray] -= 20

    反向还原:
      1. 撤销 Gamma 压暗 (加回 20 HU)
      2. 撤销 Z-score 匹配
    """
    raw = calibrated_pixels.copy()

    # 1. 撤销 Gamma 压暗: [-980, -820] 区间的像素曾被减去 20
    #    正向: 如果校准后的值在 [-960, -800] 之前(即加回20后落入[-960,-800])
    #    反向近似: 对 [-980, -820] 范围加回 20 HU
    gamma_mask = (raw > -980) & (raw < -820)
    raw[gamma_mask] += 20.0

    # 2. 撤销 Z-score: raw_original = (calibrated - ref_mean) / scale + src_mean
    scale = np.clip(CAL_TARGET_STD / (RAW_SRC_STD + 1e-6), 0.5, 2.0)
    raw = (raw - CAL_TARGET_MEAN) / scale + RAW_SRC_MEAN

    return raw

def generate_single_histogram(model, patient_id, base_dir):
    """为指定的 模型×患者 组合生成一张 HU 分布对比直方图"""
    pid = patient_id
    fused_path = os.path.join(base_dir, "data", "04_final_viz", model, f"{pid}_fused.nii.gz")
    mask_path  = os.path.join(base_dir, "data", "03_mapped", pid, f"{pid}_warped_lesion.nii.gz")
    real_path  = os.path.join(base_dir, "data", "03_mapped", pid, f"{pid}_warped.nii.gz")

    out_dir = os.path.join(base_dir, "results", model, pid)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "chart_histogram_precal_vs_postcal.png")

    for p in [fused_path, mask_path, real_path]:
        if not os.path.exists(p):
            print(f"  [SKIP] File not found: {p}")
            return None

    print(f"  Loading NIfTI files...")
    fused_data = load_nifti_data(fused_path)
    mask_data  = load_nifti_data(mask_path)
    real_data  = load_nifti_data(real_path)

    lesion_idx = mask_data > 0
    n_voxels = int(np.sum(lesion_idx))
    print(f"  Lesion voxels: {n_voxels:,}")

    fused_lesion = fused_data[lesion_idx]
    real_lesion  = real_data[lesion_idx]

    print(f"  Reverse-engineering pre-calibration distribution...")
    raw_lesion = reverse_calibration(fused_lesion)

    # ─── 绘图 ───
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            pass
    _setup_academic_fonts()
    plt.rcParams['font.family'] = 'serif'

    fig = plt.figure(figsize=(16, 8), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.15,
                          left=0.06, right=0.97, top=0.90, bottom=0.10)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])

    colors = {
        'Pre-Cal (AI Raw)':     '#4682B4',
        'Post-Cal (AI Fused)':  '#E74C3C',
        'Real COPD (Ref)':      '#27ae60',
    }

    draw_order = [
        ('Pre-Cal (AI Raw)',     raw_lesion),
        ('Post-Cal (AI Fused)',  fused_lesion),
        ('Real COPD (Ref)',      real_lesion),
    ]
    all_stats = {}
    for label, data in draw_order:
        color = colors[label]
        data_flat = data.flatten()
        ax_hist.hist(data_flat, bins=80, range=(-1024, 0),
                     alpha=0.35, label=None, color=color,
                     density=True, histtype='stepfilled')
        ax_hist.hist(data_flat, bins=80, range=(-1024, 0),
                     alpha=1.0, color=color, density=True,
                     histtype='step', linewidth=2, label=label)
        all_stats[label] = {
            'mean': float(np.mean(data_flat)),
            'std':  float(np.std(data_flat)),
            'min':  float(np.min(data_flat)),
            'max':  float(np.max(data_flat)),
            'emphysema_ratio': float(np.sum(data_flat < -950) / len(data_flat) * 100),
        }

    ax_hist.axvline(x=-950, color='#2c3e50', linestyle='--', linewidth=2,
                    alpha=0.8, label='Emphysema Threshold (-950)')

    ax_hist.set_xlim(-1024, 0)
    ax_hist.set_xlabel("HU Value", fontsize=16, fontweight='bold')
    ax_hist.set_ylabel("Density", fontsize=16, fontweight='bold')
    ax_hist.set_title("HU Distribution in 3D Lesion", fontsize=20, fontweight='bold')
    ax_hist.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE, framealpha=0.9,
                   prop={'family': 'Times New Roman', 'size': LEGEND_FONT_SIZE})
    ax_hist.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_hist.set_axisbelow(True)
    ax_hist.tick_params(axis='both', which='major', labelsize=14)

    # ─── 右侧统计文本框（精简字段 + 超大字号）───
    ax_text.axis('off')
    # 标题
    ax_text.text(0.02, 0.98, "HU Statistics",
                 transform=ax_text.transAxes, fontsize=22, fontweight='bold',
                 verticalalignment='top', fontname='Times New Roman')
    ax_text.text(0.02, 0.92, "=" * 20,
                 transform=ax_text.transAxes, fontsize=18,
                 verticalalignment='top', fontname='Times New Roman',
                 color='#666666')

    # 精简正文：仅保留 Mean + EI（最核心的两项）
    body_lines = []
    panel_order = [
        ('Pre-Cal (AI Raw)',     'Pre-Cal (AI Raw)'),
        ('Post-Cal (AI Fused)', 'Post-Cal (Fused)'),
        ('Real COPD (Ref)',     'Real COPD (Ref)'),
    ]
    for key, header in panel_order:
        s = all_stats[key]
        body_lines.extend([
            f"{header}:",
            f"  Mean:    {s['mean']:.1f} HU",
            f"  EI rate: {s['emphysema_ratio']:.1f}%",
            "",
        ])
    pre_s = all_stats['Pre-Cal (AI Raw)']
    post_s = all_stats['Post-Cal (AI Fused)']
    real_s = all_stats['Real COPD (Ref)']
    body_lines.extend([
        "Calibration Effect:",
        f"  Mean shift:",
        f"    {pre_s['mean']:.1f} → {post_s['mean']:.1f}",
        f"  Real ref: {real_s['mean']:.1f}",
        f"  ΔEI gap:  {abs(post_s['emphysema_ratio'] - real_s['emphysema_ratio']):.1f}%",
    ])
    ax_text.text(0.02, 0.865, '\n'.join(body_lines), transform=ax_text.transAxes,
                 fontsize=17, verticalalignment='top', fontname='Times New Roman',
                 linespacing=1.35)

    title_label = MODEL_LABELS.get(model, model)
    fig.suptitle(f"{pid} ({title_label}) — HU Distribution: Pre-Cal vs Post-Cal vs Real COPD",
                 fontsize=20, fontweight='bold', y=0.98)

    fig.savefig(out_path, bbox_inches='tight', facecolor='white',
                edgecolor='none', dpi=300)
    plt.close(fig)
    plt.style.use('default')
    print(f"  [OK] Saved -> {out_path}")
    return all_stats


def main():
    """批量生成 3 模型×3患者 + PatchGAN额外6例(024~029) 直方图，并输出汇总。"""
    import json

    summary_stats = {}

    # 1) 默认批量：3 模型 × 3 患者
    for model in MODELS:
        for pid in PATIENTS:
            print(f"\n{'='*60}")
            print(f"  Generating: {MODEL_LABELS[model]} x {pid}")
            print(f"{'='*60}")
            s = generate_single_histogram(model, pid, BASE)
            if s:
                summary_stats[f"{model}_{pid}"] = {
                    k: {"mean": round(v['mean'], 2), "std": round(v['std'], 2),
                         "emphysema_pct": round(v['emphysema_ratio'], 2)}
                    for k, v in s.items()
                }

    # 2) PatchGAN 扩展批量：copd_024~copd_029
    for pid in PATCHGAN_EXTRA_PATIENTS:
        print(f"\n{'='*60}")
        print(f"  Generating: {MODEL_LABELS['patchgan']} x {pid}")
        print(f"{'='*60}")
        s = generate_single_histogram("patchgan", pid, BASE)
        if s:
            summary_stats[f"patchgan_{pid}"] = {
                k: {"mean": round(v['mean'], 2), "std": round(v['std'], 2),
                     "emphysema_pct": round(v['emphysema_ratio'], 2)}
                for k, v in s.items()
            }

    # 将 patchgan×copd_001 复制为报告引用的汇总图
    src = os.path.join(BASE, "results", "patchgan", "copd_001",
                       "chart_histogram_precal_vs_postcal.png")
    dst = os.path.join(BASE, "results", "chart_histogram_precal_vs_postcal.png")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"\n[DONE] Summary chart copied -> {dst}")

    # 保存统计数据
    stats_path = os.path.join(BASE, "results", "histogram_comparison_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Stats saved -> {stats_path}")
    print(f"\n[DONE] All {len(summary_stats)} histogram charts generated.")


if __name__ == "__main__":
    main()

