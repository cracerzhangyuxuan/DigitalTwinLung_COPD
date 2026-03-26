#!/usr/bin/env python3
"""
验证集 (copd_024~026) 模型对比实验 — 学术可视化图表生成

输出（文件名带 _val 后缀，不覆盖训练集图表）:
  results/chart_radar_l2l4_val.png         - 多维能力雷达图
  results/chart_glcm_contrast_val.png      - GLCM Contrast 分组柱状图
  results/chart_delta_ei_val.png           - ΔEI 肺气肿指数偏差柱状图
  results/chart_train_vs_val_val.png       - 训练集 vs 验证集衰减对比图
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------- 学术字体配置 ----------
def _setup_academic_fonts():
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

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 12,
    "axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 11,
    "figure.facecolor": "white",
})

COLORS = {"unet": "#5B9BD5", "partial_conv": "#70AD47", "patchgan": "#ED7D31"}
LABELS = {"unet": "U-Net", "partial_conv": "PartialConv", "patchgan": "PatchGAN"}
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "results")
MODELS = ["unet", "partial_conv", "patchgan"]
PATIENTS = ["copd_024", "copd_025", "copd_026"]

# ---------- 加载验证集数据 ----------
val_json = os.path.join(OUT_DIR, "validation_metrics.json")
with open(val_json, "r", encoding="utf-8") as f:
    val_data = json.load(f)

# 加载训练集数据（用于对比）
train_json = os.path.join(OUT_DIR, "model_comparison_metrics.json")
train_data = []
if os.path.exists(train_json):
    with open(train_json, "r", encoding="utf-8") as f:
        train_data = [r for r in json.load(f) if r.get("L1_PSNR_dB") is not None]

def mean_by_model(data, model, key):
    vals = [r[key] for r in data if r["model"] == model]
    return np.mean(vals) if vals else 0.0

def ref_mean(data, key):
    seen = set()
    vals = []
    for r in data:
        if r["patient_id"] not in seen:
            seen.add(r["patient_id"])
            vals.append(r[key])
    return np.mean(vals) if vals else 1.0

# ============================================================
#  图1: 多维能力雷达图 (L2-L4) — 验证集
# ============================================================
def make_radar_val():
    ref_psd   = ref_mean(val_data, "Ref_PSD_HF")
    ref_glcm  = ref_mean(val_data, "Ref_GLCM_C")
    ref_lvcov = ref_mean(val_data, "Ref_LV_CoV")

    dims = [
        "PSD-HF\n(Freq. Fidelity)",
        "GLCM Contrast\n(Texture Roughness)",
        "LV-CoV\n(Spatial Heterogeneity)",
        "1/W-dist\n(Distribution Match)",
        "1/\u0394EI\n(Clinical Fidelity)",
    ]
    n = len(dims)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model in MODELS:
        psd_score = 1.0 - abs(mean_by_model(val_data, model, "L2_PSD_HF_Ratio") - ref_psd) / ref_psd
        glcm_score = mean_by_model(val_data, model, "L2_GLCM_Contrast") / ref_glcm
        lvcov_score = mean_by_model(val_data, model, "L3_LV_CoV") / ref_lvcov
        wdist_val = mean_by_model(val_data, model, "L3_Wasserstein_HU")
        wdist_score = 1.0 / (1.0 + wdist_val / 50.0)
        dei_val = mean_by_model(val_data, model, "L4_Delta_EI_pct")
        dei_score = 1.0 / (1.0 + dei_val)

        vals = [psd_score, glcm_score, lvcov_score, wdist_score, dei_score]
        vals += vals[:1]

        ax.plot(angles, vals, "o-", linewidth=2.5, label=LABELS[model],
                color=COLORS[model], markersize=7)
        ax.fill(angles, vals, alpha=0.12, color=COLORS[model])

    ax.set_thetagrids(np.degrees(angles[:-1]), dims, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10, color="grey")
    ax.plot(angles, [1.0] * len(angles), '--', color='#888888', linewidth=1.5,
            alpha=0.6, label='Real COPD (Target)')
    ax.set_title("Validation Set: Multi-Dimensional Radar (L2-L4)\n"
                 "(copd_024 / copd_025 / copd_026 — Unseen Data)",
                 pad=24, fontweight="bold", fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.14), frameon=True,
              fancybox=True, fontsize=11)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUT_DIR, "chart_radar_l2l4_val.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Validation radar chart -> {out}")


# ============================================================
#  图2: GLCM Contrast 分组柱状图 — 验证集
# ============================================================
def make_glcm_bar_val():
    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(PATIENTS))
    w = 0.22

    for i, model in enumerate(MODELS):
        vals = [next(r["L2_GLCM_Contrast"] for r in val_data
                     if r["model"] == model and r["patient_id"] == p)
                for p in PATIENTS]
        bars = ax.bar(x + i * w, vals, w, label=LABELS[model],
                      color=COLORS[model], edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=14,
                    fontweight="bold")

    refs = [next(r["Ref_GLCM_C"] for r in val_data if r["patient_id"] == p)
            for p in PATIENTS]
    for j, rv in enumerate(refs):
        ax.plot([x[j] - 0.1, x[j] + 3*w + 0.1], [rv, rv], "--",
                color="#C00000", linewidth=1.8, alpha=0.8)
        ax.text(x[j] + 3*w + 0.12, rv, f"Real={rv:.2f}",
                color="#C00000", fontsize=13, va="center", fontweight="bold")

    ax.set_xlabel("Patient (Validation Set)", fontsize=16, fontweight="bold")
    ax.set_ylabel("GLCM Contrast", fontsize=16, fontweight="bold")
    ax.set_title("Validation Set: GLCM Contrast per Patient\n"
                 "(higher = closer to Real COPD)",
                 fontweight="bold", fontsize=18)
    ax.set_xticks(x + w)
    ax.set_xticklabels(PATIENTS, fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(frameon=True, fancybox=True, fontsize=13)
    ax.set_ylim(0, max(refs) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    out = os.path.join(OUT_DIR, "chart_glcm_contrast_val.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Validation GLCM bar chart -> {out}")

# ============================================================
#  图3: ΔEI 肺气肿指数偏差柱状图 — 验证集
# ============================================================
def make_dei_bar_val():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(PATIENTS))
    w = 0.22

    for i, model in enumerate(MODELS):
        vals = [next(r["L4_Delta_EI_pct"] for r in val_data
                     if r["model"] == model and r["patient_id"] == p)
                for p in PATIENTS]
        bars = ax.bar(x + i * w, vals, w, label=LABELS[model],
                      color=COLORS[model], edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=11,
                    fontweight="bold")

    ax.axhline(y=5.0, color="#C00000", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(len(PATIENTS) - 0.5, 5.15, "Clinical Safety Threshold (5%)",
            color="#C00000", fontsize=11, ha="right", fontweight="bold")

    ax.set_xlabel("Patient (Validation Set)", fontsize=13)
    ax.set_ylabel("\u0394EI (%)", fontsize=13)
    ax.set_title("Validation Set: Emphysema Index Deviation (\u0394EI)\n(lower = better)",
                 fontweight="bold", fontsize=15)
    ax.set_xticks(x + w)
    ax.set_xticklabels(PATIENTS, fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(frameon=True, fancybox=True, fontsize=11)
    ax.set_ylim(0, 5.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    out = os.path.join(OUT_DIR, "chart_delta_ei_val.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Validation \u0394EI bar chart -> {out}")

# ============================================================
#  图4: 训练集 vs 验证集 衰减对比柱状图
# ============================================================
def make_train_vs_val():
    if not train_data:
        print("[SKIP] No training data for comparison chart")
        return

    metrics = [
        ("PSNR (dB) \u2191", "L1_PSNR_dB", True),
        ("MAE (HU) \u2193", "L1_MAE_HU", False),
        ("GLCM-C \u2191", "L2_GLCM_Contrast", True),
        ("W-dist (HU) \u2193", "L3_Wasserstein_HU", False),
        ("\u0394EI (%) \u2193", "L4_Delta_EI_pct", False),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))
    fig.suptitle("Training Set (copd_001~003) vs Validation Set (copd_024~026)\n"
                 "Generalization Performance Comparison",
                 fontweight="bold", fontsize=16, y=1.02)

    for ax_idx, (title, key, higher_better) in enumerate(metrics):
        ax = axes[ax_idx]
        x = np.arange(len(MODELS))
        w = 0.32

        t_vals = [mean_by_model(train_data, m, key) for m in MODELS]
        v_vals = [mean_by_model(val_data, m, key) for m in MODELS]

        bars1 = ax.bar(x - w/2, t_vals, w, label="Train Set",
                       color="#7FB3D8", edgecolor="white", linewidth=0.8)
        bars2 = ax.bar(x + w/2, v_vals, w, label="Val Set",
                       color="#F4A460", edgecolor="white", linewidth=0.8)

        for bar, v in zip(bars1, t_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for bar, v in zip(bars2, v_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in MODELS], fontsize=10, rotation=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        if ax_idx == 0:
            ax.legend(fontsize=10, frameon=True)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "chart_train_vs_val_val.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Train vs Val comparison chart -> {out}")

# ---------- 执行 ----------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    make_radar_val()
    make_glcm_bar_val()
    make_dei_bar_val()
    make_train_vs_val()
    print("\n[DONE] All validation charts generated.")

