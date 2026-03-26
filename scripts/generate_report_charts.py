#!/usr/bin/env python3
"""
生成模型对比实验报告的学术可视化图表
输出:
  results/chart_radar_l2l4.png      - 多维能力雷达图
  results/chart_glcm_contrast.png   - GLCM Contrast 分组柱状图
  results/chart_delta_ei.png        - ΔEI 肺气肿指数偏差柱状图
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------- 学术字体配置: Times New Roman (英文/数字) + SimSun (中文) ----------
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

# ---------- 全局设置 ----------
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
})

COLORS = {"unet": "#5B9BD5", "partial_conv": "#70AD47", "patchgan": "#ED7D31"}
LABELS = {"unet": "U-Net", "partial_conv": "PartialConv", "patchgan": "PatchGAN"}
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ---------- 加载数据 ----------
json_path = os.path.join(OUT_DIR, "model_comparison_metrics.json")
with open(json_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

def mean_by_model(model, key):
    return np.mean([r[key] for r in raw if r["model"] == model])

def ref_mean(key):
    seen = set()
    vals = []
    for r in raw:
        if r["patient_id"] not in seen:
            seen.add(r["patient_id"])
            vals.append(r[key])
    return np.mean(vals)

models = ["unet", "partial_conv", "patchgan"]

# ============================================================
#  图1: 多维能力雷达图 (L2-L4)
# ============================================================
def make_radar():
    # 指标: PSD-HF接近度, GLCM-C达标率, LV-CoV达标率, 1/W-dist, 1/ΔEI
    ref_psd   = ref_mean("Ref_PSD_HF")
    ref_glcm  = ref_mean("Ref_GLCM_C")
    ref_lvcov = ref_mean("Ref_LV_CoV")

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

    for model in models:
        psd_score = 1.0 - abs(mean_by_model(model, "L2_PSD_HF_Ratio") - ref_psd) / ref_psd
        glcm_score = mean_by_model(model, "L2_GLCM_Contrast") / ref_glcm
        lvcov_score = mean_by_model(model, "L3_LV_CoV") / ref_lvcov
        wdist_val = mean_by_model(model, "L3_Wasserstein_HU")
        wdist_score = 1.0 / (1.0 + wdist_val / 50.0)
        dei_val = mean_by_model(model, "L4_Delta_EI_pct")
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

    # Real COPD 参照圈线 (r=1.0): 所有维度归一化后 Real COPD 均映射为 1.0
    ax.plot(angles, [1.0] * len(angles), '--', color='#888888', linewidth=1.5,
            alpha=0.6, label='Real COPD (Target)')

    ax.set_title("Multi-Dimensional Model Capability Radar (L2-L4)",
                 pad=24, fontweight="bold", fontsize=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.14), frameon=True,
              fancybox=True, fontsize=11)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUT_DIR, "chart_radar_l2l4.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Radar chart -> {out}")

# ============================================================
#  图2: GLCM Contrast 分组柱状图
# ============================================================
def make_glcm_bar():
    patients = ["copd_001", "copd_002", "copd_003"]
    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(patients))
    w = 0.22

    for i, model in enumerate(models):
        vals = [next(r["L2_GLCM_Contrast"] for r in raw
                     if r["model"] == model and r["patient_id"] == p)
                for p in patients]
        bars = ax.bar(x + i * w, vals, w, label=LABELS[model],
                      color=COLORS[model], edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=14,
                    fontweight="bold")

    refs = [next(r["Ref_GLCM_C"] for r in raw if r["patient_id"] == p) for p in patients]
    for j, rv in enumerate(refs):
        ax.plot([x[j] - 0.1, x[j] + 3*w + 0.1], [rv, rv], "--",
                color="#C00000", linewidth=1.8, alpha=0.8)
        ax.text(x[j] + 3*w + 0.12, rv, f"Real={rv:.2f}",
                color="#C00000", fontsize=13, va="center", fontweight="bold")

    ax.set_xlabel("Patient", fontsize=16, fontweight="bold")
    ax.set_ylabel("GLCM Contrast", fontsize=16, fontweight="bold")
    ax.set_title("GLCM Contrast: Texture Roughness per Patient\n(higher = closer to Real COPD)",
                 fontweight="bold", fontsize=18)
    ax.set_xticks(x + w)
    ax.set_xticklabels(patients, fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(frameon=True, fancybox=True, fontsize=13)
    ax.set_ylim(0, max(refs) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    out = os.path.join(OUT_DIR, "chart_glcm_contrast.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] GLCM bar chart -> {out}")

# ============================================================
#  图3: ΔEI 肺气肿指数偏差柱状图
# ============================================================
def make_dei_bar():
    patients = ["copd_001", "copd_002", "copd_003"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(patients))
    w = 0.22

    for i, model in enumerate(models):
        vals = [next(r["L4_Delta_EI_pct"] for r in raw
                     if r["model"] == model and r["patient_id"] == p)
                for p in patients]
        bars = ax.bar(x + i * w, vals, w, label=LABELS[model],
                      color=COLORS[model], edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=11,
                    fontweight="bold")

    ax.axhline(y=5.0, color="#C00000", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(len(patients) - 0.5, 5.15, "Clinical Safety Threshold (5%)",
            color="#C00000", fontsize=11, ha="right", fontweight="bold")

    ax.set_xlabel("Patient", fontsize=13)
    ax.set_ylabel("ΔEI (%)", fontsize=13)
    ax.set_title("Emphysema Index Deviation (ΔEI): Clinical Fidelity\n(lower = better)",
                 fontweight="bold", fontsize=15)
    ax.set_xticks(x + w)
    ax.set_xticklabels(patients, fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(frameon=True, fancybox=True, fontsize=11)
    ax.set_ylim(0, 5.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    out = os.path.join(OUT_DIR, "chart_delta_ei.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] ΔEI bar chart -> {out}")

# ---------- 执行 ----------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    make_radar()
    make_glcm_bar()
    make_dei_bar()
    print("\n[DONE] All charts generated.")

