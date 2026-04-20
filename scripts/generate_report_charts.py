#!/usr/bin/env python3
"""
训练集 (copd_001~003) 模型对比实验 — 学术可视化图表生成（支持动态6模型）

输出（存放在 charts_train/ 子目录）:
  results/charts_train/chart_radar_l2l4.png        - 多维能力雷达图
  results/charts_train/chart_glcm_contrast.png     - GLCM Contrast 分组柱状图
  results/charts_train/chart_delta_ei.png          - ΔEI 肺气肿指数偏差柱状图
  results/charts_train/chart_composite_score.png   - 综合得分排名图
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
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 12,
    "axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 11,
    "figure.facecolor": "white",
})

COLORS = {
    "unet": "#5B9BD5", "partial_conv": "#70AD47", "patchgan": "#ED7D31",
    "attgan": "#9B59B6", "mae_patchgan": "#E74C3C", "ddpm": "#1ABC9C",
}
LABELS = {
    "unet": "U-Net", "partial_conv": "PartialConv", "patchgan": "PatchGAN",
    "attgan": "AttGAN", "mae_patchgan": "MAE-PatchGAN", "ddpm": "DDPM",
}
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "results", "charts_train")
_ALL_MODELS = ["unet", "partial_conv", "patchgan", "attgan", "mae_patchgan", "ddpm"]
PATIENTS = ["copd_001", "copd_002", "copd_003"]

# ---------- 加载数据 ----------
json_path = os.path.join(ROOT, "results", "model_comparison_metrics.json")
with open(json_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

# 动态检测数据中实际包含的模型（跳过null记录）
MODELS = [m for m in _ALL_MODELS
          if any(r["model"] == m and r.get("L1_PSNR_dB") is not None for r in raw)]
print(f"[INFO] 检测到 {len(MODELS)} 个模型: {MODELS}")

def mean_by_model(model, key):
    vals = [r[key] for r in raw if r["model"] == model and r.get(key) is not None]
    return np.mean(vals) if vals else 0.0

def ref_mean(key):
    seen = set()
    vals = []
    for r in raw:
        if r["patient_id"] not in seen and r.get(key) is not None:
            seen.add(r["patient_id"])
            vals.append(r[key])
    return np.mean(vals) if vals else 1.0

# ============================================================
#  图1: 多维能力雷达图 (L2-L4)
# ============================================================
def make_radar():
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

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    lw = 2.0 if len(MODELS) > 3 else 2.5
    ms = 5 if len(MODELS) > 3 else 7
    fill_alpha = 0.08 if len(MODELS) > 3 else 0.12

    for model in MODELS:
        psd_score = 1.0 - abs(mean_by_model(model, "L2_PSD_HF_Ratio") - ref_psd) / max(ref_psd, 1e-6)
        glcm_score = mean_by_model(model, "L2_GLCM_Contrast") / max(ref_glcm, 1e-6)
        lvcov_score = mean_by_model(model, "L3_LV_CoV") / max(ref_lvcov, 1e-6)
        wdist_val = mean_by_model(model, "L3_Wasserstein_HU")
        wdist_score = 1.0 / (1.0 + wdist_val / 50.0)
        dei_val = mean_by_model(model, "L4_Delta_EI_pct")
        dei_score = 1.0 / (1.0 + dei_val)

        vals = [psd_score, glcm_score, lvcov_score, wdist_score, dei_score]
        vals += vals[:1]

        ax.plot(angles, vals, "o-", linewidth=lw, label=LABELS[model],
                color=COLORS[model], markersize=ms)
        ax.fill(angles, vals, alpha=fill_alpha, color=COLORS[model])

    ax.set_thetagrids(np.degrees(angles[:-1]), dims, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="grey")

    ax.plot(angles, [1.0] * len(angles), '--', color='#888888', linewidth=1.5,
            alpha=0.6, label='Real COPD (Target)')

    nm = len(MODELS)
    ax.set_title(f"Training Set: Multi-Dimensional Radar (L2-L4)\n"
                 f"({nm} Models — copd_001 / copd_002 / copd_003)",
                 pad=24, fontweight="bold", fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.18), frameon=True,
              fancybox=True, fontsize=10, ncol=1)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUT_DIR, "chart_radar_l2l4.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Radar chart -> {out}")

# ============================================================
#  图2: GLCM Contrast 分组柱状图
# ============================================================
def make_glcm_bar():
    nm = len(MODELS)
    w = 0.7 / nm
    fig, ax = plt.subplots(figsize=(max(12, 3 + nm * 2), 7))
    x = np.arange(len(PATIENTS))

    for i, model in enumerate(MODELS):
        vals = [next((r["L2_GLCM_Contrast"] for r in raw
                     if r["model"] == model and r["patient_id"] == p), 0.0)
                for p in PATIENTS]
        bars = ax.bar(x + i * w, vals, w, label=LABELS[model],
                      color=COLORS[model], edgecolor="white", linewidth=0.8)
        fs = 11 if nm > 4 else 14
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=fs,
                    fontweight="bold", rotation=45 if nm > 4 else 0)

    refs = [next((r["Ref_GLCM_C"] for r in raw if r["patient_id"] == p), 0.0)
            for p in PATIENTS]
    for j, rv in enumerate(refs):
        ax.plot([x[j] - 0.05, x[j] + nm*w + 0.05], [rv, rv], "--",
                color="#C00000", linewidth=1.8, alpha=0.8)
        ax.text(x[j] + nm*w + 0.08, rv, f"Real={rv:.2f}",
                color="#C00000", fontsize=11, va="center", fontweight="bold")

    ax.set_xlabel("Patient (Training Set)", fontsize=14, fontweight="bold")
    ax.set_ylabel("GLCM Contrast", fontsize=14, fontweight="bold")
    ax.set_title(f"Training Set: GLCM Contrast per Patient ({nm} Models)\n"
                 "(higher = closer to Real COPD)",
                 fontweight="bold", fontsize=16)
    ax.set_xticks(x + w * (nm - 1) / 2)
    ax.set_xticklabels(PATIENTS, fontsize=13)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(frameon=True, fancybox=True, fontsize=10, ncol=2 if nm > 4 else 1)
    ax.set_ylim(0, max(refs) * 1.22)
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
    nm = len(MODELS)
    w = 0.7 / nm
    fig, ax = plt.subplots(figsize=(max(11, 3 + nm * 2), 6))
    x = np.arange(len(PATIENTS))

    for i, model in enumerate(MODELS):
        vals = [next((r["L4_Delta_EI_pct"] for r in raw
                     if r["model"] == model and r["patient_id"] == p), 0.0)
                for p in PATIENTS]
        bars = ax.bar(x + i * w, vals, w, label=LABELS[model],
                      color=COLORS[model], edgecolor="white", linewidth=0.8)
        fs = 9 if nm > 4 else 11
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=fs,
                    fontweight="bold", rotation=45 if nm > 5 else 0)

    ax.axhline(y=5.0, color="#C00000", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(len(PATIENTS) - 0.3, 5.15, "Clinical Safety Threshold (5%)",
            color="#C00000", fontsize=10, ha="right", fontweight="bold")

    ax.set_xlabel("Patient (Training Set)", fontsize=13)
    ax.set_ylabel("\u0394EI (%)", fontsize=13)
    ax.set_title(f"Training Set: Emphysema Index Deviation (\u0394EI) — {nm} Models\n(lower = better)",
                 fontweight="bold", fontsize=14)
    ax.set_xticks(x + w * (nm - 1) / 2)
    ax.set_xticklabels(PATIENTS, fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(frameon=True, fancybox=True, fontsize=9, ncol=2 if nm > 4 else 1)
    ax.set_ylim(0, 6.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    out = os.path.join(OUT_DIR, "chart_delta_ei.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] \u0394EI bar chart -> {out}")

# ============================================================
#  图4: 综合得分横向排名 — L1-L4 加权
# ============================================================
def make_composite_score():
    keys_higher = ["L1_PSNR_dB", "L2_GLCM_Contrast"]
    keys_lower = ["L1_MAE_HU", "L3_Wasserstein_HU", "L4_Delta_EI_pct"]

    model_scores = {}
    for model in MODELS:
        score = 0.0
        n = 0
        for k in keys_higher:
            v = mean_by_model(model, k)
            best_v = max(mean_by_model(m, k) for m in MODELS)
            if v > 0 and best_v > 0:
                score += v / best_v
                n += 1
        for k in keys_lower:
            v = mean_by_model(model, k)
            best_v = min(mean_by_model(m, k) for m in MODELS)
            if v > 0:
                score += best_v / v
                n += 1
        model_scores[model] = score / max(n, 1) * 100

    sorted_models = sorted(model_scores.keys(), key=lambda m: model_scores[m], reverse=True)
    scores = [model_scores[m] for m in sorted_models]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(sorted_models)), scores,
                   color=[COLORS[m] for m in sorted_models],
                   edgecolor="white", linewidth=1.2, height=0.6)
    for bar, s, m in zip(bars, scores, sorted_models):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{s:.1f}", va="center", fontsize=13, fontweight="bold")

    ax.set_yticks(range(len(sorted_models)))
    ax.set_yticklabels([LABELS[m] for m in sorted_models], fontsize=13)
    ax.set_xlabel("Composite Score (0-100)", fontsize=13, fontweight="bold")
    ax.set_title(f"Training Set: Composite L1-L4 Score Ranking ({len(MODELS)} Models)\n"
                 "(higher = better overall performance on training data)",
                 fontweight="bold", fontsize=14)
    ax.set_xlim(0, max(scores) * 1.12)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)

    out = os.path.join(OUT_DIR, "chart_composite_score.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Composite score chart -> {out}")

# ---------- 执行 ----------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    make_radar()
    make_glcm_bar()
    make_dei_bar()
    make_composite_score()
    print("\n[DONE] All charts generated.")

