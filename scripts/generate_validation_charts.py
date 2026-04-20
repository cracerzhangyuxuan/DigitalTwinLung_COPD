#!/usr/bin/env python3
"""
验证集 (copd_024~026) 模型对比实验 — 学术可视化图表生成

输出（存放在 charts_val/ 子目录）:
  results/charts_val/chart_radar_l2l4.png         - 多维能力雷达图
  results/charts_val/chart_glcm_contrast.png      - GLCM Contrast 分组柱状图
  results/charts_val/chart_delta_ei.png           - ΔEI 肺气肿指数偏差柱状图
  results/charts_val/chart_train_vs_val.png       - 训练集 vs 验证集衰减对比图
  results/charts_val/chart_composite_score.png    - 综合得分排名图
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

COLORS = {
    "unet": "#5B9BD5", "partial_conv": "#70AD47", "patchgan": "#ED7D31",
    "attgan": "#9B59B6", "mae_patchgan": "#E74C3C", "ddpm": "#1ABC9C",
}
LABELS = {
    "unet": "U-Net", "partial_conv": "PartialConv", "patchgan": "PatchGAN",
    "attgan": "AttGAN", "mae_patchgan": "MAE-PatchGAN", "ddpm": "DDPM",
}
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "results", "charts_val")
# 动态检测数据中包含的模型（兼容3模型和6模型场景）
_ALL_MODELS = ["unet", "partial_conv", "patchgan", "attgan", "mae_patchgan", "ddpm"]
# ---------- 加载验证集数据 ----------
val_json = os.path.join(ROOT, "results", "validation_metrics.json")
with open(val_json, "r", encoding="utf-8") as f:
    val_data = json.load(f)

# 动态检测数据中实际包含的模型和患者
MODELS = [m for m in _ALL_MODELS if any(r["model"] == m for r in val_data)]
PATIENTS = sorted(set(r["patient_id"] for r in val_data))
print(f"[INFO] 检测到 {len(MODELS)} 个模型: {MODELS}")
print(f"[INFO] 检测到 {len(PATIENTS)} 个患者: {PATIENTS}")

# 加载训练集数据（用于对比）
train_json = os.path.join(ROOT, "results", "model_comparison_metrics.json")
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
#  图0: 五模型多维能力雷达图 (L2-L4) — 排除 DDPM
# ============================================================
def make_radar_val_no_ddpm():
    """Generate a radar chart with only 5 models (excluding DDPM)."""
    models_5 = [m for m in MODELS if m != "ddpm"]
    if len(models_5) == 0:
        print("[SKIP] No non-DDPM models found, skipping 5-model radar.")
        return

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

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    lw = 2.2
    ms = 6
    fill_alpha = 0.10

    for model in models_5:
        psd_score = 1.0 - abs(mean_by_model(val_data, model, "L2_PSD_HF_Ratio") - ref_psd) / max(ref_psd, 1e-6)
        glcm_score = mean_by_model(val_data, model, "L2_GLCM_Contrast") / max(ref_glcm, 1e-6)
        lvcov_score = mean_by_model(val_data, model, "L3_LV_CoV") / max(ref_lvcov, 1e-6)
        wdist_val = mean_by_model(val_data, model, "L3_Wasserstein_HU")
        wdist_score = 1.0 / (1.0 + wdist_val / 50.0)
        dei_val = mean_by_model(val_data, model, "L4_Delta_EI_pct")
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
    patient_label = " / ".join(PATIENTS)
    ax.set_title(f"Validation Set: Multi-Dimensional Radar (L2-L4)\n"
                 f"(5 Models excl. DDPM \u2014 {patient_label} \u2014 Unseen Data)",
                 pad=24, fontweight="bold", fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.18), frameon=True,
              fancybox=True, fontsize=10, ncol=1)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUT_DIR, "chart_radar_l2l4_no_ddpm.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] 5-model radar chart (no DDPM) -> {out}")


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

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    lw = 2.0 if len(MODELS) > 3 else 2.5
    ms = 5 if len(MODELS) > 3 else 7
    fill_alpha = 0.08 if len(MODELS) > 3 else 0.12

    for model in MODELS:
        psd_score = 1.0 - abs(mean_by_model(val_data, model, "L2_PSD_HF_Ratio") - ref_psd) / max(ref_psd, 1e-6)
        glcm_score = mean_by_model(val_data, model, "L2_GLCM_Contrast") / max(ref_glcm, 1e-6)
        lvcov_score = mean_by_model(val_data, model, "L3_LV_CoV") / max(ref_lvcov, 1e-6)
        wdist_val = mean_by_model(val_data, model, "L3_Wasserstein_HU")
        wdist_score = 1.0 / (1.0 + wdist_val / 50.0)
        dei_val = mean_by_model(val_data, model, "L4_Delta_EI_pct")
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
    n_models = len(MODELS)
    patient_label = " / ".join(PATIENTS)
    ax.set_title(f"Validation Set: Multi-Dimensional Radar (L2-L4)\n"
                 f"({n_models} Models — {patient_label} — Unseen Data)",
                 pad=24, fontweight="bold", fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.18), frameon=True,
              fancybox=True, fontsize=10, ncol=1)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUT_DIR, "chart_radar_l2l4.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Validation radar chart -> {out}")


# ============================================================
#  图2: GLCM Contrast 分组柱状图 — 验证集
# ============================================================
def make_glcm_bar_val():
    nm = len(MODELS)
    w = 0.7 / nm  # 动态柱宽
    fig, ax = plt.subplots(figsize=(max(12, 3 + nm * 2), 7))
    x = np.arange(len(PATIENTS))

    for i, model in enumerate(MODELS):
        vals = [next((r["L2_GLCM_Contrast"] for r in val_data
                     if r["model"] == model and r["patient_id"] == p), 0.0)
                for p in PATIENTS]
        bars = ax.bar(x + i * w, vals, w, label=LABELS[model],
                      color=COLORS[model], edgecolor="white", linewidth=0.8)
        fs = 11 if nm > 4 else 14
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=fs,
                    fontweight="bold", rotation=45 if nm > 4 else 0)

    refs = [next((r["Ref_GLCM_C"] for r in val_data if r["patient_id"] == p), 0.0)
            for p in PATIENTS]
    for j, rv in enumerate(refs):
        ax.plot([x[j] - 0.05, x[j] + nm*w + 0.05], [rv, rv], "--",
                color="#C00000", linewidth=1.8, alpha=0.8)
        ax.text(x[j] + nm*w + 0.08, rv, f"Real={rv:.2f}",
                color="#C00000", fontsize=11, va="center", fontweight="bold")

    ax.set_xlabel("Patient (Validation Set)", fontsize=14, fontweight="bold")
    ax.set_ylabel("GLCM Contrast", fontsize=14, fontweight="bold")
    ax.set_title(f"Validation Set: GLCM Contrast per Patient ({nm} Models)\n"
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
    print(f"[OK] Validation GLCM bar chart -> {out}")

# ============================================================
#  图3: ΔEI 肺气肿指数偏差柱状图 — 验证集
# ============================================================
def make_dei_bar_val():
    nm = len(MODELS)
    w = 0.7 / nm
    fig, ax = plt.subplots(figsize=(max(11, 3 + nm * 2), 6))
    x = np.arange(len(PATIENTS))

    for i, model in enumerate(MODELS):
        vals = [next((r["L4_Delta_EI_pct"] for r in val_data
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

    ax.set_xlabel("Patient (Validation Set)", fontsize=13)
    ax.set_ylabel("\u0394EI (%)", fontsize=13)
    ax.set_title(f"Validation Set: Emphysema Index Deviation (\u0394EI) — {nm} Models\n(lower = better)",
                 fontweight="bold", fontsize=14)
    ax.set_xticks(x + w * (nm - 1) / 2)
    ax.set_xticklabels(PATIENTS, fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(frameon=True, fancybox=True, fontsize=9, ncol=2 if nm > 4 else 1)
    # 动态 y 上限：若有 DDPM 超出 6%，扩大范围
    max_dei = max(
        (next((r["L4_Delta_EI_pct"] for r in val_data
               if r["model"] == model and r["patient_id"] == p), 0.0)
         for model in MODELS for p in PATIENTS), default=6.0)
    ax.set_ylim(0, max(6.0, max_dei * 1.15))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    out = os.path.join(OUT_DIR, "chart_delta_ei.png")
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

    # 只对比训练集中存在的模型
    train_models_set = set(r["model"] for r in train_data)
    compare_models = [m for m in MODELS if m in train_models_set]
    if not compare_models:
        print("[SKIP] No overlapping models between train and val")
        return

    metrics = [
        ("PSNR (dB) \u2191", "L1_PSNR_dB", True),
        ("MAE (HU) \u2193", "L1_MAE_HU", False),
        ("GLCM-C \u2191", "L2_GLCM_Contrast", True),
        ("W-dist (HU) \u2193", "L3_Wasserstein_HU", False),
        ("\u0394EI (%) \u2193", "L4_Delta_EI_pct", False),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(24, 6))
    fig.suptitle("Training Set (copd_001~003) vs Validation Set (copd_024~029)\n"
                 "Generalization Performance Comparison",
                 fontweight="bold", fontsize=15, y=1.02)

    for ax_idx, (title, key, _higher_better) in enumerate(metrics):
        ax = axes[ax_idx]
        x = np.arange(len(compare_models))
        w = 0.32

        t_vals = [mean_by_model(train_data, m, key) for m in compare_models]
        v_vals = [mean_by_model(val_data, m, key) for m in compare_models]

        bars1 = ax.bar(x - w/2, t_vals, w, label="Train Set",
                       color="#7FB3D8", edgecolor="white", linewidth=0.8)
        bars2 = ax.bar(x + w/2, v_vals, w, label="Val Set",
                       color="#F4A460", edgecolor="white", linewidth=0.8)

        for bar, v in zip(bars1, t_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        for bar, v in zip(bars2, v_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in compare_models], fontsize=9, rotation=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        if ax_idx == 0:
            ax.legend(fontsize=9, frameon=True)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "chart_train_vs_val.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Train vs Val comparison chart -> {out}")

# ============================================================
#  图5: 六模型综合得分横向对比 — L1-L4 加权排名
# ============================================================
def make_composite_score():
    """综合得分 = 归一化(PSNR_inv_rank + MAE_rank + GLCM_score + Wdist_rank + DEI_rank)"""
    keys_higher = ["L1_PSNR_dB", "L2_GLCM_Contrast"]  # 越高越好
    keys_lower = ["L1_MAE_HU", "L3_Wasserstein_HU", "L4_Delta_EI_pct"]  # 越低越好

    model_scores = {}
    for model in MODELS:
        score = 0.0
        n = 0
        for k in keys_higher:
            v = mean_by_model(val_data, model, k)
            if v > 0:
                score += v / max(mean_by_model(val_data, m, k) for m in MODELS)
                n += 1
        for k in keys_lower:
            v = mean_by_model(val_data, model, k)
            best_v = min(mean_by_model(val_data, m, k) for m in MODELS)
            if v > 0:
                score += best_v / v
                n += 1
        model_scores[model] = score / max(n, 1) * 100  # 百分制

    # 按得分排序
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
    ax.set_title(f"Validation Set: Composite L1-L4 Score Ranking ({len(MODELS)} Models)\n"
                 "(higher = better overall performance on unseen data)",
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


# ============================================================
#  图6: GLCM Contrast 分组柱状图 — 验证集（5模型，排除 DDPM）
# ============================================================
def make_glcm_bar_val_no_ddpm():
    models_5 = [m for m in MODELS if m != "ddpm"]
    nm = len(models_5)
    if nm == 0:
        print("[SKIP] No non-DDPM models for GLCM bar chart.")
        return
    w = 0.7 / nm
    fig, ax = plt.subplots(figsize=(max(12, 3 + nm * 2), 7))
    x = np.arange(len(PATIENTS))

    for i, model in enumerate(models_5):
        vals = [next((r["L2_GLCM_Contrast"] for r in val_data
                     if r["model"] == model and r["patient_id"] == p), 0.0)
                for p in PATIENTS]
        bars = ax.bar(x + i * w, vals, w, label=LABELS[model],
                      color=COLORS[model], edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=11,
                    fontweight="bold", rotation=45 if nm > 4 else 0)

    refs = [next((r["Ref_GLCM_C"] for r in val_data if r["patient_id"] == p), 0.0)
            for p in PATIENTS]
    for j, rv in enumerate(refs):
        ax.plot([x[j] - 0.05, x[j] + nm*w + 0.05], [rv, rv], "--",
                color="#C00000", linewidth=1.8, alpha=0.8)
        ax.text(x[j] + nm*w + 0.08, rv, f"Real={rv:.2f}",
                color="#C00000", fontsize=11, va="center", fontweight="bold")

    ax.set_xlabel("Patient (Validation Set)", fontsize=14, fontweight="bold")
    ax.set_ylabel("GLCM Contrast", fontsize=14, fontweight="bold")
    ax.set_title(f"Validation Set: GLCM Contrast per Patient (5 Models excl. DDPM)\n"
                 "(higher = closer to Real COPD)",
                 fontweight="bold", fontsize=16)
    ax.set_xticks(x + w * (nm - 1) / 2)
    ax.set_xticklabels(PATIENTS, fontsize=13)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(frameon=True, fancybox=True, fontsize=10, ncol=2)
    ax.set_ylim(0, max(refs) * 1.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    out = os.path.join(OUT_DIR, "chart_glcm_contrast_no_ddpm.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] GLCM bar chart (no DDPM) -> {out}")


# ============================================================
#  图7: ΔEI 柱状图 — 验证集（5模型，排除 DDPM）
# ============================================================
def make_dei_bar_val_no_ddpm():
    models_5 = [m for m in MODELS if m != "ddpm"]
    nm = len(models_5)
    if nm == 0:
        print("[SKIP] No non-DDPM models for DEI bar chart.")
        return
    w = 0.7 / nm
    fig, ax = plt.subplots(figsize=(max(11, 3 + nm * 2), 6))
    x = np.arange(len(PATIENTS))

    for i, model in enumerate(models_5):
        vals = [next((r["L4_Delta_EI_pct"] for r in val_data
                     if r["model"] == model and r["patient_id"] == p), 0.0)
                for p in PATIENTS]
        bars = ax.bar(x + i * w, vals, w, label=LABELS[model],
                      color=COLORS[model], edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=10,
                    fontweight="bold", rotation=45 if nm > 4 else 0)

    ax.axhline(y=5.0, color="#C00000", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(len(PATIENTS) - 0.3, 5.15, "Clinical Safety Threshold (5%)",
            color="#C00000", fontsize=10, ha="right", fontweight="bold")

    ax.set_xlabel("Patient (Validation Set)", fontsize=13)
    ax.set_ylabel("\u0394EI (%)", fontsize=13)
    ax.set_title(f"Validation Set: \u0394EI per Patient (5 Models excl. DDPM)\n(lower = better)",
                 fontweight="bold", fontsize=14)
    ax.set_xticks(x + w * (nm - 1) / 2)
    ax.set_xticklabels(PATIENTS, fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(frameon=True, fancybox=True, fontsize=9, ncol=2)
    max_dei = max(
        (next((r["L4_Delta_EI_pct"] for r in val_data
               if r["model"] == model and r["patient_id"] == p), 0.0)
         for model in models_5 for p in PATIENTS), default=6.0)
    ax.set_ylim(0, max(6.0, max_dei * 1.18))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    out = os.path.join(OUT_DIR, "chart_delta_ei_no_ddpm.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] DEI bar chart (no DDPM) -> {out}")


# ============================================================
#  图8: 综合得分 — 验证集（5模型，排除 DDPM）
# ============================================================
def make_composite_score_no_ddpm():
    models_5 = [m for m in MODELS if m != "ddpm"]
    if not models_5:
        print("[SKIP] No non-DDPM models for composite score.")
        return
    keys_higher = ["L1_PSNR_dB", "L2_GLCM_Contrast"]
    keys_lower = ["L1_MAE_HU", "L3_Wasserstein_HU", "L4_Delta_EI_pct"]

    model_scores = {}
    for model in models_5:
        score = 0.0
        n = 0
        for k in keys_higher:
            v = mean_by_model(val_data, model, k)
            best = max(mean_by_model(val_data, m, k) for m in models_5)
            if best > 0:
                score += v / best
                n += 1
        for k in keys_lower:
            v = mean_by_model(val_data, model, k)
            best_v = min(mean_by_model(val_data, m, k) for m in models_5)
            if v > 0:
                score += best_v / v
                n += 1
        model_scores[model] = score / max(n, 1) * 100

    sorted_models = sorted(model_scores.keys(), key=lambda m: model_scores[m], reverse=True)
    scores = [model_scores[m] for m in sorted_models]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(range(len(sorted_models)), scores,
                   color=[COLORS[m] for m in sorted_models],
                   edgecolor="white", linewidth=1.2, height=0.6)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{s:.1f}", va="center", fontsize=13, fontweight="bold")

    ax.set_yticks(range(len(sorted_models)))
    ax.set_yticklabels([LABELS[m] for m in sorted_models], fontsize=13)
    ax.set_xlabel("Composite Score (0-100)", fontsize=13, fontweight="bold")
    ax.set_title("Validation Set: Composite L1-L4 Score Ranking (5 Models excl. DDPM)\n"
                 "(higher = better overall performance on unseen data)",
                 fontweight="bold", fontsize=14)
    ax.set_xlim(0, max(scores) * 1.12)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)

    out = os.path.join(OUT_DIR, "chart_composite_score_no_ddpm.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Composite score chart (no DDPM) -> {out}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    make_radar_val_no_ddpm()
    make_radar_val()
    make_glcm_bar_val()
    make_dei_bar_val()
    make_train_vs_val()
    make_composite_score()
    make_glcm_bar_val_no_ddpm()
    make_dei_bar_val_no_ddpm()
    make_composite_score_no_ddpm()
    print("\n[DONE] All validation charts generated.")

