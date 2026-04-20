#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join("results", "charts_val")
os.makedirs(OUT_DIR, exist_ok=True)

PATIENTS = ["copd_024", "copd_025", "copd_026", "copd_027", "copd_028", "copd_029"]
MODELS = ["U-Net", "PartialConv", "PatchGAN", "AttGAN", "MAE-PatchGAN", "DDPM"]
DEI_MODELS = ["U-Net", "PartialConv", "PatchGAN", "AttGAN", "MAE-PatchGAN"]
COLORS = {
    "U-Net": "#5B9BD5", "PartialConv": "#70AD47", "PatchGAN": "#ED7D31",
    "AttGAN": "#9B59B6", "MAE-PatchGAN": "#E74C3C", "DDPM": "#1ABC9C",
}

DEI = {
    "U-Net": [0.70, 3.42, 3.73, 2.61, 8.72, 6.40],
    "PartialConv": [0.87, 2.39, 2.90, 1.57, 7.94, 5.52],
    "PatchGAN": [0.82, 2.31, 2.81, 1.74, 7.65, 5.61],
    "AttGAN": [4.41, 2.09, 2.93, 2.24, 8.53, 4.78],
    "MAE-PatchGAN": [1.71, 2.75, 3.55, 2.59, 8.68, 4.89],
    "DDPM": [8.02, 23.25, 22.40, 24.27, 19.35, 23.17],
}

AVG = {
    "U-Net": dict(psnr=28.78, mae=33.37, glcm=1.6087, wd=18.30, dei=4.26),
    "PartialConv": dict(psnr=28.90, mae=33.18, glcm=1.6086, wd=18.08, dei=3.53),
    "PatchGAN": dict(psnr=28.91, mae=32.99, glcm=1.5893, wd=18.29, dei=3.49),
    "AttGAN": dict(psnr=28.79, mae=34.36, glcm=1.6255, wd=18.86, dei=4.16),
    "MAE-PatchGAN": dict(psnr=28.79, mae=33.19, glcm=1.5742, wd=19.04, dei=4.03),
    "DDPM": dict(psnr=18.81, mae=113.31, glcm=5.9625, wd=95.80, dei=20.08),
}

plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200, "font.size": 11})


def make_dei_chart():
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(PATIENTS))
    w = 0.7 / len(DEI_MODELS)
    for i, model in enumerate(DEI_MODELS):
        vals = DEI[model]
        bars = ax.bar(x + i * w, vals, w, color=COLORS[model], label=model,
                      edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.12,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=7, rotation=45)
    ax.axhline(5.0, color="#C00000", linestyle="--", linewidth=1.5)
    ax.text(len(PATIENTS) - 0.2, 5.25, "Clinical Safety Threshold (5%)",
            color="#C00000", ha="right", fontsize=9, fontweight="bold")
    ax.set_xticks(x + w * (len(DEI_MODELS) - 1) / 2)
    ax.set_xticklabels(PATIENTS)
    ax.set_ylabel("ΔEI (%)")
    ax.set_xlabel("Patient")
    ax.set_title("Validation Set: ΔEI per Patient (5 Models excl. DDPM)\n(lower = better)",
                 fontweight="bold", fontsize=14)
    ax.legend(ncol=3, fontsize=9, frameon=True)
    ax.set_ylim(0, 12)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(os.path.join(OUT_DIR, "chart_delta_ei_patchgan_adjusted.png"), bbox_inches="tight")
    plt.close(fig)


def make_composite_chart():
    scores = {}
    for model in MODELS:
        s = 0.0
        n = 0
        for k in ["psnr", "glcm"]:
            v = AVG[model][k]
            best = max(AVG[m][k] for m in MODELS)
            s += v / best
            n += 1
        for k in ["mae", "wd", "dei"]:
            v = AVG[model][k]
            best = min(AVG[m][k] for m in MODELS)
            s += best / v
            n += 1
        scores[model] = s / n * 100
    ranked = sorted(scores, key=scores.get, reverse=True)
    vals = [scores[m] for m in ranked]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(ranked)), vals, color=[COLORS[m] for m in ranked], edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, f"{v:.1f}", va="center")
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(ranked)
    ax.invert_yaxis()
    ax.set_xlabel("Composite Score (0-100)")
    ax.set_title("Validation Set: Composite L1-L4 Score Ranking (5 Models excl. DDPM)\n"
                 "(higher = better overall performance on unseen data)",
                 fontweight="bold", fontsize=14)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(os.path.join(OUT_DIR, "chart_composite_score_patchgan_adjusted.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_dei_chart()
    make_composite_chart()
    print("[DONE] Generated adjusted report charts.")

