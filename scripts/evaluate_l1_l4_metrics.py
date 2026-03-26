#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三模型四层评估脚本 (L1-L4)

针对 U-Net / PartialConv / PatchGAN 三种生成模型，
计算精选的 L1~L4 核心指标，验证理论预期。

输出:
  results/model_comparison_metrics.json   — 结构化指标数据
  results/model_comparison_metrics.csv    — 平铺表格
  results/model_comparison_report.md      — 实验报告
"""

import sys, json, csv
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage, stats
from skimage.feature import graycomatrix, graycoprops

# ── 项目根目录 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── 路径常量 ────────────────────────────────────────────────
MODELS = ["unet", "partial_conv", "patchgan"]
PATIENTS = ["copd_001", "copd_002", "copd_003"]
FUSED_DIR = ROOT / "data" / "04_final_viz"
MAPPED_DIR = ROOT / "data" / "03_mapped"
TEMPLATE_PATH = ROOT / "data" / "02_atlas" / "standard_template_with_airway.nii.gz"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 辅助函数 ────────────────────────────────────────────────

def load_vol(path):
    return nib.load(str(path)).get_fdata()

def compute_psnr(real, fused, dynamic_range=1400.0):
    mse = np.mean((real - fused) ** 2)
    if mse < 1e-12:
        return float("inf")
    return 20.0 * np.log10(dynamic_range / np.sqrt(mse))

def compute_mae(real, fused):
    return float(np.mean(np.abs(real - fused)))

def compute_psd_hf_ratio(image_2d, mask_2d, hf_threshold=0.25):
    """PSD 高频功率占比: f > hf_threshold * f_Nyquist 的功率 / 总功率"""
    roi = image_2d.copy()
    roi[mask_2d == 0] = 0
    coords = np.argwhere(mask_2d > 0)
    if len(coords) < 16:
        return 0.0
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    crop = roi[y0:y1+1, x0:x1+1]
    if crop.size < 16:
        return 0.0
    fft2 = np.fft.fft2(crop - crop.mean())
    psd2 = np.abs(np.fft.fftshift(fft2)) ** 2
    cy, cx = psd2.shape[0] // 2, psd2.shape[1] // 2
    Y, X = np.ogrid[:psd2.shape[0], :psd2.shape[1]]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    f_nyquist = min(cy, cx)
    if f_nyquist < 1:
        return 0.0
    total = psd2.sum()
    hf = psd2[r > hf_threshold * f_nyquist].sum()
    return float(hf / (total + 1e-12))

def compute_sobel_median(image_2d, mask_2d):
    sx = ndimage.sobel(image_2d, axis=0)
    sy = ndimage.sobel(image_2d, axis=1)
    mag = np.sqrt(sx**2 + sy**2)
    vals = mag[mask_2d > 0]
    if len(vals) == 0:
        return 0.0
    return float(np.median(vals))

def compute_glcm_contrast(image_2d, mask_2d):
    coords = np.argwhere(mask_2d > 0)
    if len(coords) < 4:
        return 0.0
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    crop = image_2d[y0:y1+1, x0:x1+1]
    if crop.size == 0 or crop.max() == crop.min():
        return 0.0
    normed = (crop - (-1000)) / 1000.0
    ci = (normed * 31).astype(np.uint8)
    ci = np.clip(ci, 0, 31)
    glcm = graycomatrix(ci, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=32, symmetric=True, normed=True)
    return float(graycoprops(glcm, "contrast").mean())

def compute_wasserstein(real_hu, fused_hu):
    return float(stats.wasserstein_distance(real_hu, fused_hu))

def compute_local_var_cov(volume_3d, mask_3d, win=5):
    """局部方差的变异系数 CoV = std(LV) / mean(LV)"""
    from scipy.ndimage import uniform_filter
    m = mask_3d > 0
    if m.sum() < win**3:
        return 0.0
    v = volume_3d.astype(np.float64)
    local_mean = uniform_filter(v, size=win)
    local_sq = uniform_filter(v**2, size=win)
    lv = local_sq - local_mean**2
    lv = np.clip(lv, 0, None)
    vals = lv[m]
    mean_lv = vals.mean()
    if mean_lv < 1e-12:
        return 0.0
    return float(vals.std() / mean_lv)

def compute_ei(volume, mask):
    m = mask > 0
    lesion_hu = volume[m]
    if len(lesion_hu) == 0:
        return 0.0
    return float((lesion_hu < -950).sum() / len(lesion_hu))

def get_best_slice(mask_3d):
    sums = mask_3d.sum(axis=(0, 1))
    return int(np.argmax(sums))


# ── 主逻辑 ──────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  三模型四层评估 (L1-L4)")
    print("=" * 70)

    # 加载标准模板（用于 Direct Warp 基线参考）
    template_data = load_vol(TEMPLATE_PATH)
    print(f"[INFO] 模板已加载: {TEMPLATE_PATH.name}, shape={template_data.shape}")

    all_results = []

    for pid in PATIENTS:
        # 加载 Real COPD + Mask
        real_path = MAPPED_DIR / pid / f"{pid}_warped.nii.gz"
        mask_path = MAPPED_DIR / pid / f"{pid}_warped_lesion.nii.gz"
        if not real_path.exists() or not mask_path.exists():
            print(f"[WARN] 跳过 {pid}: 缺少 real/mask 文件")
            continue

        real_vol = load_vol(real_path)
        mask_vol = load_vol(mask_path)
        mask_bool = mask_vol > 0
        real_lesion = real_vol[mask_bool]

        # 获取最佳切片（用于 2D 指标）
        best_z = get_best_slice(mask_vol)
        real_slice = real_vol[:, :, best_z]
        mask_slice = (mask_vol[:, :, best_z] > 0).astype(np.float64)

        # Real COPD 自身的 L2-L3 参照值
        ref_psd_hf = compute_psd_hf_ratio(real_slice, mask_slice)
        ref_sobel = compute_sobel_median(real_slice, mask_slice)
        ref_glcm_c = compute_glcm_contrast(real_slice, mask_slice)
        ref_lv_cov = compute_local_var_cov(real_vol, mask_vol)
        ref_ei = compute_ei(real_vol, mask_vol)

        print(f"\n{'─'*60}")
        print(f"  患者: {pid}  |  病灶体素: {int(mask_bool.sum()):,}  |  最佳 Z={best_z}")
        print(f"  Real COPD 参照  PSD-HF={ref_psd_hf:.4f}  Sobel={ref_sobel:.2f}"
              f"  GLCM-C={ref_glcm_c:.4f}  LV-CoV={ref_lv_cov:.4f}  EI={ref_ei:.4f}")

        for model in MODELS:
            fused_path = FUSED_DIR / model / f"{pid}_fused.nii.gz"
            if not fused_path.exists():
                print(f"  [WARN] 跳过 {model}/{pid}: 缺少 fused 文件")
                continue

            fused_vol = load_vol(fused_path)
            fused_lesion = fused_vol[mask_bool]
            fused_slice = fused_vol[:, :, best_z]

            # ── L1: 基础准确率 ──
            psnr = compute_psnr(real_lesion, fused_lesion)
            mae = compute_mae(real_lesion, fused_lesion)

            # ── L2: 纹理结构 ──
            psd_hf = compute_psd_hf_ratio(fused_slice, mask_slice)
            sobel_med = compute_sobel_median(fused_slice, mask_slice)
            glcm_c = compute_glcm_contrast(fused_slice, mask_slice)

            # ── L3: 感知真实度 ──
            w_dist = compute_wasserstein(real_lesion, fused_lesion)
            lv_cov = compute_local_var_cov(fused_vol, mask_vol)

            # ── L4: 临床有效性 ──
            ei_fused = compute_ei(fused_vol, mask_vol)
            delta_ei = abs(ei_fused - ref_ei) / (ref_ei + 1e-10) * 100.0

            row = {
                "patient_id": pid,
                "model": model,
                # L1
                "L1_PSNR_dB": round(psnr, 2),
                "L1_MAE_HU": round(mae, 2),
                # L2
                "L2_PSD_HF_Ratio": round(psd_hf, 6),
                "L2_Sobel_Median": round(sobel_med, 2),
                "L2_GLCM_Contrast": round(glcm_c, 4),
                # L3
                "L3_Wasserstein_HU": round(w_dist, 2),
                "L3_LV_CoV": round(lv_cov, 4),
                # L4
                "L4_EI_Fused": round(ei_fused, 4),
                "L4_Delta_EI_pct": round(delta_ei, 2),
                # Ref
                "Ref_PSD_HF": round(ref_psd_hf, 6),
                "Ref_Sobel": round(ref_sobel, 2),
                "Ref_GLCM_C": round(ref_glcm_c, 4),
                "Ref_LV_CoV": round(ref_lv_cov, 4),
                "Ref_EI": round(ref_ei, 4),
            }
            all_results.append(row)
            print(f"  [{model:>13s}]  PSNR={psnr:6.2f}  MAE={mae:5.2f}  "
                  f"PSD-HF={psd_hf:.4f}  Sobel={sobel_med:6.2f}  "
                  f"GLCM-C={glcm_c:.4f}  W-dist={w_dist:5.2f}  "
                  f"LV-CoV={lv_cov:.4f}  ΔEI={delta_ei:.2f}%")

    # ── 保存 JSON ───────────────────────────────────────────
    json_path = OUTPUT_DIR / "model_comparison_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] JSON -> {json_path}")

    # ── 保存 CSV ────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "model_comparison_metrics.csv"
    if all_results:
        keys = all_results[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_results)
    print(f"[SAVE] CSV  -> {csv_path}")

    # ── 汇总均值 ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  三模型均值汇总")
    print("=" * 70)
    for model in MODELS:
        rows = [r for r in all_results if r["model"] == model]
        if not rows:
            continue
        n = len(rows)
        avg = {k: sum(r[k] for r in rows) / n
               for k in rows[0] if k not in ("patient_id", "model")}
        print(f"\n  [{model}] (N={n})")
        print(f"    L1  PSNR={avg['L1_PSNR_dB']:.2f} dB   MAE={avg['L1_MAE_HU']:.2f} HU")
        print(f"    L2  PSD-HF={avg['L2_PSD_HF_Ratio']:.6f}   "
              f"Sobel={avg['L2_Sobel_Median']:.2f}   GLCM-C={avg['L2_GLCM_Contrast']:.4f}")
        print(f"    L3  W-dist={avg['L3_Wasserstein_HU']:.2f} HU   LV-CoV={avg['L3_LV_CoV']:.4f}")
        print(f"    L4  ΔEI={avg['L4_Delta_EI_pct']:.2f}%")
        print(f"    Ref PSD-HF={avg['Ref_PSD_HF']:.6f}   Sobel={avg['Ref_Sobel']:.2f}   "
              f"GLCM-C={avg['Ref_GLCM_C']:.4f}   LV-CoV={avg['Ref_LV_CoV']:.4f}")

    print("\n[DONE] 评估完成")
    return all_results


if __name__ == "__main__":
    main()

