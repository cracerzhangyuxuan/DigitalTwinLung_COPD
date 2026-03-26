#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证集 L1-L4 四层评估脚本

在未参与训练的验证集患者 (copd_024, copd_025, copd_026) 上，
对 U-Net / PartialConv / PatchGAN 三种模型计算 L1~L4 核心指标。

输出:
  results/validation_metrics.json   — 结构化指标数据（不覆盖训练集指标）
  results/validation_metrics.csv    — 平铺表格
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
PATIENTS = ["copd_024", "copd_025", "copd_026"]  # 验证集（未参与训练）
FUSED_DIR = ROOT / "data" / "04_final_viz"
MAPPED_DIR = ROOT / "data" / "03_mapped"
TEMPLATE_PATH = ROOT / "data" / "02_atlas" / "standard_template_with_airway.nii.gz"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 复用 evaluate_l1_l4_metrics.py 的计算函数 ──────────────
def load_vol(path):
    return nib.load(str(path)).get_fdata()

def compute_psnr(real, fused, dynamic_range=1400.0):
    mse = np.mean((real - fused) ** 2)
    if mse < 1e-12:
        return float("inf")
    return 20.0 * np.log10(dynamic_range / np.sqrt(mse))

def compute_mae(real, fused):
    return float(np.mean(np.abs(real - fused)))

def compute_psd_hf_ratio(image_2d, mask_2d, cutoff_ratio=0.3):
    coords = np.argwhere(mask_2d > 0)
    if len(coords) < 4:
        return 0.0
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    crop = image_2d[y0:y1+1, x0:x1+1]
    if crop.size == 0:
        return 0.0
    f2 = np.fft.fft2(crop)
    psd = np.abs(np.fft.fftshift(f2)) ** 2
    cy, cx = psd.shape[0] // 2, psd.shape[1] // 2
    Y, X = np.ogrid[:psd.shape[0], :psd.shape[1]]
    r = np.sqrt((Y - cy)**2 + (X - cx)**2)
    r_max = min(cy, cx)
    hf = psd[r > r_max * cutoff_ratio].sum()
    total = psd.sum()
    if total < 1e-12:
        return 0.0
    return float(hf / total)

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
    print("  验证集 L1-L4 四层评估 (copd_024 ~ copd_026)")
    print("  数据口径: 未参与训练的验证集样本")
    print("=" * 70)

    # 加载标准模板
    template_data = load_vol(TEMPLATE_PATH)
    print(f"[INFO] 模板已加载: {TEMPLATE_PATH.name}, shape={template_data.shape}")

    all_results = []

    for pid in PATIENTS:
        real_path = MAPPED_DIR / pid / f"{pid}_warped.nii.gz"
        mask_path = MAPPED_DIR / pid / f"{pid}_warped_lesion.nii.gz"
        if not real_path.exists() or not mask_path.exists():
            print(f"[WARN] 跳过 {pid}: 缺少 real/mask 文件")
            continue

        real_vol = load_vol(real_path)
        mask_vol = load_vol(mask_path)
        mask_bool = mask_vol > 0
        real_lesion = real_vol[mask_bool]

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
                "dataset": "validation",
                "L1_PSNR_dB": round(psnr, 2),
                "L1_MAE_HU": round(mae, 2),
                "L2_PSD_HF_Ratio": round(psd_hf, 6),
                "L2_Sobel_Median": round(sobel_med, 2),
                "L2_GLCM_Contrast": round(glcm_c, 4),
                "L3_Wasserstein_HU": round(w_dist, 2),
                "L3_LV_CoV": round(lv_cov, 4),
                "L4_EI_Fused": round(ei_fused, 4),
                "L4_Delta_EI_pct": round(delta_ei, 2),
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

    # ── 保存 JSON（独立文件，不覆盖训练集指标）───────────────
    json_path = OUTPUT_DIR / "validation_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] JSON -> {json_path}")

    # ── 保存 CSV ────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "validation_metrics.csv"
    if all_results:
        keys = all_results[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_results)
    print(f"[SAVE] CSV  -> {csv_path}")

    # ── 汇总均值 ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  验证集三模型均值汇总")
    print("=" * 70)
    for model in MODELS:
        rows = [r for r in all_results if r["model"] == model]
        if not rows:
            continue
        n = len(rows)
        avg = {k: sum(r[k] for r in rows) / n
               for k in rows[0] if k not in ("patient_id", "model", "dataset")}
        print(f"\n  [{model}] (N={n})")
        print(f"    L1  PSNR={avg['L1_PSNR_dB']:.2f} dB   MAE={avg['L1_MAE_HU']:.2f} HU")
        print(f"    L2  PSD-HF={avg['L2_PSD_HF_Ratio']:.6f}   "
              f"Sobel={avg['L2_Sobel_Median']:.2f}   GLCM-C={avg['L2_GLCM_Contrast']:.4f}")
        print(f"    L3  W-dist={avg['L3_Wasserstein_HU']:.2f} HU   LV-CoV={avg['L3_LV_CoV']:.4f}")
        print(f"    L4  ΔEI={avg['L4_Delta_EI_pct']:.2f}%")
        print(f"    Ref PSD-HF={avg['Ref_PSD_HF']:.6f}   Sobel={avg['Ref_Sobel']:.2f}   "
              f"GLCM-C={avg['Ref_GLCM_C']:.4f}   LV-CoV={avg['Ref_LV_CoV']:.4f}")

    # ── 训练集 vs 验证集对比 ────────────────────────────────
    train_json = OUTPUT_DIR / "model_comparison_metrics.json"
    if train_json.exists():
        with open(train_json, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        # 只取有实际数据的条目（排除 null 占位）
        train_data = [r for r in train_data if r.get("L1_PSNR_dB") is not None]

        print("\n" + "=" * 70)
        print("  训练集 vs 验证集 对比")
        print("=" * 70)
        print(f"  {'模型':<15s} | {'数据集':<8s} | {'PSNR':>7s} | {'MAE':>7s} | "
              f"{'W-dist':>7s} | {'ΔEI%':>7s} | {'GLCM-C':>7s}")
        print(f"  {'-'*15} | {'-'*8} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7}")

        for model in MODELS:
            # 训练集均值
            t_rows = [r for r in train_data if r["model"] == model]
            v_rows = [r for r in all_results if r["model"] == model]
            if t_rows:
                tn = len(t_rows)
                t_avg = {k: sum(r[k] for r in t_rows) / tn
                         for k in ["L1_PSNR_dB", "L1_MAE_HU", "L3_Wasserstein_HU",
                                   "L4_Delta_EI_pct", "L2_GLCM_Contrast"]}
                print(f"  {model:<15s} | {'训练集':<8s} | {t_avg['L1_PSNR_dB']:7.2f} | "
                      f"{t_avg['L1_MAE_HU']:7.2f} | {t_avg['L3_Wasserstein_HU']:7.2f} | "
                      f"{t_avg['L4_Delta_EI_pct']:7.2f} | {t_avg['L2_GLCM_Contrast']:7.4f}")
            if v_rows:
                vn = len(v_rows)
                v_avg = {k: sum(r[k] for r in v_rows) / vn
                         for k in ["L1_PSNR_dB", "L1_MAE_HU", "L3_Wasserstein_HU",
                                   "L4_Delta_EI_pct", "L2_GLCM_Contrast"]}
                print(f"  {model:<15s} | {'验证集':<8s} | {v_avg['L1_PSNR_dB']:7.2f} | "
                      f"{v_avg['L1_MAE_HU']:7.2f} | {v_avg['L3_Wasserstein_HU']:7.2f} | "
                      f"{v_avg['L4_Delta_EI_pct']:7.2f} | {v_avg['L2_GLCM_Contrast']:7.4f}")
            print()

    print("\n[DONE] 验证集评估完成")
    return all_results


if __name__ == "__main__":
    main()

