#!/usr/bin/env python3
"""
为验证集三例代表性患者生成"校准前 vs 校准后 vs Real COPD"三重直方图。
用于第四章图4-4/4-5/4-6。

选定患者：
  copd_024 (23,704 voxels) - 小范围局灶性气肿
  copd_025 (66,185 voxels) - 中等规模气肿
  copd_028 (929,827 voxels) - 超大弥漫性气肿
"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAL_TARGET_MEAN = -965.0; CAL_TARGET_STD = 45.0
RAW_SRC_MEAN = -918.0; RAW_SRC_STD = 28.0

PATIENTS = ["copd_024", "copd_025", "copd_028"]

def _setup_fonts():
    available = {f.name for f in fm.fontManager.ttflist}
    serif = 'Times New Roman' if 'Times New Roman' in available else 'DejaVu Serif'
    cjk = 'SimSun' if 'SimSun' in available else 'Microsoft YaHei'
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': [serif, cjk, 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'axes.unicode_minus': False,
    })

def load(path):
    import nibabel as nib
    return nib.load(path).get_fdata(dtype=np.float32)

def reverse_cal(pix):
    raw = pix.copy()
    gm = (raw > -980) & (raw < -820); raw[gm] += 20.0
    sc = np.clip(CAL_TARGET_STD / (RAW_SRC_STD + 1e-6), 0.5, 2.0)
    return (raw - CAL_TARGET_MEAN) / sc + RAW_SRC_MEAN

def gen_histogram(pid):
    fused_path = os.path.join(ROOT,"data","04_final_viz","patchgan",f"{pid}_fused.nii.gz")
    mask_path  = os.path.join(ROOT,"data","03_mapped",pid,f"{pid}_warped_lesion.nii.gz")
    real_path  = os.path.join(ROOT,"data","03_mapped",pid,f"{pid}_warped.nii.gz")
    for p in [fused_path, mask_path, real_path]:
        if not os.path.exists(p):
            print(f"  [SKIP] {p} not found"); return None
    
    fd = load(fused_path); md = load(mask_path); rd = load(real_path)
    idx = md > 0
    fl = fd[idx]; rl = rd[idx]; raw = reverse_cal(fl)
    
    try: plt.style.use('seaborn-v0_8-whitegrid')
    except: pass
    _setup_fonts()
    
    fig = plt.figure(figsize=(16, 8), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.15,
                          left=0.06, right=0.97, top=0.90, bottom=0.10)
    ax_h = fig.add_subplot(gs[0, 0]); ax_t = fig.add_subplot(gs[0, 1])
    
    colors = {'Pre-Cal (AI Raw)': '#4682B4', 'Post-Cal (AI Fused)': '#E74C3C',
              'Real COPD (Ref)': '#27ae60'}
    draw = [('Pre-Cal (AI Raw)', raw), ('Post-Cal (AI Fused)', fl), ('Real COPD (Ref)', rl)]
    stats = {}
    for label, data in draw:
        c = colors[label]; d = data.flatten()
        ax_h.hist(d, bins=80, range=(-1024,0), alpha=0.35, color=c, density=True, histtype='stepfilled')
        ax_h.hist(d, bins=80, range=(-1024,0), alpha=1.0, color=c, density=True, histtype='step',
                  linewidth=2, label=label)
        stats[label] = {'mean': float(np.mean(d)), 'std': float(np.std(d)),
                        'ei': float(np.sum(d<-950)/len(d)*100)}
    
    ax_h.axvline(x=-950, color='#2c3e50', ls='--', lw=2, alpha=0.8, label='Emphysema Threshold (-950)')
    ax_h.set_xlim(-1024, 0); ax_h.set_xlabel("HU Value", fontsize=14, fontweight='bold')
    ax_h.set_ylabel("Density", fontsize=14, fontweight='bold')
    ax_h.set_title("HU Distribution in 3D Lesion (Pre-Cal vs Post-Cal vs Real)", fontsize=16, fontweight='bold')
    ax_h.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax_h.grid(True, alpha=0.3); ax_h.tick_params(labelsize=12)
    
    ax_t.axis('off')
    lines = ["Statistics (Lesion Volume)", "=" * 36, ""]
    for key, hdr in [('Pre-Cal (AI Raw)','Pre-Calibration'),
                     ('Post-Cal (AI Fused)','Post-Calibration'),
                     ('Real COPD (Ref)','Real COPD')]:
        s = stats[key]
        lines += [f"{hdr}:", f"  Mean:    {s['mean']:.1f} HU", f"  EI rate: {s['ei']:.1f}%", ""]
    pre_s = stats['Pre-Cal (AI Raw)']; post_s = stats['Post-Cal (AI Fused)']
    real_s = stats['Real COPD (Ref)']
    lines += ["Calibration Effect:", f"  Mean shift:",
              f"    {pre_s['mean']:.1f} -> {post_s['mean']:.1f}",
              f"  Real ref: {real_s['mean']:.1f}",
              f"  dEI gap:  {abs(post_s['ei']-real_s['ei']):.1f}%"]
    ax_t.text(0.02, 0.865, '\n'.join(lines), transform=ax_t.transAxes,
              fontsize=15, va='top', fontname='Times New Roman', linespacing=1.35)
    
    n_vox = int(np.sum(idx))
    fig.suptitle(f"{pid} (PatchGAN) - Pre-Cal vs Post-Cal vs Real COPD  [{n_vox:,} voxels]",
                 fontsize=18, fontweight='bold', y=0.98)
    
    out_dir = os.path.join(ROOT, "results", "patchgan", pid)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "chart_histogram_precal_vs_postcal.png")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close(fig); plt.style.use('default')
    print(f"  [OK] {out_path}")
    return stats

if __name__ == "__main__":
    for pid in PATIENTS:
        print(f"\n{'='*60}\n  Generating: {pid}\n{'='*60}")
        gen_histogram(pid)

