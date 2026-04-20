#!/usr/bin/env python3
"""
Figure 4-9: multi-model slice comparison with lesion-masked |AI-Real| diff.

Layout: 2 rows x 8 cols
  Row 1 – global axial gray slice (+ red ROI box)
  Row 2 – col 1-2: ROI gray zoom; col 3-8: |AI-Real| jet heatmap on lung BG

Diff style mirrors run_phase3_pipeline.plot_comparison():
  base layer  = Real COPD gray (alpha=0.5)
  overlay     = |AI - Real| * mask, cmap='jet', alpha=0.7
  color range = unified 97.5-pctl x 1.5 across 5 converged models
"""
import os, sys, numpy as np, nibabel as nib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PID = "copd_029"; SLICE_Z = 130; ROI_SIZE = 80
HU_MIN, HU_MAX = -1024, 200

def _p(*parts): return os.path.join(ROOT, *parts)

REF = [
    ("Original COPD", _p("data","01_cleaned","copd_clean",f"{PID}_clean.nii.gz")),
    ("Real COPD",     _p("data","03_mapped",PID,f"{PID}_warped.nii.gz")),
]
MDL = [
    ("U-Net",        _p("data","04_final_viz","unet",f"{PID}_fused.nii.gz")),
    ("PartialConv",  _p("data","04_final_viz","partial_conv",f"{PID}_fused.nii.gz")),
    ("PatchGAN",     _p("data","04_final_viz","patchgan",f"{PID}_fused.nii.gz")),
    ("AttGAN",       _p("data","04_final_viz","attgan",f"{PID}_fused.nii.gz")),
    ("MAE-PatchGAN", _p("data","04_final_viz","mae_patchgan",f"{PID}_fused.nii.gz")),
    ("DDPM",         _p("data","04_final_viz","ddpm",f"{PID}_fused.nii.gz")),
]
ALL = REF + MDL
MASK = _p("data","03_mapped",PID,f"{PID}_warped_lesion.nii.gz")
OUT  = _p("results","charts_val","chart_slice_comparison_copd029.png")

def ld(path, z):
    v = nib.load(path).get_fdata(dtype=np.float32)
    if v.ndim == 4: v = v[...,0]
    return v[:,:,min(z, v.shape[2]-1)]

def lw(img):
    return np.clip((img - HU_MIN)/(HU_MAX - HU_MIN), 0, 1)

def roi_box(msk):
    ys,xs = np.where(msk > 0)
    if len(ys)==0: cy,cx = msk.shape[0]//2, msk.shape[1]//2
    else: cy,cx = int(np.median(ys)), int(np.median(xs))
    h = ROI_SIZE//2; H,W = msk.shape
    return max(cy-h,0), min(cy+h,H), max(cx-h,0), min(cx+h,W)

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for n,p in ALL:
        if not os.path.exists(p): print(f"[ERR] {p}"); sys.exit(1)
    S = {n: ld(p, SLICE_Z) for n,p in ALL}
    ms = ld(MASK, SLICE_Z) if os.path.exists(MASK) else None
    mb = (ms > 0) if ms is not None else None
    R = S["Real COPD"]
    y0,y1,x0,x1 = roi_box(ms) if ms is not None else (
        R.shape[0]//4, 3*R.shape[0]//4, R.shape[1]//4, 3*R.shape[1]//4)

    DM = {}; vl = []
    for n,_ in MDL:
        d = np.abs(S[n].astype(float) - R.astype(float))
        if mb is not None: d *= mb.astype(float)
        DM[n] = d
        nz = d[mb][d[mb]>0] if mb is not None else d[d>0]
        if nz.size>0: vl.append(float(np.percentile(nz, 97.5))*1.5)
    vm = max(max(vl[:5], default=50), 10.0)

    nc = len(ALL)
    fig, ax = plt.subplots(2, nc, figsize=(nc*2.6, 2*2.6+1.4),
                           gridspec_kw={"hspace":0.28,"wspace":0.06})
    li = None
    for j,(n,_) in enumerate(ALL):
        s = S[n]; im = j >= len(REF)
        at,ab = ax[0,j], ax[1,j]
        at.imshow(lw(s).T, cmap="gray", origin="lower", aspect="equal")
        at.add_patch(Rectangle((x0,y0), x1-x0, y1-y0,
                     lw=1.8, edgecolor="red", facecolor="none"))
        at.set_title(n, fontsize=9, fontweight="bold", pad=4); at.axis("off")
        if not im:
            ab.imshow(lw(s[x0:x1,y0:y1]).T, cmap="gray", origin="lower",
                      aspect="equal")
            ab.set_title(f"ROI: {n}", fontsize=7, fontweight="bold",
                         pad=3, color="#333")
        else:
            ab.imshow(lw(R[x0:x1,y0:y1]).T, cmap="gray", alpha=0.5,
                      origin="lower", aspect="equal")
            li = ab.imshow(DM[n][x0:x1,y0:y1].T, cmap="jet", alpha=0.7,
                           origin="lower", aspect="equal", vmin=0, vmax=vm)
            ab.set_title(f"|AI-Real|: {n}", fontsize=7, fontweight="bold",
                         pad=3, color="#CC0000")
        for sp in ab.spines.values():
            sp.set_edgecolor("red" if im else "#666")
            sp.set_linewidth(1.2); sp.set_visible(True)
        ab.set_xticks([]); ab.set_yticks([])

    ax[0,0].text(-0.22,0.5,"Global",transform=ax[0,0].transAxes,
                 fontsize=10,fontweight="bold",rotation=90,va="center",ha="center")
    ax[1,0].text(-0.22,0.5,"ROI/Diff",transform=ax[1,0].transAxes,
                 fontsize=10,fontweight="bold",rotation=90,va="center",ha="center")
    if li:
        ca = fig.add_axes([0.92,0.08,0.012,0.35])
        cb = fig.colorbar(li, cax=ca)
        cb.set_label("|AI - Real| (HU)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"Validation {PID} Axial Z={SLICE_Z}  "
                 f"(Row2 Col3-8: Lesion |AI-Real| on Lung BG)",
                 fontsize=11, fontweight="bold", y=0.99)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[DONE] {OUT}  (vmax={vm:.1f} HU)")

if __name__ == "__main__":
    main()

