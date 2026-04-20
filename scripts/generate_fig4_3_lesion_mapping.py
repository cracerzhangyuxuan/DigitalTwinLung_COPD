"""
图4-3 病灶空间映射过程示意图
左侧: COPD患者原始CT空间中的病灶掩膜(红色叠加在肺CT上)
中间: SyN配准产生的非线性变形场(以彩色位移向量场可视化)
右侧: 映射到标准模板空间后的病灶掩膜(经模板肺掩膜约束后的最终结果)
箭头表示变换方向, 标注变换前后的体素数量和保留率
"""
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# === 学术字体配置: 硬编码系统字体路径, 避免CJK fallback失败 ===
_FP_CN = None
_FP_EN = None
def _setup_fonts():
    global _FP_CN, _FP_EN
    cn_paths = ['C:/Windows/Fonts/simsun.ttc', 'C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc']
    for p in cn_paths:
        if Path(p).exists():
            _FP_CN = fm.FontProperties(fname=p); break
    if _FP_CN is None:
        _FP_CN = fm.FontProperties(family='sans-serif')
    en_paths = ['C:/Windows/Fonts/times.ttf', 'C:/Windows/Fonts/TIMES.TTF']
    for p in en_paths:
        if Path(p).exists():
            _FP_EN = fm.FontProperties(fname=p); break
    if _FP_EN is None:
        _FP_EN = fm.FontProperties(family='serif')
    matplotlib.rcParams['axes.unicode_minus'] = False
_setup_fonts()

def fp_cn(size=12, weight='normal'):
    fp = _FP_CN.copy(); fp.set_size(size); fp.set_weight(weight); return fp
def fp_en(size=12, weight='normal'):
    fp = _FP_EN.copy(); fp.set_size(size); fp.set_weight(weight); return fp

# === 选择验证集患者 copd_029 ===
PID = "copd_029"
BASE = Path(r"D:\DigitalTwinLung_COPD")
ct_orig = nib.load(BASE / f"data/01_cleaned/copd_clean/{PID}_clean.nii.gz").get_fdata()
mask_orig = nib.load(BASE / f"data/01_cleaned/copd_emphysema/{PID}_emphysema.nii.gz").get_fdata()
warp_field = nib.load(BASE / f"data/03_mapped/{PID}/{PID}_transform_0.nii.gz").get_fdata()
mask_warped = nib.load(BASE / f"data/03_mapped/{PID}/{PID}_warped_lesion.nii.gz").get_fdata()
template = nib.load(BASE / "data/02_atlas/standard_template.nii.gz").get_fdata()
template_mask = nib.load(BASE / "data/02_atlas/standard_mask.nii.gz").get_fdata()

# 体素统计
voxels_before = int(np.sum(mask_orig > 0.5))
voxels_after = int(np.sum(mask_warped > 0.5))
retention = voxels_after / voxels_before * 100 if voxels_before > 0 else 0
print(f"  {PID}: before={voxels_before:,}  after={voxels_after:,}  retention={retention:.1f}%")

# 选取冠状面切片 (病灶最多的切片)
lesion_counts_orig = np.sum(mask_orig > 0.5, axis=(0, 2))
slice_y_orig = int(np.argmax(lesion_counts_orig))
lesion_counts_warped = np.sum(mask_warped > 0.5, axis=(0, 2))
slice_y_warped = int(np.argmax(lesion_counts_warped))

# === 绘图: 使用GridSpec在列间预留箭头空间 ===
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(21, 7))
gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 0.18, 1, 0.18, 1],
              wspace=0.02, left=0.02, right=0.98, top=0.88, bottom=0.06)

# --- 左图: 患者原始CT + 病灶掩膜(红色, 高对比) ---
ax = fig.add_subplot(gs[0, 0])
ct_slice = ct_orig[:, slice_y_orig, :]
mask_slice = mask_orig[:, slice_y_orig, :]
ax.imshow(ct_slice.T, cmap='gray', origin='lower', vmin=-1024, vmax=200, aspect='equal')
# 使用纯红色叠加, alpha提高, 让小病灶更醒目
from matplotlib.colors import ListedColormap
red_cmap = ListedColormap([[1, 0, 0, 0.75]])  # 纯红色, 高不透明度
lesion_overlay = np.ma.masked_where(mask_slice.T < 0.5, np.ones_like(mask_slice.T))
ax.imshow(lesion_overlay, cmap=red_cmap, origin='lower', aspect='equal')
ax.set_title("COPD患者原始CT空间\n病灶掩膜（红色叠加）", fontproperties=fp_cn(14, 'bold'), pad=10)
ax.axis('off')

# --- 箭头区1: 左→中 (箭头在下1/4处, 文字在箭头正下方) ---
ax_arr1 = fig.add_subplot(gs[0, 1])
ax_arr1.set_xlim(0, 1); ax_arr1.set_ylim(0, 1); ax_arr1.axis('off')
ax_arr1.annotate('', xy=(0.95, 0.25), xytext=(0.05, 0.25),
    arrowprops=dict(arrowstyle='->', color='#333333', lw=3.5, mutation_scale=32))
ax_arr1.text(0.5, 0.10, "SyN\n配准", ha='center', va='top',
             fontproperties=fp_cn(18, 'bold'), color='#333333')

# --- 中图: 变形场可视化 ---
ax = fig.add_subplot(gs[0, 2])
wf = np.squeeze(warp_field)
mid_y = wf.shape[1] // 2
disp = wf[:, mid_y, :, :]
magnitude = np.sqrt(disp[:, :, 0]**2 + disp[:, :, 1]**2 + disp[:, :, 2]**2)
im = ax.imshow(magnitude.T, cmap='jet', origin='lower', aspect='equal',
               vmin=0, vmax=np.percentile(magnitude, 99))
step = 18
xs = np.arange(0, disp.shape[0], step)
zs = np.arange(0, disp.shape[1], step)
XX, ZZ = np.meshgrid(xs, zs)
U = disp[XX, ZZ, 0]; V = disp[XX, ZZ, 2]
mag_sampled = magnitude[XX, ZZ]
valid = mag_sampled > 0.5
U_masked = np.where(valid, U, 0); V_masked = np.where(valid, V, 0)
ax.quiver(XX, ZZ, U_masked, V_masked, color='white', alpha=0.6,
          scale=None, scale_units='xy', angles='xy', width=0.003, headwidth=3)
ax.set_title("SyN非线性变形场\n（位移向量场可视化）", fontproperties=fp_cn(14, 'bold'), pad=10)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
cbar.set_label("位移幅度 (mm)", fontproperties=fp_cn(11))
ax.axis('off')

# --- 箭头区2: 中→右 (箭头在下1/4处, 文字在箭头正下方) ---
ax_arr2 = fig.add_subplot(gs[0, 3])
ax_arr2.set_xlim(0, 1); ax_arr2.set_ylim(0, 1); ax_arr2.axis('off')
ax_arr2.annotate('', xy=(0.95, 0.25), xytext=(0.05, 0.25),
    arrowprops=dict(arrowstyle='->', color='#333333', lw=3.5, mutation_scale=32))
ax_arr2.text(0.5, 0.10, "空间\n映射", ha='center', va='top',
             fontproperties=fp_cn(18, 'bold'), color='#333333')

# --- 右图: 模板空间 + 映射后病灶掩膜(红色, 高对比) ---
ax = fig.add_subplot(gs[0, 4])
tmpl_slice = template[:, slice_y_warped, :]
mask_w_slice = mask_warped[:, slice_y_warped, :]
ax.imshow(tmpl_slice.T, cmap='gray', origin='lower', vmin=-1024, vmax=200, aspect='equal')
warped_overlay = np.ma.masked_where(mask_w_slice.T < 0.5, np.ones_like(mask_w_slice.T))
ax.imshow(warped_overlay, cmap=red_cmap, origin='lower', aspect='equal')
ax.set_title("标准模板空间\n映射后病灶掩膜（模板肺掩膜约束）", fontproperties=fp_cn(14, 'bold'), pad=10)
ax.axis('off')

out_path = BASE / "results" / "charts_val" / "fig4_3_lesion_mapping.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] 图4-3已保存: {out_path}")
print(f"  患者: {PID}")
print(f"  变换前体素: {voxels_before:,}")
print(f"  变换后体素: {voxels_after:,}")
print(f"  保留率: {retention:.1f}%")

