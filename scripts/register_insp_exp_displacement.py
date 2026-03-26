#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
吸气相 ↔ 呼气相肺模板配准 & 位移场可视化
==========================================

功能:
    1. ANTs SyN 非刚性配准：吸气相 (fixed) → 呼气相 (moving)
    2. 配准前后三视图对比（Axial / Coronal / Sagittal, 2×3）
    3. 3D 肺叶表面 + 位移场箭头向量可视化

输出:
    results/displacement/
        ├── registration_triview.png        # 配准前后三视图
        ├── displacement_3d_quiver.png      # 3D 位移场箭头图
        ├── warp_field.nii.gz               # 原始形变场
        └── displacement_field.nii.gz       # 位移场 (mm)

用法:
    python scripts/register_insp_exp_displacement.py [--fast] [--skip-reg]
"""

import sys, os, time, argparse, warnings
from pathlib import Path

import numpy as np
import nibabel as nib

warnings.filterwarnings('ignore')

# ──────────────────────── 路径常量 ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ATLAS_DIR    = PROJECT_ROOT / 'data' / '02_atlas'

INSP_TEMPLATE = ATLAS_DIR / 'standard_template.nii.gz'
EXP_TEMPLATE  = ATLAS_DIR / 'exp' / 'standard_template.nii.gz'
INSP_MASK     = ATLAS_DIR / 'standard_mask.nii.gz'
EXP_MASK      = ATLAS_DIR / 'exp' / 'standard_mask.nii.gz'
INSP_LOBES    = ATLAS_DIR / 'standard_lung_lobes_labeled.nii.gz'
EXP_LOBES     = ATLAS_DIR / 'exp' / 'standard_lung_lobes_labeled.nii.gz'

OUTPUT_DIR    = PROJECT_ROOT / 'results' / 'displacement'


# ──────────────────────── 1. ANTs 配准 ────────────────────────
def run_registration(fast: bool = False):
    """
    使用 ANTsPy 进行 SyN 非刚性配准:
      fixed  = 吸气相模板 (充气膨胀态)
      moving = 呼气相模板 (萎缩态)

    策略: 先将 512³ 图像降采样到 ~192³ 进行配准 (大幅加速),
          然后将 warped 结果和位移场上采样回原始分辨率。
    返回: (warped_path_fullres, warp_path)
    """
    import ants

    print('[1/3] 加载影像...')
    fixed_full  = ants.image_read(str(INSP_TEMPLATE))
    moving_full = ants.image_read(str(EXP_TEMPLATE))
    f_mask_full = ants.image_read(str(INSP_MASK))

    # 降采样加速 — 原始 512x512x364, 降到约 192x192x128
    ds_factor = 3  # ~2.7x 线性降采样
    orig_shape = fixed_full.shape
    target_shape = tuple(max(s // ds_factor, 32) for s in orig_shape)
    print(f'      原始分辨率: {orig_shape}')
    print(f'      配准分辨率: {target_shape} (降采样 {ds_factor}x 加速)')

    fixed  = ants.resample_image(fixed_full,  target_shape, use_voxels=True, interp_type=1)
    moving = ants.resample_image(moving_full, target_shape, use_voxels=True, interp_type=1)
    f_mask = ants.resample_image(f_mask_full, target_shape, use_voxels=True, interp_type=0)

    # 配准参数
    if fast:
        reg_iterations = (40, 20, 0)
        print('      模式: FAST (迭代 40,20,0)')
    else:
        reg_iterations = (80, 50, 30, 0)
        print('      模式: FULL (迭代 80,50,30,0)')

    print('[1/3] 开始 SyN 配准...')
    sys.stdout.flush()
    t0 = time.time()

    result = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform='SyNRA',
        mask=f_mask,
        reg_iterations=reg_iterations,
        syn_metric='CC',
        syn_sampling=4,
        outprefix=str(OUTPUT_DIR / 'syn_'),
        write_composite_transform=False,
        verbose=False,
    )
    elapsed = time.time() - t0
    print(f'      配准完成 ({elapsed/60:.1f} min)')
    sys.stdout.flush()

    # ANTs SyNRA 输出:
    #   syn_0GenericAffine.mat    → 仿射变换
    #   syn_1Warp.nii.gz          → SyN 非线性形变场 (forward, 降采样分辨率)
    #   syn_1InverseWarp.nii.gz   → SyN 非线性形变场 (inverse)
    warp_path = OUTPUT_DIR / 'syn_1Warp.nii.gz'
    affine_path = OUTPUT_DIR / 'syn_0GenericAffine.mat'

    if not warp_path.exists():
        print(f'[ERROR] 形变场文件未找到: {warp_path}')
        sys.exit(1)
    print(f'      SyN 形变场: {warp_path.name}')

    # 用原始分辨率的 fixed/moving 做 apply_transforms 获得全分辨率 warped
    print('[1/3] 将配准结果应用到全分辨率...')
    sys.stdout.flush()
    tx_list = result['fwdtransforms']  # [warp, affine]
    warped_full = ants.apply_transforms(
        fixed=fixed_full, moving=moving_full,
        transformlist=tx_list, interpolator='linear'
    )
    warped_path = OUTPUT_DIR / 'exp_warped_to_insp.nii.gz'
    ants.image_write(warped_full, str(warped_path))
    print(f'      配准后呼气相 (全分辨率): {warped_path.name}')

    # 位移场保留降采样版本 (对3D可视化足够)
    import shutil
    disp_path = OUTPUT_DIR / 'displacement_field.nii.gz'
    shutil.copy2(str(warp_path), str(disp_path))
    print(f'      位移场: {disp_path.name}')
    sys.stdout.flush()

    return warped_path, warp_path


# ──────────────────────── 字体配置工具 ────────────────────────
_FONT_CN = None   # FontProperties 缓存
_FONT_EN = None

def _setup_fonts():
    """
    全局字体配置: 中文=SimSun(宋体), 英文=Times New Roman.
    使用硬编码系统字体路径 + FontProperties 强制加载，
    彻底规避 matplotlib rcParams serif fallback 不识别 CJK 的问题。
    返回 (fp_cn, fp_en) 两个 FontProperties 对象。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.font_manager as fm

    global _FONT_CN, _FONT_EN
    if _FONT_CN is not None:
        return _FONT_CN, _FONT_EN

    # ---- 中文字体 ----
    cn_candidates = [
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/msyh.ttc',
    ]
    cn_path = None
    for p in cn_candidates:
        if Path(p).exists():
            cn_path = p
            break
    if cn_path:
        _FONT_CN = fm.FontProperties(fname=cn_path)
        print(f'      字体(中文): {cn_path}')
    else:
        _FONT_CN = fm.FontProperties(family='sans-serif')
        print('      [WARN] 未找到中文字体，使用默认')

    # ---- 英文字体 ----
    en_candidates = [
        'C:/Windows/Fonts/times.ttf',
        'C:/Windows/Fonts/timesbd.ttf',
        'C:/Windows/Fonts/TIMES.TTF',
    ]
    en_path = None
    for p in en_candidates:
        if Path(p).exists():
            en_path = p
            break
    if en_path:
        _FONT_EN = fm.FontProperties(fname=en_path)
        print(f'      字体(英文): {en_path}')
    else:
        _FONT_EN = fm.FontProperties(family='serif')
        print('      [WARN] 未找到 Times New Roman，使用默认 serif')

    # 基本 rcParams (不依赖它渲染中文，仅设置 unicode_minus)
    matplotlib.rcParams['axes.unicode_minus'] = False
    return _FONT_CN, _FONT_EN


# ──────────────────────── 2. 配准前后三视图 ────────────────────────
def plot_triview_comparison(warp_path: Path):
    """
    生成 2行×3列 的三视图对比:
      上行: 配准前 (吸气相 vs 呼气相 叠加)
      下行: 配准后 (吸气相 vs warped呼气相 叠加)
    动态选择各方向上肺面积最大的切片。
    """
    fp_cn, fp_en = _setup_fonts()
    import matplotlib.pyplot as plt

    print('[2/3] 生成三视图对比...')

    insp_data = nib.load(str(INSP_TEMPLATE)).get_fdata()
    exp_data  = nib.load(str(EXP_TEMPLATE)).get_fdata()
    warped_file = OUTPUT_DIR / 'exp_warped_to_insp.nii.gz'
    warped_data = nib.load(str(warped_file)).get_fdata()
    spacing = nib.load(str(INSP_TEMPLATE)).header.get_zooms()

    # 加载肺 mask 用于智能切片选择
    if INSP_MASK.exists():
        mask_data = nib.load(str(INSP_MASK)).get_fdata()
    else:
        mask_data = (insp_data > -500).astype(float)

    def norm(x):
        p1, p99 = np.percentile(x[x > -900], [1, 99])
        return np.clip((x - p1) / (p99 - p1 + 1e-8), 0, 1)

    insp_n = norm(insp_data)
    exp_n  = norm(exp_data)
    warp_n = norm(warped_data)

    # ---- 动态选择最大肺面积切片 ----
    mask_bin = (mask_data > 0).astype(float)
    axial_areas   = np.array([mask_bin[:, :, z].sum() for z in range(mask_bin.shape[2])])
    cz = int(np.argmax(axial_areas))
    coronal_areas = np.array([mask_bin[:, y, :].sum() for y in range(mask_bin.shape[1])])
    cy = int(np.argmax(coronal_areas))
    sagittal_areas = np.array([mask_bin[x, :, :].sum() for x in range(mask_bin.shape[0])])
    cx = int(np.argmax(sagittal_areas))
    print(f'      智能切片: Axial z={cz}, Coronal y={cy}, Sagittal x={cx}')

    slices_before = [
        (insp_n[:, :, cz], exp_n[:, :, cz],  'Axial',    [spacing[0], spacing[1]]),
        (insp_n[:, cy, :], exp_n[:, cy, :],   'Coronal',  [spacing[0], spacing[2]]),
        (insp_n[cx, :, :], exp_n[cx, :, :],   'Sagittal', [spacing[1], spacing[2]]),
    ]
    slices_after = [
        (insp_n[:, :, cz], warp_n[:, :, cz],  'Axial',    [spacing[0], spacing[1]]),
        (insp_n[:, cy, :], warp_n[:, cy, :],   'Coronal',  [spacing[0], spacing[2]]),
        (insp_n[cx, :, :], warp_n[cx, :, :],   'Sagittal', [spacing[1], spacing[2]]),
    ]

    fig = plt.figure(figsize=(22, 14), facecolor='black')
    gs = fig.add_gridspec(2, 4, width_ratios=[0.08, 1, 1, 1],
                          hspace=0.10, wspace=0.06,
                          left=0.03, right=0.97, top=0.88, bottom=0.08)

    row_labels = [('配准前\nBefore', slices_before),
                  ('配准后\nAfter',  slices_after)]

    for row_idx, (row_label, row_slices) in enumerate(row_labels):
        ax_lbl = fig.add_subplot(gs[row_idx, 0])
        ax_lbl.set_facecolor('black')
        ax_lbl.axis('off')
        fp_lbl = fp_cn.copy(); fp_lbl.set_size(26); fp_lbl.set_weight('bold')
        ax_lbl.text(0.5, 0.5, row_label, color='white',
                    ha='center', va='center', rotation=90,
                    fontproperties=fp_lbl)

        for col_idx, (s_fixed, s_moving, title, asp) in enumerate(row_slices):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            ax.set_facecolor('black')
            sf = np.rot90(s_fixed)
            sm = np.rot90(s_moving)
            rgb = np.zeros((*sf.shape, 3))
            rgb[:, :, 1] = sf
            rgb[:, :, 0] = sm
            rgb[:, :, 2] = np.minimum(sf, sm) * 0.3
            ax.imshow(rgb, aspect=asp[1] / asp[0])
            ax.axis('off')
            if row_idx == 0:
                fp_t = fp_en.copy(); fp_t.set_size(24); fp_t.set_weight('bold')
                ax.set_title(title, color='white', pad=12, fontproperties=fp_t)

    fp_sup = fp_cn.copy(); fp_sup.set_size(30); fp_sup.set_weight('bold')
    fig.suptitle('吸气相 ↔ 呼气相模板配准  三视图对比',
                 color='white', y=0.96, fontproperties=fp_sup)

    fp_leg = fp_cn.copy(); fp_leg.set_size(20)
    fig.text(0.5, 0.025,
             '绿色 = 吸气相 (Fixed)      红色 = 呼气相 (Moving)      黄色 = 重叠区域',
             ha='center', va='center', color='yellow', fontproperties=fp_leg)

    out_path = OUTPUT_DIR / 'registration_triview.png'
    fig.savefig(str(out_path), dpi=200, facecolor='black',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f'      三视图已保存: {out_path.name}')



# ──────────────────────── 3. 3D 位移场箭头可视化 ────────────────────────
def plot_3d_displacement(warp_path: Path):
    """
    3D 肺叶表面 + 位移场箭头向量可视化 (多视角版本):
      - 从 5 个肺叶标签提取等值面 (marching cubes)
      - 每个肺叶用不同颜色半透明渲染
      - 在肺表面上采样位移向量，用箭头 (quiver) 叠加
      - 箭头颜色编码位移幅度 (mm)，添加 colorbar
      - 生成 6 个不同视角的图片供对比挑选
    """
    fp_cn, fp_en = _setup_fonts()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes
    from matplotlib.patches import Patch

    print('[3/3] 生成 3D 位移场可视化 (多视角)...')

    # ---- 加载数据 ----
    lobes_img = nib.load(str(INSP_LOBES))
    lobes_full = lobes_img.get_fdata().astype(int)

    warp_img = nib.load(str(warp_path))
    warp_data = warp_img.get_fdata()
    if warp_data.ndim == 5:
        warp_data = warp_data[:, :, :, 0, :]
    warp_spacing = np.array(warp_img.header.get_zooms()[:3])

    warp_shape = warp_data.shape[:3]
    from scipy.ndimage import zoom as ndizoom
    # 如果形状匹配则无需降采样
    if warp_shape == lobes_full.shape:
        lobes_ds = lobes_full
        sp = warp_spacing
    else:
        zoom_factors = [ws / ls for ws, ls in zip(warp_shape, lobes_full.shape)]
        lobes_ds = ndizoom(lobes_full, zoom_factors, order=0)
        sp = warp_spacing
    print(f'      肺叶标签: {lobes_full.shape} -> {lobes_ds.shape}')

    disp_mag = np.sqrt(np.sum(warp_data ** 2, axis=-1))

    # 统计数据 (处理形状匹配/不匹配两种情况)
    if disp_mag.shape == lobes_full.shape:
        disp_for_stats = disp_mag
        lobes_for_stats = lobes_full
    else:
        disp_for_stats = ndizoom(disp_mag,
            [ls / ws for ws, ls in zip(warp_shape, lobes_full.shape)], order=1)
        min_shape = tuple(min(a, b) for a, b in zip(disp_for_stats.shape, lobes_full.shape))
        disp_for_stats = disp_for_stats[:min_shape[0], :min_shape[1], :min_shape[2]]
        lobes_for_stats = lobes_full[:min_shape[0], :min_shape[1], :min_shape[2]]
    lung_disp = disp_for_stats[lobes_for_stats > 0]

    lobe_info = {
        1: ('LUL 左肺上叶', [0.90, 0.55, 0.25, 0.15]),
        2: ('LLL 左肺下叶', [0.95, 0.75, 0.20, 0.15]),
        3: ('RUL 右肺上叶', [0.40, 0.65, 0.90, 0.15]),
        4: ('RML 右肺中叶', [0.55, 0.85, 0.65, 0.15]),
        5: ('RLL 右肺下叶', [0.85, 0.45, 0.65, 0.15]),
    }

    # ---- 预计算表面 mesh (所有视角共用) ----
    # 对肺叶标签做降采样用于 marching cubes (全分辨率太耗内存)
    ds_factor = 3
    lobes_mc = lobes_ds[::ds_factor, ::ds_factor, ::ds_factor]
    sp_mc = sp * ds_factor

    meshes = []
    for lobe_id, (name, color) in lobe_info.items():
        mask = (lobes_mc == lobe_id).astype(float)
        if mask.sum() < 100:
            continue
        try:
            verts, faces, _, _ = marching_cubes(mask, level=0.5, spacing=sp_mc)
            meshes.append((lobe_id, name, color, verts, faces))
        except Exception as e:
            print(f'      [WARN] 肺叶 {lobe_id} 表面提取失败: {e}')

    # ---- 预计算箭头数据 (所有视角共用) ----
    lung_coords = np.argwhere(lobes_ds > 0)
    n_arrows = min(2000, len(lung_coords))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(lung_coords), size=n_arrows, replace=False)
    sample_pts = lung_coords[idx]

    px = sample_pts[:, 0] * sp[0]
    py = sample_pts[:, 1] * sp[1]
    pz = sample_pts[:, 2] * sp[2]
    dx = warp_data[sample_pts[:, 0], sample_pts[:, 1], sample_pts[:, 2], 0]
    dy = warp_data[sample_pts[:, 0], sample_pts[:, 1], sample_pts[:, 2], 1]
    dz = warp_data[sample_pts[:, 0], sample_pts[:, 1], sample_pts[:, 2], 2]
    dm = np.sqrt(dx**2 + dy**2 + dz**2)

    p95 = np.percentile(dm, 95) if np.percentile(dm, 95) > 0.01 else 1.0

    # ---- 箭头缩放：基于肺部物理尺寸而非体素间距 ----
    all_pts = np.argwhere(lobes_ds > 0)
    mins = all_pts.min(axis=0) * sp
    maxs = all_pts.max(axis=0) * sp
    lung_extent = np.max(maxs - mins)  # 肺部最大尺寸 (mm)
    arrow_len = lung_extent * 0.05     # 箭头长度 = 肺部尺寸的5%

    dm_clip = np.minimum(dm, p95)
    scale_per_arrow = dm_clip / (p95 + 1e-8) * arrow_len
    dx_n = dx / (dm + 1e-8) * scale_per_arrow
    dy_n = dy / (dm + 1e-8) * scale_per_arrow
    dz_n = dz / (dm + 1e-8) * scale_per_arrow
    print(f'      箭头缩放: lung_extent={lung_extent:.1f}mm, arrow_len={arrow_len:.1f}mm, P95_disp={p95:.2f}mm')

    cmap = plt.cm.jet
    norm_cm = plt.Normalize(vmin=np.percentile(dm, 5), vmax=p95)
    colors = cmap(norm_cm(dm))

    pad_ax = lung_extent * 0.05

    # 统计文本
    fp_stats = fp_en.copy(); fp_stats.set_size(14)
    stats_text = (f'Mean = {lung_disp.mean():.2f} mm\n'
                  f'Median = {np.median(lung_disp):.2f} mm\n'
                  f'Max = {lung_disp.max():.2f} mm\n'
                  f'P95 = {np.percentile(lung_disp, 95):.2f} mm')

    # ---- 多视角定义 ----
    view_configs = [
        ('front',        20,  -60,  '正面偏左 Front-Left'),
        ('front_right',  20,  -120, '正面偏右 Front-Right'),
        ('top',          75,  -60,  '俯视 Top-Down'),
        ('side_left',    10,  -10,  '左侧视 Left Side'),
        ('side_right',   10,  -170, '右侧视 Right Side'),
        ('oblique',      35,  -45,  '斜角视 Oblique 45°'),
    ]

    for vi, (tag, elev, azim, view_label) in enumerate(view_configs):
        fig = plt.figure(figsize=(18, 15), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')

        # 绘制肺叶表面
        for lobe_id, name, color, verts, faces_arr in meshes:
            mesh = Poly3DCollection(verts[faces_arr], alpha=color[3],
                                     zorder=1)
            mesh.set_facecolor(color[:3])
            mesh.set_edgecolor('none')
            ax.add_collection3d(mesh)

        # 绘制位移箭头 (高 zorder 确保在半透明 mesh 之上可见)
        batch_size = 200
        for i in range(0, n_arrows, batch_size):
            end = min(i + batch_size, n_arrows)
            ax.quiver(
                px[i:end], py[i:end], pz[i:end],
                dx_n[i:end], dy_n[i:end], dz_n[i:end],
                colors=colors[i:end],
                arrow_length_ratio=0.25,
                linewidth=1.8,
                normalize=False,
                zorder=10,
            )

        # 坐标轴
        ax.set_xlim(mins[0] - pad_ax, maxs[0] + pad_ax)
        ax.set_ylim(mins[1] - pad_ax, maxs[1] + pad_ax)
        ax.set_zlim(mins[2] - pad_ax, maxs[2] + pad_ax)

        fp_ax = fp_en.copy(); fp_ax.set_size(16)
        ax.set_xlabel('X (mm)', fontproperties=fp_ax, labelpad=14)
        ax.set_ylabel('Y (mm)', fontproperties=fp_ax, labelpad=14)
        ax.set_zlabel('Z (mm)', fontproperties=fp_ax, labelpad=14)
        ax.tick_params(axis='both', labelsize=12)

        fp_title = fp_cn.copy(); fp_title.set_size(20); fp_title.set_weight('bold')
        ax.set_title(f'肺部位移场 3D 可视化  ({view_label})\n'
                     f'Inspiratory \u2192 Expiratory Displacement Field',
                     fontproperties=fp_title, pad=30)

        ax.view_init(elev=elev, azim=azim)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_cm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.10, aspect=22)
        fp_cbar = fp_en.copy(); fp_cbar.set_size(18); fp_cbar.set_weight('bold')
        cbar.set_label('Displacement Magnitude (mm)',
                       fontproperties=fp_cbar, labelpad=12)
        cbar.ax.tick_params(labelsize=14)

        # 图例
        legend_patches = [Patch(facecolor=c[:3], alpha=0.5, label=n)
                          for _, n, c, _, _ in meshes]
        fp_leg = fp_cn.copy(); fp_leg.set_size(14)
        fp_leg_title = fp_en.copy(); fp_leg_title.set_size(16)
        leg = ax.legend(handles=legend_patches, loc='upper left',
                        prop=fp_leg, framealpha=0.85,
                        title='Lung Lobes', title_fontproperties=fp_leg_title,
                        borderpad=1.0, handlelength=2.0, handleheight=1.5)
        leg.get_frame().set_linewidth(1.5)

        # 统计标注
        fig.text(0.02, 0.06, stats_text, fontproperties=fp_stats,
                 verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='lightyellow', alpha=0.9,
                           edgecolor='gray', linewidth=1.2))

        fig.subplots_adjust(left=0.02, right=0.88, top=0.90, bottom=0.05)

        out_path = OUTPUT_DIR / f'displacement_3d_{tag}.png'
        fig.savefig(str(out_path), dpi=200, bbox_inches='tight')
        plt.close()
        print(f'      [{vi+1}/6] {out_path.name}  (elev={elev}, azim={azim})')

    # ---- 打印位移统计 ----
    print(f'\n  📊 位移统计 (肺区域):')
    print(f'      均值: {lung_disp.mean():.2f} mm')
    print(f'      中位: {np.median(lung_disp):.2f} mm')
    print(f'      P5:   {np.percentile(lung_disp, 5):.2f} mm')
    print(f'      P95:  {np.percentile(lung_disp, 95):.2f} mm')
    print(f'      最大: {lung_disp.max():.2f} mm')

    print(f'\n  📊 各肺叶位移统计:')
    for lobe_id, (name, _) in lobe_info.items():
        lobe_disp = disp_for_stats[lobes_for_stats == lobe_id]
        if len(lobe_disp) > 0:
            print(f'      {name}: 均值={lobe_disp.mean():.2f} mm, '
                  f'中位={np.median(lobe_disp):.2f} mm, '
                  f'最大={lobe_disp.max():.2f} mm')


# ──────────────────────── main ────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='吸气相↔呼气相肺模板配准 & 位移场可视化')
    parser.add_argument('--fast', action='store_true',
                        help='使用快速模式 (减少迭代次数，约5分钟)')
    parser.add_argument('--skip-reg', action='store_true',
                        help='跳过配准步骤 (使用已有的形变场)')
    args = parser.parse_args()

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  吸气相 ↔ 呼气相肺模板配准 & 位移场可视化')
    print('=' * 60)
    print(f'  吸气相模板: {INSP_TEMPLATE}')
    print(f'  呼气相模板: {EXP_TEMPLATE}')
    print(f'  输出目录:   {OUTPUT_DIR}')
    print('=' * 60)

    # Step 1: 配准
    warp_path = OUTPUT_DIR / 'syn_1Warp.nii.gz'
    if args.skip_reg:
        if not warp_path.exists():
            print(f'[ERROR] --skip-reg 但形变场不存在: {warp_path}')
            sys.exit(1)
        print('[1/3] 跳过配准 (使用已有形变场)')
    else:
        _, warp_path = run_registration(fast=args.fast)

    # Step 2: 三视图对比
    plot_triview_comparison(warp_path)

    # Step 3: 3D 位移场可视化
    plot_3d_displacement(warp_path)

    print('\n' + '=' * 60)
    print('  全部完成！输出文件:')
    for f in sorted(OUTPUT_DIR.glob('*')):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f'    {f.name}  ({size_mb:.1f} MB)')
    print('=' * 60)


if __name__ == '__main__':
    main()