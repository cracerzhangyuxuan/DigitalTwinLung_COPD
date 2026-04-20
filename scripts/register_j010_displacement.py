#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J010_20170807 标签数据配准 & 位移场生成 & 3D可视化
==================================================
输入为分割标签文件(值0-7)，使用 Mattes MI 度量替代 CC。
输出格式与 results/displacement/ 完全一致。

用法:
    python scripts/register_j010_displacement.py [--fast] [--skip-reg]
"""
import sys, time, shutil, argparse
import numpy as np
import nibabel as nib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
J010_DIR   = PROJECT_ROOT / 'data' / 'J010_20170807_I036_E035'
J010_INSP  = J010_DIR / 'J010_20170807_I_036.nii.gz'
J010_EXP   = J010_DIR / 'J010_20170807_E_035.nii.gz'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'displacement_J010'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def _setup_fonts():
    cn = 'C:/Windows/Fonts/simsun.ttc'
    en = 'C:/Windows/Fonts/times.ttf'
    fp_cn = fm.FontProperties(fname=cn) if Path(cn).exists() else fm.FontProperties()
    fp_en = fm.FontProperties(fname=en) if Path(en).exists() else fm.FontProperties()
    matplotlib.rcParams['axes.unicode_minus'] = False
    return fp_cn, fp_en


# ════════════════════════ 1. 配准 ════════════════════════
def run_registration(fast=False):
    import ants
    print('[1/3] 加载标签影像...')
    fixed_full  = ants.image_read(str(J010_INSP))
    moving_full = ants.image_read(str(J010_EXP))

    ds_factor = 3
    orig_shape = fixed_full.shape
    target_shape = tuple(max(s // ds_factor, 32) for s in orig_shape)
    print(f'      原始: {orig_shape} -> 配准: {target_shape}')

    fixed  = ants.resample_image(fixed_full,  target_shape, use_voxels=True, interp_type=0)
    moving = ants.resample_image(moving_full, target_shape, use_voxels=True, interp_type=0)

    mask_np = (fixed.numpy() > 0).astype('float32')
    f_mask = fixed.new_image_like(mask_np)

    reg_iterations = (40, 20, 0) if fast else (80, 50, 30, 0)
    mode = 'FAST' if fast else 'FULL'
    print(f'      模式: {mode}, 迭代: {reg_iterations}')
    print('[1/3] SyNRA + Mattes MI 配准...')
    sys.stdout.flush()
    t0 = time.time()

    result = ants.registration(
        fixed=fixed, moving=moving,
        type_of_transform='SyNRA',
        mask=f_mask,
        reg_iterations=reg_iterations,
        syn_metric='mattes',
        syn_sampling=32,
        outprefix=str(OUTPUT_DIR / 'syn_'),
        write_composite_transform=False,
        verbose=False,
    )
    print(f'      配准完成 ({(time.time()-t0)/60:.1f} min)')

    warp_path = OUTPUT_DIR / 'syn_1Warp.nii.gz'
    if not warp_path.exists():
        print(f'[ERROR] 形变场不存在: {warp_path}'); sys.exit(1)

    print('[1/3] 应用到全分辨率...')
    warped_full = ants.apply_transforms(
        fixed=fixed_full, moving=moving_full,
        transformlist=result['fwdtransforms'],
        interpolator='nearestNeighbor'
    )
    warped_path = OUTPUT_DIR / 'exp_warped_to_insp.nii.gz'
    ants.image_write(warped_full, str(warped_path))

    disp_path = OUTPUT_DIR / 'displacement_field.nii.gz'
    shutil.copy2(str(warp_path), str(disp_path))

    wd = nib.load(str(warp_path)).get_fdata()
    if wd.ndim == 5: wd = wd[:,:,:,0,:]
    mag = np.sqrt(np.sum(wd**2, axis=-1))
    print(f'      位移场: range=[{mag.min():.4f}, {mag.max():.4f}], mean={mag.mean():.4f}')
    return warped_path, warp_path


# ════════════════════════ 2. 三视图 ════════════════════════
def plot_triview(warp_path):
    fp_cn, fp_en = _setup_fonts()
    print('[2/3] 生成三视图...')

    insp_data = nib.load(str(J010_INSP)).get_fdata()
    exp_data  = nib.load(str(J010_EXP)).get_fdata()
    warped_data = nib.load(str(OUTPUT_DIR / 'exp_warped_to_insp.nii.gz')).get_fdata()
    spacing = nib.load(str(J010_INSP)).header.get_zooms()

    mask_bin = (insp_data > 0).astype(float)

    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8) if mx > mn else x

    insp_n, exp_n, warp_n = norm(insp_data), norm(exp_data), norm(warped_data)

    cz = int(np.argmax([mask_bin[:,:,z].sum() for z in range(mask_bin.shape[2])]))
    cy = int(np.argmax([mask_bin[:,y,:].sum() for y in range(mask_bin.shape[1])]))
    cx = int(np.argmax([mask_bin[x,:,:].sum() for x in range(mask_bin.shape[0])]))
    print(f'      切片: Axial z={cz}, Coronal y={cy}, Sagittal x={cx}')

    ez = min(cz, exp_data.shape[2]-1)
    slices_b = [
        (insp_n[:,:,cz], exp_n[:,:,ez],  'Axial',    [spacing[0],spacing[1]]),
        (insp_n[:,cy,:],  exp_n[:,cy,:],  'Coronal',  [spacing[0],spacing[2]]),
        (insp_n[cx,:,:],  exp_n[cx,:,:],  'Sagittal', [spacing[1],spacing[2]]),
    ]
    slices_a = [
        (insp_n[:,:,cz], warp_n[:,:,cz], 'Axial',    [spacing[0],spacing[1]]),
        (insp_n[:,cy,:],  warp_n[:,cy,:], 'Coronal',  [spacing[0],spacing[2]]),
        (insp_n[cx,:,:],  warp_n[cx,:,:], 'Sagittal', [spacing[1],spacing[2]]),
    ]

    fig = plt.figure(figsize=(22,14), facecolor='black')
    gs = fig.add_gridspec(2, 4, width_ratios=[0.08,1,1,1],
                          hspace=0.10, wspace=0.06,
                          left=0.03, right=0.97, top=0.88, bottom=0.08)

    for ri, (rl, rs) in enumerate([('配准前\nBefore', slices_b),
                                    ('配准后\nAfter',  slices_a)]):
        ax_l = fig.add_subplot(gs[ri, 0]); ax_l.set_facecolor('black'); ax_l.axis('off')
        fp_l = fp_cn.copy(); fp_l.set_size(26); fp_l.set_weight('bold')
        ax_l.text(0.5, 0.5, rl, color='white', ha='center', va='center',
                  rotation=90, fontproperties=fp_l)
        for ci, (sf, sm, title, asp) in enumerate(rs):
            ax = fig.add_subplot(gs[ri, ci+1]); ax.set_facecolor('black')
            sf2, sm2 = np.rot90(sf), np.rot90(sm)
            if sf2.shape != sm2.shape:
                h, w = min(sf2.shape[0],sm2.shape[0]), min(sf2.shape[1],sm2.shape[1])
                sf2, sm2 = sf2[:h,:w], sm2[:h,:w]
            rgb = np.zeros((*sf2.shape, 3))
            rgb[:,:,1] = sf2; rgb[:,:,0] = sm2; rgb[:,:,2] = np.minimum(sf2,sm2)*0.3
            ax.imshow(rgb, aspect=asp[1]/asp[0]); ax.axis('off')
            if ri == 0:
                fp_t = fp_en.copy(); fp_t.set_size(24); fp_t.set_weight('bold')
                ax.set_title(title, color='white', pad=12, fontproperties=fp_t)

    fp_s = fp_cn.copy(); fp_s.set_size(30); fp_s.set_weight('bold')
    fig.suptitle('J010 吸气相 \u2194 呼气相标签配准  三视图对比',
                 color='white', y=0.96, fontproperties=fp_s)
    fp_lg = fp_cn.copy(); fp_lg.set_size(20)
    fig.text(0.5, 0.025,
             '绿色 = 吸气相 (Fixed)      红色 = 呼气相 (Moving)      黄色 = 重叠区域',
             ha='center', va='center', color='yellow', fontproperties=fp_lg)
    out = OUTPUT_DIR / 'registration_triview.png'
    fig.savefig(str(out), dpi=200, facecolor='black', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f'      三视图已保存: {out.name}')


# ════════════════════════ 3. 3D 位移场 ════════════════════════
def plot_3d_displacement(warp_path):
    """
    参考原始脚本 plot_3d_displacement() 风格:
    - J010 自带肺叶标签(1-5)，可直接做五肺叶彩色渲染
    - 叠加位移场箭头 (quiver)
    """
    fp_cn, fp_en = _setup_fonts()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes
    from matplotlib.patches import Patch
    from scipy.ndimage import zoom as ndizoom

    print('[3/3] 生成 3D 位移场可视化 (多视角)...')

    # 加载吸气相标签 (作为肺叶分割)
    insp_data = nib.load(str(J010_INSP)).get_fdata().astype(int)

    # 加载位移场
    warp_img = nib.load(str(warp_path))
    warp_data = warp_img.get_fdata()
    if warp_data.ndim == 5:
        warp_data = warp_data[:, :, :, 0, :]
    warp_spacing = np.array(warp_img.header.get_zooms()[:3])
    warp_shape = warp_data.shape[:3]

    # 将肺叶标签降采样到 warp 分辨率
    zoom_factors = [ws / ls for ws, ls in zip(warp_shape, insp_data.shape)]
    lobes_ds = ndizoom(insp_data, zoom_factors, order=0)
    # 只保留肺叶标签 1-5 (排除气道 6,7)
    lobes_ds[lobes_ds > 5] = 0

    disp_mag = np.sqrt(np.sum(warp_data ** 2, axis=-1))

    # 位移统计 (肺区域)
    lung_mask = lobes_ds > 0
    lung_disp = disp_mag[lung_mask]
    if len(lung_disp) == 0:
        print('      [WARN] 肺区域为空，跳过'); return

    lobe_info = {
        1: ('LUL 左肺上叶', [0.90, 0.55, 0.25, 0.15]),
        2: ('LLL 左肺下叶', [0.95, 0.75, 0.20, 0.15]),
        3: ('RUL 右肺上叶', [0.40, 0.65, 0.90, 0.15]),
        4: ('RML 右肺中叶', [0.55, 0.85, 0.65, 0.15]),
        5: ('RLL 右肺下叶', [0.85, 0.45, 0.65, 0.15]),
    }

    # Marching cubes 提取每个肺叶
    ds_mc = 2
    lobes_mc = lobes_ds[::ds_mc, ::ds_mc, ::ds_mc]
    sp_mc = warp_spacing * ds_mc

    meshes = []
    for lid, (name, color) in lobe_info.items():
        mask = (lobes_mc == lid).astype(float)
        if mask.sum() < 100:
            continue
        try:
            verts, faces, _, _ = marching_cubes(mask, level=0.5, spacing=sp_mc)
            meshes.append((lid, name, color, verts, faces))
        except Exception as e:
            print(f'      [WARN] 肺叶 {lid} 提取失败: {e}')

    print(f'      提取到 {len(meshes)} 个肺叶表面')

    # 箭头数据
    lung_coords = np.argwhere(lung_mask)
    n_arrows = min(2000, len(lung_coords))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(lung_coords), size=n_arrows, replace=False)
    pts = lung_coords[idx]

    sp = warp_spacing
    px, py, pz = pts[:,0]*sp[0], pts[:,1]*sp[1], pts[:,2]*sp[2]
    dx = warp_data[pts[:,0], pts[:,1], pts[:,2], 0]
    dy = warp_data[pts[:,0], pts[:,1], pts[:,2], 1]
    dz = warp_data[pts[:,0], pts[:,1], pts[:,2], 2]
    dm = np.sqrt(dx**2 + dy**2 + dz**2)

    p95 = np.percentile(dm, 95) if np.percentile(dm, 95) > 0.01 else 1.0

    all_pts = np.argwhere(lung_mask)
    mins = all_pts.min(axis=0) * sp
    maxs = all_pts.max(axis=0) * sp
    lung_extent = np.max(maxs - mins)
    arrow_len = lung_extent * 0.05

    dm_clip = np.minimum(dm, p95)
    scale = dm_clip / (p95 + 1e-8) * arrow_len
    dx_n = dx / (dm + 1e-8) * scale
    dy_n = dy / (dm + 1e-8) * scale
    dz_n = dz / (dm + 1e-8) * scale
    print(f'      箭头: extent={lung_extent:.1f}mm, arrow_len={arrow_len:.1f}mm, P95={p95:.2f}mm')

    cmap = plt.cm.jet
    norm_cm = plt.Normalize(vmin=np.percentile(dm, 5), vmax=p95)
    colors = cmap(norm_cm(dm))
    pad_ax = lung_extent * 0.05

    fp_st = fp_en.copy(); fp_st.set_size(14)
    stats_text = (f'Mean = {lung_disp.mean():.2f} mm\n'
                  f'Median = {np.median(lung_disp):.2f} mm\n'
                  f'Max = {lung_disp.max():.2f} mm\n'
                  f'P95 = {np.percentile(lung_disp, 95):.2f} mm')

    view_configs = [
        ('front',       20,  -60,  '正面偏左 Front-Left'),
        ('front_right', 20,  -120, '正面偏右 Front-Right'),
        ('top',         75,  -60,  '俯视 Top-Down'),
        ('side_left',   10,  -10,  '左侧视 Left Side'),
        ('side_right',  10,  -170, '右侧视 Right Side'),
        ('oblique',     35,  -45,  '斜角视 Oblique 45°'),
    ]

    for vi, (tag, elev, azim, view_label) in enumerate(view_configs):
        fig = plt.figure(figsize=(18, 15), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')

        for lid, name, color, verts, faces_arr in meshes:
            mesh_obj = Poly3DCollection(verts[faces_arr], alpha=color[3], zorder=1)
            mesh_obj.set_facecolor(color[:3])
            mesh_obj.set_edgecolor('none')
            ax.add_collection3d(mesh_obj)

        batch = 200
        for i in range(0, n_arrows, batch):
            e = min(i + batch, n_arrows)
            ax.quiver(px[i:e], py[i:e], pz[i:e],
                      dx_n[i:e], dy_n[i:e], dz_n[i:e],
                      colors=colors[i:e], arrow_length_ratio=0.25,
                      linewidth=1.8, normalize=False, zorder=10)

        ax.set_xlim(mins[0]-pad_ax, maxs[0]+pad_ax)
        ax.set_ylim(mins[1]-pad_ax, maxs[1]+pad_ax)
        ax.set_zlim(mins[2]-pad_ax, maxs[2]+pad_ax)

        fp_ax = fp_en.copy(); fp_ax.set_size(16)
        ax.set_xlabel('X (mm)', fontproperties=fp_ax, labelpad=14)
        ax.set_ylabel('Y (mm)', fontproperties=fp_ax, labelpad=14)
        ax.set_zlabel('Z (mm)', fontproperties=fp_ax, labelpad=14)
        ax.tick_params(axis='both', labelsize=12)

        fp_ti = fp_cn.copy(); fp_ti.set_size(20); fp_ti.set_weight('bold')
        ax.set_title(f'J010 肺部位移场 3D 可视化  ({view_label})\n'
                     f'Inspiratory \u2192 Expiratory Displacement Field',
                     fontproperties=fp_ti, pad=30)
        ax.view_init(elev=elev, azim=azim)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_cm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.10, aspect=22)
        fp_cb = fp_en.copy(); fp_cb.set_size(18); fp_cb.set_weight('bold')
        cbar.set_label('Displacement Magnitude (mm)', fontproperties=fp_cb, labelpad=12)
        cbar.ax.tick_params(labelsize=14)

        legend_patches = [Patch(facecolor=c[:3], alpha=0.5, label=n)
                          for _, n, c, _, _ in meshes]
        fp_lg = fp_cn.copy(); fp_lg.set_size(14)
        fp_lt = fp_en.copy(); fp_lt.set_size(16)
        leg = ax.legend(handles=legend_patches, loc='upper left',
                        prop=fp_lg, framealpha=0.85,
                        title='Lung Lobes', title_fontproperties=fp_lt,
                        borderpad=1.0, handlelength=2.0, handleheight=1.5)
        leg.get_frame().set_linewidth(1.5)

        fig.text(0.02, 0.06, stats_text, fontproperties=fp_st,
                 verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                           alpha=0.9, edgecolor='gray', linewidth=1.2))
        fig.subplots_adjust(left=0.02, right=0.88, top=0.90, bottom=0.05)

        out = OUTPUT_DIR / f'displacement_3d_{tag}.png'
        fig.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close()
        print(f'      [{vi+1}/6] {out.name}  (elev={elev}, azim={azim})')

    print(f'\n  \U0001f4ca 位移统计 (肺区域):')
    print(f'      均值: {lung_disp.mean():.2f} mm')
    print(f'      中位: {np.median(lung_disp):.2f} mm')
    print(f'      P95:  {np.percentile(lung_disp, 95):.2f} mm')
    print(f'      最大: {lung_disp.max():.2f} mm')

    print(f'\n  \U0001f4ca 各肺叶位移统计:')
    for lid, (name, _) in lobe_info.items():
        ld = disp_mag[lobes_ds == lid]
        if len(ld) > 0:
            print(f'      {name}: 均值={ld.mean():.2f}mm, '
                  f'中位={np.median(ld):.2f}mm, 最大={ld.max():.2f}mm')


# ════════════════════════ main ════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='J010_20170807 标签配准 & 位移场 & 3D可视化')
    parser.add_argument('--fast', action='store_true',
                        help='快速模式 (减少迭代)')
    parser.add_argument('--skip-reg', action='store_true',
                        help='跳过配准 (使用已有形变场)')
    args = parser.parse_args()

    if not J010_INSP.exists() or not J010_EXP.exists():
        print('[ERROR] 输入文件不存在'); sys.exit(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  J010_20170807 标签配准 & 位移场可视化')
    print('  度量函数: Mattes Mutual Information (适合标签数据)')
    print('=' * 60)
    print(f'  吸气相: {J010_INSP.name}')
    print(f'  呼气相: {J010_EXP.name}')
    print(f'  输出:   {OUTPUT_DIR}')
    print('=' * 60)

    warp_path = OUTPUT_DIR / 'syn_1Warp.nii.gz'
    if args.skip_reg:
        if not warp_path.exists():
            print(f'[ERROR] --skip-reg 但形变场不存在'); sys.exit(1)
        print('[1/3] 跳过配准')
    else:
        _, warp_path = run_registration(fast=args.fast)

    plot_triview(warp_path)
    plot_3d_displacement(warp_path)

    print('\n' + '=' * 60)
    print('  全部完成！输出文件:')
    for f in sorted(OUTPUT_DIR.glob('*')):
        print(f'    {f.name}  ({f.stat().st_size/1024/1024:.1f} MB)')
    print('=' * 60)


if __name__ == '__main__':
    main()

