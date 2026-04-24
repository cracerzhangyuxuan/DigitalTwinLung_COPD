#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任意患者呼吸双相标签配准 & 位移场生成 & 3D可视化

用法:
  python scripts/register_patient_displacement.py --patient-dir data/K096 --patient-id K096 [--fast]
"""
import sys, time, shutil, argparse
import numpy as np
import nibabel as nib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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


def _find_phase_files(patient_dir: Path):
    insp = sorted(patient_dir.glob('*_I_*.nii.gz'))
    exp = sorted(patient_dir.glob('*_E_*.nii.gz'))
    if len(insp) != 1 or len(exp) != 1:
        raise FileNotFoundError(f'未能唯一定位吸/呼气文件: insp={insp}, exp={exp}')
    return insp[0], exp[0]


def run_registration(insp_path: Path, exp_path: Path, output_dir: Path, fast=False):
    import ants
    print('[1/3] 加载标签影像...')
    fixed_full = ants.image_read(str(insp_path))
    moving_full = ants.image_read(str(exp_path))
    ds_factor = 3
    orig_shape = fixed_full.shape
    target_shape = tuple(max(s // ds_factor, 32) for s in orig_shape)
    print(f'      原始: {orig_shape} -> 配准: {target_shape}')
    fixed = ants.resample_image(fixed_full, target_shape, use_voxels=True, interp_type=0)
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
        type_of_transform='SyNRA', mask=f_mask,
        reg_iterations=reg_iterations,
        syn_metric='mattes', syn_sampling=32,
        outprefix=str(output_dir / 'syn_'),
        write_composite_transform=False, verbose=False,
    )
    print(f'      配准完成 ({(time.time()-t0)/60:.1f} min)')
    warp_path = output_dir / 'syn_1Warp.nii.gz'
    if not warp_path.exists():
        raise FileNotFoundError(f'形变场不存在: {warp_path}')
    print('[1/3] 应用到全分辨率...')
    warped_full = ants.apply_transforms(
        fixed=fixed_full, moving=moving_full,
        transformlist=result['fwdtransforms'], interpolator='nearestNeighbor'
    )
    warped_path = output_dir / 'exp_warped_to_insp.nii.gz'
    ants.image_write(warped_full, str(warped_path))
    shutil.copy2(str(warp_path), str(output_dir / 'displacement_field.nii.gz'))
    return warped_path, warp_path


def plot_triview(insp_path: Path, exp_path: Path, output_dir: Path):
    fp_cn, fp_en = _setup_fonts()
    print('[2/3] 生成三视图...')
    insp_data = nib.load(str(insp_path)).get_fdata()
    exp_data = nib.load(str(exp_path)).get_fdata()
    warped_data = nib.load(str(output_dir / 'exp_warped_to_insp.nii.gz')).get_fdata()
    spacing = nib.load(str(insp_path)).header.get_zooms()
    mask_bin = (insp_data > 0).astype(float)
    norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8) if x.max() > x.min() else x
    insp_n, exp_n, warp_n = norm(insp_data), norm(exp_data), norm(warped_data)
    cz = int(np.argmax([mask_bin[:, :, z].sum() for z in range(mask_bin.shape[2])]))
    cy = int(np.argmax([mask_bin[:, y, :].sum() for y in range(mask_bin.shape[1])]))
    cx = int(np.argmax([mask_bin[x, :, :].sum() for x in range(mask_bin.shape[0])]))
    ez = min(cz, exp_data.shape[2] - 1)
    slices_b = [(insp_n[:, :, cz], exp_n[:, :, ez], 'Axial', [spacing[0], spacing[1]]), (insp_n[:, cy, :], exp_n[:, cy, :], 'Coronal', [spacing[0], spacing[2]]), (insp_n[cx, :, :], exp_n[cx, :, :], 'Sagittal', [spacing[1], spacing[2]])]
    slices_a = [(insp_n[:, :, cz], warp_n[:, :, cz], 'Axial', [spacing[0], spacing[1]]), (insp_n[:, cy, :], warp_n[:, cy, :], 'Coronal', [spacing[0], spacing[2]]), (insp_n[cx, :, :], warp_n[cx, :, :], 'Sagittal', [spacing[1], spacing[2]])]
    fig = plt.figure(figsize=(22, 14), facecolor='black')
    gs = fig.add_gridspec(2, 4, width_ratios=[0.08, 1, 1, 1], hspace=0.10, wspace=0.06, left=0.03, right=0.97, top=0.88, bottom=0.08)
    for ri, (rl, rs) in enumerate([('配准前\nBefore', slices_b), ('配准后\nAfter', slices_a)]):
        ax_l = fig.add_subplot(gs[ri, 0]); ax_l.set_facecolor('black'); ax_l.axis('off')
        fp_l = fp_cn.copy(); fp_l.set_size(26); fp_l.set_weight('bold')
        ax_l.text(0.5, 0.5, rl, color='white', ha='center', va='center', rotation=90, fontproperties=fp_l)
        for ci, (sf, sm, title, asp) in enumerate(rs):
            ax = fig.add_subplot(gs[ri, ci + 1]); ax.set_facecolor('black')
            sf2, sm2 = np.rot90(sf), np.rot90(sm)
            h, w = min(sf2.shape[0], sm2.shape[0]), min(sf2.shape[1], sm2.shape[1])
            sf2, sm2 = sf2[:h, :w], sm2[:h, :w]
            rgb = np.zeros((*sf2.shape, 3)); rgb[:, :, 1] = sf2; rgb[:, :, 0] = sm2; rgb[:, :, 2] = np.minimum(sf2, sm2) * 0.3
            ax.imshow(rgb, aspect=asp[1] / asp[0]); ax.axis('off')
            if ri == 0:
                fp_t = fp_en.copy(); fp_t.set_size(24); fp_t.set_weight('bold')
                ax.set_title(title, color='white', pad=12, fontproperties=fp_t)
    fig.savefig(str(output_dir / 'registration_triview.png'), dpi=200, facecolor='black', edgecolor='none', bbox_inches='tight')
    plt.close()


def plot_3d_displacement(insp_path: Path, warp_path: Path, output_dir: Path, patient_id: str):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes
    from matplotlib.patches import Patch
    from scipy.ndimage import zoom as ndizoom
    fp_cn, fp_en = _setup_fonts()
    print('[3/3] 生成 3D 位移场可视化 (多视角)...')
    insp_data = nib.load(str(insp_path)).get_fdata().astype(int)
    warp_img = nib.load(str(warp_path)); warp_data = warp_img.get_fdata()
    if warp_data.ndim == 5: warp_data = warp_data[:, :, :, 0, :]
    warp_spacing = np.array(warp_img.header.get_zooms()[:3]); warp_shape = warp_data.shape[:3]
    lobes_ds = ndizoom(insp_data, [ws / ls for ws, ls in zip(warp_shape, insp_data.shape)], order=0); lobes_ds[lobes_ds > 5] = 0
    disp_mag = np.sqrt(np.sum(warp_data ** 2, axis=-1)); lung_mask = lobes_ds > 0; lung_disp = disp_mag[lung_mask]
    if len(lung_disp) == 0: print('      [WARN] 肺区域为空，跳过'); return
    lobe_info = {1: ('LUL 左肺上叶', [0.90, 0.55, 0.25, 0.15]), 2: ('LLL 左肺下叶', [0.95, 0.75, 0.20, 0.15]), 3: ('RUL 右肺上叶', [0.40, 0.65, 0.90, 0.15]), 4: ('RML 右肺中叶', [0.55, 0.85, 0.65, 0.15]), 5: ('RLL 右肺下叶', [0.85, 0.45, 0.65, 0.15])}
    ds_mc = 2; lobes_mc = lobes_ds[::ds_mc, ::ds_mc, ::ds_mc]; sp_mc = warp_spacing * ds_mc; meshes = []
    for lid, (name, color) in lobe_info.items():
        mask = (lobes_mc == lid).astype(float)
        if mask.sum() < 100: continue
        try: verts, faces, _, _ = marching_cubes(mask, level=0.5, spacing=sp_mc); meshes.append((lid, name, color, verts, faces))
        except Exception as e: print(f'      [WARN] 肺叶 {lid} 提取失败: {e}')
    lung_coords = np.argwhere(lung_mask); n_arrows = min(2000, len(lung_coords)); rng = np.random.default_rng(42); pts = lung_coords[rng.choice(len(lung_coords), size=n_arrows, replace=False)]
    sp = warp_spacing; px, py, pz = pts[:, 0] * sp[0], pts[:, 1] * sp[1], pts[:, 2] * sp[2]; dx = warp_data[pts[:, 0], pts[:, 1], pts[:, 2], 0]; dy = warp_data[pts[:, 0], pts[:, 1], pts[:, 2], 1]; dz = warp_data[pts[:, 0], pts[:, 1], pts[:, 2], 2]; dm = np.sqrt(dx**2 + dy**2 + dz**2)
    p95 = np.percentile(dm, 95) if np.percentile(dm, 95) > 0.01 else 1.0; mins = np.argwhere(lung_mask).min(axis=0) * sp; maxs = np.argwhere(lung_mask).max(axis=0) * sp; lung_extent = np.max(maxs - mins); arrow_len = lung_extent * 0.05
    scale = np.minimum(dm, p95) / (p95 + 1e-8) * arrow_len; dx_n = dx / (dm + 1e-8) * scale; dy_n = dy / (dm + 1e-8) * scale; dz_n = dz / (dm + 1e-8) * scale
    cmap = plt.cm.jet; norm_cm = plt.Normalize(vmin=np.percentile(dm, 5), vmax=p95); colors = cmap(norm_cm(dm)); pad_ax = lung_extent * 0.05
    for tag, elev, azim, view_label in [('front', 20, -60, '正面偏左 Front-Left'), ('front_right', 20, -120, '正面偏右 Front-Right'), ('top', 75, -60, '俯视 Top-Down'), ('side_left', 10, -10, '左侧视 Left Side'), ('side_right', 10, -170, '右侧视 Right Side'), ('oblique', 35, -45, '斜角视 Oblique 45°')]:
        fig = plt.figure(figsize=(18, 15), facecolor='white'); ax = fig.add_subplot(111, projection='3d')
        for _, name, color, verts, faces_arr in meshes:
            mesh_obj = Poly3DCollection(verts[faces_arr], alpha=color[3], zorder=1); mesh_obj.set_facecolor(color[:3]); mesh_obj.set_edgecolor('none'); ax.add_collection3d(mesh_obj)
        ax.quiver(px, py, pz, dx_n, dy_n, dz_n, colors=colors, arrow_length_ratio=0.25, linewidth=1.3, normalize=False, zorder=10)
        ax.set_xlim(mins[0]-pad_ax, maxs[0]+pad_ax); ax.set_ylim(mins[1]-pad_ax, maxs[1]+pad_ax); ax.set_zlim(mins[2]-pad_ax, maxs[2]+pad_ax)
        ax.set_xlabel('X (mm)', fontproperties=fp_en, labelpad=14); ax.set_ylabel('Y (mm)', fontproperties=fp_en, labelpad=14); ax.set_zlabel('Z (mm)', fontproperties=fp_en, labelpad=14); ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'{patient_id} 肺部位移场 3D 可视化  ({view_label})\nInspiratory → Expiratory Displacement Field', fontproperties=fp_cn, pad=30)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_cm); sm.set_array([]); cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.10, aspect=22); cbar.set_label('Displacement Magnitude (mm)', fontproperties=fp_en, labelpad=12)
        leg = ax.legend(handles=[Patch(facecolor=c[:3], alpha=0.5, label=n) for _, n, c, _, _ in meshes], loc='upper left', prop=fp_cn, framealpha=0.85, title='Lung Lobes', title_fontproperties=fp_en); leg.get_frame().set_linewidth(1.5)
        fig.savefig(str(output_dir / f'displacement_3d_{tag}.png'), dpi=200, bbox_inches='tight'); plt.close()
    print(f'      位移统计: mean={lung_disp.mean():.2f} mm, p95={np.percentile(lung_disp,95):.2f} mm, max={lung_disp.max():.2f} mm')


def main():
    parser = argparse.ArgumentParser(description='任意患者吸呼气标签配准 & 位移场 & 3D可视化')
    parser.add_argument('--patient-dir', required=True, help='患者目录，如 data/K096')
    parser.add_argument('--patient-id', required=True, help='患者ID，如 K096')
    parser.add_argument('--fast', action='store_true', help='快速模式')
    parser.add_argument('--skip-reg', action='store_true', help='跳过配准，复用已有形变场')
    args = parser.parse_args()
    patient_dir = PROJECT_ROOT / args.patient_dir if not Path(args.patient_dir).is_absolute() else Path(args.patient_dir)
    insp_path, exp_path = _find_phase_files(patient_dir)
    output_dir = PROJECT_ROOT / 'results' / f'displacement_{args.patient_id}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print('=' * 60); print(f'  {args.patient_id} 标签配准 & 位移场可视化'); print('  度量函数: Mattes Mutual Information (适合标签数据)'); print('=' * 60)
    print(f'  吸气相: {insp_path.name}'); print(f'  呼气相: {exp_path.name}'); print(f'  输出:   {output_dir}'); print('=' * 60)
    warp_path = output_dir / 'syn_1Warp.nii.gz'
    if args.skip_reg:
        if not warp_path.exists(): raise FileNotFoundError(f'--skip-reg 但形变场不存在: {warp_path}')
        print('[1/3] 跳过配准')
    else:
        _, warp_path = run_registration(insp_path, exp_path, output_dir, fast=args.fast)
    plot_triview(insp_path, exp_path, output_dir)
    plot_3d_displacement(insp_path, warp_path, output_dir, args.patient_id)
    print('\n' + '=' * 60); print('  全部完成！输出文件:')
    for f in sorted(output_dir.glob('*')): print(f'    {f.name}  ({f.stat().st_size/1024/1024:.1f} MB)')
    print('=' * 60)


if __name__ == '__main__':
    main()
