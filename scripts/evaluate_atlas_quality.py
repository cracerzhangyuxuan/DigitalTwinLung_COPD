#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数字孪生底座 (Digital Twin Lung Atlas) 质量评估脚本

评估体系 (3 维度 7 指标):
  A. 解剖与形态学精度 (Anatomical Accuracy)
     A1. 肺叶重叠度 mDSC       — 多受试者配准后的解剖标签对齐精度
     A2. 组织对比度 CNR          — 气道壁 vs 肺实质 / 气道壁 vs 气腔 的可分离度
  B. 纹理与精细拓扑 (Texture & Topology)
     B1. 边界清晰度 Sharpness    — 拉普拉斯方差，反映高频细节保留
     B2. 管状结构度 Frangi       — 3D Frangi 滤波响应，衡量气道/血管树完整性
     B3. 强度保真度 Wasserstein  — 模板 vs 个体 CT 的 HU 直方图距离
  C. 形变场物理属性 (Deformation Physics)
     C1. 雅可比折叠率             — |J| <= 0 的体素占比（理想 0%）
     C2. 形变平滑度              — std(log|J|)，衡量形变剧烈程度

使用方法:
    python scripts/evaluate_atlas_quality.py
    python scripts/evaluate_atlas_quality.py --limit 5
    python scripts/evaluate_atlas_quality.py --output results/atlas_eval
"""

import sys
import json
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import nibabel as nib
except ImportError:
    print("请安装 nibabel: pip install nibabel"); sys.exit(1)

try:
    from scipy import ndimage
    from scipy.stats import wasserstein_distance
except ImportError:
    print("请安装 scipy: pip install scipy"); sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    # 全局 CJK 字体配置（Windows: SimHei，否则回退 DejaVu Sans）
    matplotlib.rcParams['font.sans-serif'] = [
        'SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans'
    ]
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("请安装 matplotlib: pip install matplotlib"); sys.exit(1)

warnings.filterwarnings('ignore')


def setup_logger(log_level='INFO'):
    logger = logging.getLogger('atlas_eval')
    logger.setLevel(getattr(logging, log_level))
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(h)
    return logger


# ============================================================================
# A. 解剖与形态学精度
# ============================================================================

class AnatomicalMetrics:

    @staticmethod
    def compute_mdsc(template_lobes, subject_lobes_warped, lobe_labels=None):
        """A1. 多类别 Dice 相似系数 (mDSC)"""
        if lobe_labels is None:
            lobe_labels = [1, 2, 3, 4, 5]
        dices = {}
        for lb in lobe_labels:
            t = (template_lobes == lb)
            s = (subject_lobes_warped == lb)
            inter = np.sum(t & s)
            vol = np.sum(t) + np.sum(s)
            dices[f'dice_lobe_{lb}'] = round(2.0 * inter / vol, 4) if vol > 0 else 0.0
        dices['mean_dice'] = round(float(np.mean(list(dices.values()))), 4)
        return dices

    @staticmethod
    def compute_cnr(template_data, airway_mask, lung_mask, wall_hu_thresh=-700.0):
        """A2. 组织对比度噪声比 (CNR): 气道壁 vs 肺实质, 气道壁 vs 气腔"""
        aw = template_data[airway_mask > 0]
        lung_bool = (lung_mask > 0)
        airway_bool = (airway_mask > 0)
        lo = lung_bool & (~airway_bool)
        lu = template_data[lo]
        if len(aw) == 0 or len(lu) == 0:
            return {'cnr_wall_lung': 0.0, 'cnr_wall_lumen': 0.0,
                    'wall_mean_hu': 0.0, 'lumen_mean_hu': 0.0, 'lung_mean_hu': 0.0}
        # 自适应阈值: 使用 K-means 风格的双峰分离
        # 气道融合后有两个峰: 壁(-400 HU) 和 腔(-995 HU)
        # 阈值设为两峰的中间值
        sorted_aw = np.sort(aw)
        # 使用百分位值估计双峰位置
        low_peak = np.percentile(aw, 25)   # 腔峰估计
        high_peak = np.percentile(aw, 90)  # 壁峰估计
        adaptive_thresh = (low_peak + high_peak) / 2.0
        wall = aw[aw > adaptive_thresh]
        lumen = aw[aw <= adaptive_thresh]
        w_mu = float(np.mean(wall)) if len(wall) > 0 else float(np.mean(aw))
        l_mu = float(np.mean(lumen)) if len(lumen) > 0 else w_mu
        p_mu = float(np.mean(lu))
        w_s = max(float(np.std(wall)), 1.0) if len(wall) > 1 else 1.0
        l_s = max(float(np.std(lumen)), 1.0) if len(lumen) > 1 else 1.0
        p_s = max(float(np.std(lu)), 1.0) if len(lu) > 1 else 1.0
        d1 = np.sqrt(w_s**2 + p_s**2)
        d2 = np.sqrt(w_s**2 + l_s**2)
        return {
            'cnr_wall_lung': round(abs(w_mu - p_mu) / d1, 4) if d1 > 0 else 0.0,
            'cnr_wall_lumen': round(abs(w_mu - l_mu) / d2, 4) if d2 > 0 else 0.0,
            'wall_mean_hu': round(w_mu, 1),
            'lumen_mean_hu': round(l_mu, 1),
            'lung_mean_hu': round(p_mu, 1),
        }


# ============================================================================
# B. 纹理与精细拓扑
# ============================================================================

class TextureTopologyMetrics:

    @staticmethod
    def compute_sharpness(template_data, lung_mask):
        """B1. 边界清晰度: 拉普拉斯方差 + 平均梯度幅值"""
        m = lung_mask > 0
        if m.sum() < 100:
            return {'sharpness_laplacian_var': 0.0, 'sharpness_gradient_mag': 0.0}
        lap = ndimage.laplace(template_data.astype(np.float32))
        grad = ndimage.gaussian_gradient_magnitude(template_data.astype(np.float32), sigma=1.0)
        return {
            'sharpness_laplacian_var': round(float(np.var(lap[m])), 2),
            'sharpness_gradient_mag': round(float(np.mean(grad[m])), 4),
        }

    @staticmethod
    def compute_frangi(template_data, airway_mask, lung_mask, sigmas=None):
        """B2. 管状结构度: 3D Frangi 滤波响应"""
        if sigmas is None:
            sigmas = [1.0, 2.0, 3.0]
        m = lung_mask > 0
        try:
            from skimage.filters import frangi
        except ImportError:
            return {'frangi_mean_airway': 0.0, 'frangi_mean_lung': 0.0,
                    'frangi_ratio': 0.0, 'frangi_coverage': 0.0}
        fr = frangi(template_data.astype(np.float32), sigmas=sigmas, black_ridges=False)
        aw = airway_mask > 0
        lo = m & (~aw)
        fa = fr[aw]
        fl = fr[lo]
        ma = float(np.mean(fa)) if len(fa) > 0 else 0.0
        ml = float(np.mean(fl)) if len(fl) > 0 else 0.0
        ratio = ma / ml if ml > 1e-10 else 0.0
        if len(fa) > 0:
            thr = np.percentile(fr[m], 90)
            cov = float(np.sum(fa > thr) / len(fa))
        else:
            cov = 0.0
        return {
            'frangi_mean_airway': round(ma, 6), 'frangi_mean_lung': round(ml, 6),
            'frangi_ratio': round(ratio, 2), 'frangi_coverage': round(cov, 4),
        }

    @staticmethod
    def compute_wasserstein(template_data, subject_data, lung_mask):
        """B3. 强度保真度: Wasserstein 距离"""
        m = lung_mask > 0
        t = template_data[m]
        s = subject_data[m]
        if len(t) == 0 or len(s) == 0:
            return {'wasserstein_dist': 0.0, 'hu_mean_diff': 0.0, 'hu_std_diff': 0.0}
        return {
            'wasserstein_dist': round(float(wasserstein_distance(t, s)), 2),
            'hu_mean_diff': round(float(abs(np.mean(t) - np.mean(s))), 2),
            'hu_std_diff': round(float(abs(np.std(t) - np.std(s))), 2),
        }


# ============================================================================
# C. 形变场物理属性
# ============================================================================

class DeformationMetrics:

    @staticmethod
    def compute_jacobian(transform_path, template_path, lung_mask=None):
        """C1+C2. 雅可比行列式: 折叠率 + 形变平滑度"""
        default = {'jacobian_folding_rate': -1.0, 'jacobian_log_std': -1.0,
                    'jacobian_mean': -1.0, 'jacobian_min': -1.0, 'jacobian_max': -1.0}
        try:
            import ants
        except ImportError:
            default['jacobian_note'] = 'ANTsPy not installed'
            return default
        try:
            dom = ants.image_read(template_path)
            jac = ants.create_jacobian_determinant_image(dom, transform_path, do_log=False)
            jd = jac.numpy()
            if lung_mask is not None and jd.shape == lung_mask.shape:
                jv = jd[lung_mask > 0]
            else:
                jv = jd.flatten()
            fr = float(np.sum(jv <= 0) / len(jv))
            pos = jv[jv > 0]
            ls = float(np.std(np.log(pos))) if len(pos) > 0 else float('inf')
            return {
                'jacobian_folding_rate': round(fr, 6),
                'jacobian_log_std': round(ls, 4),
                'jacobian_mean': round(float(np.mean(jv)), 4),
                'jacobian_min': round(float(np.min(jv)), 4),
                'jacobian_max': round(float(np.max(jv)), 4),
            }
        except Exception as e:
            default['jacobian_note'] = str(e)
            return default


# ============================================================================
# D. 解剖形态学 — 体积 / 表面积 / 球形度
# ============================================================================

class MorphologyMetrics:
    """
    计算左肺、右肺、气道三个结构的体积、表面积和球形度。
    所有计算在模板配准空间中进行（spacing 统一为模板的 0.754×0.754×1.0 mm）。
    """

    STRUCTURE_NAMES = {
        'left_lung':  '左肺 (标签 1+2)',
        'right_lung': '右肺 (标签 3+4+5)',
        'airway':     '气道树',
    }

    @staticmethod
    def compute_structures(lobes_data, trachea_data, spacing):
        """
        计算三个解剖结构的形态学指标。

        参数:
            lobes_data  : ndarray, 肺叶标签 (1=LUL, 2=LLL, 3=RUL, 4=RML, 5=RLL)
            trachea_data: ndarray, 气管树二值 mask
            spacing     : tuple, 体素间距 (mm), e.g. (0.754, 0.754, 1.0)

        返回:
            dict  包含每个结构的 *_volume_cc, *_surface_cm2, *_sphericity
        """
        try:
            from skimage.measure import marching_cubes, mesh_surface_area
            mc_available = True
        except ImportError:
            mc_available = False

        voxel_vol_cc = float(spacing[0]) * float(spacing[1]) * float(spacing[2]) / 1000.0
        sp = tuple(float(s) for s in spacing[:3])

        structures = {
            'left_lung':  np.isin(lobes_data, [1, 2]).astype(np.uint8),
            'right_lung': np.isin(lobes_data, [3, 4, 5]).astype(np.uint8),
            'airway':     (trachea_data > 0).astype(np.uint8),
        }

        results = {}
        for name, mask in structures.items():
            nvox = int(mask.sum())
            vol_cc = round(nvox * voxel_vol_cc, 2)
            results[f'{name}_voxels'] = nvox
            results[f'{name}_volume_cc'] = vol_cc

            if nvox < 10 or not mc_available:
                results[f'{name}_surface_cm2'] = 0.0
                results[f'{name}_sphericity'] = 0.0
                continue

            try:
                verts, faces, _, _ = marching_cubes(
                    mask.astype(np.float32), level=0.5, spacing=sp
                )
                area_mm2 = float(mesh_surface_area(verts, faces))
                area_cm2 = round(area_mm2 / 100.0, 2)
                vol_mm3 = vol_cc * 1000.0
                # 球形度 Ψ = π^(1/3) × (6V)^(2/3) / A
                sphericity = (np.pi ** (1 / 3) * (6 * vol_mm3) ** (2 / 3)) / area_mm2
                sphericity = round(float(np.clip(sphericity, 0, 1)), 4)
            except Exception:
                area_cm2 = 0.0
                sphericity = 0.0

            results[f'{name}_surface_cm2'] = area_cm2
            results[f'{name}_sphericity'] = sphericity

        return results


# ============================================================================
# 可视化模块
# ============================================================================

class AtlasVisualizer:
    """图谱评估可视化"""

    LOBE_NAMES = {1: '左上叶', 2: '左下叶', 3: '右上叶', 4: '右中叶', 5: '右下叶'}
    LOBE_COLORS = {1: '#e74c3c', 2: '#3498db', 3: '#2ecc71', 4: '#f39c12', 5: '#9b59b6'}

    @staticmethod
    def plot_radar_chart(metrics, output_path, population_metrics=None):
        """绘制 7 指标雷达图总览（双层标注: 归一化分数 + 原始值）"""
        labels = [
            'A1\nmDSC', 'A2\nCNR', 'B1\nSharpness',
            'B2\nFrangi', 'B3\nWasserstein\n(↓)', 'C1\nFolding\n(↓)', 'C2\nSmoothness\n(↓)'
        ]

        # ---- 原始值 ----
        raw_vals = [
            metrics.get('mean_dice', 0),
            metrics.get('cnr_wall_lung', 0),
            metrics.get('sharpness_laplacian_var', 0),
            metrics.get('frangi_ratio', 0),
            metrics.get('wasserstein_dist', 0),
            metrics.get('jacobian_folding_rate', 0),
            metrics.get('jacobian_log_std', 0),
        ]
        # 用于标注的原始值字符串 (带单位)
        raw_strs = [
            f'{raw_vals[0]:.2f}',
            f'{raw_vals[1]:.2f}',
            f'{raw_vals[2]:.0f}',
            f'{raw_vals[3]:.2f}' if raw_vals[3] >= 0 else 'N/A',
            f'{raw_vals[4]:.1f} HU',
            f'{raw_vals[5]*100:.3f}%' if raw_vals[5] >= 0 else 'N/A',
            f'{raw_vals[6]:.3f}' if raw_vals[6] >= 0 else 'N/A',
        ]

        # ---- 归一化到 0-1 (越高越好) ----
        scores = [
            np.clip(raw_vals[0], 0, 1),                          # A1: 直接使用
            np.clip(raw_vals[1] / 5.0, 0, 1),                   # A2: CNR/5
            np.clip(raw_vals[2] / 5000.0, 0, 1),                # B1: Sharpness/5000
            np.clip(raw_vals[3] / 10.0, 0, 1),                  # B2: Frangi/10
            np.clip(1.0 - raw_vals[4] / 200.0, 0, 1),           # B3: 反转
            np.clip(1.0 - raw_vals[5] * 100, 0, 1),             # C1: 反转
            np.clip(1.0 - raw_vals[6] / 2.0, 0, 1),             # C2: 反转
        ]
        # 不可用指标统一设为 0.5 (灰色占位)
        unavailable = [False] * 7
        if metrics.get('frangi_ratio', -1) < 0:
            scores[3] = 0.5
            unavailable[3] = True
        if metrics.get('jacobian_folding_rate', -1) < 0:
            scores[5] = 0.5
            scores[6] = 0.5
            unavailable[5] = True
            unavailable[6] = True

        N = len(labels)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        values = scores + scores[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=2.5, color='#2980b9', markersize=10)
        ax.fill(angles, values, alpha=0.25, color='#3498db')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.set_title('Digital Twin Atlas Quality Radar\n(3 Dimensions, 7 Metrics)',
                      fontsize=16, fontweight='bold', pad=30)

        # 双层标注: 原始值 (红, 主标注) + 归一化分数 (灰, 次标注)
        for i in range(N):
            angle = angles[i]
            val = values[i]
            # 【修复】原始值作为主标注 (红色粗体)，与实际计算值完全一致
            color = '#888888' if unavailable[i] else '#c0392b'
            suffix = ' (N/A)' if unavailable[i] else ''
            ax.annotate(f'{raw_strs[i]}{suffix}', xy=(angle, val),
                        fontsize=10, ha='center', va='bottom',
                        color=color, fontweight='bold')
            # 归一化分数作为次标注 (灰色斜体, 向外偏移)
            offset = min(val + 0.13, 1.02)
            ax.annotate(f'score:{val:.2f}', xy=(angle, offset),
                        fontsize=8, ha='center', va='bottom',
                        color='#7f8c8d', fontstyle='italic')

        # 如果提供了群体均值指标，叠加绘制对比基准线
        if population_metrics is not None:
            pop_raw = [
                population_metrics.get('mean_dice', 0),
                population_metrics.get('cnr_wall_lung', 0),
                population_metrics.get('sharpness_laplacian_var', 0),
                population_metrics.get('frangi_ratio', 0),
                population_metrics.get('wasserstein_dist', 0),
                population_metrics.get('jacobian_folding_rate', 0),
                population_metrics.get('jacobian_log_std', 0),
            ]
            pop_scores = [
                np.clip(pop_raw[0], 0, 1),
                np.clip(pop_raw[1] / 5.0, 0, 1),
                np.clip(pop_raw[2] / 5000.0, 0, 1),
                np.clip(pop_raw[3] / 10.0, 0, 1),
                np.clip(1.0 - pop_raw[4] / 200.0, 0, 1),
                np.clip(1.0 - pop_raw[5] * 100, 0, 1),
                np.clip(1.0 - pop_raw[6] / 2.0, 0, 1),
            ]
            pop_values = pop_scores + pop_scores[:1]
            ax.plot(angles, pop_values, 'o--', linewidth=2, color='#e67e22',
                    markersize=7, label='群体样本均值')
            ax.fill(angles, pop_values, alpha=0.12, color='#e67e22')

        # 图例：区分当前模板与群体均值
        handles = [plt.Line2D([0], [0], color='#2980b9', linewidth=2.5,
                               marker='o', markersize=8, label='当前模板')]
        if population_metrics is not None:
            handles.append(plt.Line2D([0], [0], color='#e67e22', linewidth=2,
                                       marker='o', markersize=7, linestyle='--',
                                       label='群体样本均值'))
        ax.legend(handles=handles, loc='upper right',
                  bbox_to_anchor=(1.3, 1.1), fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_triview_slices(template_data, lung_mask, airway_mask, output_path):
        """绘制三视图切片展示 (轴位/冠状/矢状)"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        shape = template_data.shape
        slices = {
            'Axial': (2, shape[2] // 2),
            'Coronal': (1, shape[1] // 2),
            'Sagittal': (0, shape[0] // 2),
        }

        for col, (view_name, (axis, idx)) in enumerate(slices.items()):
            # 提取切片并转置
            ct_slice = np.take(template_data, idx, axis=axis).T

            # 如果是轴位 (Axial)，在 origin='lower' 下使用 k=1 实现视觉上的顺时针旋转 90 度
            if view_name == 'Axial':
                ct_slice = np.rot90(ct_slice, k=1)

            # 第一行: 模板 CT
            axes[0, col].imshow(ct_slice, cmap='gray', vmin=-1100, vmax=200, origin='lower')
            axes[0, col].set_title(f'{view_name} - Template CT', fontsize=12, fontweight='bold')
            axes[0, col].axis('off')

            # 第二行: 结构叠加 (肺 mask + 气道 mask)
            axes[1, col].imshow(ct_slice, cmap='gray', vmin=-1100, vmax=200, origin='lower')
            if lung_mask is not None:
                lm = np.take(lung_mask, idx, axis=axis).T
                if view_name == 'Axial':
                    lm = np.rot90(lm, k=1)
                axes[1, col].contour(lm, levels=[0.5], colors='#2ecc71', linewidths=1.5)
            if airway_mask is not None:
                am = np.take(airway_mask, idx, axis=axis).T
                if view_name == 'Axial':
                    am = np.rot90(am, k=1)
                overlay = np.ma.masked_where(am < 0.5, am)
                axes[1, col].imshow(overlay, cmap='autumn', alpha=0.6, origin='lower')
            axes[1, col].set_title(f'{view_name} - Lung + Airway Overlay', fontsize=12)
            axes[1, col].axis('off')

        plt.suptitle('Atlas Quality - Tri-View Inspection', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_hu_histogram(template_data, lung_mask, airway_mask, output_path,
                          subject_data=None):
        """绘制 HU 直方图对比"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        m = lung_mask > 0
        tmpl_hu = template_data[m]

        # 1. 模板肺实质 HU 分布
        axes[0].hist(tmpl_hu, bins=200, range=(-1100, 100), color='#3498db',
                     alpha=0.7, density=True, label='Template')
        if subject_data is not None:
            subj_hu = subject_data[m]
            axes[0].hist(subj_hu, bins=200, range=(-1100, 100), color='#e74c3c',
                         alpha=0.5, density=True, label='Subject (warped)')
        axes[0].set_xlabel('HU Value', fontsize=11)
        axes[0].set_ylabel('Density', fontsize=11)
        axes[0].set_title('Lung Parenchyma HU Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].axvline(x=-950, color='red', linestyle='--', alpha=0.5, label='Emphysema (-950)')

        # 2. 气道区域 HU 分布
        aw = airway_mask > 0
        if aw.sum() > 0:
            aw_hu = template_data[aw]
            axes[1].hist(aw_hu, bins=100, range=(-1100, 200), color='#e67e22',
                         alpha=0.7, density=True)
            axes[1].axvline(x=-400, color='red', linestyle='--', alpha=0.7, label='Wall (-400)')
            axes[1].axvline(x=-995, color='blue', linestyle='--', alpha=0.7, label='Lumen (-995)')
        axes[1].set_xlabel('HU Value', fontsize=11)
        axes[1].set_title('Airway Region HU Distribution', fontsize=12, fontweight='bold')
        axes[1].legend()

        # 3. 壁-腔-肺三组织 Box Plot
        lung_only = m & (~aw)
        wall_hu = template_data[aw & (template_data > -700)]
        lumen_hu = template_data[aw & (template_data <= -700)]
        parenchyma_hu = template_data[lung_only]
        bp_data = []
        bp_labels = []
        for d, lbl in [(wall_hu, 'Wall'), (lumen_hu, 'Lumen'), (parenchyma_hu, 'Parenchyma')]:
            if len(d) > 0:
                bp_data.append(d[::max(1, len(d)//5000)])  # 子采样
                bp_labels.append(lbl)
        if bp_data:
            bp = axes[2].boxplot(bp_data, labels=bp_labels, patch_artist=True)
            colors = ['#e74c3c', '#3498db', '#2ecc71']
            for patch, c in zip(bp['boxes'], colors[:len(bp_data)]):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)
        axes[2].set_ylabel('HU Value', fontsize=11)
        axes[2].set_title('Tissue HU Comparison', fontsize=12, fontweight='bold')

        plt.suptitle('HU Distribution Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_dice_bar(all_dice_results, output_path):
        """绘制肺叶 Dice 条形图"""
        if not all_dice_results:
            return
        lobe_ids = [1, 2, 3, 4, 5]
        lobe_names = ['LUL', 'LLL', 'RUL', 'RML', 'RLL']
        n_subj = len(all_dice_results)

        means = []
        stds = []
        for lb in lobe_ids:
            vals = [d.get(f'dice_lobe_{lb}', 0) for d in all_dice_results]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(lobe_names))
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')

        ax.set_xlabel('Lung Lobe', fontsize=12)
        ax.set_ylabel('Dice Score', fontsize=12)
        ax.set_title(f'A1. Lobe-wise Dice Similarity (n={n_subj} subjects)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(lobe_names, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='Threshold (0.85)')
        ax.legend()

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{m:.3f}', ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_volume_comparison(template_morpho, subjects_morpho, output_path):
        """
        绘制体积对比图：3 结构 × 散点+均值条+误差带+模板水平线。

        参数:
            template_morpho : dict, 模板形态学指标 (MorphologyMetrics.compute_structures 输出)
            subjects_morpho : list[dict], 每个受试者的形态学指标
            output_path     : Path, 输出 PNG 路径
        """
        structures = [
            ('left_lung',  '左肺 Left Lung',  '#3498db'),
            ('right_lung', '右肺 Right Lung', '#e74c3c'),
            ('airway',     '气道 Airway',     '#2ecc71'),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        fig.suptitle('D1. 解剖结构体积对比 (配准后空间)\nAnatomical Volume Comparison (Registered Space)',
                     fontsize=14, fontweight='bold', y=1.02)

        for ax, (key, label, color) in zip(axes, structures):
            vol_key = f'{key}_volume_cc'
            tmpl_val = template_morpho.get(vol_key, 0)
            subj_vals = [s.get(vol_key, 0) for s in subjects_morpho if s.get(vol_key, 0) > 0]

            if subj_vals:
                mean_v, std_v = float(np.mean(subj_vals)), float(np.std(subj_vals))
                x_pos = 1.0
                # 误差带（±1 SD）
                ax.bar(x_pos, mean_v, width=0.5, color=color, alpha=0.55,
                       edgecolor='black', linewidth=1.2, label=f'均值 ± SD\n{mean_v:.0f}±{std_v:.0f} cc')
                ax.errorbar(x_pos, mean_v, yerr=std_v, fmt='none',
                            ecolor='black', elinewidth=2, capsize=8)
                # 个体散点
                jitter = np.random.uniform(-0.08, 0.08, len(subj_vals))
                ax.scatter(np.full(len(subj_vals), x_pos) + jitter, subj_vals,
                           color='black', s=40, zorder=5, alpha=0.8, label='个体值')
            else:
                mean_v, std_v = 0, 0
                ax.text(1.0, 0.5, '无受试者数据', ha='center', va='center',
                        transform=ax.transAxes, fontsize=11, color='gray')

            # 模板水平线
            ax.axhline(tmpl_val, color='red', linewidth=2, linestyle='--',
                       label=f'模板 {tmpl_val:.0f} cc')

            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.set_ylabel('体积 (cc)', fontsize=11)
            ax.set_xlim(0.5, 1.5)
            ax.set_xticks([])
            ax.legend(fontsize=9, loc='upper right')
            ax.yaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_sphericity_comparison(template_morpho, subjects_morpho, output_path):
        """
        绘制球形度对比图：3 结构 × 群体分布（条形+散点+误差棒）+ 模板红色虚线。
        参考 plot_volume_comparison 的绘图形式。

        球形度 Ψ = π^(1/3) × (6V)^(2/3) / A ∈ (0,1]
        球=1.0, 肺≈0.3-0.6, 气道树≈0.05-0.15 (管状)
        """
        structures = [
            ('left_lung',  '左肺 Left Lung',  '#3498db'),
            ('right_lung', '右肺 Right Lung', '#e74c3c'),
            ('airway',     '气道 Airway',     '#2ecc71'),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        fig.suptitle('D3. 球形度对比 Sphericity Comparison\n'
                     r'$\Psi = \pi^{1/3}(6V)^{2/3}/A$ (球=1.0, 管状≈0)',
                     fontsize=13, fontweight='bold', y=1.02)

        for ax, (key, label, color) in zip(axes, structures):
            sp_key = f'{key}_sphericity'
            tmpl_val = template_morpho.get(sp_key, 0)
            subj_vals = [s.get(sp_key, 0) for s in subjects_morpho if s.get(sp_key, 0) > 0]

            x_pos = 1.0
            if subj_vals:
                mean_v = float(np.mean(subj_vals))
                std_v  = float(np.std(subj_vals))
                # 群体均值条形 + 误差棒
                ax.bar(x_pos, mean_v, width=0.5, color=color, alpha=0.55,
                       edgecolor='black', linewidth=1.2,
                       label=f'均值 ± SD\n{mean_v:.3f}±{std_v:.3f}')
                ax.errorbar(x_pos, mean_v, yerr=std_v, fmt='none',
                            ecolor='black', elinewidth=2, capsize=8)
                # 个体散点
                jitter = np.random.uniform(-0.08, 0.08, len(subj_vals))
                ax.scatter(np.full(len(subj_vals), x_pos) + jitter, subj_vals,
                           color='black', s=40, zorder=5, alpha=0.8, label='个体值')
                y_max = max(max(subj_vals), tmpl_val) * 1.35
            else:
                ax.text(0.5, 0.5, '无受试者数据', ha='center', va='center',
                        transform=ax.transAxes, fontsize=11, color='gray')
                y_max = max(tmpl_val * 1.5, 0.5)

            # 模板值：红色虚线（与 plot_volume_comparison 保持一致）
            ax.axhline(tmpl_val, color='red', linewidth=2, linestyle='--',
                       label=f'模板实测值 {tmpl_val:.3f}')

            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.set_ylabel('球形度 Sphericity (Ψ)', fontsize=11)
            ax.set_xlim(0.5, 1.5)
            ax.set_xticks([])
            ax.set_ylim(0, y_max)
            ax.legend(fontsize=9, loc='upper right')
            ax.yaxis.grid(True, alpha=0.4)
            ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def _build_report_text(metrics, timestamp, fv, pct, frangi_note, jac_note,
                       morpho_table_rows, has_dice, morpho_data):
    """
    构建完整的 Markdown 评估报告文本。
    设计目标：任何 AI 模型或研究人员读完文档即可独立理解所有指标，
              无需查阅源代码或其他文档。
    """
    n_subj = morpho_data['n_subjects'] if morpho_data else 0
    dice_img = '\n![肺叶 Dice 条形图](dice_bar_chart.png)\n' if has_dice else \
               '\n> ℹ️ 暂无配准肺叶标签数据，Dice 条形图未生成。请先运行 `register_normals_to_atlas.py`。\n'
    morpho_section = ''
    if morpho_table_rows:
        morpho_section = f"""
---

## D. 解剖结构形态学对比

> **评估目的**：验证模板的解剖尺寸是否代表健康人群体的中心趋势（群体代表性验证）。
> 所有计算均在模板配准空间中进行（统一 spacing = 0.754×0.754×1.0 mm），
> 故个体间体积差异可直接对比，不受原始分辨率影响。

### D1. 体积对比 (Volume Comparison)

**含义**：左肺、右肺、气道树的三维体积（单位：cc = cm³）。体积衡量的是解剖结构的大小，
是最直接的形态学属性。如果模板体积偏离群体均值超过 2 个标准差（Z-score > |2|），
说明模板可能对某一特定体型的受试者过度代表，存在**样本代表性偏差**。

**计算方法**：体素计数 × 体素体积。体素体积 = spacing_x × spacing_y × spacing_z / 1000
= 0.754 × 0.754 × 1.0 / 1000 ≈ 0.000568 cc。

**Z-score 解读**：
- |Z| < 1.0 → ✅ 模板处于群体中心，代表性良好
- 1.0 ≤ |Z| < 2.0 → ⚠️ 轻度偏离，可接受但建议关注
- |Z| ≥ 2.0 → ❌ 显著偏离，模板可能代表性不足

### D2. 表面积对比 (Surface Area)

**含义**：使用 Marching Cubes 算法（skimage.measure.marching_cubes）提取等值面后，
通过三角网格面积求和得到表面积（单位：cm²）。Marching Cubes 的精度受分割边缘平滑度影响。

**注意**：裸表面积对比的学术价值有限（受分割算法的边缘处理方式影响），建议配合球形度指标一起解读。

### D3. 球形度 (Sphericity)

**含义**：球形度 Ψ 量化了一个三维结构与球体的相似程度。
- 完美球体：Ψ = 1.0
- 肺叶（不规则叶状）：Ψ ≈ 0.3–0.6
- 气道树（高度分支管状）：Ψ ≈ 0.01–0.10

**公式**：Ψ = π^(1/3) × (6V)^(2/3) / A
其中 V 为体积（mm³），A 为表面积（mm²）。
球形度归一化了体积和表面积的关系，比裸表面积更能反映结构的**形态学特征**。

### 形态学汇总表 (n={n_subj} 个受试者)

| 指标 | 模板值 | 群体均值 ± SD | Z-score | 解读 |
|:-----|:-------|:-------------|:--------|:-----|
{morpho_table_rows}

> **Z-score 解读**：Z = (模板值 - 群体均值) / 群体SD。|Z| < 1 → ✅ 模板处于群体中心；
> 1 ≤ |Z| < 2 → ⚠️ 轻度偏离，可接受；|Z| ≥ 2 → ❌ 显著偏离，模板可能对特定体型过度代表。

![体积对比图](volume_comparison.png)

> **体积对比图解读**：每组包含 3 个结构（左肺/右肺/气道）。
> 蓝色条=群体均值，误差棒=±1SD，黑点=每个受试者个体值，红色虚线=模板值。
> 红线若落在条形内（误差棒范围内）说明模板体积具有良好的群体代表性；
> 红线明显偏高/低则表示模板可能偏向大肺或小肺的受试者。

![球形度对比图](sphericity_comparison.png)

> **球形度对比图解读**：球形度 Ψ = π^(1/3)×(6V)^(2/3)/A ∈ (0,1]。
> 预期值：左/右肺 Ψ ≈ 0.35–0.55（不规则叶状），气道树 Ψ ≈ 0.05–0.15（高度分支管状）。
> 模板与群体柱高差异越小，说明模板在几何形态上越具代表性。
"""

    lines = [
        f"# 数字孪生底座 (Digital Twin Lung Atlas) 质量评估报告",
        f"",
        f"**评估时间**: {timestamp}",
        f"**模板文件**: `standard_template_with_airway.nii.gz`  ",
        f"**受试者数**: {n_subj} 例正常人（配准后空间）",
        f"",
        f"> **文档说明**：本报告为 AI 可读格式。每个指标均包含：定义、数学公式、",
        f"> 计算方法、解读标准和本次实测值。无需查阅源代码即可完整理解所有评估内容。",
        f"",
        f"---",
        f"",
        f"## 评估体系总览",
        f"",
        f"本报告采用 **3 大维度、7 核心指标 + 形态学附加维度** 的评估框架：",
        f"",
        f"| 维度 | 代号 | 指标名称 | 实测值 | 优化方向 |",
        f"|:-----|:-----|:---------|:-------|:---------|",
        f"| A. 解剖精度 | A1 | 肺叶重叠度 mDSC | {fv('mean_dice')} | ↑ 越高越好 |",
        f"| A. 解剖精度 | A2 | 组织对比度 CNR (壁-肺) | {fv('cnr_wall_lung')} | ↑ 越高越好 |",
        f"| B. 纹理拓扑 | B1 | 边界清晰度 Sharpness | {fv('sharpness_laplacian_var','.1f')} | ↑ 越高越好 |",
        f"| B. 纹理拓扑 | B2 | 管状结构度 Frangi Ratio | {fv('frangi_ratio')} | ↑ 越高越好 |",
        f"| B. 纹理拓扑 | B3 | 强度保真度 Wasserstein | {fv('wasserstein_dist','.2f')} HU | ↓ 越低越好 |",
        f"| C. 形变物理 | C1 | 雅可比折叠率 | {pct('jacobian_folding_rate')} | ↓ 理想值 0% |",
        f"| C. 形变物理 | C2 | 形变平滑度 std(log\|J\|) | {fv('jacobian_log_std')} | ↓ 越低越好 |",
        f"",
        f"---",
        f"",
        f"## A. 解剖与形态学精度",
        f"",
        f"### A1. 肺叶重叠度 — 多类别 Dice 相似系数 (mDSC)",
        f"",
        f"**背景**：图谱模板的核心价值在于将多个个体的解剖结构对齐到统一空间。",
        f"如果配准精度不足，各受试者的肺叶边界在模板空间中就会不对齐，",
        f"导致后续的 COPD 纹理合成出现解剖错位。mDSC 是量化这种对齐程度的标准指标。",
        f"",
        f"**数学定义**（对每个肺叶 k）：",
        f"```",
        f"DSC_k = 2 × |T_k ∩ P_k| / (|T_k| + |P_k|)",
        f"mDSC  = mean(DSC_1, DSC_2, ..., DSC_5)   [5 个肺叶]",
        f"```",
        f"其中 T_k 为模板肺叶 k 的体素集合，P_k 为受试者配准后肺叶 k 的体素集合。",
        f"",
        f"**计算方法**：",
        f"1. 使用 ANTs SyNRA 将每个健康人 CT 配准到模板空间",
        f"2. 用相同形变场将个体肺叶标签（标签 1=LUL, 2=LLL, 3=RUL, 4=RML, 5=RLL）",
        f"   warp 到模板空间（最近邻插值，保持整数标签）",
        f"3. 逐肺叶计算 Dice，取 5 叶平均值",
        f"",
        f"**解读标准**：",
        f"- DSC > 0.90：优秀，解剖对齐精确",
        f"- DSC 0.80–0.90：良好，适合大多数用途",
        f"- DSC < 0.80：较差，可能导致后续分析误差",
        f"",
        f"**实测值**：平均 mDSC = **{fv('mean_dice')}**",
        f"- LUL={fv('dice_lobe_1','.4f','-')}  LLL={fv('dice_lobe_2','.4f','-')}  "
        f"RUL={fv('dice_lobe_3','.4f','-')}  RML={fv('dice_lobe_4','.4f','-')}  "
        f"RLL={fv('dice_lobe_5','.4f','-')}",
        f"- 注: {metrics.get('dice_note', '基于配准后肺叶标签计算')}",
        f"",
        dice_img,
        f"> **图表解读**：每根条形代表一个肺叶在 {n_subj} 名受试者中的平均 Dice 相似系数（误差棒=SD）。",
        f"> 红色虚线（0.85）为良好配准阈值（DSC=0.85）。",
        f"> 当前平均 mDSC = **{fv('mean_dice')}**，{'高于' if float(metrics.get('mean_dice',0) or 0) >= 0.85 else '低于'}阈值 0.85。",
        f"> - 若 mDSC 整体偏低，首要原因通常是配准迭代次数不足（当前 `reg_iterations=(20,10,0)`，",
        f">   建议增至 `(100,70,50,0)` 重新运行 `register_normals_to_atlas.py`）",
        f"> - 个别受试者 Dice 极低（如 < 0.6）往往说明该受试者肺部形态特殊，配准陷入局部最优",
        f"",
        f"---",
        f"",
        f"### A2. 组织对比度 — 对比噪声比 (CNR)",
        f"",
        f"**背景**：数字孪生肺模板使用「双层气道合成算法」：",
        f"气道壁 HU = -400，气道腔 HU = -995，与周围肺实质（约 -820 HU）形成清晰分界。",
        f"CNR 量化了这种分界的可分离程度，CNR 越高意味着气道在模板中越清晰可辨。",
        f"",
        f"**公式**：",
        f"```",
        f"CNR = |μ_A - μ_B| / sqrt(σ_A² + σ_B²)",
        f"```",
        f"其中 A、B 为两个组织区域，μ 为均值，σ 为标准差。",
        f"",
        f"**自适应阈值**：使用 P25 和 P90 百分位数的二分法，自动将气道像素",
        f"分为壁（较高 HU，≈ -400）和腔（较低 HU，≈ -995）两个区域。",
        f"",
        f"**实测值**：",
        f"- 气道壁 vs 肺实质 CNR = **{fv('cnr_wall_lung')}**（参考标准：> 2.0 为良好）",
        f"- 气道壁 vs 气腔 CNR   = **{fv('cnr_wall_lumen')}**（气道内部对比）",
        f"- 壁均值 = {fv('wall_mean_hu','.0f')} HU，腔均值 = {fv('lumen_mean_hu','.0f')} HU，",
        f"  肺均值 = {fv('lung_mean_hu','.1f')} HU",
        f"",
        f"---",
        f"",
        f"## B. 纹理与精细拓扑",
        f"",
        f"### B1. 边界清晰度 — 拉普拉斯方差 (Sharpness)",
        f"",
        f"**背景**：清晰的图像意味着保留了丰富的高频细节（边缘、纹理）。",
        f"拉普拉斯算子是二阶微分算子，对图像中的边缘和细节区域高度敏感。",
        f"拉普拉斯响应的方差越大，说明图像中边缘信息越丰富，图像越清晰。",
        f"",
        f"**公式**：",
        f"```",
        f"Sharpness = Var(∇²I)  其中 ∇² 为拉普拉斯算子",
        f"```",
        f"实现：scipy.ndimage.laplace(I)，在肺实质 mask 内计算方差。",
        f"",
        f"**解读标准**：",
        f"- > 5000：非常清晰（高细节保留）",
        f"- 2000–5000：清晰（正常水平）",
        f"- < 1000：模糊（细节丢失，通常由过度平滑引起）",
        f"",
        f"**实测值**：拉普拉斯方差 = **{fv('sharpness_laplacian_var','.2f')}**，",
        f"平均梯度幅值 = {fv('sharpness_gradient_mag','.4f')}",
        f"",
        f"---",
        f"",
        f"### B2. 管状结构度 — Frangi Vesselness 滤波 (Frangi Ratio)",
        f"",
        f"**背景**：气道树和血管树是高度分支的管状结构。Frangi 滤波器基于",
        f"Hessian 矩阵的特征值来检测管状结构，能区分球状、平板状和管状形态。",
        f"管状比 (Frangi Ratio) = 气道区域响应 / 肺实质区域响应，",
        f"比值越高说明气道与背景的分辨越清晰。",
        f"",
        f"**Hessian 特征值分析**（Frangi 1998）：",
        f"```",
        f"R_A = |λ_2| / |λ_3|       (平板性)",
        f"R_B = |λ_1| / sqrt(|λ_2 × λ_3|)   (球形性)",
        f"S   = sqrt(λ_1² + λ_2² + λ_3²)    (结构强度)",
        f"V_0 = exp(-R_A²/2α²) × (1-exp(-R_B²/2β²)) × (1-exp(-S²/2c²))",
        f"```",
        f"",
        f"**实测值**：Frangi 管状比 = **{fv('frangi_ratio')}** {frangi_note}",
        f"- 气道区域 Frangi 响应 = {fv('frangi_mean_airway','g','-')}",
        f"- 肺实质 Frangi 响应   = {fv('frangi_mean_lung','g','-')}",
        f"- 注：-1.0 表示已跳过（--skip-frangi），雷达图显示为占位符 0.5",
        f"",
        f"---",
        f"",
        f"### B3. 强度保真度 — Wasserstein 距离",
        f"",
        f"**背景**：HU 值（Hounsfield Unit）是 CT 图像中组织密度的度量。",
        f"一个好的图谱模板应该具有与源数据相近的 HU 分布，",
        f"表明模板在合成过程中保留了真实的组织密度信息。",
        f"Wasserstein 距离（地球搬运距离，Earth Mover's Distance）",
        f"度量了两个分布之间的最优传输代价。",
        f"",
        f"**公式**：",
        f"```",
        f"W₁(p, q) = ∫₋∞⁺∞ |F_p(x) - F_q(x)| dx",
        f"```",
        f"其中 F_p、F_q 为两个分布的累积分布函数（CDF）。",
        f"实现：scipy.stats.wasserstein_distance，在肺实质 mask 区域内的 HU 分布上计算。",
        f"",
        f"**数据来源**：模板 HU 分布 vs 每个配准后的正常人 CT HU 分布（肺实质区域）。",
        f"",
        f"**解读标准**：",
        f"- < 30 HU：优秀，模板 HU 与群体高度一致",
        f"- 30–80 HU：良好，正常范围",
        f"- > 100 HU：偏差较大，需检查模板构建过程",
        f"",
        f"**实测值**（{n_subj} 个受试者平均）：",
        f"- Wasserstein 距离 = **{fv('wasserstein_dist','.2f','N/A')} HU**",
        f"- HU 均值差 = {fv('hu_mean_diff','.2f','N/A')} HU，HU 标准差差 = {fv('hu_std_diff','.2f','N/A')} HU",
        f"",
        f"![HU 直方图](hu_histogram.png)",
        f"",
        f"---",
        f"",
        f"## C. 形变场物理属性",
        f"",
        f"> **背景**：非线性配准（ANTs SyN）将每个受试者 CT 变形对齐到模板。",
        f"> 形变场的物理合理性是评估配准质量的关键——不物理的形变会产生解剖错位，",
        f"> 干扰后续 COPD 纹理合成的位置精度。雅可比行列式 J(x) 描述了局部体积的缩放比例。",
        f"",
        f"### C1. 雅可比折叠率 (Jacobian Folding Rate)",
        f"",
        f"**含义**：当形变场在某个体素位置「折叠」时，雅可比行列式 |J(x)| ≤ 0，",
        f"这意味着该位置的空间映射发生了拓扑反转（类似翻转手套），在物理上不可能发生。",
        f"折叠率 = 折叠体素数 / 总体素数，理想值为 **0%**。",
        f"",
        f"**公式**：",
        f"```",
        f"Folding Rate = count(J(x) <= 0) / count(total voxels in lung mask)",
        f"J(x) = det(grad phi(x))   phi 为形变场",
        f"```",
        f"实现：ants.create_jacobian_determinant_image(domain, transform, do_log=False)，",
        f"在肺实质 mask 内统计 |J| ≤ 0 的体素比例。",
        f"",
        f"**实测值**（{n_subj} 个受试者平均）：折叠率 = **{pct('jacobian_folding_rate')}** {jac_note}",
        f"- 参考范围：< 0.1% 为可接受，0% 为理想值",
        f"",
        f"---",
        f"",
        f"### C2. 形变平滑度 — std(log|J|)",
        f"",
        f"**含义**：log|J| 表示局部体积的对数变化量（正值=膨胀，负值=压缩）。",
        f"std(log|J|) 衡量整个肺部形变的均匀程度：",
        f"值越小表示形变越平滑（接近刚体变换），",
        f"值越大表示局部存在剧烈的拉伸或压缩，可能导致解剖结构扭曲。",
        f"",
        f"**公式**：",
        f"```",
        f"Smoothness = std(log(J(x)))   对 x ∈ 肺实质区域",
        f"```",
        f"",
        f"**实测值**（{n_subj} 个受试者平均）：std(log|J|) = **{fv('jacobian_log_std')}** {jac_note}",
        f"- J 均值 = {fv('jacobian_mean')}，J 范围 = [{fv('jacobian_min')}, {fv('jacobian_max')}]",
        f"- 解读：< 0.3 非常平滑，0.3–0.7 正常，> 1.0 形变剧烈需关注",
        f"",
        morpho_section,
        f"---",
        f"",
        f"## 可视化总览",
        f"",
        f"![综合评估雷达图](radar_chart.png)",
        f"",
        f"> 雷达图各顶点显示归一化分数（红色，0–1 标准化）和原始值（灰色斜体）。",
        f"> 标有 (↓) 的指标表示原始值越低越好，归一化时已做反转处理。",
        f"",
        f"![模板三视图](triview_slices.png)",
        f"",
        f"> 三视图展示模板的轴状面（Axial）、冠状面（Coronal）、矢状面（Sagittal）",
        f"> 以及气道 mask 叠加效果。",
        f"",
        f"---",
        f"",
        f"## 综合评估结论",
        f"",
        f"| 评估项 | 实测值 | 参考范围 | 状态 |",
        f"|:-------|:-------|:---------|:-----|",
        f"| A1 mDSC | {fv('mean_dice')} | > 0.85 优秀 | {'✅' if float(metrics.get('mean_dice',0) or 0) > 0.85 else '⚠️'} |",
        f"| A2 CNR (壁-肺) | {fv('cnr_wall_lung')} | > 2.0 良好 | {'✅' if float(metrics.get('cnr_wall_lung',0) or 0) > 2 else '⚠️'} |",
        f"| B1 Sharpness | {fv('sharpness_laplacian_var','.0f')} | > 2000 清晰 | {'✅' if float(metrics.get('sharpness_laplacian_var',0) or 0) > 2000 else '⚠️'} |",
        f"| B3 Wasserstein | {fv('wasserstein_dist','.1f','N/A')} HU | < 80 HU 良好 | {'✅' if float(metrics.get('wasserstein_dist',99) or 99) < 80 else '⚠️'} |",
        f"| C1 折叠率 | {pct('jacobian_folding_rate')} | < 0.1% | {'✅' if float(metrics.get('jacobian_folding_rate',1) or 1) < 0.001 else '⚠️'} |",
        f"| C2 std(log\|J\|) | {fv('jacobian_log_std')} | < 0.7 正常 | {'✅' if 0 <= float(metrics.get('jacobian_log_std',1) or 1) < 0.7 else '⚠️'} |",
        f"",
        f"---",
        f"",
        f"*由 DigitalTwinLung_COPD 图谱质量评估器自动生成 | {timestamp}*",
    ]
    return '\n'.join(lines) + '\n'


def _compute_morpho_stats(template_morpho, subjects_morpho):
    """
    计算形态学群体统计量与 Z-score。

    参数:
        template_morpho  : dict, 模板的形态学指标
        subjects_morpho  : list[dict], 每个受试者的形态学指标
    返回:
        dict 包含 template / population / zscores / n_subjects
        若模板数据为空则返回 None
    """
    if not template_morpho:
        return None

    keys_labels = [
        ('left_lung_volume_cc',    '左肺体积 (cc)'),
        ('right_lung_volume_cc',   '右肺体积 (cc)'),
        ('airway_volume_cc',       '气道体积 (cc)'),
        ('left_lung_surface_cm2',  '左肺表面积 (cm²)'),
        ('right_lung_surface_cm2', '右肺表面积 (cm²)'),
        ('airway_surface_cm2',     '气道表面积 (cm²)'),
        ('left_lung_sphericity',   '左肺球形度'),
        ('right_lung_sphericity',  '右肺球形度'),
        ('airway_sphericity',      '气道球形度'),
    ]

    pop_stats = {}
    zscores = {}

    for key, label in keys_labels:
        tmpl_val = template_morpho.get(key)
        if tmpl_val is None:
            continue
        if subjects_morpho:
            vals = [s.get(key, 0) for s in subjects_morpho if s.get(key, 0) > 0]
            if vals:
                mean_v = float(np.mean(vals))
                std_v  = float(np.std(vals))
                pop_stats[key] = {
                    'mean':  round(mean_v, 3),
                    'std':   round(std_v, 3),
                    'n':     len(vals),
                    'label': label,
                }
                zscores[key] = round((tmpl_val - mean_v) / std_v, 2) if std_v > 0 else 0.0

    return {
        'template':   template_morpho,
        'population': pop_stats,
        'zscores':    zscores,
        'n_subjects': len(subjects_morpho),
    }


def generate_markdown_report(metrics, output_dir, timestamp,
                              morpho_data=None, has_dice=False):
    """
    生成详细的 Markdown 质量评估报告。
    每个指标包含定义、公式、计算方法、解读标准和实测值，
    设计目标：任何 AI 模型或研究人员读完文档即可独立理解所有指标。

    参数:
        metrics    : dict, 所有量化指标 (run_evaluation 汇总的 all_metrics)
        output_dir : Path, 报告输出目录
        timestamp  : str,  评估时间字符串
        morpho_data: dict or None, 包含 template/subjects/zscore 的形态学数据
        has_dice   : bool, 是否存在真实 Dice 计算结果（有则嵌入图表）
    """
    # ---- 辅助格式函数 ----
    def fv(key, fmt='.4f', fallback='N/A'):
        v = metrics.get(key)
        if v is None:
            return fallback
        try:
            return format(float(v), fmt)
        except Exception:
            return str(v)

    def pct(key, fallback='N/A'):
        v = metrics.get(key)
        if v is None:
            return fallback
        try:
            return f'{float(v)*100:.4f}%'
        except Exception:
            return str(v)

    frangi_note = ('（已跳过 --skip-frangi，值为占位符 -1.0）'
                   if metrics.get('frangi_ratio', 0) < 0 else '')
    jac_note = ('（无配准数据，值为 -1.0）'
                if metrics.get('jacobian_folding_rate', 0) < 0 else '')

    # ---- 形态学表格 ----
    morpho_table_rows = ''
    if morpho_data:
        tmpl = morpho_data.get('template', {})
        zsc  = morpho_data.get('zscores', {})
        pop  = morpho_data.get('population', {})
        # struct_map 的键必须与 MorphologyMetrics.compute_structures() 输出完全一致
        struct_map = [
            ('left_lung_volume_cc',    '左肺体积 (cc)'),
            ('right_lung_volume_cc',   '右肺体积 (cc)'),
            ('airway_volume_cc',       '气道体积 (cc)'),
            ('left_lung_surface_cm2',  '左肺表面积 (cm²)'),
            ('right_lung_surface_cm2', '右肺表面积 (cm²)'),
            ('airway_surface_cm2',     '气道表面积 (cm²)'),
            ('left_lung_sphericity',   '左肺球形度'),
            ('right_lung_sphericity',  '右肺球形度'),
            ('airway_sphericity',      '气道球形度'),
        ]
        rows = []
        for key, label in struct_map:
            tv    = tmpl.get(key, 'N/A')
            # population dict 结构: {key: {'mean':..., 'std':..., 'n':...}}
            pinfo = pop.get(key, {})
            pm    = pinfo.get('mean', 'N/A')
            ps    = pinfo.get('std', 'N/A')
            z     = zsc.get(key, 'N/A')
            interp = ('✅ 正常'    if isinstance(z, float) and abs(z) < 1 else
                      '⚠️ 轻度偏离' if isinstance(z, float) and abs(z) < 2 else
                      '❌ 显著偏离' if isinstance(z, float) else '—')
            tv_s = f'{tv:.2f}' if isinstance(tv, float) else str(tv)
            pm_s = f'{pm:.2f} ± {ps:.2f}' if isinstance(pm, float) else 'N/A'
            z_s  = f'{z:+.2f}' if isinstance(z, float) else str(z)
            rows.append(f'| {label} | {tv_s} | {pm_s} | {z_s} | {interp} |')
        morpho_table_rows = '\n'.join(rows)

    report_path = output_dir / 'atlas_quality_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(_build_report_text(
            metrics, timestamp, fv, pct, frangi_note, jac_note,
            morpho_table_rows, has_dice, morpho_data
        ))
    return report_path


# ============================================================================
# 主评估流程
# ============================================================================

def run_evaluation(args):
    """执行完整的图谱质量评估"""
    logger = setup_logger(args.log_level)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info("=" * 70)
    logger.info("数字孪生底座 (Digital Twin Atlas) 质量评估")
    logger.info("=" * 70)

    # ---- 配置路径 ----
    atlas_dir = Path(args.atlas_dir)
    template_path = atlas_dir / 'standard_template_with_airway.nii.gz'
    lung_mask_path = atlas_dir / 'standard_mask.nii.gz'
    airway_mask_path = atlas_dir / 'standard_trachea_mask.nii.gz'
    lobes_path = atlas_dir / 'standard_lung_lobes_labeled.nii.gz'
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 加载模板数据 ----
    logger.info(f"\n  模板: {template_path.name}")
    if not template_path.exists():
        logger.error(f"  模板不存在: {template_path}")
        return False

    template_nii = nib.load(str(template_path))
    template_data = template_nii.get_fdata()
    logger.info(f"  尺寸: {template_data.shape}, HU范围: [{template_data.min():.0f}, {template_data.max():.0f}]")

    lung_mask = nib.load(str(lung_mask_path)).get_fdata() if lung_mask_path.exists() else (template_data > -1024)
    airway_mask = nib.load(str(airway_mask_path)).get_fdata() if airway_mask_path.exists() else np.zeros_like(template_data)
    template_lobes = nib.load(str(lobes_path)).get_fdata() if lobes_path.exists() else None

    all_metrics = {}

    # ===========================================================
    # A2. CNR
    # ===========================================================
    logger.info("\n[A2] 计算组织对比度 CNR...")
    cnr = AnatomicalMetrics.compute_cnr(template_data, airway_mask, lung_mask)
    all_metrics.update(cnr)
    logger.info(f"  气道壁-肺实质 CNR = {cnr['cnr_wall_lung']:.4f}")
    logger.info(f"  气道壁-气腔 CNR   = {cnr['cnr_wall_lumen']:.4f}")
    logger.info(f"  壁={cnr['wall_mean_hu']:.0f} HU, 腔={cnr['lumen_mean_hu']:.0f} HU, 肺={cnr['lung_mean_hu']:.0f} HU")

    # ===========================================================
    # B1. Sharpness
    # ===========================================================
    logger.info("\n[B1] 计算边界清晰度 Sharpness...")
    sharp = TextureTopologyMetrics.compute_sharpness(template_data, lung_mask)
    all_metrics.update(sharp)
    logger.info(f"  拉普拉斯方差 = {sharp['sharpness_laplacian_var']:.2f}")
    logger.info(f"  平均梯度幅值 = {sharp['sharpness_gradient_mag']:.4f}")

    # ===========================================================
    # B2. Frangi Vesselness
    # ===========================================================
    if not args.skip_frangi:
        logger.info("\n[B2] 计算管状结构度 Frangi (可能需要几分钟)...")
        frangi = TextureTopologyMetrics.compute_frangi(template_data, airway_mask, lung_mask)
        all_metrics.update(frangi)
        logger.info(f"  气管 Frangi 响应 = {frangi['frangi_mean_airway']:.6f}")
        logger.info(f"  肺实质 Frangi 响应 = {frangi['frangi_mean_lung']:.6f}")
        logger.info(f"  管状比 (气管/肺) = {frangi['frangi_ratio']:.2f}")
    else:
        logger.info("\n[B2] 跳过 Frangi (--skip-frangi)")
        all_metrics['frangi_ratio'] = -1.0

    # ===========================================================
    # A1 mDSC + B3 Wasserstein + C1/C2 Jacobian + D Morphology
    # 数据来源: 健康人配准到模板空间后的结果 (data/04_normal_mapped)
    # 请先运行: python scripts/register_normals_to_atlas.py
    # ===========================================================
    normal_mapped_dir = Path(args.normal_mapped_dir)
    subject_dirs = sorted(normal_mapped_dir.glob('normal_*')) if normal_mapped_dir.exists() else []
    if args.limit:
        subject_dirs = subject_dirs[:args.limit]

    all_dice_results = []
    all_wasserstein = []
    all_jacobian = []
    all_morpho_subjects = []

    if not subject_dirs:
        logger.warning(f"\n  ⚠ 未找到正常人配准数据: {normal_mapped_dir}")
        logger.warning("    请先运行: python scripts/register_normals_to_atlas.py")
    else:
        logger.info(f"\n{'='*50}")
        logger.info(f"正常人配准数据评估 ({len(subject_dirs)} 个受试者)")
        logger.info(f"{'='*50}")

        for i, subj_dir in enumerate(subject_dirs):
            pid = subj_dir.name
            logger.info(f"\n  [{i+1}/{len(subject_dirs)}] {pid}")

            # B3. Wasserstein Distance — 正常人 warped CT vs 模板 HU 分布
            warped_path = subj_dir / f"{pid}_warped.nii.gz"
            if warped_path.exists():
                subj_data = nib.load(str(warped_path)).get_fdata()
                ws = TextureTopologyMetrics.compute_wasserstein(template_data, subj_data, lung_mask)
                all_wasserstein.append(ws)
                logger.info(f"    B3 Wasserstein = {ws['wasserstein_dist']:.2f} HU")
                del subj_data

            # A1. Dice — 配准后肺叶标签 vs 模板肺叶标签
            warped_lobes_path = subj_dir / f"{pid}_warped_lobes.nii.gz"
            if template_lobes is not None and warped_lobes_path.exists():
                subj_lobes = nib.load(str(warped_lobes_path)).get_fdata()
                dice_result = AnatomicalMetrics.compute_mdsc(template_lobes, subj_lobes)
                all_dice_results.append(dice_result)
                logger.info(f"    A1 mDSC = {dice_result.get('mean_dice', 'N/A'):.4f}")

            # C1+C2. Jacobian — 从 ANTs SyN 非线性形变场计算
            # ANTs SyNRA 输出命名规则:
            #   {outprefix}0GenericAffine.mat  ← 仿射部分（不含局部形变，不用于 Jacobian）
            #   {outprefix}1Warp.nii.gz        ← SyN 非线性形变场（正向，用于 Jacobian）
            #   {outprefix}1InverseWarp.nii.gz ← SyN 非线性形变场（逆向）
            transform_path = subj_dir / f"{pid}_transform_1Warp.nii.gz"
            if transform_path.exists():
                jac = DeformationMetrics.compute_jacobian(
                    str(transform_path), str(template_path), lung_mask
                )
                all_jacobian.append(jac)
                if jac['jacobian_folding_rate'] >= 0:
                    logger.info(f"    C1 折叠率 = {jac['jacobian_folding_rate']*100:.4f}%")
                    logger.info(f"    C2 std(log|J|) = {jac['jacobian_log_std']:.4f}")
                else:
                    logger.info(f"    C1/C2 Jacobian: {jac.get('jacobian_note', 'N/A')}")

            # D. 形态学 — 体积 / 表面积 / 球形度（配准后空间）
            warped_trachea_path = subj_dir / f"{pid}_warped_trachea.nii.gz"
            if warped_lobes_path.exists() and warped_trachea_path.exists():
                s_lobes = nib.load(str(warped_lobes_path)).get_fdata()
                s_trachea = nib.load(str(warped_trachea_path)).get_fdata()
                sp = tuple(float(x) for x in template_nii.header.get_zooms()[:3])
                morpho = MorphologyMetrics.compute_structures(s_lobes, s_trachea, sp)
                all_morpho_subjects.append(morpho)

            import gc
            gc.collect()

    # ------ 汇总 B3 Wasserstein ------
    if all_wasserstein:
        avg_ws = round(float(np.mean([w['wasserstein_dist'] for w in all_wasserstein])), 2)
        avg_hd = round(float(np.mean([w['hu_mean_diff'] for w in all_wasserstein])), 2)
        avg_sd = round(float(np.mean([w['hu_std_diff'] for w in all_wasserstein])), 2)
        all_metrics['wasserstein_dist'] = avg_ws
        all_metrics['hu_mean_diff'] = avg_hd
        all_metrics['hu_std_diff'] = avg_sd
        logger.info(f"\n  B3 平均 Wasserstein = {avg_ws:.2f} HU (n={len(all_wasserstein)})")

    # ------ 汇总 C1/C2 Jacobian ------
    if all_jacobian:
        valid_jac = [j for j in all_jacobian if j['jacobian_folding_rate'] >= 0]
        if valid_jac:
            avg_fr = round(float(np.mean([j['jacobian_folding_rate'] for j in valid_jac])), 6)
            avg_ls = round(float(np.mean([j['jacobian_log_std'] for j in valid_jac])), 4)
            all_metrics['jacobian_folding_rate'] = avg_fr
            all_metrics['jacobian_log_std'] = avg_ls
            all_metrics['jacobian_mean'] = round(float(np.mean([j['jacobian_mean'] for j in valid_jac])), 4)
            all_metrics['jacobian_min'] = round(float(np.min([j['jacobian_min'] for j in valid_jac])), 4)
            all_metrics['jacobian_max'] = round(float(np.max([j['jacobian_max'] for j in valid_jac])), 4)
            logger.info(f"  C1 平均折叠率 = {avg_fr*100:.4f}% (n={len(valid_jac)})")
            logger.info(f"  C2 平均 std(log|J|) = {avg_ls:.4f}")
    else:
        all_metrics.setdefault('jacobian_folding_rate', -1.0)
        all_metrics.setdefault('jacobian_log_std', -1.0)

    # ------ A1. mDSC ------
    lobe_labels = [1, 2, 3, 4, 5]
    if template_lobes is not None:
        logger.info("\n[A1] 汇总肺叶 mDSC...")
        for lb in lobe_labels:
            all_metrics[f'lobe_{lb}_voxels'] = int(np.sum(template_lobes == lb))
        if all_dice_results:
            for lb in lobe_labels:
                vals = [d.get(f'dice_lobe_{lb}', 0) for d in all_dice_results]
                all_metrics[f'dice_lobe_{lb}'] = round(float(np.mean(vals)), 4)
            all_metrics['mean_dice'] = round(float(np.mean(
                [all_metrics[f'dice_lobe_{lb}'] for lb in lobe_labels])), 4)
            logger.info(f"  平均 mDSC = {all_metrics['mean_dice']:.4f} (n={len(all_dice_results)})")
        else:
            total_lung = np.sum(lung_mask > 0)
            total_labeled = sum(np.sum(template_lobes == lb) for lb in lobe_labels)
            coverage = min(total_labeled / total_lung, 1.0) if total_lung > 0 else 0
            all_metrics['mean_dice'] = round(float(coverage), 4)
            all_metrics['dice_note'] = 'label_coverage (no warped lobe labels)'
            logger.info(f"  标签覆盖率 = {coverage:.4f} (无配准肺叶标签，覆盖率代替 Dice)")

    # ------ D. 形态学 — 模板 + 群体统计 ------
    template_morpho = {}
    morpho_data = None
    if template_lobes is not None:
        logger.info("\n[D] 计算模板形态学 (体积/表面积/球形度)...")
        sp = tuple(float(x) for x in template_nii.header.get_zooms()[:3])
        template_morpho = MorphologyMetrics.compute_structures(template_lobes, airway_mask, sp)
        logger.info(f"  左肺: {template_morpho.get('left_lung_volume_cc', 0):.0f} cc  "
                    f"表面积 {template_morpho.get('left_lung_surface_cm2', 0):.0f} cm²  "
                    f"球形度 {template_morpho.get('left_lung_sphericity', 0):.4f}")
        logger.info(f"  右肺: {template_morpho.get('right_lung_volume_cc', 0):.0f} cc  "
                    f"表面积 {template_morpho.get('right_lung_surface_cm2', 0):.0f} cm²  "
                    f"球形度 {template_morpho.get('right_lung_sphericity', 0):.4f}")
        logger.info(f"  气道: {template_morpho.get('airway_volume_cc', 0):.0f} cc  "
                    f"表面积 {template_morpho.get('airway_surface_cm2', 0):.0f} cm²  "
                    f"球形度 {template_morpho.get('airway_sphericity', 0):.4f}")
        morpho_data = _compute_morpho_stats(template_morpho, all_morpho_subjects)

    # ===========================================================
    # 生成可视化
    # ===========================================================
    logger.info(f"\n{'='*50}")
    logger.info("生成可视化...")
    logger.info(f"{'='*50}")
    viz = AtlasVisualizer

    radar_path = output_dir / 'radar_chart.png'
    viz.plot_radar_chart(all_metrics, radar_path)
    logger.info(f"  ✓ 雷达图: {radar_path.name}")

    triview_path = output_dir / 'triview_slices.png'
    viz.plot_triview_slices(template_data, lung_mask, airway_mask, triview_path)
    logger.info(f"  ✓ 三视图: {triview_path.name}")

    hist_path = output_dir / 'hu_histogram.png'
    sample_subj = None
    if subject_dirs:
        sp_path = subject_dirs[0] / f"{subject_dirs[0].name}_warped.nii.gz"
        if sp_path.exists():
            sample_subj = nib.load(str(sp_path)).get_fdata()
    viz.plot_hu_histogram(template_data, lung_mask, airway_mask, hist_path, sample_subj)
    logger.info(f"  ✓ HU直方图: {hist_path.name}")

    if all_dice_results:
        dice_path = output_dir / 'dice_bar_chart.png'
        viz.plot_dice_bar(all_dice_results, dice_path)
        logger.info(f"  ✓ Dice条形图: {dice_path.name}")

    if template_morpho and all_morpho_subjects:
        vol_path = output_dir / 'volume_comparison.png'
        viz.plot_volume_comparison(template_morpho, all_morpho_subjects, vol_path)
        logger.info(f"  ✓ 体积对比图: {vol_path.name}")
        sph_path = output_dir / 'sphericity_comparison.png'
        viz.plot_sphericity_comparison(template_morpho, all_morpho_subjects, sph_path)
        logger.info(f"  ✓ 球形度对比图: {sph_path.name}")

    # ===========================================================
    # 保存 JSON + Markdown 报告
    # ===========================================================
    json_path = output_dir / 'atlas_quality_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'template': str(template_path),
            'n_subjects': len(subject_dirs),
            'metrics': all_metrics,
            'per_subject_wasserstein': all_wasserstein,
            'per_subject_jacobian': all_jacobian,
            'template_morpho': template_morpho,
            'morpho_subjects': all_morpho_subjects,
        }, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n  ✓ JSON 指标: {json_path.name}")

    md_path = generate_markdown_report(
        all_metrics, output_dir, timestamp,
        morpho_data=morpho_data,
        has_dice=bool(all_dice_results),
    )
    logger.info(f"  ✓ Markdown 报告: {md_path.name}")

    # ===========================================================
    # 打印总结
    # ===========================================================
    logger.info(f"\n{'='*70}")
    logger.info("评估完成 — 结果总结")
    logger.info(f"{'='*70}")
    logger.info(f"  A1. mDSC           = {all_metrics.get('mean_dice', 'N/A')}")
    logger.info(f"  A2. CNR (壁-肺)    = {all_metrics.get('cnr_wall_lung', 'N/A')}")
    logger.info(f"  B1. Sharpness      = {all_metrics.get('sharpness_laplacian_var', 'N/A')}")
    logger.info(f"  B2. Frangi Ratio   = {all_metrics.get('frangi_ratio', 'N/A')}")
    logger.info(f"  B3. Wasserstein    = {all_metrics.get('wasserstein_dist', 'N/A')} HU")
    logger.info(f"  C1. Folding Rate   = {all_metrics.get('jacobian_folding_rate', 'N/A')}")
    logger.info(f"  C2. std(log|J|)    = {all_metrics.get('jacobian_log_std', 'N/A')}")
    if template_morpho:
        logger.info(f"  D.  左肺体积       = {template_morpho.get('left_lung_volume_cc', 'N/A')} cc")
        logger.info(f"  D.  右肺体积       = {template_morpho.get('right_lung_volume_cc', 'N/A')} cc")
    logger.info(f"\n  输出目录: {output_dir}")
    logger.info(f"{'='*70}")


    return True


def main():
    parser = argparse.ArgumentParser(
        description='数字孪生底座 (Digital Twin Atlas) 质量评估',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--atlas-dir', default='data/02_atlas',
                        help='图谱数据目录 (default: data/02_atlas)')
    parser.add_argument('--normal-mapped-dir', default='data/04_normal_mapped',
                        help='健康人配准结果目录 (default: data/04_normal_mapped)\n'
                             '请先运行 register_normals_to_atlas.py 生成此目录')
    parser.add_argument('--normal-mask-dir', default='data/01_cleaned/normal_mask',
                        help='正常肺原始 mask 目录 (default: data/01_cleaned/normal_mask)')
    parser.add_argument('--output', default='results/atlas_eval',
                        help='输出目录 (default: results/atlas_eval)')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制评估的受试者数量')
    parser.add_argument('--skip-frangi', action='store_true',
                        help='跳过 Frangi 滤波 (加速, 适合快速调试)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING'],
                        help='日志级别')
    args = parser.parse_args()
    success = run_evaluation(args)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

