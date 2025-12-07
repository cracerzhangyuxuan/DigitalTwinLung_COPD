#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成多视角对比可视化图片

输出：
- 原始 COPD 肺部（配准前）：X/Y/Z 三个视角
- 病灶映射后的肺部（配准后）：X/Y/Z 三个视角
"""

import os
import sys
from pathlib import Path

# 确保项目根目录在路径中
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import pyvista as pv
from scipy import ndimage

from src.utils.io import load_nifti
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_lung_mask(mask: np.ndarray, keep_largest_n: int = 2) -> np.ndarray:
    """
    清理肺 mask，只保留最大的 N 个连通分量（通常是左右肺）

    Args:
        mask: 二值 mask
        keep_largest_n: 保留的最大连通分量数量

    Returns:
        清理后的 mask
    """
    binary_mask = (mask > 0).astype(np.uint8)
    labeled, num_features = ndimage.label(binary_mask)

    if num_features <= keep_largest_n:
        return binary_mask

    # 计算每个连通分量的大小
    component_sizes = []
    for i in range(1, num_features + 1):
        size = np.sum(labeled == i)
        component_sizes.append((i, size))

    # 按大小排序，保留最大的 N 个
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    keep_labels = [x[0] for x in component_sizes[:keep_largest_n]]

    # 创建清理后的 mask
    cleaned_mask = np.zeros_like(binary_mask)
    for label in keep_labels:
        cleaned_mask[labeled == label] = 1

    return cleaned_mask


def render_multiview(
    ct_path: str,
    lesion_mask_path: str,
    lung_mask_path: str,
    output_prefix: str,
    output_dir: str = "data/04_final_viz",
    lung_color=(0.8, 0.8, 0.8),
    lesion_color=(0.9, 0.1, 0.1),
    lung_opacity=0.25,
    lesion_opacity=0.9,
    lung_threshold=-500,
    window_size=(1920, 1080),
    use_mask_surface: bool = False,  # 是否使用 mask 表面渲染
    auto_threshold: bool = True  # 是否自动计算最佳阈值
):
    """
    生成 X/Y/Z 三个视角的渲染图

    Args:
        ct_path: CT 文件路径
        lesion_mask_path: 病灶 mask 路径
        lung_mask_path: 肺部 mask 路径
        output_prefix: 输出文件前缀
        output_dir: 输出目录
        use_mask_surface: 是否使用 mask 表面渲染（更清晰的边界）
        auto_threshold: 是否自动计算最佳阈值（基于肺内HU分布）
    """
    print(f"\n{'='*60}")
    print(f"渲染: {output_prefix}")
    print(f"{'='*60}")
    
    # 检查文件
    if not os.path.exists(ct_path):
        print(f"❌ CT 文件不存在: {ct_path}")
        return False
    if not os.path.exists(lesion_mask_path):
        print(f"❌ 病灶 mask 不存在: {lesion_mask_path}")
        return False
    if not os.path.exists(lung_mask_path):
        print(f"❌ 肺 mask 不存在: {lung_mask_path}")
        return False
    
    # 加载数据
    print("加载数据...")
    ct_data = load_nifti(ct_path)
    lesion_mask = load_nifti(lesion_mask_path)
    lung_mask = load_nifti(lung_mask_path)
    
    print(f"  CT 形状: {ct_data.shape}")
    print(f"  肺 mask: {np.sum(lung_mask > 0):,} 体素")
    print(f"  病灶 mask: {np.sum(lesion_mask > 0):,} 体素")
    
    # 确保 mask 是二值的
    lung_mask = (lung_mask > 0).astype(np.uint8)
    lesion_mask = (lesion_mask > 0).astype(np.uint8)

    # 清理肺 mask，只保留最大的1个连通分量（主肺部区域）
    # 注意：某些 mask 中左右肺是连通的，第二大分量可能是胸腔边界
    lung_mask_original_voxels = np.sum(lung_mask > 0)
    lung_mask = clean_lung_mask(lung_mask, keep_largest_n=1)
    lung_mask_cleaned_voxels = np.sum(lung_mask > 0)
    if lung_mask_original_voxels != lung_mask_cleaned_voxels:
        removed = lung_mask_original_voxels - lung_mask_cleaned_voxels
        print(f"  清理 mask: 移除 {removed:,} 体素 ({removed/lung_mask_original_voxels*100:.1f}%)")

    # 约束病灶在肺内
    lesion_mask = lesion_mask & lung_mask

    # 将肺外区域设为空气
    ct_data_masked = ct_data.copy()
    ct_data_masked[lung_mask == 0] = -1000

    # 自动计算最佳阈值（基于肺内 HU 值分布）
    if auto_threshold:
        lung_values = ct_data[lung_mask > 0]
        # 使用第90百分位作为阈值，确保捕获足够的肺边界组织
        optimal_threshold = np.percentile(lung_values, 90)
        # 限制阈值范围
        optimal_threshold = max(min(optimal_threshold, -400), -800)
        print(f"  自动计算阈值: {optimal_threshold:.0f} HU (原始: {lung_threshold})")
        lung_threshold = optimal_threshold

    # 创建网格
    print("创建网格...")
    spacing = (1.0, 1.0, 1.0)

    # 肺部网格 - 优先使用 mask 表面（更清晰）
    if use_mask_surface:
        # 使用 lung_mask 直接创建表面（清晰的边界）
        print("  使用 mask 表面渲染...")
        lung_grid = pv.ImageData()
        lung_grid.dimensions = lung_mask.shape
        lung_grid.spacing = spacing
        lung_grid.point_data["values"] = lung_mask.astype(float).flatten(order="F")
        lung_mesh = lung_grid.contour([0.5], scalars="values")
        if lung_mesh.n_points > 0:
            # 减少平滑次数，保留更多表面细节和质感
            lung_mesh = lung_mesh.smooth(n_iter=30, relaxation_factor=0.05)
    else:
        # 使用 CT 值等值面
        lung_grid = pv.ImageData()
        lung_grid.dimensions = ct_data_masked.shape
        lung_grid.spacing = spacing
        lung_grid.point_data["values"] = ct_data_masked.flatten(order="F")
        lung_mesh = lung_grid.contour([lung_threshold], scalars="values")
        if lung_mesh.n_points > 0:
            # 减少平滑迭代次数，保留更多细节
            lung_mesh = lung_mesh.smooth(n_iter=20, relaxation_factor=0.03)

    print(f"  肺部网格: {lung_mesh.n_points:,} 点")
    
    # 病灶网格
    lesion_mesh = None
    if np.sum(lesion_mask) > 0:
        lesion_grid = pv.ImageData()
        lesion_grid.dimensions = lesion_mask.shape
        lesion_grid.spacing = spacing
        lesion_grid.point_data["values"] = lesion_mask.astype(float).flatten(order="F")
        lesion_mesh = lesion_grid.contour([0.5], scalars="values")
        if lesion_mesh.n_points > 0:
            lesion_mesh = lesion_mesh.smooth(n_iter=30, relaxation_factor=0.1)
        print(f"  病灶网格: {lesion_mesh.n_points:,} 点")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义视角
    views = {
        'x': ('yz', 'X 轴视角（矢状面）'),  # 从右侧看
        'y': ('xz', 'Y 轴视角（冠状面）'),  # 从前方看
        'z': ('xy', 'Z 轴视角（横断面）'),  # 从上方看
    }
    
    # 渲染三个视角
    for view_name, (camera_pos, desc) in views.items():
        print(f"\n渲染 {desc}...")
        
        plotter = pv.Plotter(window_size=window_size, off_screen=True)
        plotter.set_background("white")
        
        # 添加肺部
        if lung_mesh.n_points > 0:
            plotter.add_mesh(
                lung_mesh,
                color=lung_color,
                opacity=lung_opacity,
                smooth_shading=True
            )
        
        # 添加病灶
        if lesion_mesh is not None and lesion_mesh.n_points > 0:
            plotter.add_mesh(
                lesion_mesh,
                color=lesion_color,
                opacity=lesion_opacity,
                smooth_shading=True
            )
        
        # 设置相机
        plotter.camera_position = camera_pos
        plotter.camera.zoom(1.3)
        
        # 添加坐标轴
        plotter.add_axes()
        
        # 保存
        output_path = os.path.join(output_dir, f"{output_prefix}_view_{view_name}.png")
        plotter.screenshot(output_path)
        plotter.close()
        
        print(f"  ✅ 已保存: {output_path}")
    
    return True


def render_template_only(
    ct_path: str,
    lung_mask_path: str,
    output_prefix: str,
    output_dir: str = "data/04_final_viz",
    lung_color=(0.8, 0.8, 0.8),
    lung_opacity=0.3,  # 提高默认透明度
    lung_threshold=-500,
    window_size=(1920, 1080),
    use_mask_surface: bool = True  # 默认使用 mask 表面
):
    """
    生成模板肺部（仅肺，无病灶）的 X/Y/Z 三个视角的渲染图
    """
    print(f"\n{'=' * 60}")
    print(f"渲染: {output_prefix}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    print("加载数据...")
    ct_data = load_nifti(ct_path)
    lung_mask = load_nifti(lung_mask_path)

    print(f"  CT 形状: {ct_data.shape}")
    print(f"  肺 mask: {np.sum(lung_mask > 0):,} 体素")

    # 创建网格
    print("创建网格...")
    spacing = (1, 1, 1)

    # 清理 mask，只保留主要肺部（去除胸腔边界等结构）
    lung_mask_binary = clean_lung_mask(lung_mask, keep_largest_n=1).astype(float)

    if use_mask_surface:
        # 使用 mask 表面（更清晰的边界）
        print("  使用 mask 表面渲染...")
        grid = pv.ImageData()
        grid.dimensions = lung_mask_binary.shape
        grid.spacing = spacing
        grid.origin = (0, 0, 0)
        grid.point_data["values"] = lung_mask_binary.flatten(order="F")
        lung_surface = grid.contour([0.5], scalars="values")
        if lung_surface.n_points > 0:
            # 减少平滑次数，保留更多表面细节
            lung_surface = lung_surface.smooth(n_iter=30, relaxation_factor=0.05)
    else:
        # 使用 CT 值等值面
        ct_masked = ct_data.copy()
        ct_masked[lung_mask_binary == 0] = -1000
        grid = pv.ImageData()
        grid.dimensions = ct_masked.shape
        grid.spacing = spacing
        grid.origin = (0, 0, 0)
        grid.point_data["values"] = ct_masked.flatten(order="F")
        lung_surface = grid.contour([lung_threshold], scalars="values")
        if lung_surface.n_points > 0:
            lung_surface = lung_surface.smooth(n_iter=20, relaxation_factor=0.03)

    print(f"  肺部网格: {lung_surface.n_points:,} 点")

    # 视角定义
    views = {
        "x": {"position": "yz", "viewup": (0, 0, 1)},
        "y": {"position": "xz", "viewup": (0, 0, 1)},
        "z": {"position": "xy", "viewup": (0, 1, 0)},
    }

    for view_name, view_params in views.items():
        print(f"\n渲染 {view_name.upper()} 轴视角...")

        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.set_background("white")

        plotter.add_mesh(
            lung_surface,
            color=lung_color,
            opacity=lung_opacity,
            smooth_shading=True
        )

        plotter.view_vector(
            vector={'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}[view_name],
            viewup=view_params["viewup"]
        )
        plotter.camera.zoom(1.3)
        plotter.add_axes()

        output_path = os.path.join(output_dir, f"{output_prefix}_view_{view_name}.png")
        plotter.screenshot(output_path)
        plotter.close()

        print(f"  ✅ 已保存: {output_path}")

    return True


def main():
    print("=" * 70)
    print("生成多视角对比可视化图片")
    print("=" * 70)
    
    output_dir = "data/04_final_viz"
    
    # =========================================================================
    # 1. 原始 COPD 肺部（配准前）
    # =========================================================================
    print("\n" + "=" * 60)
    print("[1] 原始 COPD 肺部（配准前）")
    print("=" * 60)
    
    render_multiview(
        ct_path="data/01_cleaned/copd_clean/copd_001_clean_v3.nii.gz",
        lesion_mask_path="data/01_cleaned/copd_emphysema/copd_001_emphysema_v4.nii.gz",
        lung_mask_path="data/01_cleaned/copd_mask/copd_001_mask_v3.nii.gz",
        output_prefix="copd_001_original",
        output_dir=output_dir,
        use_mask_surface=True,  # 使用 mask 表面，获得更清晰的轮廓
        lung_opacity=0.3  # 稍微提高不透明度
    )
    
    # =========================================================================
    # 2. 病灶映射后的肺部（配准后）
    # =========================================================================
    print("\n" + "=" * 60)
    print("[2] 病灶映射后的肺部（配准后）")
    print("=" * 60)

    # 检查配准后的数据是否存在
    mapped_ct = "data/03_mapped/copd_001/copd_001_clean_warped.nii.gz"
    mapped_lesion = "data/03_mapped/copd_001/copd_001_emphysema_warped.nii.gz"
    # 使用精确肺 mask（如果存在），否则使用旧的模板 mask
    template_lung_mask = "data/02_atlas/temp_template_lung_mask.nii.gz"
    template_mask_old = "data/02_atlas/temp_template_mask.nii.gz"
    template_mask = template_lung_mask if os.path.exists(template_lung_mask) else template_mask_old

    # 检查配准数据的版本
    if os.path.exists(mapped_lesion):
        mapped_lesion_data = load_nifti(mapped_lesion)
        mapped_voxels = np.sum(mapped_lesion_data > 0)

        if mapped_voxels > 100000:  # 如果病灶太多，说明是旧版本
            print(f"⚠️ 配准后的病灶 mask ({mapped_voxels:,} 体素) 是旧版本")
            print("   需要使用 v4 版本病灶重新运行配准")
            print("   当前跳过配准后数据的渲染")
            print("")
            print("   重新配准需要安装 ANTsPy:")
            print("   pip install antspyx")
        else:
            render_multiview(
                ct_path=mapped_ct,
                lesion_mask_path=mapped_lesion,
                lung_mask_path=template_mask,
                output_prefix="copd_001_mapped",
                output_dir=output_dir,
                use_mask_surface=True,  # 使用 mask 表面，获得更清晰的轮廓
                lung_opacity=0.3  # 稍微提高不透明度
            )
    else:
        print("⚠️ 配准后的数据不存在，跳过此步骤")

    # =========================================================================
    # 3. 模板肺部（底座）
    # =========================================================================
    print("\n" + "=" * 60)
    print("[3] 模板肺部（底座）")
    print("=" * 60)

    template_ct = "data/02_atlas/temp_template.nii.gz"
    template_lung_mask = "data/02_atlas/temp_template_lung_mask.nii.gz"

    if os.path.exists(template_ct) and os.path.exists(template_lung_mask):
        render_template_only(
            ct_path=template_ct,
            lung_mask_path=template_lung_mask,
            output_prefix="template",
            output_dir=output_dir
        )
    else:
        print("⚠️ 模板数据不存在，跳过此步骤")

    # =========================================================================
    # 完成
    # =========================================================================
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n输出目录: {output_dir}/")

    # 列出生成的文件
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        if files:
            print("\n生成的文件:")
            for f in sorted(files):
                filepath = os.path.join(output_dir, f)
                size = os.path.getsize(filepath) / 1024
                print(f"  - {f} ({size:.1f} KB)")


if __name__ == "__main__":
    main()

