#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
静态 3D 渲染模块

使用 PyVista 进行肺部 + 病灶的双通道体渲染

功能特性：
- 支持多视角渲染（X/Y/Z 三轴）
- 支持 mask 表面渲染（清晰边界）
- 支持自动阈值计算（基于肺内 HU 分布）
- 支持肺 mask 清理（保留最大连通分量）
"""

from pathlib import Path
from typing import Union, Tuple, Optional, List

import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

from ..utils.io import load_nifti
from ..utils.logger import get_logger

logger = get_logger(__name__)


def check_pyvista_available() -> bool:
    """检查 PyVista 是否可用"""
    if pv is None:
        logger.error("PyVista 未安装，请运行: pip install pyvista")
        return False
    return True


def clean_lung_mask(mask: np.ndarray, keep_largest_n: int = 2) -> np.ndarray:
    """
    清理肺 mask，只保留最大的 N 个连通分量（通常是左右肺）

    Args:
        mask: 二值 mask 数组
        keep_largest_n: 保留的最大连通分量数量，默认 2（左右肺）

    Returns:
        cleaned_mask: 清理后的 mask

    Example:
        >>> lung_mask = load_nifti("lung_mask.nii.gz")
        >>> cleaned = clean_lung_mask(lung_mask, keep_largest_n=1)
    """
    if ndimage is None:
        logger.warning("scipy.ndimage 未安装，无法清理 mask")
        return (mask > 0).astype(np.uint8)

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

    removed_voxels = np.sum(binary_mask) - np.sum(cleaned_mask)
    if removed_voxels > 0:
        logger.debug(f"清理 mask: 移除 {removed_voxels} 体素")

    return cleaned_mask


def create_lung_mesh(
    ct_data: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    threshold: float = -500,
    lung_mask: Optional[np.ndarray] = None
) -> 'pv.PolyData':
    """
    从 CT 数据创建肺部表面网格（优化版）

    优化：支持使用 lung_mask 限制渲染范围，避免渲染肺外组织

    Args:
        ct_data: CT 体数据
        spacing: 体素间距
        threshold: 等值面阈值
        lung_mask: 可选，肺部 mask，用于限制渲染范围

    Returns:
        mesh: PyVista 网格
    """
    if not check_pyvista_available():
        raise ImportError("PyVista 不可用")

    # 如果提供了 lung_mask，将肺外区域设置为背景值
    if lung_mask is not None:
        ct_data = ct_data.copy()
        ct_data[lung_mask == 0] = -1000  # 肺外设为空气
        logger.debug("已应用 lung_mask 限制渲染范围")

    # 创建 ImageData - 使用 point_data（contour 需要 point_data）
    grid = pv.ImageData()
    grid.dimensions = ct_data.shape
    grid.spacing = spacing
    grid.point_data["values"] = ct_data.flatten(order="F")

    # 提取等值面
    mesh = grid.contour([threshold], scalars="values")

    # 平滑网格以获得更好的视觉效果
    if mesh.n_points > 0:
        mesh = mesh.smooth(n_iter=50, relaxation_factor=0.1)

    return mesh


def create_lesion_mesh(
    lesion_mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    lung_mask: Optional[np.ndarray] = None,
    smooth: bool = True
) -> Optional['pv.PolyData']:
    """
    从病灶 mask 创建表面网格（优化版）

    优化：
    1. 可选用 lung_mask 约束病灶范围
    2. 添加平滑处理改善视觉效果
    3. 添加空 mask 检查

    Args:
        lesion_mask: 病灶二值 mask
        spacing: 体素间距
        lung_mask: 可选，肺部 mask，用于约束病灶
        smooth: 是否平滑网格

    Returns:
        mesh: PyVista 网格，如果 mask 为空则返回 None
    """
    if not check_pyvista_available():
        raise ImportError("PyVista 不可用")

    # 确保是二值 mask
    lesion_mask = (lesion_mask > 0).astype(np.uint8)

    # 如果提供了 lung_mask，确保病灶只在肺内
    if lung_mask is not None:
        lung_mask_binary = (lung_mask > 0).astype(np.uint8)
        before_voxels = np.sum(lesion_mask)
        lesion_mask = lesion_mask & lung_mask_binary
        after_voxels = np.sum(lesion_mask)
        if before_voxels > after_voxels:
            logger.info(
                f"病灶 mask 约束: {before_voxels} -> {after_voxels} 体素 "
                f"(移除 {before_voxels - after_voxels} 个肺外体素)"
            )

    # 检查是否有有效的病灶
    lesion_voxels = np.sum(lesion_mask)
    if lesion_voxels == 0:
        logger.warning("病灶 mask 为空，无法创建网格")
        return None

    logger.info(f"创建病灶网格: {lesion_voxels} 体素")

    # 创建 ImageData - 使用 point_data（contour 需要 point_data）
    grid = pv.ImageData()
    grid.dimensions = lesion_mask.shape
    grid.spacing = spacing
    grid.point_data["values"] = lesion_mask.astype(float).flatten(order="F")

    # 提取等值面
    mesh = grid.contour([0.5], scalars="values")

    # 平滑网格以获得更好的视觉效果
    if smooth and mesh.n_points > 0:
        mesh = mesh.smooth(n_iter=30, relaxation_factor=0.1)

    return mesh


def render_static(
    ct_path: Union[str, Path],
    lesion_mask_path: Optional[Union[str, Path]] = None,
    lung_mask_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    lung_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),  # 更浅的灰色
    lesion_color: Tuple[float, float, float] = (0.9, 0.1, 0.1),  # 更鲜艳的红色
    lung_opacity: float = 0.25,  # 降低透明度，让肺更"透"
    lesion_opacity: float = 0.9,  # 提高病灶不透明度，更突出
    lung_threshold: float = -500,  # 肺组织等值面阈值
    window_size: Tuple[int, int] = (1920, 1080),
    background_color: str = "white",
    show: bool = True
) -> Optional[str]:
    """
    静态渲染肺部 + 病灶（优化版）

    优化:
    1. 支持 lung_mask 限制渲染范围
    2. 调整默认透明度使肺更透、病灶更突出
    3. 添加病灶验证和约束

    Args:
        ct_path: CT 文件路径
        lesion_mask_path: 病灶 mask 路径 (可选)
        lung_mask_path: 肺部 mask 路径 (可选，用于约束渲染范围)
        output_path: 截图保存路径 (可选)
        spacing: 体素间距
        lung_color: 肺部颜色 (RGB, 0-1)
        lesion_color: 病灶颜色 (RGB, 0-1)
        lung_opacity: 肺部透明度 (0-1，越小越透明)
        lesion_opacity: 病灶透明度
        lung_threshold: 肺组织等值面 HU 阈值
        window_size: 窗口大小
        background_color: 背景颜色
        show: 是否显示窗口

    Returns:
        output_path: 截图路径 (如果保存)
    """
    if not check_pyvista_available():
        return None

    logger.info(f"渲染: {Path(ct_path).name}")

    # 加载数据
    ct_data = load_nifti(ct_path)
    logger.info(f"  CT 形状: {ct_data.shape}, 范围: [{ct_data.min():.0f}, {ct_data.max():.0f}]")

    # 加载肺 mask（如果有）
    lung_mask = None
    if lung_mask_path is not None:
        lung_mask_path = Path(lung_mask_path)
        if lung_mask_path.exists():
            lung_mask = load_nifti(lung_mask_path)
            lung_mask = (lung_mask > 0).astype(np.uint8)
            logger.info(f"  已加载肺 mask: {np.sum(lung_mask)} 体素")

    # 创建渲染器
    plotter = pv.Plotter(window_size=window_size, off_screen=not show)
    plotter.set_background(background_color)

    # 添加肺部（使用 lung_mask 限制范围）
    lung_mesh = create_lung_mesh(ct_data, spacing, threshold=lung_threshold, lung_mask=lung_mask)
    if lung_mesh.n_points > 0:
        plotter.add_mesh(
            lung_mesh,
            color=lung_color,
            opacity=lung_opacity,
            smooth_shading=True
        )
        logger.info(f"  肺部网格: {lung_mesh.n_points} 点")
    else:
        logger.warning("  肺部网格为空")

    # 添加病灶 (如果有)
    if lesion_mask_path is not None:
        lesion_mask_path = Path(lesion_mask_path)
        if lesion_mask_path.exists():
            lesion_mask = load_nifti(lesion_mask_path)
            lesion_voxels = np.sum(lesion_mask > 0)
            logger.info(f"  病灶 mask: {lesion_voxels} 体素")

            if lesion_voxels > 0:
                # 创建病灶网格（用 lung_mask 约束）
                lesion_mesh = create_lesion_mesh(lesion_mask, spacing, lung_mask=lung_mask)
                if lesion_mesh is not None and lesion_mesh.n_points > 0:
                    plotter.add_mesh(
                        lesion_mesh,
                        color=lesion_color,
                        opacity=lesion_opacity,
                        smooth_shading=True
                    )
                    logger.info(f"  病灶网格: {lesion_mesh.n_points} 点")
                else:
                    logger.warning("  病灶网格为空（可能被 lung_mask 完全过滤）")
            else:
                logger.warning("  病灶 mask 为空")
        else:
            logger.warning(f"  病灶 mask 文件不存在: {lesion_mask_path}")

    # 设置相机 - 使用前视图
    plotter.camera_position = 'xy'
    plotter.camera.zoom(1.3)  # 稍微放大一点

    # 添加坐标轴
    plotter.add_axes()

    # 保存截图
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(output_path))
        logger.info(f"截图已保存: {output_path}")
    
    # 显示
    if show:
        plotter.show()
    
    plotter.close()
    
    return str(output_path) if output_path else None


def render_comparison(
    ct_paths: List[Union[str, Path]],
    titles: List[str],
    output_path: Optional[Union[str, Path]] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    window_size: Tuple[int, int] = (1920, 1080),
    show: bool = True
) -> Optional[str]:
    """
    并排比较渲染多个 CT
    
    Args:
        ct_paths: CT 文件路径列表
        titles: 每个 CT 的标题
        output_path: 截图保存路径
        spacing: 体素间距
        window_size: 窗口大小
        show: 是否显示
        
    Returns:
        output_path: 截图路径
    """
    if not check_pyvista_available():
        return None
    
    n = len(ct_paths)
    
    # 创建多子图渲染器
    plotter = pv.Plotter(
        shape=(1, n),
        window_size=window_size,
        off_screen=not show
    )
    
    for i, (ct_path, title) in enumerate(zip(ct_paths, titles)):
        plotter.subplot(0, i)
        
        ct_data = load_nifti(ct_path)
        lung_mesh = create_lung_mesh(ct_data, spacing)
        
        plotter.add_mesh(lung_mesh, color='gray', opacity=0.5)
        plotter.add_text(title, font_size=12)
        plotter.camera_position = 'xy'
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(output_path))
    
    if show:
        plotter.show()
    
    plotter.close()
    
    return str(output_path) if output_path else None


def batch_render(
    ct_dir: Union[str, Path],
    mask_dir: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "renders",
    pattern: str = "*.nii.gz"
) -> int:
    """
    批量渲染
    
    Args:
        ct_dir: CT 目录
        mask_dir: Mask 目录 (可选)
        output_dir: 输出目录
        pattern: 文件匹配模式
        
    Returns:
        count: 成功渲染的数量
    """
    ct_dir = Path(ct_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ct_files = list(ct_dir.glob(pattern))
    logger.info(f"找到 {len(ct_files)} 个 CT 文件")
    
    count = 0
    for ct_path in ct_files:
        stem = ct_path.stem.replace('.nii', '')
        output_path = output_dir / f"{stem}_render.png"
        
        # 查找对应的 mask
        mask_path = None
        if mask_dir is not None:
            mask_dir = Path(mask_dir)
            for suffix in ['_lesion', '_emphysema', '_mask']:
                candidate = mask_dir / f"{stem}{suffix}.nii.gz"
                if candidate.exists():
                    mask_path = candidate
                    break
        
        try:
            render_static(
                ct_path=ct_path,
                lesion_mask_path=mask_path,
                output_path=output_path,
                show=False
            )
            count += 1
        except Exception as e:
            logger.error(f"渲染失败 {ct_path.name}: {e}")
    
    logger.info(f"批量渲染完成: {count}/{len(ct_files)}")
    return count


def render_multiview(
    ct_path: Union[str, Path],
    lesion_mask_path: Union[str, Path],
    lung_mask_path: Union[str, Path],
    output_prefix: str,
    output_dir: Union[str, Path] = "data/04_final_viz",
    lung_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
    lesion_color: Tuple[float, float, float] = (0.9, 0.1, 0.1),
    lung_opacity: float = 0.3,
    lesion_opacity: float = 0.9,
    lung_threshold: float = -500,
    window_size: Tuple[int, int] = (1920, 1080),
    use_mask_surface: bool = True,
    auto_threshold: bool = True
) -> bool:
    """
    生成 X/Y/Z 三个视角的渲染图

    此函数从 generate_multiview_comparison.py 整合而来，提供多视角渲染能力。

    Args:
        ct_path: CT 文件路径
        lesion_mask_path: 病灶 mask 路径
        lung_mask_path: 肺部 mask 路径
        output_prefix: 输出文件前缀
        output_dir: 输出目录
        lung_color: 肺部颜色 RGB
        lesion_color: 病灶颜色 RGB
        lung_opacity: 肺部透明度 (0-1)
        lesion_opacity: 病灶透明度 (0-1)
        lung_threshold: 肺组织等值面 HU 阈值
        window_size: 窗口大小
        use_mask_surface: 是否使用 mask 表面渲染（更清晰的边界）
        auto_threshold: 是否自动计算最佳阈值（基于肺内 HU 分布）

    Returns:
        success: 是否成功

    Example:
        >>> render_multiview(
        ...     ct_path="data/01_cleaned/copd_clean/copd_001_clean.nii.gz",
        ...     lesion_mask_path="data/01_cleaned/copd_emphysema/copd_001_emphysema.nii.gz",
        ...     lung_mask_path="data/01_cleaned/copd_mask/copd_001_mask.nii.gz",
        ...     output_prefix="copd_001",
        ...     output_dir="data/04_final_viz"
        ... )
    """
    if not check_pyvista_available():
        return False

    logger.info(f"多视角渲染: {output_prefix}")

    # 检查文件
    ct_path = Path(ct_path)
    lesion_mask_path = Path(lesion_mask_path)
    lung_mask_path = Path(lung_mask_path)

    for p, name in [(ct_path, "CT"), (lesion_mask_path, "病灶 mask"), (lung_mask_path, "肺 mask")]:
        if not p.exists():
            logger.error(f"{name} 文件不存在: {p}")
            return False

    # 加载数据
    ct_data = load_nifti(ct_path)
    lesion_mask = load_nifti(lesion_mask_path)
    lung_mask = load_nifti(lung_mask_path)

    logger.info(f"  CT 形状: {ct_data.shape}")
    logger.info(f"  肺 mask: {np.sum(lung_mask > 0):,} 体素")
    logger.info(f"  病灶 mask: {np.sum(lesion_mask > 0):,} 体素")

    # 确保 mask 是二值的
    lung_mask = (lung_mask > 0).astype(np.uint8)
    lesion_mask = (lesion_mask > 0).astype(np.uint8)

    # 清理肺 mask，只保留最大的 1 个连通分量
    lung_mask = clean_lung_mask(lung_mask, keep_largest_n=1)

    # 约束病灶在肺内
    lesion_mask = lesion_mask & lung_mask

    # 将肺外区域设为空气
    ct_data_masked = ct_data.copy()
    ct_data_masked[lung_mask == 0] = -1000

    # 自动计算最佳阈值
    if auto_threshold:
        lung_values = ct_data[lung_mask > 0]
        optimal_threshold = np.percentile(lung_values, 90)
        optimal_threshold = max(min(optimal_threshold, -400), -800)
        logger.debug(f"  自动计算阈值: {optimal_threshold:.0f} HU")
        lung_threshold = optimal_threshold

    # 创建网格
    spacing = (1.0, 1.0, 1.0)

    if use_mask_surface:
        # 使用 lung_mask 直接创建表面
        lung_grid = pv.ImageData()
        lung_grid.dimensions = lung_mask.shape
        lung_grid.spacing = spacing
        lung_grid.point_data["values"] = lung_mask.astype(float).flatten(order="F")
        lung_mesh = lung_grid.contour([0.5], scalars="values")
        if lung_mesh.n_points > 0:
            lung_mesh = lung_mesh.smooth(n_iter=30, relaxation_factor=0.05)
    else:
        lung_grid = pv.ImageData()
        lung_grid.dimensions = ct_data_masked.shape
        lung_grid.spacing = spacing
        lung_grid.point_data["values"] = ct_data_masked.flatten(order="F")
        lung_mesh = lung_grid.contour([lung_threshold], scalars="values")
        if lung_mesh.n_points > 0:
            lung_mesh = lung_mesh.smooth(n_iter=20, relaxation_factor=0.03)

    logger.info(f"  肺部网格: {lung_mesh.n_points:,} 点")

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
        logger.info(f"  病灶网格: {lesion_mesh.n_points:,} 点")

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 定义视角
    views = {
        'x': ('yz', 'X 轴视角（矢状面）'),
        'y': ('xz', 'Y 轴视角（冠状面）'),
        'z': ('xy', 'Z 轴视角（横断面）'),
    }

    # 渲染三个视角
    for view_name, (camera_pos, desc) in views.items():
        logger.debug(f"渲染 {desc}...")

        plotter = pv.Plotter(window_size=window_size, off_screen=True)
        plotter.set_background("white")

        if lung_mesh.n_points > 0:
            plotter.add_mesh(lung_mesh, color=lung_color, opacity=lung_opacity, smooth_shading=True)

        if lesion_mesh is not None and lesion_mesh.n_points > 0:
            plotter.add_mesh(lesion_mesh, color=lesion_color, opacity=lesion_opacity, smooth_shading=True)

        plotter.camera_position = camera_pos
        plotter.camera.zoom(1.3)
        plotter.add_axes()

        output_path = output_dir / f"{output_prefix}_view_{view_name}.png"
        plotter.screenshot(str(output_path))
        plotter.close()

        logger.info(f"  ✅ 已保存: {output_path}")

    return True


def render_template_only(
    ct_path: Union[str, Path],
    lung_mask_path: Union[str, Path],
    output_prefix: str,
    output_dir: Union[str, Path] = "data/04_final_viz",
    lung_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
    lung_opacity: float = 0.3,
    window_size: Tuple[int, int] = (1920, 1080),
    use_mask_surface: bool = True
) -> bool:
    """
    生成模板肺部（仅肺，无病灶）的 X/Y/Z 三个视角的渲染图

    此函数从 generate_multiview_comparison.py 整合而来。

    Args:
        ct_path: CT 文件路径
        lung_mask_path: 肺部 mask 路径
        output_prefix: 输出文件前缀
        output_dir: 输出目录
        lung_color: 肺部颜色 RGB
        lung_opacity: 肺部透明度 (0-1)
        window_size: 窗口大小
        use_mask_surface: 是否使用 mask 表面渲染

    Returns:
        success: 是否成功

    Example:
        >>> render_template_only(
        ...     ct_path="data/02_atlas/temp_template.nii.gz",
        ...     lung_mask_path="data/02_atlas/temp_template_lung_mask.nii.gz",
        ...     output_prefix="template",
        ...     output_dir="data/04_final_viz"
        ... )
    """
    if not check_pyvista_available():
        return False

    logger.info(f"模板渲染: {output_prefix}")

    ct_path = Path(ct_path)
    lung_mask_path = Path(lung_mask_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ct_path.exists() or not lung_mask_path.exists():
        logger.error("CT 或肺 mask 文件不存在")
        return False

    # 加载数据
    ct_data = load_nifti(ct_path)
    lung_mask = load_nifti(lung_mask_path)

    logger.info(f"  CT 形状: {ct_data.shape}")
    logger.info(f"  肺 mask: {np.sum(lung_mask > 0):,} 体素")

    # 清理 mask - 保留左右肺两个连通分量
    lung_mask_binary = clean_lung_mask(lung_mask, keep_largest_n=2).astype(float)

    # 创建网格
    spacing = (1, 1, 1)

    if use_mask_surface:
        grid = pv.ImageData()
        grid.dimensions = lung_mask_binary.shape
        grid.spacing = spacing
        grid.origin = (0, 0, 0)
        grid.point_data["values"] = lung_mask_binary.flatten(order="F")
        lung_surface = grid.contour([0.5], scalars="values")
        if lung_surface.n_points > 0:
            lung_surface = lung_surface.smooth(n_iter=30, relaxation_factor=0.05)
    else:
        ct_masked = ct_data.copy()
        ct_masked[lung_mask_binary == 0] = -1000
        grid = pv.ImageData()
        grid.dimensions = ct_masked.shape
        grid.spacing = spacing
        grid.origin = (0, 0, 0)
        grid.point_data["values"] = ct_masked.flatten(order="F")
        lung_surface = grid.contour([-500], scalars="values")
        if lung_surface.n_points > 0:
            lung_surface = lung_surface.smooth(n_iter=20, relaxation_factor=0.03)

    logger.info(f"  肺部网格: {lung_surface.n_points:,} 点")

    # 视角定义
    views = {
        "x": {"vector": (1, 0, 0), "viewup": (0, 0, 1)},
        "y": {"vector": (0, 1, 0), "viewup": (0, 0, 1)},
        "z": {"vector": (0, 0, 1), "viewup": (0, 1, 0)},
    }

    for view_name, view_params in views.items():
        logger.debug(f"渲染 {view_name.upper()} 轴视角...")

        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.set_background("white")
        plotter.add_mesh(lung_surface, color=lung_color, opacity=lung_opacity, smooth_shading=True)
        plotter.view_vector(vector=view_params["vector"], viewup=view_params["viewup"])
        plotter.camera.zoom(1.3)
        plotter.add_axes()

        output_path = output_dir / f"{output_prefix}_view_{view_name}.png"
        plotter.screenshot(str(output_path))
        plotter.close()

        logger.info(f"  ✅ 已保存: {output_path}")

    return True


def generate_slice_visualization(
    nifti_path: Union[str, Path],
    output_dir: Union[str, Path],
    prefix: str = "template",
    window_center: float = -600,
    window_width: float = 1500,
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 150,
    cmap: str = "gray"
) -> List[Path]:
    """
    生成 NIfTI 文件的三轴切片可视化图像

    从 X、Y、Z 三个轴向分别生成中心切片的 PNG 图像，
    使用指定的窗宽窗位设置（默认为肺窗）。

    Args:
        nifti_path: NIfTI 文件路径
        output_dir: 输出目录
        prefix: 文件名前缀
        window_center: 窗位（默认 -600 HU，肺窗）
        window_width: 窗宽（默认 1500 HU）
        figsize: 图像尺寸
        dpi: 图像分辨率
        cmap: 颜色映射（默认灰度）

    Returns:
        output_paths: 生成的 PNG 文件路径列表

    Example:
        >>> paths = generate_slice_visualization(
        ...     "data/02_atlas/standard_template.nii.gz",
        ...     "data/02_atlas/visualizations",
        ...     prefix="template"
        ... )
        >>> print(paths)
        ['template_axial.png', 'template_coronal.png', 'template_sagittal.png']
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非 GUI 后端
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib 未安装，请运行: pip install matplotlib")
        return []

    nifti_path = Path(nifti_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not nifti_path.exists():
        logger.error(f"文件不存在: {nifti_path}")
        return []

    # 加载数据
    data = load_nifti(nifti_path)
    logger.info(f"加载数据: {nifti_path.name}")
    logger.info(f"  形状: {data.shape}")
    logger.info(f"  HU 范围: [{data.min():.0f}, {data.max():.0f}]")

    # 计算窗宽窗位的显示范围
    vmin = window_center - window_width / 2
    vmax = window_center + window_width / 2

    # 定义三个轴向的切片
    slices_info = [
        {
            "name": "axial",
            "title": "Axial View (Z-axis)",
            "axis": 2,
            "xlabel": "X (Left -> Right)",
            "ylabel": "Y (Posterior -> Anterior)"
        },
        {
            "name": "coronal",
            "title": "Coronal View (Y-axis)",
            "axis": 1,
            "xlabel": "X (Left -> Right)",
            "ylabel": "Z (Inferior -> Superior)"
        },
        {
            "name": "sagittal",
            "title": "Sagittal View (X-axis)",
            "axis": 0,
            "xlabel": "Y (Posterior -> Anterior)",
            "ylabel": "Z (Inferior -> Superior)"
        }
    ]

    output_paths = []

    for info in slices_info:
        # 获取中心切片索引
        axis = info["axis"]
        center_idx = data.shape[axis] // 2

        # 提取切片
        if axis == 0:
            slice_data = data[center_idx, :, :]
        elif axis == 1:
            slice_data = data[:, center_idx, :]
        else:  # axis == 2
            slice_data = data[:, :, center_idx]

        # 对于冠状面和矢状面，需要旋转以正确显示
        if axis in [1, 2]:
            slice_data = np.rot90(slice_data)

        # 创建图像
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        im = ax.imshow(
            slice_data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='equal'
        )

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='HU')

        # 添加标题和标签
        ax.set_title(f"{info['title']}\nSlice {center_idx}/{data.shape[axis]}", fontsize=14)
        ax.set_xlabel(info['xlabel'], fontsize=11)
        ax.set_ylabel(info['ylabel'], fontsize=11)

        # 添加窗宽窗位信息
        ax.text(
            0.02, 0.98,
            f"Window: C={window_center:.0f} W={window_width:.0f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        plt.tight_layout()

        # 保存
        output_path = output_dir / f"{prefix}_{info['name']}.png"
        plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)

        output_paths.append(output_path)
        logger.info(f"  ✅ 已保存: {output_path}")

    logger.info(f"切片可视化完成: 共 {len(output_paths)} 张图像")
    return output_paths


def main(config: dict) -> None:
    """主函数"""
    paths = config.get('paths', {})

    ct_dir = Path(paths.get('final_viz', 'data/04_final_viz'))
    output_dir = ct_dir / 'renders'

    batch_render(ct_dir, output_dir=output_dir)


if __name__ == "__main__":
    import yaml

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    main(config)

