#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
静态 3D 渲染模块

使用 PyVista 进行肺部 + 病灶的双通道体渲染
"""

from pathlib import Path
from typing import Union, Tuple, Optional, List

import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None

from ..utils.io import load_nifti
from ..utils.logger import get_logger

logger = get_logger(__name__)


def check_pyvista_available() -> bool:
    """检查 PyVista 是否可用"""
    if pv is None:
        logger.error("PyVista 未安装，请运行: pip install pyvista")
        return False
    return True


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


def main(config: dict) -> None:
    """主函数"""
    paths = config.get('paths', {})
    viz_config = config.get('visualization', {})
    
    ct_dir = Path(paths.get('final_viz', 'data/04_final_viz'))
    output_dir = ct_dir / 'renders'
    
    batch_render(ct_dir, output_dir=output_dir)


if __name__ == "__main__":
    import yaml
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)

