#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化工具模块

基于 PyVista 的 3D 可视化封装函数
"""

from typing import Tuple, Optional, List, Union

import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None


def create_volume_actor(
    data: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    opacity: Union[float, str, List] = 'linear',
    cmap: str = 'gray',
    clim: Optional[Tuple[float, float]] = None
) -> 'pv.Actor':
    """
    创建体渲染 Actor
    
    Args:
        data: 3D 体数据
        spacing: 体素间距
        opacity: 透明度设置
        cmap: 颜色映射
        clim: 颜色范围
        
    Returns:
        actor: PyVista Actor
    """
    if pv is None:
        raise ImportError("请安装 pyvista: pip install pyvista")
    
    grid = pv.ImageData()
    grid.dimensions = np.array(data.shape) + 1
    grid.spacing = spacing
    grid.cell_data['values'] = data.flatten(order='F')
    
    return grid


def render_lung_with_lesion(
    lung_data: np.ndarray,
    lesion_mask: Optional[np.ndarray] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    lung_opacity: float = 0.3,
    lesion_opacity: float = 0.8,
    lung_color: Tuple[float, float, float] = (0.7, 0.7, 0.7),
    lesion_color: Tuple[float, float, float] = (1.0, 0.2, 0.2),
    window_size: Tuple[int, int] = (1920, 1080),
    background: str = 'white',
    show: bool = True,
    screenshot_path: Optional[str] = None
) -> Optional['pv.Plotter']:
    """
    渲染肺部 + 病灶的双通道可视化
    
    Args:
        lung_data: 肺部 CT 数据
        lesion_mask: 病灶 mask
        spacing: 体素间距
        lung_opacity: 肺部透明度
        lesion_opacity: 病灶透明度
        lung_color: 肺部颜色 (R, G, B)
        lesion_color: 病灶颜色 (R, G, B)
        window_size: 窗口大小
        background: 背景颜色
        show: 是否显示窗口
        screenshot_path: 截图保存路径
        
    Returns:
        plotter: PyVista Plotter 对象
    """
    if pv is None:
        raise ImportError("请安装 pyvista: pip install pyvista")
    
    plotter = pv.Plotter(window_size=window_size, off_screen=not show)
    plotter.set_background(background)
    
    # 创建肺部网格
    lung_grid = pv.ImageData()
    lung_grid.dimensions = np.array(lung_data.shape) + 1
    lung_grid.spacing = spacing
    lung_grid.cell_data['lung'] = lung_data.flatten(order='F')
    
    # 添加肺部体渲染
    plotter.add_volume(
        lung_grid,
        scalars='lung',
        opacity='linear',
        cmap='gray',
        shade=True,
        opacity_unit_distance=10,
    )
    
    # 如果有病灶 mask，添加病灶渲染
    if lesion_mask is not None:
        # 提取病灶表面
        lesion_grid = pv.ImageData()
        lesion_grid.dimensions = np.array(lesion_mask.shape) + 1
        lesion_grid.spacing = spacing
        lesion_grid.cell_data['lesion'] = lesion_mask.flatten(order='F')
        
        # 使用等值面提取病灶表面
        contour = lesion_grid.contour([0.5], scalars='lesion')
        if contour.n_points > 0:
            plotter.add_mesh(
                contour,
                color=lesion_color,
                opacity=lesion_opacity,
                smooth_shading=True,
            )
    
    # 设置相机
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.2)
    
    # 保存截图
    if screenshot_path:
        plotter.screenshot(screenshot_path)
    
    if show:
        plotter.show()
    
    return plotter


def create_animation_frames(
    lung_data: np.ndarray,
    lesion_mask: Optional[np.ndarray] = None,
    num_frames: int = 60,
    amplitude: float = 0.1,
    copd_delay: float = 0.3,
    output_dir: Optional[str] = None
) -> List[np.ndarray]:
    """
    生成呼吸动画帧
    
    Args:
        lung_data: 肺部数据
        lesion_mask: 病灶 mask
        num_frames: 帧数
        amplitude: 形变幅度
        copd_delay: COPD 呼气延迟
        output_dir: 输出目录
        
    Returns:
        frames: 帧列表
    """
    if pv is None:
        raise ImportError("请安装 pyvista: pip install pyvista")
    
    frames = []
    
    for i in range(num_frames):
        t = i / num_frames * 2 * np.pi
        
        # 正弦波形变因子
        scale_factor = 1.0 + amplitude * np.sin(t)
        
        # 简单的缩放形变 (实际应用中需要更复杂的形变)
        # 这里只是示例
        frame_data = lung_data * scale_factor
        
        frames.append(frame_data)
    
    return frames


def save_screenshot(
    plotter: 'pv.Plotter',
    filepath: str,
    scale: int = 2
) -> None:
    """
    保存高清截图
    
    Args:
        plotter: PyVista Plotter
        filepath: 保存路径
        scale: 分辨率倍数
    """
    plotter.screenshot(filepath, scale=scale)

