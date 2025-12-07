#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
呼吸动画模块

使用 sin(t) 函数模拟肺部呼吸运动，生成动态 3D 动画
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
from .static_render import create_lung_mesh, create_lesion_mesh

logger = get_logger(__name__)


def apply_breathing_deformation(
    mesh: 'pv.PolyData',
    t: float,
    amplitude: float = 0.1,
    frequency: float = 1.0
) -> 'pv.PolyData':
    """
    应用呼吸变形
    
    使用 sin(t) 函数模拟呼吸运动:
    - 吸气时肺部膨胀 (向外扩张)
    - 呼气时肺部收缩 (向内收缩)
    
    Args:
        mesh: 输入网格
        t: 时间参数 (0 到 2π 为一个呼吸周期)
        amplitude: 变形幅度 (相对于原始大小的比例)
        frequency: 呼吸频率
        
    Returns:
        deformed_mesh: 变形后的网格
    """
    # 计算缩放因子
    scale = 1.0 + amplitude * np.sin(frequency * t)
    
    # 获取网格中心
    center = mesh.center
    
    # 复制网格
    deformed = mesh.copy()
    
    # 应用缩放变形 (相对于中心)
    points = deformed.points.copy()
    points = center + (points - center) * scale
    deformed.points = points
    
    return deformed


def create_breathing_animation(
    ct_path: Union[str, Path],
    lesion_mask_path: Optional[Union[str, Path]] = None,
    output_path: Union[str, Path] = "breathing_animation.gif",
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_frames: int = 30,
    fps: int = 10,
    amplitude: float = 0.08,
    lung_color: Tuple[float, float, float] = (0.7, 0.7, 0.7),
    lesion_color: Tuple[float, float, float] = (1.0, 0.2, 0.2),
    lung_opacity: float = 0.3,
    lesion_opacity: float = 0.8,
    window_size: Tuple[int, int] = (800, 600)
) -> str:
    """
    创建呼吸动画 GIF
    
    Args:
        ct_path: CT 文件路径
        lesion_mask_path: 病灶 mask 路径 (可选)
        output_path: 输出 GIF 路径
        spacing: 体素间距
        num_frames: 帧数
        fps: 帧率
        amplitude: 呼吸幅度
        lung_color: 肺部颜色
        lesion_color: 病灶颜色
        lung_opacity: 肺部透明度
        lesion_opacity: 病灶透明度
        window_size: 窗口大小
        
    Returns:
        output_path: 生成的 GIF 路径
    """
    if pv is None:
        raise ImportError("PyVista 未安装")
    
    logger.info(f"创建呼吸动画: {Path(ct_path).name}")
    
    # 加载数据
    ct_data = load_nifti(ct_path)
    
    # 创建基础网格
    lung_mesh = create_lung_mesh(ct_data, spacing)
    
    lesion_mesh = None
    if lesion_mask_path is not None:
        lesion_mask = load_nifti(lesion_mask_path)
        if np.sum(lesion_mask) > 0:
            lesion_mesh = create_lesion_mesh(lesion_mask, spacing)
    
    # 创建渲染器
    plotter = pv.Plotter(window_size=window_size, off_screen=True)
    plotter.set_background("white")
    
    # 打开 GIF 写入
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.open_gif(str(output_path), fps=fps)
    
    # 生成每一帧
    for frame in range(num_frames):
        t = 2 * np.pi * frame / num_frames
        
        plotter.clear()
        
        # 变形肺部
        deformed_lung = apply_breathing_deformation(lung_mesh, t, amplitude)
        plotter.add_mesh(
            deformed_lung,
            color=lung_color,
            opacity=lung_opacity,
            smooth_shading=True
        )
        
        # 变形病灶
        if lesion_mesh is not None:
            deformed_lesion = apply_breathing_deformation(lesion_mesh, t, amplitude)
            plotter.add_mesh(
                deformed_lesion,
                color=lesion_color,
                opacity=lesion_opacity,
                smooth_shading=True
            )
        
        # 添加时间标签
        phase = "吸气" if np.sin(t) > 0 else "呼气"
        plotter.add_text(f"呼吸周期: {phase}", font_size=10)
        
        # 设置相机
        plotter.camera_position = 'xy'
        
        # 写入帧
        plotter.write_frame()
    
    plotter.close()
    
    logger.info(f"动画已保存: {output_path}")
    return str(output_path)


def render_with_breathing(
    ct_path: Union[str, Path],
    lesion_mask_path: Optional[Union[str, Path]] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    amplitude: float = 0.08,
    frequency: float = 1.0,
    lung_color: Tuple[float, float, float] = (0.7, 0.7, 0.7),
    lesion_color: Tuple[float, float, float] = (1.0, 0.2, 0.2),
    window_size: Tuple[int, int] = (1920, 1080)
) -> None:
    """
    交互式呼吸动画渲染
    
    实时显示带呼吸动画的 3D 肺部模型
    
    Args:
        ct_path: CT 文件路径
        lesion_mask_path: 病灶 mask 路径
        spacing: 体素间距
        amplitude: 呼吸幅度
        frequency: 呼吸频率
        lung_color: 肺部颜色
        lesion_color: 病灶颜色
        window_size: 窗口大小
    """
    if pv is None:
        raise ImportError("PyVista 未安装")
    
    logger.info("启动交互式呼吸动画...")
    
    # 加载数据
    ct_data = load_nifti(ct_path)
    lung_mesh = create_lung_mesh(ct_data, spacing)
    
    lesion_mesh = None
    if lesion_mask_path is not None:
        lesion_mask = load_nifti(lesion_mask_path)
        if np.sum(lesion_mask) > 0:
            lesion_mesh = create_lesion_mesh(lesion_mask, spacing)
    
    # 创建渲染器
    plotter = pv.Plotter(window_size=window_size)
    plotter.set_background("white")
    
    # 添加初始网格
    lung_actor = plotter.add_mesh(
        lung_mesh,
        color=lung_color,
        opacity=0.3,
        smooth_shading=True,
        name="lung"
    )
    
    if lesion_mesh is not None:
        lesion_actor = plotter.add_mesh(
            lesion_mesh,
            color=lesion_color,
            opacity=0.8,
            smooth_shading=True,
            name="lesion"
        )
    
    plotter.camera_position = 'xy'
    plotter.add_axes()
    
    # 动画回调
    t = [0.0]  # 使用列表以便在闭包中修改
    
    def update_animation():
        t[0] += 0.1
        
        # 更新肺部
        deformed_lung = apply_breathing_deformation(lung_mesh, t[0], amplitude, frequency)
        plotter.update_coordinates(deformed_lung.points, mesh=lung_mesh, render=False)
        
        # 更新病灶
        if lesion_mesh is not None:
            deformed_lesion = apply_breathing_deformation(lesion_mesh, t[0], amplitude, frequency)
            plotter.update_coordinates(deformed_lesion.points, mesh=lesion_mesh, render=False)
    
    # 添加定时器回调
    plotter.add_callback(update_animation, interval=50)
    
    # 显示
    plotter.show()


def batch_create_animations(
    ct_dir: Union[str, Path],
    mask_dir: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "animations",
    pattern: str = "*_fused.nii.gz"
) -> int:
    """
    批量创建呼吸动画
    """
    ct_dir = Path(ct_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ct_files = list(ct_dir.glob(pattern))
    logger.info(f"找到 {len(ct_files)} 个 CT 文件")
    
    count = 0
    for ct_path in ct_files:
        stem = ct_path.stem.replace('.nii', '')
        output_path = output_dir / f"{stem}_breathing.gif"
        
        # 查找对应的 mask
        mask_path = None
        if mask_dir is not None:
            mask_dir = Path(mask_dir)
            for suffix in ['_lesion', '_emphysema']:
                candidate = mask_dir / f"{stem}{suffix}.nii.gz"
                if candidate.exists():
                    mask_path = candidate
                    break
        
        try:
            create_breathing_animation(
                ct_path=ct_path,
                lesion_mask_path=mask_path,
                output_path=output_path
            )
            count += 1
        except Exception as e:
            logger.error(f"创建动画失败 {ct_path.name}: {e}")
    
    logger.info(f"批量创建完成: {count}/{len(ct_files)}")
    return count


def main(config: dict) -> None:
    """主函数"""
    paths = config.get('paths', {})
    viz_config = config.get('visualization', {})
    
    ct_dir = Path(paths.get('final_viz', 'data/04_final_viz'))
    output_dir = ct_dir / 'animations'
    
    batch_create_animations(ct_dir, output_dir=output_dir)


if __name__ == "__main__":
    import yaml
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)

