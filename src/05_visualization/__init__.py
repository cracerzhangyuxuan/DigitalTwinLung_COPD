"""
阶段四：3D 可视化模块

使用 PyVista 进行 3D 体渲染和呼吸动画
"""

from .static_render import (
    render_static,
    render_comparison,
    render_multiview,
    render_template_only,
    generate_slice_visualization,
    render_atlas_complete_3views
)
from .dynamic_breath import create_breathing_animation, render_with_breathing

__all__ = [
    'render_static',
    'render_comparison',
    'render_multiview',
    'render_template_only',
    'generate_slice_visualization',
    'render_atlas_complete_3views',
    'create_breathing_animation',
    'render_with_breathing',
]

