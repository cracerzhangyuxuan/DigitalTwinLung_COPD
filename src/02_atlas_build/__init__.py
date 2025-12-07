"""
阶段二：标准底座构建模块

使用 ANTsPy 从多例正常肺 CT 构建标准模板 (Atlas)
"""

from .build_template_ants import build_template, load_template

__all__ = ['build_template', 'load_template']

