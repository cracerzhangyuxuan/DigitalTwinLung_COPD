"""
阶段一：数据预处理模块

包含:
- run_segmentation: 调用 TotalSegmentator 进行肺部分割
- clean_background: 去除骨骼等背景，保留纯净肺部
- extract_emphysema: 使用 LAA-950 算法提取肺气肿病灶
"""

from .run_segmentation import run_segmentation, batch_segmentation
from .clean_background import clean_background, replace_background
from .extract_emphysema import extract_emphysema_mask, compute_laa950

__all__ = [
    'run_segmentation',
    'batch_segmentation',
    'clean_background',
    'replace_background',
    'extract_emphysema_mask',
    'compute_laa950',
]

