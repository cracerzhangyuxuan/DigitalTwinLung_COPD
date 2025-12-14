"""
阶段一：数据预处理模块

包含:
- run_segmentation: 调用 TotalSegmentator 进行肺部分割
- clean_background: 去除骨骼等背景，保留纯净肺部
- extract_emphysema: 使用 LAA-950 算法提取肺气肿病灶
- simple_lung_segment: 简化版肺部分割
"""

from .run_segmentation import (
    run_segmentation,
    batch_segmentation,
    combine_lung_masks,
    check_gpu_available,
    check_totalsegmentator_available,
    get_default_method,
    get_default_device,
    run_totalsegmentator_batch,
    run_threshold_batch,
)
from .clean_background import clean_background, replace_background
from .extract_emphysema import extract_emphysema_mask, compute_laa950
from .simple_lung_segment import (
    segment_lung_from_file,
    batch_segment_lungs,
    threshold_lung_segmentation,
)

__all__ = [
    # 环境检查
    'check_gpu_available',
    'check_totalsegmentator_available',
    'get_default_method',
    'get_default_device',
    # 批量分割
    'run_totalsegmentator_batch',
    'run_threshold_batch',
    'run_segmentation',
    'batch_segmentation',
    'batch_segment_lungs',
    'combine_lung_masks',
    # 单文件分割
    'segment_lung_from_file',
    'threshold_lung_segmentation',
    # 背景清洗
    'clean_background',
    'replace_background',
    # 病灶提取
    'extract_emphysema_mask',
    'compute_laa950',
]

