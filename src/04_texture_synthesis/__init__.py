"""
阶段三(下)：AI 纹理融合模块

使用深度学习 (U-Net Inpainting / Partial Conv / Patch-GAN) 在病灶区域生成真实纹理

支持三种模型架构：
- 基线方案: InpaintingUNet (3D U-Net)
- 进阶方案: PartialConvUNet (3D Partial Convolution)
- 高级方案: PatchGAN (InpaintingUNet + PatchDiscriminator)
"""

from .dataset import LungPatchDataset
from .network import (
    InpaintingUNet,
    PartialConvUNet,
    PatchDiscriminator,
    create_model,
    count_parameters,
)
from .losses import InpaintingLoss
from .train import Trainer
from .inference_fuse import fuse_lesion

__all__ = [
    'LungPatchDataset',
    'InpaintingUNet',
    'PartialConvUNet',
    'PatchDiscriminator',
    'create_model',
    'count_parameters',
    'InpaintingLoss',
    'Trainer',
    'fuse_lesion',
]

