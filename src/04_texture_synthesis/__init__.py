"""
阶段三(下)：AI 纹理融合模块

使用深度学习 (U-Net Inpainting / Patch-GAN) 在病灶区域生成真实纹理
"""

from .dataset import LungPatchDataset
from .network import InpaintingUNet, PatchDiscriminator
from .losses import InpaintingLoss
from .train import Trainer
from .inference_fuse import fuse_lesion

__all__ = [
    'LungPatchDataset',
    'InpaintingUNet',
    'PatchDiscriminator',
    'InpaintingLoss',
    'Trainer',
    'fuse_lesion',
]

