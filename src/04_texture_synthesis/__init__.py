"""
阶段三(下)：AI 纹理融合模块

使用深度学习在病灶区域生成真实纹理

支持六种模型架构：
- 基线方案: InpaintingUNet (3D U-Net)
- 进阶方案: PartialConvUNet (3D Partial Convolution)
- 高级方案: PatchGAN (InpaintingUNet + PatchDiscriminator)
- 注意力增强方案: AttGAN (AttentionUNet + PatchDiscriminator)
- 自监督预训练方案: MAE-PatchGAN (InpaintingUNet + PatchDiscriminator, 预训练权重)
- 扩散模型方案: DDPM (DiffusionUNet)
"""

from .dataset import LungPatchDataset
from .network import (
    InpaintingUNet,
    PartialConvUNet,
    PatchDiscriminator,
    AttentionUNet,
    DiffusionUNet,
    create_model,
    count_parameters,
)
from .losses import InpaintingLoss
from .train import Trainer
from .diffusion_trainer import DiffusionTrainer, NoiseScheduler
from .mae_pretrain import MAEPretrainer, MAEMaskGenerator
from .inference_fuse import fuse_lesion, ddpm_inpaint_patch

__all__ = [
    'LungPatchDataset',
    'InpaintingUNet',
    'PartialConvUNet',
    'PatchDiscriminator',
    'AttentionUNet',
    'DiffusionUNet',
    'create_model',
    'count_parameters',
    'InpaintingLoss',
    'Trainer',
    'DiffusionTrainer',
    'NoiseScheduler',
    'MAEPretrainer',
    'MAEMaskGenerator',
    'fuse_lesion',
    'ddpm_inpaint_patch',
]

