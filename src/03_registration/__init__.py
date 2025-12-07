"""
阶段三(上)：空间配准模块

将 COPD 患者的病灶 mask 配准到标准模板空间
"""

from .register_lesions import register_to_template, warp_mask, batch_register

__all__ = ['register_to_template', 'warp_mask', 'batch_register']

