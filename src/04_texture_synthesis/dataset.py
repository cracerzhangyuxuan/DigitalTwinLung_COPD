#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集模块

用于 Inpainting 训练的 Patch 数据集
"""

from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    Dataset = object

from ..utils.io import load_nifti
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LungPatchDataset(Dataset):
    """
    肺部 Patch 数据集
    
    从肺部 CT 中提取 3D patch，用于训练 Inpainting 网络
    """
    
    def __init__(
        self,
        ct_paths: List[Union[str, Path]],
        mask_paths: List[Union[str, Path]],
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        patches_per_volume: int = 50,
        min_mask_ratio: float = 0.1,
        max_mask_ratio: float = 0.5,
        augment: bool = True,
        normalize: bool = True,
        hu_min: float = -1000,
        hu_max: float = 400
    ):
        """
        Args:
            ct_paths: CT 文件路径列表
            mask_paths: 对应的病灶 mask 路径列表
            patch_size: Patch 大小 (D, H, W)
            patches_per_volume: 每个体积提取的 patch 数
            min_mask_ratio: patch 中 mask 的最小比例
            max_mask_ratio: patch 中 mask 的最大比例
            augment: 是否进行数据增强
            normalize: 是否归一化
            hu_min: HU 归一化最小值
            hu_max: HU 归一化最大值
        """
        if torch is None:
            raise ImportError("请安装 PyTorch: pip install torch")
        
        self.ct_paths = [Path(p) for p in ct_paths]
        self.mask_paths = [Path(p) for p in mask_paths]
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.augment = augment
        self.normalize = normalize
        self.hu_min = hu_min
        self.hu_max = hu_max
        
        # 预提取 patch 索引
        self.patch_indices = self._generate_patch_indices()
        
        logger.info(
            f"数据集初始化: {len(self.ct_paths)} 个体积, "
            f"{len(self.patch_indices)} 个 patch"
        )
    
    def _generate_patch_indices(self) -> List[Tuple[int, Tuple[int, int, int]]]:
        """生成有效的 patch 索引"""
        indices = []
        
        for vol_idx, (ct_path, mask_path) in enumerate(zip(self.ct_paths, self.mask_paths)):
            try:
                ct = load_nifti(ct_path)
                mask = load_nifti(mask_path)
                
                # 在 mask 区域内采样
                valid_positions = self._find_valid_positions(ct, mask)
                
                # 随机选择位置
                if len(valid_positions) > 0:
                    selected = np.random.choice(
                        len(valid_positions),
                        min(self.patches_per_volume, len(valid_positions)),
                        replace=False
                    )
                    for idx in selected:
                        indices.append((vol_idx, valid_positions[idx]))
                        
            except Exception as e:
                logger.warning(f"处理失败 {ct_path.name}: {e}")
        
        return indices
    
    def _find_valid_positions(
        self,
        ct: np.ndarray,
        mask: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """找到有效的 patch 中心位置"""
        positions = []
        
        d, h, w = ct.shape
        pd, ph, pw = self.patch_size
        
        # 在包含 mask 的区域内采样
        mask_coords = np.where(mask > 0)
        
        if len(mask_coords[0]) == 0:
            return positions
        
        # 随机采样候选位置
        num_candidates = min(1000, len(mask_coords[0]))
        candidate_indices = np.random.choice(
            len(mask_coords[0]), num_candidates, replace=False
        )
        
        for idx in candidate_indices:
            z = mask_coords[0][idx]
            y = mask_coords[1][idx]
            x = mask_coords[2][idx]
            
            # 确保 patch 在体积范围内
            z = max(pd // 2, min(z, d - pd // 2))
            y = max(ph // 2, min(y, h - ph // 2))
            x = max(pw // 2, min(x, w - pw // 2))
            
            positions.append((z, y, x))
        
        return positions
    
    def _extract_patch(
        self,
        volume: np.ndarray,
        center: Tuple[int, int, int]
    ) -> np.ndarray:
        """从体积中提取 patch"""
        z, y, x = center
        pd, ph, pw = self.patch_size
        
        patch = volume[
            z - pd // 2: z + pd // 2,
            y - ph // 2: y + ph // 2,
            x - pw // 2: x + pw // 2
        ]
        
        return patch
    
    def _normalize_ct(self, data: np.ndarray) -> np.ndarray:
        """归一化 CT 数据到 [0, 1]"""
        data = np.clip(data, self.hu_min, self.hu_max)
        return (data - self.hu_min) / (self.hu_max - self.hu_min)
    
    def _augment(
        self,
        ct_patch: np.ndarray,
        mask_patch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据增强

        包含：随机翻转、随机旋转、高斯噪声、强度缩放
        """
        # 随机翻转
        for axis in range(3):
            if np.random.random() > 0.5:
                ct_patch = np.flip(ct_patch, axis=axis)
                mask_patch = np.flip(mask_patch, axis=axis)

        # 随机旋转 (90度增量)
        k = np.random.randint(0, 4)
        ct_patch = np.rot90(ct_patch, k, axes=(1, 2))
        mask_patch = np.rot90(mask_patch, k, axes=(1, 2))

        # 高斯噪声 (30% 概率)
        if np.random.random() < 0.3:
            noise_std = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_std, ct_patch.shape)
            ct_patch = ct_patch + noise

        # 强度缩放 (30% 概率)
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            ct_patch = ct_patch * scale

        return ct_patch.copy(), mask_patch.copy()
    
    def __len__(self) -> int:
        return len(self.patch_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, 'torch.Tensor']:
        vol_idx, center = self.patch_indices[idx]
        
        # 加载数据
        ct = load_nifti(self.ct_paths[vol_idx])
        mask = load_nifti(self.mask_paths[vol_idx])
        
        # 提取 patch
        ct_patch = self._extract_patch(ct, center)
        mask_patch = self._extract_patch(mask, center)
        
        # 数据增强
        if self.augment:
            ct_patch, mask_patch = self._augment(ct_patch, mask_patch)
        
        # 归一化
        if self.normalize:
            ct_patch = self._normalize_ct(ct_patch)
        
        # 创建输入 (mask 区域置为 0)
        input_patch = ct_patch.copy()
        input_patch[mask_patch > 0] = 0
        
        # 转换为 tensor
        return {
            'input': torch.from_numpy(input_patch[np.newaxis]).float(),
            'target': torch.from_numpy(ct_patch[np.newaxis]).float(),
            'mask': torch.from_numpy(mask_patch[np.newaxis]).float(),
        }


def create_dataloader(
    ct_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    batch_size: int = 4,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    num_workers: int = 4,
    train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证 DataLoader
    """
    ct_dir = Path(ct_dir)
    mask_dir = Path(mask_dir)
    
    ct_files = sorted(ct_dir.glob("*.nii.gz"))
    mask_files = sorted(mask_dir.glob("*.nii.gz"))
    
    # 划分训练/验证集
    n_train = int(len(ct_files) * train_ratio)
    
    train_dataset = LungPatchDataset(
        ct_paths=ct_files[:n_train],
        mask_paths=mask_files[:n_train],
        patch_size=patch_size,
        augment=True
    )
    
    val_dataset = LungPatchDataset(
        ct_paths=ct_files[n_train:],
        mask_paths=mask_files[n_train:],
        patch_size=patch_size,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader

