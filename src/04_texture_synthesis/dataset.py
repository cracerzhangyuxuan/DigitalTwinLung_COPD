#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集模块

用于 Inpainting 训练的 Patch 数据集
"""

from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict
from functools import lru_cache

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


def _load_nifti_cached(path_str: str) -> np.ndarray:
    """带缓存的 NIfTI 加载，避免同一体积被重复读取磁盘

    缓存策略：LRU 缓存最多 64 个体积（CT + mask 各算一个）
    29 例患者 × 2 文件 = 58 个，64 足够覆盖全部训练+验证数据
    首次加载后全部驻留内存，后续 epoch 零磁盘 IO
    """
    return load_nifti(path_str).copy()  # copy 确保缓存数据不被意外修改


# 使用模块级 lru_cache 装饰（不能直接装饰因为 np.ndarray 不可哈希作为参数，
# 但 path_str 是字符串，可以作为 key）
_load_nifti_cached = lru_cache(maxsize=64)(_load_nifti_cached)


class LungPatchDataset(Dataset):
    """
    肺部 Patch 数据集（CICI-FiLM 条件化版本）

    从肺部 CT 中提取 3D patch，用于训练 Inpainting 网络

    CICI-FiLM 阶段②改造：
        - 支持加载患者级 5 维条件向量（c₁~c₅）
        - 在 __getitem__ 中返回归一化后的条件向量
        - 条件向量来源：patient_features.json（由 extract_patient_features.py 生成）
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
        hu_max: float = 400,
        patient_features_path: Optional[Union[str, Path]] = None,
        use_condition: bool = False
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
            patient_features_path: 患者特征 JSON 路径（CICI-FiLM 阶段②）
            use_condition: 是否使用条件向量（False=Exp-0, True=Exp-2）
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
        self.use_condition = use_condition

        # 加载患者特征（CICI-FiLM 阶段②）
        self.patient_features = {}
        if use_condition and patient_features_path:
            import json
            with open(patient_features_path, 'r') as f:
                self.patient_features = json.load(f)
            logger.info(f"[CICI-FiLM] 加载患者特征: {len(self.patient_features)} 例")

        # 预提取 patch 索引
        self.patch_indices = self._generate_patch_indices()

        logger.info(
            f"数据集初始化: {len(self.ct_paths)} 个体积, "
            f"{len(self.patch_indices)} 个 patch, "
            f"条件化={'启用' if use_condition else '禁用'}"
        )
    
    def _generate_patch_indices(self) -> List[Tuple[int, Tuple[int, int, int]]]:
        """生成有效的 patch 索引"""
        indices = []

        for vol_idx, (ct_path, mask_path) in enumerate(zip(self.ct_paths, self.mask_paths)):
            try:
                ct = _load_nifti_cached(str(ct_path))
                mask = _load_nifti_cached(str(mask_path))
                
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
    
    def _extract_patient_id(self, ct_path: Path) -> str:
        """从 CT 文件路径提取患者 ID"""
        # 支持格式: copd_XXX_clean.nii.gz, copd_XXX_warped.nii.gz, XXX.nii.gz
        stem = ct_path.stem.replace('.nii', '')
        patient_id = stem.replace('copd_', '').replace('_clean', '').replace('_warped', '')
        return patient_id

    def _get_condition_vector(self, vol_idx: int) -> Optional[np.ndarray]:
        """
        获取患者的 5 维条件向量（归一化后）

        归一化策略（严格对照文档 §2.4）:
            c₁ global_EI:       比率值，/100 即可 → [0, ~0.5]
            c₂ lesion_vol_ratio: 比率值，/100 即可 → [0, ~0.5]
            c₃ lesion_HU_mean:  z-score → sigmoid → (0, 1)
            c₄ lesion_HU_std:   z-score → sigmoid → (0, 1)
            c₅ GOLD:            x / 4 → {0.25, 0.5, 0.75, 1.0}

        c₃/c₄ 的 z-score 参数（μ, σ）应从训练集 23 例计算并固定。
        此处提供初始默认值，可在 patient_features.json 中附带
        'norm_stats' 字段覆盖。

        Returns:
            condition: (5,) 数组，或 None（如果不使用条件）
        """
        if not self.use_condition or not self.patient_features:
            return None

        # 提取患者 ID
        patient_id = self._extract_patient_id(self.ct_paths[vol_idx])

        # 查找患者特征
        if patient_id not in self.patient_features:
            logger.warning(f"未找到患者特征: {patient_id}，使用默认值")
            # 默认值（GOLD 2 中度 COPD）
            features = {
                'global_EI': 20.0,
                'lesion_vol_ratio': 15.0,
                'lesion_HU_mean': -965.0,
                'lesion_HU_std': 45.0,
                'GOLD': 2
            }
        else:
            features = self.patient_features[patient_id]

        # 提取原始值
        c1 = features['global_EI']          # 0-100 (%)
        c2 = features['lesion_vol_ratio']   # 0-100 (%)
        c3 = features['lesion_HU_mean']     # 约 -990 ~ -930 HU
        c4 = features['lesion_HU_std']      # 约 20 ~ 70 HU
        c5 = features['GOLD']               # 1-4

        # ---- 归一化（严格对照文档 §2.4）----

        # c₁, c₂: 比率值，除以 100 → [0, ~0.5]
        c1_norm = c1 / 100.0
        c2_norm = c2 / 100.0

        # c₃, c₄: z-score → sigmoid（对异常值鲁棒，不依赖硬编码上下界）
        # μ, σ 默认值取自文档 §2.4 描述的典型范围中心和跨度:
        #   c₃ 范围 [-990, -930] → μ≈-960, σ≈15
        #   c₄ 范围 [20, 70]    → μ≈45,   σ≈12.5
        # 实际使用时应替换为训练集 23 例的真实统计量
        norm_stats = self.patient_features.get('_norm_stats', {})
        c3_mu = norm_stats.get('c3_mean', -960.0)
        c3_sigma = norm_stats.get('c3_std', 15.0)
        c4_mu = norm_stats.get('c4_mean', 45.0)
        c4_sigma = norm_stats.get('c4_std', 12.5)

        def _sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        c3_norm = float(_sigmoid((c3 - c3_mu) / (c3_sigma + 1e-8)))
        c4_norm = float(_sigmoid((c4 - c4_mu) / (c4_sigma + 1e-8)))

        # c₅: GOLD / 4 → {0.25, 0.5, 0.75, 1.0}
        c5_norm = float(c5) / 4.0

        condition = np.array([c1_norm, c2_norm, c3_norm, c4_norm, c5_norm], dtype=np.float32)
        return condition

    def __getitem__(self, idx: int) -> Dict[str, 'torch.Tensor']:
        vol_idx, center = self.patch_indices[idx]

        # 从缓存加载 CT 和 mask（避免每个 patch 重复读取磁盘）
        ct = _load_nifti_cached(str(self.ct_paths[vol_idx]))
        mask = _load_nifti_cached(str(self.mask_paths[vol_idx]))

        # 提取 patch
        ct_patch = self._extract_patch(ct, center)
        mask_patch = self._extract_patch(mask, center)

        # 数据增强
        if self.augment:
            ct_patch, mask_patch = self._augment(ct_patch, mask_patch)

        # 归一化 CT patch (Target)
        ct_patch_norm = self._normalize_ct(ct_patch)

        # 创建 Input：复制 Target，在 mask 区域填充均匀噪声
        # 使用宽范围的均匀噪声 [0, 1]，防止模型依赖固定的"灰色"均值
        # 这迫使模型学习从上下文推断正确的低 HU 值，而非简单填充中间值
        input_patch = ct_patch_norm.copy()
        if np.sum(mask_patch) > 0:
            # 使用均匀噪声 [0, 1] 填充 mask 区域
            # 注意：0 对应 HU_min (-1000)，1 对应 HU_max (400)
            # 宽范围噪声防止模型偏向任何特定值
            uniform_noise = np.random.uniform(0.0, 1.0, input_patch.shape)
            input_patch[mask_patch > 0] = uniform_noise[mask_patch > 0]

        # 获取条件向量（CICI-FiLM 阶段②）
        condition = self._get_condition_vector(vol_idx)

        # 转换为 tensor 并返回
        result = {
            'input': torch.from_numpy(input_patch[np.newaxis]).float(),
            'target': torch.from_numpy(ct_patch_norm[np.newaxis]).float(),
            'mask': torch.from_numpy(mask_patch[np.newaxis]).float()
        }

        # 添加条件向量（如果启用）
        if condition is not None:
            result['condition'] = torch.from_numpy(condition).float()

        return result


def create_dataloader(
    ct_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    batch_size: int = 4,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    num_workers: int = 4,
    train_ratio: float = 0.8,
    patient_features_path: Optional[Union[str, Path]] = None,
    use_condition: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证 DataLoader（CICI-FiLM 条件化版本）

    Args:
        ct_dir: CT 目录
        mask_dir: mask 目录
        batch_size: 批大小
        patch_size: patch 大小
        num_workers: 数据加载线程数
        train_ratio: 训练集比例
        patient_features_path: 患者特征 JSON 路径（CICI-FiLM 阶段②）
        use_condition: 是否使用条件向量（False=Exp-0, True=Exp-2）
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
        augment=True,
        patient_features_path=patient_features_path,
        use_condition=use_condition
    )

    val_dataset = LungPatchDataset(
        ct_paths=ct_files[n_train:],
        mask_paths=mask_files[n_train:],
        patch_size=patch_size,
        augment=False,
        patient_features_path=patient_features_path,
        use_condition=use_condition
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

