#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytest 配置和共享 fixtures

提供测试所需的模拟数据和配置
"""

import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest


@pytest.fixture(scope="session")
def temp_dir():
    """创建临时目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_ct_data() -> np.ndarray:
    """
    生成模拟 CT 数据
    
    Returns:
        ct_data: 模拟的 3D CT 数据 (HU 单位)
    """
    # 创建 64x64x64 的模拟 CT
    shape = (64, 64, 64)
    
    # 背景为空气 (-1000 HU)
    ct = np.full(shape, -1000, dtype=np.float32)
    
    # 添加模拟肺部区域 (约 -700 HU)
    center = np.array(shape) // 2
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
                if dist < 25:
                    ct[z, y, x] = -700 + np.random.randn() * 50
    
    return ct


@pytest.fixture
def sample_lung_mask(sample_ct_data) -> np.ndarray:
    """
    生成模拟肺部 mask
    
    Returns:
        mask: 二值肺部 mask
    """
    mask = (sample_ct_data > -900).astype(np.uint8)
    return mask


@pytest.fixture
def sample_lesion_mask(sample_lung_mask) -> np.ndarray:
    """
    生成模拟病灶 mask
    
    Returns:
        mask: 二值病灶 mask
    """
    mask = np.zeros_like(sample_lung_mask)
    
    # 在肺部区域内添加几个小病灶
    lung_coords = np.where(sample_lung_mask > 0)
    if len(lung_coords[0]) > 0:
        # 随机选择几个位置作为病灶中心
        n_lesions = 3
        indices = np.random.choice(len(lung_coords[0]), n_lesions, replace=False)
        
        for idx in indices:
            z, y, x = lung_coords[0][idx], lung_coords[1][idx], lung_coords[2][idx]
            # 创建小球形病灶
            for dz in range(-3, 4):
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        if dz**2 + dy**2 + dx**2 <= 9:
                            nz, ny, nx = z + dz, y + dy, x + dx
                            if 0 <= nz < mask.shape[0] and 0 <= ny < mask.shape[1] and 0 <= nx < mask.shape[2]:
                                mask[nz, ny, nx] = 1
    
    return mask


@pytest.fixture
def sample_nifti_file(temp_dir, sample_ct_data) -> Path:
    """
    创建模拟 NIfTI 文件
    
    Returns:
        filepath: NIfTI 文件路径
    """
    try:
        import nibabel as nib
    except ImportError:
        pytest.skip("nibabel 未安装")
    
    filepath = temp_dir / "sample_ct.nii.gz"
    
    affine = np.eye(4)
    img = nib.Nifti1Image(sample_ct_data, affine)
    nib.save(img, filepath)
    
    return filepath


@pytest.fixture
def sample_mask_file(temp_dir, sample_lung_mask) -> Path:
    """
    创建模拟 mask NIfTI 文件
    """
    try:
        import nibabel as nib
    except ImportError:
        pytest.skip("nibabel 未安装")
    
    filepath = temp_dir / "sample_mask.nii.gz"
    
    affine = np.eye(4)
    img = nib.Nifti1Image(sample_lung_mask, affine)
    nib.save(img, filepath)
    
    return filepath


@pytest.fixture
def sample_config() -> dict:
    """
    生成测试配置
    
    Returns:
        config: 配置字典
    """
    return {
        'paths': {
            'raw_data': 'data/00_raw',
            'cleaned_data': 'data/01_cleaned',
            'atlas': 'data/02_atlas',
            'mapped': 'data/03_mapped',
            'final_viz': 'data/04_final_viz',
            'checkpoints': 'checkpoints',
            'logs': 'logs',
        },
        'preprocessing': {
            'segmentation': {
                'task': 'lung',
                'fast_mode': True,
            },
            'laa950': {
                'threshold': -950,
                'min_volume_mm3': 10,
            },
        },
        'registration': {
            'template_build': {
                'type_of_transform': 'SyN',
                'iteration_limit': 2,
            },
            'lesion_registration': {
                'type_of_transform': 'SyNRA',
                'reg_iterations': [10, 5],
            },
        },
        'training': {
            'batch_size': 2,
            'epochs': 2,
            'learning_rate': 0.0002,
            'patch_size': [32, 32, 32],
            'save_frequency': 1,
            'loss_weights': {
                'reconstruction': 1.0,
                'perceptual': 0.1,
                'adversarial': 0.01,
            },
        },
        'evaluation': {
            'ssim_threshold': 0.85,
            'psnr_threshold': 25.0,
            'dice_threshold': 0.85,
        },
        'visualization': {
            'lung_opacity': 0.3,
            'lesion_opacity': 0.8,
            'lung_color': [0.7, 0.7, 0.7],
            'lesion_color': [1.0, 0.2, 0.2],
        },
    }


@pytest.fixture
def sample_patch_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成用于网络测试的 patch 数据
    
    Returns:
        input_patch: 输入 patch (带空洞)
        target_patch: 目标 patch
        mask_patch: mask patch
    """
    shape = (32, 32, 32)
    
    # 创建目标 patch
    target = np.random.randn(*shape).astype(np.float32) * 0.1 + 0.5
    target = np.clip(target, 0, 1)
    
    # 创建 mask
    mask = np.zeros(shape, dtype=np.float32)
    mask[12:20, 12:20, 12:20] = 1
    
    # 创建输入 (mask 区域置零)
    input_patch = target.copy()
    input_patch[mask > 0] = 0
    
    return input_patch, target, mask


# 标记需要特定依赖的测试
def pytest_configure(config):
    """配置自定义标记"""
    config.addinivalue_line(
        "markers", "requires_torch: 需要 PyTorch"
    )
    config.addinivalue_line(
        "markers", "requires_ants: 需要 ANTsPy"
    )
    config.addinivalue_line(
        "markers", "requires_pyvista: 需要 PyVista"
    )
    config.addinivalue_line(
        "markers", "slow: 运行缓慢的测试"
    )

