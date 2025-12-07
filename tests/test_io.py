#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IO 模块测试
"""

import numpy as np
import pytest


class TestNiftiIO:
    """NIfTI 文件读写测试"""
    
    def test_load_nifti(self, sample_nifti_file):
        """测试加载 NIfTI 文件"""
        from src.utils.io import load_nifti
        
        data = load_nifti(sample_nifti_file)
        
        assert isinstance(data, np.ndarray)
        assert data.shape == (64, 64, 64)
        assert data.dtype == np.float32
    
    def test_load_nifti_with_affine(self, sample_nifti_file):
        """测试加载 NIfTI 文件并返回 affine"""
        from src.utils.io import load_nifti
        
        data, affine = load_nifti(sample_nifti_file, return_affine=True)
        
        assert isinstance(data, np.ndarray)
        assert isinstance(affine, np.ndarray)
        assert affine.shape == (4, 4)
    
    def test_save_nifti(self, temp_dir, sample_ct_data):
        """测试保存 NIfTI 文件"""
        from src.utils.io import save_nifti, load_nifti
        
        output_path = temp_dir / "test_output.nii.gz"
        
        save_nifti(sample_ct_data, output_path)
        
        assert output_path.exists()
        
        # 验证保存的数据
        loaded = load_nifti(output_path)
        np.testing.assert_array_almost_equal(loaded, sample_ct_data, decimal=5)
    
    def test_save_nifti_with_affine(self, temp_dir, sample_ct_data):
        """测试保存带 affine 的 NIfTI 文件"""
        from src.utils.io import save_nifti, load_nifti
        
        output_path = temp_dir / "test_with_affine.nii.gz"
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        
        save_nifti(sample_ct_data, output_path, affine=affine)
        
        _, loaded_affine = load_nifti(output_path, return_affine=True)
        np.testing.assert_array_almost_equal(loaded_affine, affine)
    
    def test_load_nonexistent_file(self, temp_dir):
        """测试加载不存在的文件"""
        from src.utils.io import load_nifti
        
        with pytest.raises(FileNotFoundError):
            load_nifti(temp_dir / "nonexistent.nii.gz")


class TestNiftiInfo:
    """NIfTI 信息获取测试"""
    
    def test_get_nifti_info(self, sample_nifti_file):
        """测试获取 NIfTI 文件信息"""
        from src.utils.io import get_nifti_info
        
        info = get_nifti_info(sample_nifti_file)
        
        assert 'shape' in info
        assert 'dtype' in info
        assert 'spacing' in info
        assert info['shape'] == (64, 64, 64)

