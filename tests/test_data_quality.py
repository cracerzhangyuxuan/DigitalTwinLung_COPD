#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据质量检查测试
"""

import numpy as np
import pytest


class TestQualityCheck:
    """数据质量检查测试"""
    
    def test_check_ct_quality_valid(self, sample_nifti_file, sample_config):
        """测试有效 CT 的质量检查"""
        from src.utils.data_quality import check_ct_quality
        
        result = check_ct_quality(sample_nifti_file, sample_config)
        
        assert hasattr(result, 'passed')
        assert hasattr(result, 'checks')
        assert hasattr(result, 'messages')
        assert hasattr(result, 'stats')
    
    def test_quality_check_result_structure(self, sample_nifti_file):
        """测试质量检查结果结构"""
        from src.utils.data_quality import check_ct_quality, QualityCheckResult
        
        result = check_ct_quality(sample_nifti_file)
        
        assert isinstance(result, QualityCheckResult)
        assert isinstance(result.checks, dict)
        assert isinstance(result.messages, list)
        assert isinstance(result.stats, dict)
    
    def test_generate_quality_report(self, sample_nifti_file, temp_dir):
        """测试生成质量报告"""
        from src.utils.data_quality import check_ct_quality, generate_quality_report
        
        result = check_ct_quality(sample_nifti_file)
        results = {str(sample_nifti_file): result}
        
        report = generate_quality_report(results)
        
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_generate_quality_report_to_file(self, sample_nifti_file, temp_dir):
        """测试保存质量报告到文件"""
        from src.utils.data_quality import check_ct_quality, generate_quality_report
        
        result = check_ct_quality(sample_nifti_file)
        results = {str(sample_nifti_file): result}
        
        output_path = temp_dir / "quality_report.txt"
        report = generate_quality_report(results, output_path)
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            content = f.read()
        
        assert len(content) > 0


class TestQualityCheckCriteria:
    """质量检查标准测试"""
    
    def test_slice_count_check(self, temp_dir):
        """测试切片数量检查"""
        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel 未安装")
        
        from src.utils.data_quality import check_ct_quality
        
        # 创建切片数量不足的 CT
        small_ct = np.random.randn(50, 64, 64).astype(np.float32) * 100 - 500
        filepath = temp_dir / "small_ct.nii.gz"
        
        img = nib.Nifti1Image(small_ct, np.eye(4))
        nib.save(img, filepath)
        
        config = {'data_quality': {'min_slices': 100}}
        result = check_ct_quality(filepath, config)
        
        # 应该检测到切片数量不足
        assert 'slice_count' in result.checks
    
    def test_hu_range_check(self, temp_dir):
        """测试 HU 范围检查"""
        try:
            import nibabel as nib
        except ImportError:
            pytest.skip("nibabel 未安装")
        
        from src.utils.data_quality import check_ct_quality
        
        # 创建 HU 范围异常的 CT
        abnormal_ct = np.random.randn(64, 64, 64).astype(np.float32) * 100 + 500
        filepath = temp_dir / "abnormal_ct.nii.gz"
        
        img = nib.Nifti1Image(abnormal_ct, np.eye(4))
        nib.save(img, filepath)
        
        result = check_ct_quality(filepath)
        
        assert 'hu_range' in result.checks
        assert 'min' in result.stats
        assert 'max' in result.stats

