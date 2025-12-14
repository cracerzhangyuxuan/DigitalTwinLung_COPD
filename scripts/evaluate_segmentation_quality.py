#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2 分割质量评估脚本

评估TotalSegmentator分割结果的质量，为Atlas构建提供依据

评估指标：
1. 肺部体积分布统计
2. 左右肺对称性分析
3. 分割完整性检查（空洞、断裂）
4. HU值范围验证
5. 3D可视化对比

验收标准：
- 肺部体积：2000-8000 cc（正常成人肺总容量约6L）
- 体积变异系数 CV < 30%
- 无大面积空洞（空洞率 < 5%）
- 结构连续性（主连通分量占比 > 95%）

作者: DigitalTwinLung_COPD Team
日期: 2025-12-10
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

import numpy as np

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import nibabel as nib
except ImportError:
    print("请安装 nibabel: pip install nibabel")
    sys.exit(1)

try:
    from scipy import ndimage
    from scipy.ndimage import binary_fill_holes, binary_erosion, label
except ImportError:
    print("请安装 scipy: pip install scipy")
    sys.exit(1)


class SegmentationQualityEvaluator:
    """分割质量评估器"""
    
    # 验收标准
    CRITERIA = {
        'min_volume_cc': 2000,      # 最小肺体积 (cc)
        'max_volume_cc': 8000,      # 最大肺体积 (cc)
        'max_cv': 0.30,             # 最大变异系数
        'max_hole_ratio': 0.05,     # 最大空洞率
        'min_main_component': 0.95, # 主连通分量最小占比
        'hu_min': -1100,            # CT最小HU值
        'hu_max': 100,              # 肺内最大HU值（排除异常）
        'min_samples': 10,          # 最小样本数
    }
    
    def __init__(self, clean_dir: Path, mask_dir: Path):
        self.clean_dir = Path(clean_dir)
        self.mask_dir = Path(mask_dir)
        self.results = []
        self.summary = {}
        
    def get_voxel_volume_cc(self, nii_img) -> float:
        """计算单个体素的体积（立方厘米）"""
        header = nii_img.header
        zooms = header.get_zooms()[:3]  # (x, y, z) spacing in mm
        voxel_vol_mm3 = zooms[0] * zooms[1] * zooms[2]
        return voxel_vol_mm3 / 1000.0  # mm³ -> cc
    
    def analyze_single_case(self, case_id: str) -> Dict:
        """分析单个病例的分割质量"""
        clean_path = self.clean_dir / f"{case_id}_clean.nii.gz"
        mask_path = self.mask_dir / f"{case_id}_mask.nii.gz"
        
        if not clean_path.exists() or not mask_path.exists():
            return {'case_id': case_id, 'status': 'missing', 'error': '文件不存在'}
        
        try:
            # 加载数据
            clean_nii = nib.load(str(clean_path))
            mask_nii = nib.load(str(mask_path))
            
            ct_data = clean_nii.get_fdata()
            mask_data = mask_nii.get_fdata()
            
            # 基本信息
            voxel_vol_cc = self.get_voxel_volume_cc(mask_nii)
            shape = mask_data.shape
            spacing = mask_nii.header.get_zooms()[:3]
            
            # 1. 体积统计
            lung_voxels = int(np.sum(mask_data > 0))
            total_voxels = int(mask_data.size)
            lung_volume_cc = lung_voxels * voxel_vol_cc
            lung_ratio = lung_voxels / total_voxels
            
            # 2. 左右肺对称性分析
            mid_x = shape[0] // 2
            left_mask = mask_data[:mid_x, :, :]
            right_mask = mask_data[mid_x:, :, :]
            left_volume = np.sum(left_mask > 0) * voxel_vol_cc
            right_volume = np.sum(right_mask > 0) * voxel_vol_cc
            
            if left_volume + right_volume > 0:
                symmetry_ratio = min(left_volume, right_volume) / max(left_volume, right_volume)
            else:
                symmetry_ratio = 0
            
            # 3. 连通分量分析
            binary_mask = (mask_data > 0).astype(np.uint8)
            labeled, num_components = label(binary_mask)
            
            component_sizes = []
            for i in range(1, num_components + 1):
                size = np.sum(labeled == i)
                component_sizes.append(size)
            
            if component_sizes:
                component_sizes.sort(reverse=True)
                main_component_ratio = component_sizes[0] / lung_voxels if lung_voxels > 0 else 0
                # 前两大连通分量（理想情况是左右肺）
                top2_ratio = sum(component_sizes[:2]) / lung_voxels if lung_voxels > 0 else 0
            else:
                main_component_ratio = 0
                top2_ratio = 0
            
            # 4. 空洞检测
            filled_mask = np.zeros_like(binary_mask)
            for z in range(shape[2]):
                filled_mask[:, :, z] = binary_fill_holes(binary_mask[:, :, z])
            
            holes = filled_mask.astype(int) - binary_mask.astype(int)
            hole_voxels = int(np.sum(holes > 0))
            hole_ratio = hole_voxels / lung_voxels if lung_voxels > 0 else 0
            
            # 5. HU值分析（仅肺内区域）
            lung_hu = ct_data[mask_data > 0]
            if len(lung_hu) > 0:
                hu_mean = float(np.mean(lung_hu))
                hu_std = float(np.std(lung_hu))
                hu_min = float(np.min(lung_hu))
                hu_max = float(np.max(lung_hu))
                hu_median = float(np.median(lung_hu))
                # 正常肺组织HU分布：-950 到 -700 之间应占主要部分
                normal_hu_ratio = np.sum((lung_hu >= -950) & (lung_hu <= -500)) / len(lung_hu)
            else:
                hu_mean = hu_std = hu_min = hu_max = hu_median = 0
                normal_hu_ratio = 0
            
            # 6. Z轴覆盖率（确保完整覆盖）
            z_coverage = []
            for z in range(shape[2]):
                if np.sum(mask_data[:, :, z] > 0) > 100:  # 至少100个体素
                    z_coverage.append(z)
            
            if z_coverage:
                z_start = min(z_coverage)
                z_end = max(z_coverage)
                z_span = z_end - z_start + 1
                z_continuity = len(z_coverage) / z_span if z_span > 0 else 0
            else:
                z_start = z_end = z_span = 0
                z_continuity = 0
            
            result = {
                'case_id': case_id,
                'status': 'success',
                'shape': list(shape),
                'spacing_mm': list(spacing),
                'voxel_vol_cc': voxel_vol_cc,
                # 体积指标
                'lung_voxels': lung_voxels,
                'lung_volume_cc': round(lung_volume_cc, 2),
                'lung_ratio': round(lung_ratio, 4),
                # 对称性
                'left_volume_cc': round(left_volume, 2),
                'right_volume_cc': round(right_volume, 2),
                'symmetry_ratio': round(symmetry_ratio, 4),
                # 连通性
                'num_components': num_components,
                'main_component_ratio': round(main_component_ratio, 4),
                'top2_component_ratio': round(top2_ratio, 4),
                # 空洞
                'hole_voxels': hole_voxels,
                'hole_ratio': round(hole_ratio, 4),
                # HU值
                'hu_mean': round(hu_mean, 2),
                'hu_std': round(hu_std, 2),
                'hu_min': round(hu_min, 2),
                'hu_max': round(hu_max, 2),
                'hu_median': round(hu_median, 2),
                'normal_hu_ratio': round(normal_hu_ratio, 4),
                # Z轴覆盖
                'z_start': z_start,
                'z_end': z_end,
                'z_span': z_span,
                'z_continuity': round(z_continuity, 4),
            }
            
            return result
            
        except Exception as e:
            return {'case_id': case_id, 'status': 'error', 'error': str(e)}
    
    def evaluate_all(self) -> None:
        """评估所有病例"""
        # 获取所有mask文件
        mask_files = sorted(self.mask_dir.glob("*_mask.nii.gz"))
        
        print("=" * 70)
        print("Phase 2 分割质量评估")
        print("=" * 70)
        print(f"Mask目录: {self.mask_dir}")
        print(f"Clean目录: {self.clean_dir}")
        print(f"发现 {len(mask_files)} 个文件")
        print("=" * 70)
        
        self.results = []
        
        for i, mask_path in enumerate(mask_files, 1):
            case_id = mask_path.name.replace('_mask.nii.gz', '')
            print(f"[{i}/{len(mask_files)}] 分析: {case_id}...", end=' ')
            
            result = self.analyze_single_case(case_id)
            self.results.append(result)
            
            if result['status'] == 'success':
                print(f"体积={result['lung_volume_cc']:.0f}cc, "
                      f"对称性={result['symmetry_ratio']:.2f}, "
                      f"空洞率={result['hole_ratio']*100:.1f}%")
            else:
                print(f"失败: {result.get('error', '未知错误')}")
        
        self._compute_summary()
    
    def _compute_summary(self) -> None:
        """计算汇总统计"""
        successful = [r for r in self.results if r['status'] == 'success']
        
        if not successful:
            self.summary = {'status': 'failed', 'error': '没有成功分析的病例'}
            return
        
        volumes = [r['lung_volume_cc'] for r in successful]
        symmetries = [r['symmetry_ratio'] for r in successful]
        hole_ratios = [r['hole_ratio'] for r in successful]
        main_ratios = [r['main_component_ratio'] for r in successful]
        top2_ratios = [r['top2_component_ratio'] for r in successful]
        hu_means = [r['hu_mean'] for r in successful]
        normal_hu_ratios = [r['normal_hu_ratio'] for r in successful]
        
        # 计算统计量
        vol_mean = np.mean(volumes)
        vol_std = np.std(volumes)
        vol_cv = vol_std / vol_mean if vol_mean > 0 else 0
        
        self.summary = {
            'total_cases': len(self.results),
            'successful_cases': len(successful),
            'failed_cases': len(self.results) - len(successful),
            # 体积统计
            'volume_mean_cc': round(vol_mean, 2),
            'volume_std_cc': round(vol_std, 2),
            'volume_cv': round(vol_cv, 4),
            'volume_min_cc': round(min(volumes), 2),
            'volume_max_cc': round(max(volumes), 2),
            # 对称性
            'symmetry_mean': round(np.mean(symmetries), 4),
            'symmetry_min': round(min(symmetries), 4),
            # 空洞率
            'hole_ratio_mean': round(np.mean(hole_ratios), 4),
            'hole_ratio_max': round(max(hole_ratios), 4),
            # 连通性
            'main_component_mean': round(np.mean(main_ratios), 4),
            'top2_component_mean': round(np.mean(top2_ratios), 4),
            # HU值
            'hu_mean_avg': round(np.mean(hu_means), 2),
            'normal_hu_ratio_mean': round(np.mean(normal_hu_ratios), 4),
        }
        
        # 验收检查
        self._validate()
    
    def _validate(self) -> None:
        """验收检查"""
        checks = []
        passed = True
        
        # 检查1: 样本数量
        if self.summary['successful_cases'] >= self.CRITERIA['min_samples']:
            checks.append(('✅', f"样本数量充足: {self.summary['successful_cases']} >= {self.CRITERIA['min_samples']}"))
        else:
            checks.append(('❌', f"样本数量不足: {self.summary['successful_cases']} < {self.CRITERIA['min_samples']}"))
            passed = False
        
        # 检查2: 体积范围
        vol_in_range = sum(1 for r in self.results if r['status'] == 'success' 
                          and self.CRITERIA['min_volume_cc'] <= r['lung_volume_cc'] <= self.CRITERIA['max_volume_cc'])
        vol_ratio = vol_in_range / self.summary['successful_cases']
        if vol_ratio >= 0.9:
            checks.append(('✅', f"体积范围合理: {vol_ratio*100:.0f}% 在 {self.CRITERIA['min_volume_cc']}-{self.CRITERIA['max_volume_cc']}cc"))
        else:
            checks.append(('⚠️', f"部分体积异常: 仅 {vol_ratio*100:.0f}% 在合理范围"))
        
        # 检查3: 变异系数
        if self.summary['volume_cv'] <= self.CRITERIA['max_cv']:
            checks.append(('✅', f"体积一致性好: CV={self.summary['volume_cv']:.2%} <= {self.CRITERIA['max_cv']:.0%}"))
        else:
            checks.append(('⚠️', f"体积变异较大: CV={self.summary['volume_cv']:.2%} > {self.CRITERIA['max_cv']:.0%}"))
        
        # 检查4: 空洞率
        if self.summary['hole_ratio_max'] <= self.CRITERIA['max_hole_ratio']:
            checks.append(('✅', f"无明显空洞: 最大空洞率={self.summary['hole_ratio_max']:.2%} <= {self.CRITERIA['max_hole_ratio']:.0%}"))
        else:
            checks.append(('❌', f"存在空洞问题: 最大空洞率={self.summary['hole_ratio_max']:.2%}"))
            passed = False
        
        # 检查5: 连通性
        if self.summary['top2_component_mean'] >= self.CRITERIA['min_main_component']:
            checks.append(('✅', f"结构连续性好: 前两大分量占比={self.summary['top2_component_mean']:.2%}"))
        else:
            checks.append(('⚠️', f"结构可能碎片化: 前两大分量占比={self.summary['top2_component_mean']:.2%}"))
        
        # 检查6: 对称性
        if self.summary['symmetry_min'] >= 0.6:
            checks.append(('✅', f"左右肺对称性好: 最小对称比={self.summary['symmetry_min']:.2f}"))
        else:
            checks.append(('⚠️', f"存在不对称: 最小对称比={self.summary['symmetry_min']:.2f}"))
        
        self.summary['validation_checks'] = checks
        self.summary['validation_passed'] = passed
    
    def print_report(self) -> None:
        """打印评估报告"""
        print("\n" + "=" * 70)
        print("分割质量评估报告")
        print("=" * 70)
        
        print(f"\n📊 总体统计:")
        print(f"  总病例数: {self.summary['total_cases']}")
        print(f"  成功分析: {self.summary['successful_cases']}")
        print(f"  失败病例: {self.summary['failed_cases']}")
        
        print(f"\n📏 体积统计:")
        print(f"  平均体积: {self.summary['volume_mean_cc']:.0f} cc")
        print(f"  标准差: {self.summary['volume_std_cc']:.0f} cc")
        print(f"  变异系数: {self.summary['volume_cv']:.2%}")
        print(f"  范围: {self.summary['volume_min_cc']:.0f} - {self.summary['volume_max_cc']:.0f} cc")
        
        print(f"\n🔄 对称性:")
        print(f"  平均对称比: {self.summary['symmetry_mean']:.2f}")
        print(f"  最小对称比: {self.summary['symmetry_min']:.2f}")
        
        print(f"\n🔗 连通性:")
        print(f"  主分量平均占比: {self.summary['main_component_mean']:.2%}")
        print(f"  前两大分量占比: {self.summary['top2_component_mean']:.2%}")
        
        print(f"\n🕳️ 空洞分析:")
        print(f"  平均空洞率: {self.summary['hole_ratio_mean']:.2%}")
        print(f"  最大空洞率: {self.summary['hole_ratio_max']:.2%}")
        
        print(f"\n📈 HU值统计:")
        print(f"  平均HU: {self.summary['hu_mean_avg']:.0f}")
        print(f"  正常HU占比: {self.summary['normal_hu_ratio_mean']:.2%}")
        
        print("\n" + "=" * 70)
        print("验收检查结果:")
        print("=" * 70)
        for status, msg in self.summary['validation_checks']:
            print(f"  {status} {msg}")
        
        print("\n" + "=" * 70)
        if self.summary['validation_passed']:
            print("🎉 结论: ✅ 通过 - 分割质量满足构建高质量数字孪生底座的要求")
        else:
            print("⚠️ 结论: ❌ 不通过 - 存在需要修复的质量问题")
        print("=" * 70)
    
    def save_report(self, output_path: Path) -> None:
        """保存评估报告为JSON"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'clean_dir': str(self.clean_dir),
            'mask_dir': str(self.mask_dir),
            'summary': self.summary,
            'cases': self.results
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n报告已保存: {output_path}")
    
    def get_conclusion(self) -> Tuple[bool, str]:
        """获取结论"""
        if self.summary.get('validation_passed'):
            return True, "分割质量满足要求，可以进行Atlas构建"
        else:
            return False, "分割质量存在问题，需要进行修复"


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 2 分割质量评估')
    parser.add_argument('--clean-dir', type=str, 
                       default='data/01_cleaned/normal_clean',
                       help='清洗后CT目录')
    parser.add_argument('--mask-dir', type=str,
                       default='data/01_cleaned/normal_mask', 
                       help='分割Mask目录')
    parser.add_argument('--output', type=str,
                       default='data/02_atlas/segmentation_quality_report.json',
                       help='输出报告路径')
    
    args = parser.parse_args()
    
    evaluator = SegmentationQualityEvaluator(
        clean_dir=Path(args.clean_dir),
        mask_dir=Path(args.mask_dir)
    )
    
    evaluator.evaluate_all()
    evaluator.print_report()
    evaluator.save_report(Path(args.output))
    
    passed, msg = evaluator.get_conclusion()
    print(f"\n最终结论: {msg}")
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

