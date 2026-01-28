#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证评估报告和直方图数据的一致性

用法：
    python verify_data_consistency.py --model-type unet --patient copd_001
"""

import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np


def verify_patient_data(model_type: str, patient_id: str):
    """验证单个患者的数据一致性"""
    
    # 路径设置
    results_dir = Path(f'results/{model_type}')
    patient_dir = results_dir / patient_id
    fused_dir = Path(f'data/04_final_viz/{model_type}')
    mapped_dir = Path('data/03_mapped')
    
    # 文件路径
    report_path = patient_dir / f"{patient_id}_evaluation_report.json"
    fused_path = fused_dir / f"{patient_id}_fused.nii.gz"
    mask_path = mapped_dir / patient_id / f"{patient_id}_warped_lesion.nii.gz"
    
    print(f"\n{'='*60}")
    print(f"验证患者: {patient_id}")
    print(f"{'='*60}\n")
    
    # 1. 检查文件是否存在
    print("1. 检查文件存在性:")
    files_ok = True
    for name, path in [
        ("评估报告", report_path),
        ("融合结果", fused_path),
        ("病灶Mask", mask_path)
    ]:
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"   {status} {name}: {path}")
        if not exists:
            files_ok = False
    
    if not files_ok:
        print("\n❌ 文件缺失，请先运行推理和评估")
        return False
    
    # 2. 读取评估报告
    print("\n2. 评估报告中的数据:")
    with open(report_path, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
    
    report_emph = report_data.get('fused_emphysema_ratio', 0)
    print(f"   AI肺气肿比例: {report_emph:.1%}")
    print(f"   体素数量: {report_data.get('voxel_count', 0)}")
    
    # 3. 重新计算直方图数据
    print("\n3. 重新计算直方图数据:")
    fused_data = nib.load(str(fused_path)).get_fdata()
    mask_data = nib.load(str(mask_path)).get_fdata()
    
    mask_bool = mask_data > 0
    fused_lesion = fused_data[mask_bool]
    
    calc_emph = (fused_lesion < -950).sum() / len(fused_lesion)
    print(f"   AI肺气肿比例: {calc_emph:.1%}")
    print(f"   体素数量: {len(fused_lesion)}")
    
    # 4. 对比结果
    print("\n4. 数据一致性检查:")
    emph_diff = abs(report_emph - calc_emph)
    voxel_match = report_data.get('voxel_count') == len(fused_lesion)
    
    if emph_diff < 0.001 and voxel_match:
        print(f"   ✓ 数据一致！差异: {emph_diff:.4f}")
        return True
    else:
        print(f"   ✗ 数据不一致！")
        print(f"      肺气肿比例差异: {emph_diff:.4f}")
        print(f"      体素数量匹配: {voxel_match}")
        print(f"\n   可能原因:")
        print(f"      1. 评估报告使用的是旧的 fused 文件")
        print(f"      2. 推理后没有重新运行评估")
        print(f"      3. 文件被手动修改")
        return False


def main():
    parser = argparse.ArgumentParser(description="验证数据一致性")
    parser.add_argument('--model-type', type=str, default='unet',
                       choices=['unet', 'partial_conv', 'patchgan'])
    parser.add_argument('--patient', type=str, required=True,
                       help='患者ID，例如 copd_001')
    
    args = parser.parse_args()
    
    is_consistent = verify_patient_data(args.model_type, args.patient)
    
    if is_consistent:
        print(f"\n✓ 验证通过：数据一致")
    else:
        print(f"\n✗ 验证失败：数据不一致")
        print(f"\n建议操作:")
        print(f"   1. 重新运行推理:")
        print(f"      python run_phase3_pipeline.py --inference --model-type {args.model_type}")
        print(f"   2. 重新运行评估:")
        print(f"      python run_phase3_pipeline.py --evaluate --model-type {args.model_type}")
        print(f"   3. 重新生成可视化:")
        print(f"      python run_phase3_pipeline.py --visualize --model-type {args.model_type}")


if __name__ == "__main__":
    main()

