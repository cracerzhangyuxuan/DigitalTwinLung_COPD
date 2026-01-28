#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全面诊断模型数据不一致问题

用法：
    python diagnose_model_data.py --model-type unet
"""

import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from datetime import datetime


def diagnose_model_data(model_type: str):
    """全面诊断模型数据"""
    
    print(f"\n{'='*70}")
    print(f"模型数据诊断 - {model_type}")
    print(f"{'='*70}\n")
    
    # 1. 检查模型文件
    print("1. 模型检查点信息:")
    checkpoint_dir = Path(f'checkpoints/{model_type}')
    if checkpoint_dir.exists():
        for ckpt_file in ['best.pth', 'latest.pth']:
            ckpt_path = checkpoint_dir / ckpt_file
            if ckpt_path.exists():
                stat = ckpt_path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(stat.st_mtime)
                print(f"   {ckpt_file:12s}: {size_mb:8.2f} MB, 修改时间: {mtime}")
            else:
                print(f"   {ckpt_file:12s}: ✗ 不存在")
    else:
        print(f"   ✗ 检查点目录不存在: {checkpoint_dir}")
    
    # 2. 检查评估报告
    print("\n2. 评估报告:")
    report_path = Path(f'results/{model_type}/evaluation_report.md')
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 提取关键数据
            for line in content.split('\n'):
                if 'AI肺气肿比例' in line or '生成时间' in line:
                    print(f"   {line.strip()}")
    else:
        print(f"   ✗ 评估报告不存在: {report_path}")
    
    # 3. 检查所有患者的详细数据
    print("\n3. 患者详细数据:")
    results_dir = Path(f'results/{model_type}')
    patient_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    
    if not patient_dirs:
        print("   ✗ 未找到患者数据")
        return
    
    print(f"\n   {'患者ID':<12} {'AI肺气肿%':<12} {'真实肺气肿%':<14} {'体素数':<10} {'PSNR':<8} {'SSIM':<8}")
    print(f"   {'-'*70}")
    
    all_ai_emph = []
    all_real_emph = []
    
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        report_path = patient_dir / f"{patient_id}_evaluation_report.json"
        
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                ai_emph = data.get('fused_emphysema_ratio', 0) * 100
                real_emph = data.get('real_emphysema_ratio', 0) * 100
                voxels = data.get('voxel_count', 0)
                psnr = data.get('psnr', 0)
                ssim = data.get('ssim', 0)
                
                all_ai_emph.append(ai_emph)
                all_real_emph.append(real_emph)
                
                print(f"   {patient_id:<12} {ai_emph:>10.1f}% {real_emph:>12.1f}% {voxels:>10d} {psnr:>7.2f} {ssim:>7.4f}")
    
    if all_ai_emph:
        print(f"   {'-'*70}")
        print(f"   {'平均':<12} {np.mean(all_ai_emph):>10.1f}% {np.mean(all_real_emph):>12.1f}%")
        print(f"   {'中位数':<12} {np.median(all_ai_emph):>10.1f}% {np.median(all_real_emph):>12.1f}%")
        print(f"   {'最大值':<12} {np.max(all_ai_emph):>10.1f}% {np.max(all_real_emph):>12.1f}%")
    
    # 4. 检查 fused 文件的时间戳
    print("\n4. Fused 文件时间戳:")
    fused_dir = Path(f'data/04_final_viz/{model_type}')
    if fused_dir.exists():
        fused_files = sorted(fused_dir.glob("*_fused.nii.gz"))[:5]  # 只显示前5个
        for fused_file in fused_files:
            stat = fused_file.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)
            print(f"   {fused_file.name:<30} {mtime}")
    else:
        print(f"   ✗ Fused 目录不存在: {fused_dir}")
    
    # 5. 诊断结论
    print("\n5. 诊断结论:")
    if all_ai_emph:
        avg_ai = np.mean(all_ai_emph)
        if avg_ai < 20:
            print(f"   ⚠ AI肺气肿比例平均值为 {avg_ai:.1f}%，远低于预期的60%")
            print(f"\n   可能原因:")
            print(f"   1. 模型训练不充分或训练目标不正确")
            print(f"   2. 使用了错误的模型检查点（best.pth vs latest.pth）")
            print(f"   3. 推理时使用了旧的模型权重")
            print(f"   4. 您看到的'60%'可能来自其他来源（如训练日志、其他模型等）")
            print(f"\n   建议操作:")
            print(f"   1. 确认模型训练是否完成，检查训练日志")
            print(f"   2. 检查是否有其他模型检查点目录")
            print(f"   3. 重新训练模型，确保训练目标正确")
        elif avg_ai >= 50:
            print(f"   ✓ AI肺气肿比例平均值为 {avg_ai:.1f}%，符合预期")
        else:
            print(f"   ⚠ AI肺气肿比例平均值为 {avg_ai:.1f}%，介于预期范围")


def main():
    parser = argparse.ArgumentParser(description="诊断模型数据不一致问题")
    parser.add_argument('--model-type', type=str, default='unet',
                       choices=['unet', 'partial_conv', 'patchgan'])
    
    args = parser.parse_args()
    diagnose_model_data(args.model_type)


if __name__ == "__main__":
    main()

