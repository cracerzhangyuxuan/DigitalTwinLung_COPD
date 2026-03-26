#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查验证集患者数据完整性"""
from pathlib import Path

mapped_dir = Path('data/03_mapped')
patients = ['copd_024', 'copd_025', 'copd_026']
models = ['unet', 'partial_conv', 'patchgan']

print("=== 验证集患者数据完整性检查 ===\n")

for p in patients:
    warped = mapped_dir / p / f"{p}_warped.nii.gz"
    mask = mapped_dir / p / f"{p}_warped_lesion.nii.gz"
    print(f"{p}:")
    print(f"  warped CT:    {'✓' if warped.exists() else '✗'} {warped}")
    print(f"  lesion mask:  {'✓' if mask.exists() else '✗'} {mask}")

print("\n=== 已有模型检查点 ===\n")
ckpt_dir = Path('checkpoints')
for m in models:
    best = ckpt_dir / m / 'best.pth'
    legacy = ckpt_dir / 'best.pth'
    print(f"{m}:")
    print(f"  {best}: {'✓' if best.exists() else '✗'}")
    if not best.exists():
        print(f"  {legacy}: {'✓' if legacy.exists() else '✗'}")

print("\n=== 已有融合结果 (copd_024~026) ===\n")
fused_dir = Path('data/04_final_viz')
for m in models:
    print(f"{m}:")
    for p in patients:
        fused = fused_dir / m / f"{p}_fused.nii.gz"
        print(f"  {fused.name}: {'✓ 已存在' if fused.exists() else '✗ 需生成'}")

