#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CICI-FiLM 患者特征提取脚本
=============================

从患者真实 COPD CT 和病灶 mask 中提取 5 维条件向量：
  c₁: global_EI（全局肺气肿指数，%）
  c₂: lesion_vol_ratio（病灶体积占比，%）
  c₃: lesion_HU_mean（病灶区 HU 均值）
  c₄: lesion_HU_std（病灶区 HU 标准差）
  c₅: GOLD（COPD 分期，1-4）

输出格式（JSON）：
{
  "patient_id": {
    "global_EI": float,
    "lesion_vol_ratio": float,
    "lesion_HU_mean": float,
    "lesion_HU_std": float,
    "GOLD": int,
    "source_ct": str,
    "source_mask": str
  }
}

数据来源（文档 §2.2）:
  - c₁ (EI):  copd_XXX_clean.nii.gz + copd_XXX_mask.nii.gz (肺 mask)
  - c₂ (vol): copd_XXX_emphysema.nii.gz (病灶 mask) / copd_XXX_mask.nii.gz (肺 mask)
  - c₃, c₄:  copd_XXX_clean.nii.gz 在 emphysema mask 区域内的 HU 统计
  - c₅:      clinical_data.csv

使用方法：
  python scripts/extract_patient_features.py \\
    --ct-dir data/01_cleaned/copd_clean \\
    --lung-mask-dir data/01_cleaned/copd_mask \\
    --emphysema-dir data/01_cleaned/copd_emphysema \\
    --clinical-data data/clinical_data.csv \\
    --output data/patient_features.json
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_nifti
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_emphysema_index(ct_data: np.ndarray, lung_mask: np.ndarray, threshold: float = -950.0) -> float:
    """
    计算肺气肿指数（Emphysema Index）
    
    EI = (肺区域内 HU < threshold 的体素数) / (肺区域总体素数) × 100%
    
    Args:
        ct_data: CT 数据（HU 值）
        lung_mask: 肺区域 mask（>0 表示肺）
        threshold: 气肿阈值（默认 -950 HU）
    
    Returns:
        EI: 肺气肿指数（%）
    """
    lung_voxels = ct_data[lung_mask > 0]
    if len(lung_voxels) == 0:
        return 0.0
    
    emphysema_voxels = np.sum(lung_voxels < threshold)
    ei = (emphysema_voxels / len(lung_voxels)) * 100.0
    return float(ei)


def extract_patient_features(
    ct_path: Path,
    lung_mask_path: Path,
    emphysema_mask_path: Path,
    patient_id: str,
    gold_stage: int = None
) -> dict:
    """
    提取单个患者的 5 维条件向量

    严格按照文档 §2.2 定义:
      c₁ = EI = Σ 𝟙[CT(v) < -950 ∧ LungMask(v)>0] / Σ 𝟙[LungMask(v)>0]
      c₂ = lesion_vol_ratio = Σ 𝟙[EmphMask(v)>0] / Σ 𝟙[LungMask(v)>0]
      c₃ = mean( CT(v) | EmphMask(v)>0 )
      c₄ = std( CT(v) | EmphMask(v)>0 )
      c₅ = GOLD 分期

    Args:
        ct_path: 患者 COPD CT 路径 (copd_XXX_clean.nii.gz)
        lung_mask_path: 肺区域 mask 路径 (copd_XXX_mask.nii.gz)
        emphysema_mask_path: 肺气肿病灶 mask 路径 (copd_XXX_emphysema.nii.gz)
        patient_id: 患者 ID
        gold_stage: GOLD 分期（1-4），若为 None 则默认为 2

    Returns:
        features: 包含 5 维条件向量的字典
    """
    logger.info(f"处理患者: {patient_id}")

    # 加载数据
    ct_data = load_nifti(ct_path)
    lung_mask = load_nifti(lung_mask_path)
    emph_mask = load_nifti(emphysema_mask_path)

    # c₁: global_EI（全局肺气肿指数）
    # EI = (肺区域内 HU < -950 的体素数) / (肺区域总体素数) × 100%
    global_ei = compute_emphysema_index(ct_data, lung_mask, threshold=-950.0)

    # c₂: lesion_vol_ratio（病灶体积占肺总体积的比率）
    # lesion_vol_ratio = Σ 𝟙[EmphMask(v)>0] / Σ 𝟙[LungMask(v)>0] × 100%
    total_lung_voxels = np.sum(lung_mask > 0)
    lesion_voxels = np.sum(emph_mask > 0)
    lesion_vol_ratio = (lesion_voxels / total_lung_voxels * 100.0) if total_lung_voxels > 0 else 0.0

    # c₃, c₄: lesion_HU_mean, lesion_HU_std（病灶区 HU 统计）
    # 取 CT 在 emphysema mask 标记区域内的 HU 值
    lesion_hu = ct_data[emph_mask > 0]
    if len(lesion_hu) > 0:
        lesion_hu_mean = float(np.mean(lesion_hu))
        lesion_hu_std = float(np.std(lesion_hu))
    else:
        lesion_hu_mean = -965.0  # 默认值
        lesion_hu_std = 45.0
        logger.warning(f"  患者 {patient_id} emphysema mask 为空，使用默认 c₃/c₄")

    # c₅: GOLD 分期
    gold = gold_stage if gold_stage is not None else 2  # 默认 GOLD 2

    features = {
        'global_EI': global_ei,
        'lesion_vol_ratio': lesion_vol_ratio,
        'lesion_HU_mean': lesion_hu_mean,
        'lesion_HU_std': lesion_hu_std,
        'GOLD': gold,
        'source_ct': str(ct_path),
        'source_lung_mask': str(lung_mask_path),
        'source_emphysema_mask': str(emphysema_mask_path)
    }

    logger.info(f"  c₁ (global_EI): {global_ei:.2f}%")
    logger.info(f"  c₂ (lesion_vol_ratio): {lesion_vol_ratio:.2f}%")
    logger.info(f"  c₃ (lesion_HU_mean): {lesion_hu_mean:.1f} HU")
    logger.info(f"  c₄ (lesion_HU_std): {lesion_hu_std:.1f} HU")
    logger.info(f"  c₅ (GOLD): {gold}")
    return features


def _normalize_patient_id(value) -> str:
    value = str(value).strip()
    if value.startswith('copd_'):
        value = value.replace('copd_', '')
    return value


def main():
    parser = argparse.ArgumentParser(
        description='CICI-FiLM: 提取患者 5 维条件向量',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/extract_patient_features.py \\
    --ct-dir data/01_cleaned/copd_clean \\
    --lung-mask-dir data/01_cleaned/copd_mask \\
    --emphysema-dir data/01_cleaned/copd_emphysema \\
    --clinical-data data/clinical_data.csv \\
    --output data/patient_features.json
        """
    )
    parser.add_argument('--ct-dir', required=True,
                       help='COPD CT 目录 (copd_XXX_clean.nii.gz)')
    parser.add_argument('--lung-mask-dir', required=True,
                       help='肺区域 mask 目录 (copd_XXX_mask.nii.gz)')
    parser.add_argument('--emphysema-dir', required=True,
                       help='肺气肿病灶 mask 目录 (copd_XXX_emphysema.nii.gz)')
    parser.add_argument('--clinical-data',
                       help='临床数据 CSV（自动兼容 project_id/patient_code/gold_stage 等列）')
    parser.add_argument('--output', required=True, help='输出 JSON 路径')
    args = parser.parse_args()

    ct_dir = Path(args.ct_dir)
    lung_mask_dir = Path(args.lung_mask_dir)
    emphysema_dir = Path(args.emphysema_dir)
    output_path = Path(args.output)

    clinical_data = {}
    if args.clinical_data:
        df = pd.read_csv(args.clinical_data)
        id_col = 'patient_id' if 'patient_id' in df.columns else ('project_id' if 'project_id' in df.columns else 'patient_code')
        gold_col = 'GOLD' if 'GOLD' in df.columns else 'gold_stage'
        clinical_data = {
            _normalize_patient_id(pid): int(gold)
            for pid, gold in zip(df[id_col], df[gold_col])
        }
        logger.info(f"加载临床数据: {len(clinical_data)} 例")

    # 查找所有 CT 文件
    ct_files = sorted(ct_dir.glob("*.nii.gz"))
    logger.info(f"找到 {len(ct_files)} 个 CT 文件")

    all_features = {}

    for ct_path in ct_files:
        # 提取患者 ID（文件名格式: copd_XXX_clean.nii.gz → XXX）
        # stem 得到 copd_XXX_clean，去掉 .nii 后缀
        patient_id = ct_path.name.split('.')[0].replace('_clean', '').replace('copd_', '')

        # 构造对应的 lung mask 路径: copd_XXX_mask.nii.gz
        lung_mask_path = lung_mask_dir / f"copd_{patient_id}_mask.nii.gz"
        if not lung_mask_path.exists():
            logger.warning(f"未找到肺 mask: {lung_mask_path}")
            continue

        # 构造对应的 emphysema mask 路径: copd_XXX_emphysema.nii.gz
        emph_mask_path = emphysema_dir / f"copd_{patient_id}_emphysema.nii.gz"
        if not emph_mask_path.exists():
            logger.warning(f"未找到 emphysema mask: {emph_mask_path}")
            continue

        # 获取 GOLD 分期
        gold_stage = clinical_data.get(patient_id, None)

        # 提取特征
        try:
            features = extract_patient_features(
                ct_path, lung_mask_path, emph_mask_path, patient_id, gold_stage
            )
            all_features[patient_id] = features
        except Exception as e:
            logger.error(f"处理失败 {patient_id}: {e}")

    # ---- 计算训练集的 c₃/c₄ 归一化统计量（μ, σ），用于 dataset.py z-score 归一化 ----
    if all_features:
        c3_values = [f['lesion_HU_mean'] for f in all_features.values()]
        c4_values = [f['lesion_HU_std'] for f in all_features.values()]
        norm_stats = {
            'c3_mean': float(np.mean(c3_values)),
            'c3_std': float(np.std(c3_values)),
            'c4_mean': float(np.mean(c4_values)),
            'c4_std': float(np.std(c4_values)),
            'n_patients': len(all_features)
        }
        all_features['_norm_stats'] = norm_stats
        logger.info(f"归一化统计量 (n={norm_stats['n_patients']}):")
        logger.info(f"  c₃ μ={norm_stats['c3_mean']:.1f}, σ={norm_stats['c3_std']:.1f}")
        logger.info(f"  c₄ μ={norm_stats['c4_mean']:.1f}, σ={norm_stats['c4_std']:.1f}")

    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_features, f, indent=2)

    logger.info(f"特征提取完成: {len(all_features) - (1 if '_norm_stats' in all_features else 0)} 例")
    logger.info(f"保存至: {output_path}")


if __name__ == '__main__':
    main()

