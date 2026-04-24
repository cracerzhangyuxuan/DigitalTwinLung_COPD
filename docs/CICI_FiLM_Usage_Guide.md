# CICI-FiLM 使用指南

本文档提供 CICI-FiLM 个性化病灶建模实验的完整使用流程。

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [阶段① 零训练自适应校准](#3-阶段①-零训练自适应校准)
4. [阶段② FiLM 微调训练](#4-阶段②-film-微调训练)
5. [评估与对比](#5-评估与对比)

---

## 1. 环境准备

确保已安装所有依赖：

```bash
pip install torch torchvision nibabel numpy scipy pandas matplotlib
```

---

## 2. 数据准备

### 2.1 提取患者特征

从患者真实 COPD CT 中提取 5 维条件向量（c₁~c₅）：

```bash
python scripts/extract_patient_features.py \
  --ct-dir data/01_cleaned/copd_clean \
  --lung-mask-dir data/01_cleaned/copd_mask \
  --emphysema-dir data/01_cleaned/copd_emphysema \
  --clinical-data data/clinical_data.csv \
  --output data/patient_features.json
```

**输入**：
- `data/01_cleaned/copd_clean/`: 患者真实 COPD CT（`copd_XXX_clean.nii.gz`）
- `data/01_cleaned/copd_mask/`: 肺区域 mask（`copd_XXX_mask.nii.gz`）
- `data/01_cleaned/copd_emphysema/`: 肺气肿病灶 mask（`copd_XXX_emphysema.nii.gz`）
- `data/clinical_data.csv`: 临床数据（包含 `patient_id` 和 `GOLD` 列）

**输出**：
- `data/patient_features.json`: 包含所有患者的 5 维条件向量

**示例输出**：
```json
{
  "J010": {
    "global_EI": 25.3,
    "lesion_vol_ratio": 18.7,
    "lesion_HU_mean": -968.5,
    "lesion_HU_std": 42.3,
    "GOLD": 2,
    "source_ct": "data/01_cleaned/copd_clean/copd_J010_clean.nii.gz",
    "source_lung_mask": "data/01_cleaned/copd_mask/copd_J010_mask.nii.gz",
    "source_emphysema_mask": "data/01_cleaned/copd_emphysema/copd_J010_emphysema.nii.gz"
  }
}
```

---

## 3. 阶段① 零训练自适应校准

### 3.1 实验目标

验证：将 `target_stats` 从固定值改为患者真实 c₃/c₄ 是否能提升 HU 统计一致性。

### 3.2 执行推理

**Exp-0 (Baseline)**：使用固定 HU 校准

```bash
python scripts/inference_cici_film.py \
  --mode exp1 \
  --backbone-checkpoint checkpoints/patchgan/best.pth \
  --template data/02_atlas/insp_template.nii.gz \
  --mask data/copd_masks/mask_J010.nii.gz \
  --patient-features data/patient_features.json \
  --patient-id J010 \
  --output results/exp0_J010.nii.gz
```

**Exp-1 (自适应校准)**：使用患者真实 c₃/c₄

```bash
python scripts/inference_cici_film.py \
  --mode exp1 \
  --backbone-checkpoint checkpoints/patchgan/best.pth \
  --template data/02_atlas/insp_template.nii.gz \
  --mask data/copd_masks/mask_J010.nii.gz \
  --patient-features data/patient_features.json \
  --patient-id J010 \
  --output results/exp1_J010.nii.gz
```

### 3.3 评估指标

对比 Exp-0 vs Exp-1：

- **ΔEI**：生成 CT 与真实 COPD CT 的 EI 差异
- **HU KL 散度**：病灶区 HU 分布的 KL 散度
- **HU Mean/Std Error**：病灶区 HU 均值/标准差误差

---

## 4. 阶段② FiLM 微调训练

### 4.1 实验目标

验证：冻结 backbone 训练 FiLM 分支（2,434 参数）是否能进一步提升个体化建模能力。

### 4.2 训练 CICI-FiLM

```bash
python scripts/train_cici_film.py \
  --backbone-checkpoint checkpoints/patchgan/best.pth \
  --ct-dir data/copd_ct \
  --mask-dir data/copd_masks \
  --patient-features data/patient_features.json \
  --output-dir checkpoints/cici_film \
  --epochs 50 \
  --batch-size 4 \
  --lr 0.0001
```

**训练参数说明**：
- `--epochs 50`: 训练 50 轮（可根据验证损失调整）
- `--batch-size 4`: 批大小（根据 GPU 显存调整）
- `--lr 0.0001`: 学习率（FiLM 分支较小，使用较小学习率）

**训练输出**：
- `checkpoints/cici_film/best.pth`: 最佳模型
- `checkpoints/cici_film/latest.pth`: 最新模型
- `checkpoints/cici_film/training_log.json`: 训练历史

### 4.3 推理

```bash
python scripts/inference_cici_film.py \
  --mode exp2 \
  --film-checkpoint checkpoints/cici_film/best.pth \
  --template data/02_atlas/insp_template.nii.gz \
  --mask data/copd_masks/mask_J010.nii.gz \
  --patient-features data/patient_features.json \
  --patient-id J010 \
  --output results/exp2_J010.nii.gz
```

---

## 5. 评估与对比

### 5.1 三组实验对比

| 实验 | 方法 | 可训练参数 | 预期效果 |
|:-----|:-----|:----------|:--------|
| Exp-0 | 固定 HU 校准 | 0 | Baseline |
| Exp-1 | 自适应 HU 校准 | 0 | ↓ HU Mean/Std Error |
| Exp-2 | FiLM 微调 | 2,434 | ↓ ΔEI + ↓ HU KL |

### 5.2 评估脚本

```bash
python scripts/evaluate_cici_film.py \
  --exp0-dir results/exp0 \
  --exp1-dir results/exp1 \
  --exp2-dir results/exp2 \
  --ref-dir data/copd_ct \
  --output results/evaluation_report.json
```

---

## 附录：文件结构

```
DigitalTwinLung_COPD/
├── data/
│   ├── 01_cleaned/
│   │   ├── copd_clean/       # 患者真实 COPD CT
│   │   ├── copd_mask/        # 肺区域 mask
│   │   └── copd_emphysema/   # 肺气肿病灶 mask
│   ├── clinical_data.csv     # 临床数据
│   └── patient_features.json # 5 维条件向量 + 归一化统计量
├── checkpoints/
│   ├── patchgan/best.pth     # 预训练 backbone
│   └── cici_film/best.pth    # CICI-FiLM 模型
├── src/04_texture_synthesis/
│   ├── conditioned_model.py  # FiLMBlock + ConditionedGenerator
│   ├── dataset.py            # 条件化数据集
│   ├── train.py              # 条件化训练器
│   └── inference_fuse.py     # 自适应 HU 校准
└── scripts/
    ├── extract_patient_features.py  # 特征提取
    ├── train_cici_film.py           # 训练脚本
    └── inference_cici_film.py       # 推理脚本
```

