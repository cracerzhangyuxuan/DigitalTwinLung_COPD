# Phase 3 运行命令指南

本文档提供 Phase 3（病理映射与 AI 纹理融合）的完整运行命令。

## 目录

1. [快速开始](#快速开始)
2. [Phase 3A: 空间映射](#phase-3a-空间映射)
3. [Phase 3B: AI 纹理融合](#phase-3b-ai-纹理融合)
4. [完整流水线](#完整流水线)
5. [常用参数组合](#常用参数组合)

---

## 快速开始

### 前置条件检查

```bash
# 检查环境和数据
python run_phase3_pipeline.py --check-only
```

### 最简单的运行方式

```bash
# 运行 Phase 3A（空间映射 + 可视化）
python run_phase3_pipeline.py

# 运行 Phase 3B 训练（使用默认 U-Net 模型）
python run_phase3_pipeline.py --phase3b

# 运行 Phase 3B 推理
python run_phase3_pipeline.py --inference
```

---

## Phase 3A: 空间映射

### 基本命令

```bash
# 完整 Phase 3A 流程
python run_phase3_pipeline.py

# 快速测试（仅处理 3 例）
python run_phase3_pipeline.py --quick-test

# 限制处理数量
python run_phase3_pipeline.py --limit 5
```

### 跳过/仅执行特定步骤

```bash
# 跳过配准（使用已有结果，仅执行可视化）
python run_phase3_pipeline.py --skip-registration

# 仅执行可视化
python run_phase3_pipeline.py --viz-only
```

---

## Phase 3B: AI 纹理融合

### 三种模型架构

| 模型类型 | 参数 | 特点 | 推荐场景 |
|----------|------|------|----------|
| U-Net | `--model-type unet` | 实现简单，效果稳定 | **首选方案** |
| Partial Conv | `--model-type partial_conv` | 处理不规则 Mask 更优 | 复杂病灶形状 |
| PatchGAN | `--model-type patchgan` | 效果最佳，训练复杂 | 追求最佳质量 |

### 训练命令

```bash
# 基线方案: U-Net（推荐首选）
python run_phase3_pipeline.py --phase3b --model-type unet --epochs 50

# 进阶方案: Partial Convolution
python run_phase3_pipeline.py --phase3b --model-type partial_conv --epochs 50

# 高级方案: PatchGAN
python run_phase3_pipeline.py --phase3b --model-type patchgan --epochs 100

# 自定义训练参数
python run_phase3_pipeline.py --phase3b \
    --model-type unet \
    --epochs 100 \
    --batch-size 4 \
    --lr 0.0002
```

### 推理命令

```bash
# 使用默认检查点（checkpoints/best.pth）
python run_phase3_pipeline.py --inference

# 指定检查点
python run_phase3_pipeline.py --inference --checkpoint checkpoints/best.pth

# 处理特定患者
python run_phase3_pipeline.py --inference --patient COPD001

# 使用 CPU
python run_phase3_pipeline.py --inference --device cpu
```

---

## 完整流水线

### 一键运行全部流程

```bash
# 完整流水线: 3A 空间映射 + 3B 训练 + 3B 推理
python run_phase3_pipeline.py --full --model-type unet --epochs 50
```

### 分步运行（推荐）

```bash
# Step 1: 运行 Phase 3A
python run_phase3_pipeline.py

# Step 2: 检查映射结果
# 查看 data/03_mapped/visualizations/ 中的可视化图片

# Step 3: 运行 Phase 3B 训练
python run_phase3_pipeline.py --phase3b --model-type unet --epochs 50

# Step 4: 运行 Phase 3B 推理
python run_phase3_pipeline.py --inference

# Step 5: 查看最终结果
# 输出在 data/04_final_viz/ 目录
```

---

## 常用参数组合

### 开发调试

```bash
# 快速测试完整流程
python run_phase3_pipeline.py --quick-test
python run_phase3_pipeline.py --phase3b --epochs 5 --batch-size 2
python run_phase3_pipeline.py --inference
```

### 生产环境

```bash
# 完整训练（推荐配置）
python run_phase3_pipeline.py --phase3b \
    --model-type unet \
    --epochs 100 \
    --batch-size 4 \
    --device cuda

# 高质量训练（PatchGAN）
python run_phase3_pipeline.py --phase3b \
    --model-type patchgan \
    --epochs 200 \
    --batch-size 2 \
    --device cuda
```

### 独立脚本（向后兼容）

```bash
# 使用独立训练脚本
python run_phase3b_training.py --model-type unet --epochs 50

# 使用独立推理脚本
python run_phase3b_inference.py --checkpoint checkpoints/best.pth
```

---

## 输出目录结构

```
data/
├── 03_mapped/                    # Phase 3A 输出
│   ├── COPD001/
│   │   ├── COPD001_warped.nii.gz        # 配准后的 CT
│   │   └── COPD001_warped_lesion.nii.gz # 配准后的病灶 mask
│   └── visualizations/
│       └── COPD001_view_*.png           # 三视图渲染
│
├── 04_final_viz/                 # Phase 3B 输出
│   └── COPD001_fused.nii.gz     # 融合后的数字孪生 CT
│
checkpoints/
├── best.pth                      # 最佳模型
├── latest.pth                    # 最新检查点
└── training_log.json             # 训练历史
```

---

**文档版本**: v1.0  
**最后更新**: 2026-01-07

