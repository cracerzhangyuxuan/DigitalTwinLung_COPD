# Phase 3 运行命令指南

本文档提供 Phase 3（病理映射与 AI 纹理融合）的完整运行命令。

## 目录

1. [快速开始](#快速开始)
2. [Phase 3A: 空间映射](#phase-3a-空间映射)
3. [Phase 3B: AI 纹理融合](#phase-3b-ai-纹理融合)
4. [模型评估](#模型评估)
5. [结果可视化](#结果可视化)
6. [完整流水线](#完整流水线)
7. [批量执行命令](#批量执行命令)

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

# 模型评估
python run_phase3_pipeline.py --evaluate

# 生成可视化
python run_phase3_pipeline.py --visualize
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
python run_phase3_pipeline.py --phase3b --model-type unet

# 进阶方案: Partial Convolution
python run_phase3_pipeline.py --phase3b --model-type partial_conv

# 高级方案: PatchGAN
python run_phase3_pipeline.py --phase3b --model-type patchgan

# 自定义训练参数
python run_phase3_pipeline.py --phase3b \
    --model-type unet \
    --epochs 100 \
    --batch-size 4 \
    --lr 0.0002
```

### 推理命令

```bash
# 使用 UNet 模型推理
python run_phase3_pipeline.py --inference --model-type unet

# 使用 Partial Conv 模型推理
python run_phase3_pipeline.py --inference --model-type partial_conv

# 使用 PatchGAN 模型推理
python run_phase3_pipeline.py --inference --model-type patchgan

# 指定检查点
python run_phase3_pipeline.py --inference --checkpoint checkpoints/unet/best.pth

# 处理特定患者
python run_phase3_pipeline.py --inference --model-type unet --patient copd_001

# 使用 CPU
python run_phase3_pipeline.py --inference --device cpu
```

---

## 模型评估

评估功能将 AI 融合结果与真实 COPD CT 进行对比，计算 PSNR、SSIM 和肺气肿比例等指标。

```bash
# 评估 UNet 模型
python run_phase3_pipeline.py --evaluate --model-type unet

# 评估 Partial Conv 模型
python run_phase3_pipeline.py --evaluate --model-type partial_conv

# 评估 PatchGAN 模型
python run_phase3_pipeline.py --evaluate --model-type patchgan

# 限制评估患者数量
python run_phase3_pipeline.py --evaluate --model-type unet --limit 10
```

**输出目录**: `evaluation_results/{model_type}/`

---

## 结果可视化

生成多视图对比图（模板 vs AI融合 vs 病灶叠加）。

```bash
# 可视化 UNet 结果
python run_phase3_pipeline.py --visualize --model-type unet

# 可视化 Partial Conv 结果
python run_phase3_pipeline.py --visualize --model-type partial_conv

# 可视化 PatchGAN 结果
python run_phase3_pipeline.py --visualize --model-type patchgan

# 限制可视化患者数量
python run_phase3_pipeline.py --visualize --model-type unet --limit 5
```

**输出目录**: `visualization_results/{model_type}/`

---

## 完整流水线

### 一键运行全部流程

```bash
# 完整流水线: 3A 空间映射 + 3B 训练 + 3B 推理
python run_phase3_pipeline.py --full --model-type unet
```

### 分步运行（推荐）

```bash
# Step 1: 运行 Phase 3A
python run_phase3_pipeline.py

# Step 2: 检查映射结果
# 查看 data/03_mapped/visualizations/ 中的可视化图片

# Step 3: 运行 Phase 3B 训练
python run_phase3_pipeline.py --phase3b --model-type unet

# Step 4: 运行 Phase 3B 推理
python run_phase3_pipeline.py --inference --model-type unet

# Step 5: 模型评估
python run_phase3_pipeline.py --evaluate --model-type unet

# Step 6: 生成可视化
python run_phase3_pipeline.py --visualize --model-type unet
```

---

## 批量执行命令

### PowerShell 单独执行命令（9条）

```powershell
# === 推理命令（3条）===
python run_phase3_pipeline.py --inference --model-type unet
python run_phase3_pipeline.py --inference --model-type partial_conv
python run_phase3_pipeline.py --inference --model-type patchgan

# === 评估命令（3条）===
python run_phase3_pipeline.py --evaluate --model-type unet
python run_phase3_pipeline.py --evaluate --model-type partial_conv
python run_phase3_pipeline.py --evaluate --model-type patchgan

# === 可视化命令（3条）===
python run_phase3_pipeline.py --visualize --model-type unet
python run_phase3_pipeline.py --visualize --model-type partial_conv
python run_phase3_pipeline.py --visualize --model-type patchgan
```

### PowerShell 一键批量执行

```powershell
# 依次对三个模型执行推理→评估→可视化
@("unet", "partial_conv", "patchgan") | ForEach-Object {
    Write-Host "========== 处理模型: $_ ==========" -ForegroundColor Green
    python run_phase3_pipeline.py --inference --model-type $_
    python run_phase3_pipeline.py --evaluate --model-type $_
    python run_phase3_pipeline.py --visualize --model-type $_
}
Write-Host "========== 全部完成! ==========" -ForegroundColor Green
```

### 快速测试命令

```powershell
# 仅处理少量数据的快速验证
python run_phase3_pipeline.py --inference --model-type unet --limit 3
python run_phase3_pipeline.py --evaluate --model-type unet --limit 3
python run_phase3_pipeline.py --visualize --model-type unet --limit 3
```

---

## 输出目录结构

```text
data/
├── 03_mapped/                    # Phase 3A 输出
│   ├── copd_001/
│   │   ├── copd_001_warped.nii.gz        # 配准后的 CT
│   │   └── copd_001_warped_lesion.nii.gz # 配准后的病灶 mask
│   └── visualizations/
│       └── copd_001_view_*.png           # 三视图渲染
│
├── 04_final_viz/                 # Phase 3B 推理输出
│   ├── unet/                     # UNet 模型结果
│   │   └── copd_001_fused.nii.gz
│   ├── partial_conv/             # Partial Conv 模型结果
│   │   └── copd_001_fused.nii.gz
│   └── patchgan/                 # PatchGAN 模型结果
│       └── copd_001_fused.nii.gz

checkpoints/
├── unet/                         # UNet 检查点
│   ├── best.pth
│   └── training_log.json
├── partial_conv/                 # Partial Conv 检查点
│   └── best.pth
└── patchgan/                     # PatchGAN 检查点
    └── best.pth

evaluation_results/
├── unet/                         # UNet 评估报告
│   └── evaluation_report.md
├── partial_conv/
└── patchgan/

visualization_results/
├── unet/                         # UNet 可视化
│   └── copd_001_visualization.png
├── partial_conv/
└── patchgan/
```

---

**文档版本**: v2.0
**最后更新**: 2026-01-13