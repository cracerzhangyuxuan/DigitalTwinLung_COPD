# Phase 3B 设计与实现文档

**日期**: 2026-01-07  
**版本**: v1.0

---

## 1. 整体架构

### 1.1 数据流

```
Phase 3A 输出                    Phase 3B 处理                    最终输出
─────────────────────────────────────────────────────────────────────────────
data/03_mapped/                  训练 Inpainting 模型             data/04_final_viz/
├── copd_001/                    ┌─────────────────┐              ├── copd_001_fused.nii.gz
│   ├── *_warped.nii.gz    ───▶ │  LungPatchDataset │              ├── copd_002_fused.nii.gz
│   └── *_warped_lesion.nii.gz  │  (64³ patches)   │              └── ...
├── copd_002/                    └────────┬────────┘
│   └── ...                               │
└── ...                                   ▼
                                 ┌─────────────────┐
data/02_atlas/                   │  InpaintingUNet  │
├── standard_template.nii.gz ──▶│  (3D U-Net)      │
└── standard_mask.nii.gz        └────────┬────────┘
                                         │
                                         ▼
                                 ┌─────────────────┐
                                 │  fuse_lesion()  │──▶ 融合 CT
                                 │  + 边界平滑     │
                                 └─────────────────┘
```

### 1.2 模块划分

| 模块 | 文件 | 功能 |
|------|------|------|
| 数据集 | `dataset.py` | 3D Patch 提取、数据增强 |
| 网络 | `network.py` | InpaintingUNet, PatchDiscriminator |
| 损失函数 | `losses.py` | L1, Perceptual, Adversarial, SSIM |
| 训练 | `train.py` | Trainer 类, 检查点管理 |
| 推理 | `inference_fuse.py` | 滑动窗口推理, 边界平滑 |

---

## 2. 模型架构

### 2.1 三种方案对比

| 方案 | 类名 | 特点 | 适用场景 | 命令行参数 |
|------|------|------|----------|------------|
| **基线方案** | `InpaintingUNet` | 实现简单，效果稳定 | 首选方案 | `--model-type unet` |
| **进阶方案** | `PartialConvUNet` | 处理不规则 Mask 更优 | 复杂病灶形状 | `--model-type partial_conv` |
| **高级方案** | `InpaintingUNet + PatchDiscriminator` | 效果最佳，训练复杂 | 追求最佳质量 | `--model-type patchgan` |

### 2.2 InpaintingUNet (基线方案)

```
输入: (B, 1, 64, 64, 64) - 带空洞的 CT patch
      ↓
┌─────────────────────────────────────────────────────────────┐
│  编码器 (Encoder)                                           │
│  ├── ConvBlock3D(1→32)   → skip1                           │
│  ├── DownBlock(32→64)    → skip2                           │
│  ├── DownBlock(64→128)   → skip3                           │
│  └── DownBlock(128→256)  → skip4                           │
├─────────────────────────────────────────────────────────────┤
│  瓶颈 (Bottleneck)                                          │
│  └── ConvBlock3D(256→512)                                  │
├─────────────────────────────────────────────────────────────┤
│  解码器 (Decoder)                                           │
│  ├── UpBlock(512→256) + skip4                              │
│  ├── UpBlock(256→128) + skip3                              │
│  ├── UpBlock(128→64)  + skip2                              │
│  └── UpBlock(64→32)   + skip1                              │
├─────────────────────────────────────────────────────────────┤
│  输出层                                                     │
│  └── Conv3d(32→1, kernel=1)                                │
└─────────────────────────────────────────────────────────────┘
      ↓
输出: (B, 1, 64, 64, 64) - 填充后的 CT patch

参数量: 22,581,217 (~22.6M)
```

### 2.2 PatchDiscriminator (判别器，可选)

```
输入: (B, 1, 64, 64, 64)
      ↓
Conv3d(1→64, k=4, s=2) → LeakyReLU
Conv3d(64→128, k=4, s=2) → BN → LeakyReLU
Conv3d(128→256, k=4, s=2) → BN → LeakyReLU
Conv3d(256→512, k=4, s=2) → BN → LeakyReLU
Conv3d(512→1, k=4, s=1)
      ↓
输出: (B, 1, 2, 2, 2) - patch-wise 判别结果
```

---

## 3. 损失函数

### 3.1 组合损失

```python
Total Loss = λ_rec * L_reconstruction + λ_per * L_perceptual + λ_adv * L_adversarial
```

| 损失 | 权重 | 说明 |
|------|------|------|
| L_reconstruction | 1.0 | L1 损失，mask 区域 10x 权重 |
| L_perceptual | 0.1 | 多尺度特征差异 |
| L_adversarial | 0.01 | LSGAN 损失（可选） |

### 3.2 Mask 加权

```python
weight = 1.0 + (mask_weight - 1.0) * mask  # mask 区域权重 = 10.0
loss = F.l1_loss(pred, target, reduction='none') * weight
```

---

## 4. 训练策略

### 4.1 数据集构建

- **Patch 大小**: 64×64×64 体素
- **每个体积提取**: 50 个 patch
- **采样策略**: 在病灶 mask 区域内采样
- **数据增强**: 随机翻转、90° 旋转

### 4.2 训练参数

| 参数 | 默认值 |
|------|--------|
| Batch Size | 4 |
| Learning Rate | 0.0002 |
| Optimizer | Adam (β1=0.5, β2=0.999) |
| Epochs | 100 |
| LR Scheduler | StepLR (step=50, γ=0.5) |

### 4.3 检查点保存

- `checkpoints/best.pth` - 最佳验证损失
- `checkpoints/latest.pth` - 最新检查点
- `checkpoints/training_log.json` - 训练历史

---

## 5. 推理流程

### 5.1 滑动窗口推理

```python
patch_size = (64, 64, 64)
overlap = 16
step = patch_size - overlap  # = 48

for z in range(0, D-64, 48):
    for y in range(0, H-64, 48):
        for x in range(0, W-64, 48):
            if mask[z:z+64, y:y+64, x:x+64].sum() > 0:
                patch = model(input_patch)
                output[mask_region] += patch[mask_region]
                weight[mask_region] += 1

output = output / weight  # 重叠区域平均
```

### 5.2 边界平滑

```python
# 创建边界区域 (膨胀 - 腐蚀)
dilated = binary_dilation(mask, iterations=3)
eroded = binary_erosion(mask, iterations=3)
boundary = dilated & ~eroded

# 距离加权混合
distance = distance_transform_edt(~mask)
distance = clip(distance, 0, 3) / 3

result[boundary] = output[boundary] * (1-distance) + original[boundary] * distance
```

---

## 6. 使用指南

### 6.1 训练

```bash
# 基础训练（仅 U-Net）
python run_phase3b_training.py --no-gan --epochs 50

# 完整训练（U-Net + GAN）
python run_phase3b_training.py --epochs 100

# 从检查点恢复
python run_phase3b_training.py --resume checkpoints/latest.pth

# 调整参数
python run_phase3b_training.py --batch-size 2 --lr 0.0001
```

### 6.2 推理

```bash
# 批量推理
python run_phase3b_inference.py

# 指定患者
python run_phase3b_inference.py --patient copd_001

# 禁用边界平滑
python run_phase3b_inference.py --no-smooth
```

---

## 7. 文件结构

```
src/04_texture_synthesis/
├── __init__.py           # 模块导出
├── dataset.py            # LungPatchDataset
├── network.py            # InpaintingUNet, PatchDiscriminator
├── losses.py             # InpaintingLoss, SSIMLoss
├── train.py              # Trainer 类
└── inference_fuse.py     # fuse_lesion, batch_fuse

run_phase3b_training.py   # 训练入口脚本
run_phase3b_inference.py  # 推理入口脚本
```

---

## 8. 验收标准

| 指标 | 目标值 |
|------|--------|
| SSIM | ≥ 0.85 |
| PSNR | ≥ 25 dB |
| 边界质量 | 无明显断裂 |
| 纹理真实性 | 视觉评估通过 |

---

**文档版本**: v1.0  
**最后更新**: 2026-01-07

