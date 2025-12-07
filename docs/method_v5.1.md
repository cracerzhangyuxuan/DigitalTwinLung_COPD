# COPD 数字孪生肺项目 - 方法论文档 v5.1

## 概述

本项目旨在构建 COPD（慢性阻塞性肺疾病）患者的数字孪生肺模型，通过结合传统医学图像配准技术和深度学习纹理合成方法，实现高保真的 3D 肺部可视化。

## 核心方法

### 1. 混合建模策略

本项目采用"传统配准 + AI 纹理融合"的混合策略：

- **传统配准**：使用 ANTsPy 的 SyN 算法进行非刚性配准，确保解剖结构的准确对齐
- **AI 纹理融合**：使用 U-Net Inpainting 网络在病灶区域生成真实纹理

### 2. 数据处理流程

```
原始 CT → 分割 → 背景清洗 → 病灶提取 → 配准 → AI 融合 → 3D 可视化
```

### 3. 关键算法

#### 3.1 LAA-950 病灶提取

使用 -950 HU 阈值提取肺气肿区域：

```python
emphysema_mask = (ct_data < -950) & (lung_mask > 0)
```

#### 3.2 SyN 非刚性配准

使用 ANTsPy 的 SyN 算法进行配准：

```python
registration = ants.registration(
    fixed=template,
    moving=patient_ct,
    type_of_transform='SyNRA'
)
```

#### 3.3 Inpainting 网络

使用 3D U-Net 进行病灶区域的纹理填充：

- 输入：带空洞的 CT patch
- 输出：填充后的 CT patch
- 损失：重建损失 + 感知损失 + 对抗损失

## 评估指标

### 图像质量指标

| 指标 | 阈值 | 说明 |
|------|------|------|
| SSIM | ≥ 0.85 | 结构相似性 |
| PSNR | ≥ 25.0 dB | 峰值信噪比 |
| NCC | ≥ 0.90 | 归一化互相关 |

### 分割指标

| 指标 | 阈值 | 说明 |
|------|------|------|
| Dice | ≥ 0.85 | Dice 系数 |
| HD95 | ≤ 10.0 mm | 95% Hausdorff 距离 |

## 技术栈

- **医学图像处理**：TotalSegmentator, ANTsPy, nibabel
- **深度学习**：PyTorch
- **3D 可视化**：PyVista

## 参考文献

1. Avants, B. B., et al. (2011). A reproducible evaluation of ANTs similarity metric performance in brain image registration.
2. Ronneberger, O., et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
3. Wasserthal, J., et al. (2023). TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images.

