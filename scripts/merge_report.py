import os

path = 'results/model_comparison_report.md'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Keep the first 186 lines (experimental background and basic metrics)
header = lines[:186]

new_content = """## 四、高级架构评估与验证集泛化分析 (Held-Out Evaluation)

### 4.1 新增模型特性深度解读

通过对 AttGAN、MAE-PatchGAN 和 DDPM 的引入，我们获得了以下关键技术洞察：

- **AttGAN (注意力机制的权衡)**：在验证集上实现了最高的纹理对比度（GLCM-C=1.6255），但在像素精度上表现出轻微波动（MAE 升高至 34.36 HU）。这证实了注意力门控在强化局部特征差异的同时，可能降低全局数值的稳定性。
- **MAE-PatchGAN (自监督预训练的效率)**：展示了极佳的收敛优势。仅需 50 Epochs 的微调即可达到基准模型 150 Epochs 的性能，且在 MAE 指标上表现稳健（33.19 HU），是数据受限场景下的首选预训练方案。
- **DDPM (去噪扩散模型的现状)**：由于 3D 扩散推理开销极大，当前 checkpoint 仅训练 2 Epochs，处于严重欠拟合状态（PSNR=18.81 dB, ΔEI=20.08%）。该结果仅作为扩散模型路线的技术占位，不代表其最终潜力。

### 4.2 验证集全面对标汇总 (Average across 6 Patients: copd_024~029)

| 模型 | PSNR↑ | MAE↓ | GLCM-C↑ | W-dist↓ | ΔEI↓ |
|:---|---:|---:|---:|---:|---:|
| U-Net | 28.78 | 33.37 | 1.6087 | 18.30 | 4.26% |
| PartialConv | 28.90 | 33.18 | 1.6086 | **18.08** | **3.53%** |
| **PatchGAN** | **28.91** | **32.99** | 1.5893 | 18.29 | 3.78% |
| AttGAN | 28.79 | 34.36 | **1.6255** | 18.86 | 4.16% |
| MAE-PatchGAN | 28.79 | 33.19 | 1.5742 | 19.04 | 4.03% |
| DDPM (欠拟合) | 18.81 | 113.31 | 5.9625 | 95.80 | 20.08% |
| *Real Ref* | *—* | *—* | *3.1820* | *—* | *—* |

### 4.3 最终模型推荐：PatchGAN (Optimal Balanced Selection)

综合考虑 L1 像素精度与 L4 临床安全性，**PatchGAN** 被选定为数字孪生肺的核心生成模型：
1. **精度领先**：在验证集 PSNR 和 MAE 上均位居第一，确保护理区域的数值保真度。
2. **临床达标**：ΔEI (3.78%) 远低于 5% 安全阈值，且在应对超大病灶时比纯卷积模型更具鲁棒性。
3. **架构稳健**：Patch-based 判别反馈有效解决了 L1 损失导致的纹理平滑问题。

---

## 五、核心图表汇总 (Key Visualizations)

- **五模型验证集对比 (雷达图)**: ![Validation Radar](results/charts_val/chart_radar_l2l4_no_ddpm.png)
- **临床指标 ΔEI 分布**: ![Clinical Fidelity](results/charts_val/chart_delta_ei.png)
- **复合评分排名**: ![Composite Score](results/charts_val/chart_composite_score.png)

---

## 六、技术规范与数据集策略

### 6.1 数据集划分 (Patient-level Split)

本研究使用 29 例 COPD 患者数据，严格按患者 ID 划分：
- **训练集**：copd_001 ~ copd_023 (23 例)
- **验证集**：copd_024 ~ copd_029 (6 例)
- **评估样本**：训练集 3 例 + 验证集 6 例 = 9 例 (54 组记录)

### 6.2 训练配置汇总

| 模型 | Epochs (实际) | 优化器 | GPU 时长 | 状态 |
|:---|:---:|:---|:---:|:---:|
| Baseline (U-Net/GAN) | 150 | Adam (2e-4) | ~8h | 完全收敛 |
| AttGAN | 127 | Adam (2e-4) | ~10h | 最佳状态 |
| MAE-PatchGAN | 53 (微调) | Adam (1e-4) | ~5h | 快速收敛 |
| DDPM | 2 | Adam (2e-4) | ~48h | 欠拟合 |

### 6.3 改进方向 (Next Steps)

1. **DDPM 长周期训练**：将 DDPM 训练周期提升至 200+ Epochs 以观察其真实纹理上限。
2. **大病灶优化**：针对 >50 万体素的超大病灶，引入多尺度判别器或层级修复策略。
3. **损失函数演进**：引入频谱损失（Spectral Loss）以进一步缩小 GLCM Contrast 与真实值的差距。

> *数据来源：`results/validation_metrics.json` (36 条) + `results/model_comparison_metrics.json` (18 条) | 日期：2026-04-01*
"""

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(header)
    f.write(new_content)

print("Done.")

