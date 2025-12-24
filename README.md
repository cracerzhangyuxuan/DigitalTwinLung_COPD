# COPD 数字孪生肺项目

> 基于全代码自动化的COPD数字孪生肺构建与3D可视化研究

## 📋 项目概述

本项目采用"混合建模"策略，结合传统医学图像配准算法（ANTsPy）和生成式AI（Inpainting），构建可控的COPD数字孪生肺模型，并实现全代码化的3D可视化。

### 核心特点

- **全自动化流水线**：不依赖GUI软件，纯代码实现
- **混合建模策略**：传统配准保证解剖合规性 + AI解决纹理融合
- **可控病灶生成**：精确控制病灶位置和体积
- **动态3D可视化**：呼吸模拟动画 + 高清渲染输出

## 🛠 技术栈

| 模块 | 技术 | 作用 |
|------|------|------|
| 肺叶分割 | LungMask (LTRCLobes_R231) | 5肺叶精细分割，边界清晰 |
| 气管树分割 | Raidionicsrads (AGU-Net) | 3-4级支气管分割 |
| 配准 | ANTsPy (SyN) | 非线性空间映射 |
| AI融合 | PyTorch (U-Net) | 病灶纹理Inpainting |
| 可视化 | PyVista (VTK) | 3D体渲染 |

> **注意**：2025-12-24 更新，已将 TotalSegmentator 替换为 LungMask + Raidionicsrads，原因：
> - TotalSegmentator 气管树分割仅能分割主气管，缺少分支结构
> - TotalSegmentator 肺叶分割边界碎片化严重

## 📁 项目结构

```
DigitalTwinLung_COPD/
├── data/                    # 数据层 (不纳入版本控制)
├── src/                     # 代码层
├── tests/                   # 测试层
├── notebooks/               # 实验层
├── checkpoints/             # 模型权重 (不纳入版本控制)
├── logs/                    # 运行日志 (不纳入版本控制)
├── docs/                    # 文档
├── AeroPath/                # [可选] 第三方气管树分割模型 (不纳入版本控制)
├── config.yaml              # 全局配置
├── requirements.txt         # 依赖列表
└── run_pipeline.py          # 主入口
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装基础依赖
pip install -r requirements.txt

# 安装 ANTsPy (需要从源码或 conda)
# conda install -c aramislab antspyx
```

#### 1.1 肺叶分割模型 (LungMask)

```bash
# 安装 LungMask
pip install lungmask

# 验证安装
python -c "from lungmask import LMInferer; print('LungMask 安装成功')"
```

LungMask 特点：
- 使用 LTRCLobes_R231 融合模型，肺叶边界清晰
- 支持正常肺和病理肺（COPD、COVID-19等）
- GPU 加速，单例 5-10 秒

#### 1.2 气管树分割模型 (Raidionicsrads)

```bash
# 安装 Raidionicsrads
pip install raidionicsrads

# 验证安装
python -c "from raidionicsrads.compute import run_model; print('Raidionicsrads 安装成功')"
```

Raidionicsrads 特点：
- 基于 AGU-Net 架构，可分割到 3-4 级支气管
- 分支结构完整，适合气管树模板构建
- 首次运行自动下载预训练权重

### 2. 数据准备

将原始CT数据放入 `data/00_raw/` 目录：
- 正常肺：`data/00_raw/normal/`
- COPD患者：`data/00_raw/copd/`

### 3. 运行流水线

```bash
# 一键运行全流程
python run_pipeline.py

# 或分阶段运行

# Phase 1: 预处理（含气管树分割和肺叶标记）
python -m src.01_preprocessing.run_segmentation

# Phase 2: 标准底座构建（含气管树和5肺叶标签）
python run_phase2_pipeline.py

# Phase 2 常用选项:
# 快速测试（3例）: python run_phase2_pipeline.py --quick-test
# 仅分割步骤:      python run_phase2_pipeline.py --step1-only
# 仅气管树模板:    python run_phase2_pipeline.py --step2-only
# 限制处理数量:    python run_phase2_pipeline.py --limit 5
# 强制覆盖:        python run_phase2_pipeline.py --force

# Phase 3-4: 配准、融合与可视化
python -m src.03_registration.register_lesions
python -m src.04_texture_synthesis.train
python -m src.05_visualization.static_render
```

### 4. 运行测试

```bash
pytest tests/ -v
```

## 📊 四阶段实施流程

| 阶段 | 目标 | 输入 | 输出 |
|------|------|------|------|
| Phase 1 | MVP验证 | 3+1例CT | 3D截图 |
| Phase 2 | 底座构建 | 37例正常肺 | Template + 气管树Mask |
| Phase 3 | AI融合 | 29例COPD | 融合CT |
| Phase 4 | 演示输出 | 融合CT | 视频/图片 |

**Phase 2 输出文件：**
- `standard_template.nii.gz` - 标准肺部模板
- `standard_mask.nii.gz` - 肺部模板 mask
- `standard_trachea_mask.nii.gz` - 气管树模板 mask（新增）

## 📖 文档

- [工程实施指南](Engineering_Edition.md)
- [技术方案文档](v5_1_Final.md)
- [进度追踪](Project_Progress_Tracker.md)

## 📝 许可证

本项目仅供学术研究使用。

## 👤 作者

硕士毕业设计项目

