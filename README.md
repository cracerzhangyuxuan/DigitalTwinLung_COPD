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
| 预处理 | TotalSegmentator | 自动肺部分割 |
| 配准 | ANTsPy (SyN) | 非线性空间映射 |
| AI融合 | PyTorch (U-Net) | 病灶纹理Inpainting |
| 可视化 | PyVista (VTK) | 3D体渲染 |

## 📁 项目结构

```
DigitalTwinLung_COPD/
├── data/                    # 数据层
├── src/                     # 代码层
├── tests/                   # 测试层
├── notebooks/               # 实验层
├── checkpoints/             # 模型权重
├── logs/                    # 运行日志
├── docs/                    # 文档
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

# 安装依赖
pip install -r requirements.txt

# 安装 ANTsPy (需要从源码或 conda)
# conda install -c aramislab antspyx

# 安装 TotalSegmentator
pip install TotalSegmentator
```

### 2. 数据准备

将原始CT数据放入 `data/00_raw/` 目录：
- 正常肺：`data/00_raw/normal/`
- COPD患者：`data/00_raw/copd/`

### 3. 运行流水线

```bash
# 一键运行全流程
python run_pipeline.py

# 或分阶段运行
python -m src.01_preprocessing.run_segmentation
python -m src.02_atlas_build.build_template_ants
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
| Phase 2 | 底座构建 | 15-20例正常肺 | Template |
| Phase 3 | AI融合 | 30-50例COPD | 融合CT |
| Phase 4 | 演示输出 | 融合CT | 视频/图片 |

## 📖 文档

- [工程实施指南](Engineering_Edition.md)
- [技术方案文档](v5_1_Final.md)
- [进度追踪](Project_Progress_Tracker.md)

## 📝 许可证

本项目仅供学术研究使用。

## 👤 作者

硕士毕业设计项目

