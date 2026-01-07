# Digital Twin Lung COPD 项目交接文档

## 1. 项目概述

### 1.1 项目名称与目标
- **项目名称**: Digital Twin Lung COPD (数字孪生肺 COPD 项目)
- **核心目标**: 构建 COPD（慢性阻塞性肺疾病）患者的数字孪生肺模型
- **最终产出**: 可视化的 3D 肺部模型，能够展示病灶分布和严重程度

### 1.2 核心功能
1. **标准底座构建**: 从多例正常肺 CT 构建标准化肺部模板
2. **病灶空间映射**: 将 COPD 患者病灶配准到标准空间
3. **AI 纹理融合**: 使用深度学习修复/融合病灶区域纹理（开发中）
4. **3D 可视化**: 生成多视角渲染图，支持 5 肺叶着色

### 1.3 技术栈
| 类别 | 工具/库 |
|------|---------|
| 语言 | Python 3.9+ |
| 医学影像 | nibabel, SimpleITK |
| 图像配准 | ANTsPy (SyN/SyNRA 算法) |
| 肺分割 | LungMask (LTRCLobes), TotalSegmentator |
| 深度学习 | PyTorch (GPU 加速) |
| 3D 渲染 | PyVista |
| 配置管理 | PyYAML |

### 1.4 数据流向
```
原始 COPD CT (NIfTI)
    ↓ [Phase 1: 预处理]
肺部分割 + 背景清洗 + 病灶提取 (LAA-950)
    ↓ [Phase 2: 底座构建]
从正常肺构建标准模板 + 肺叶/气管 mask
    ↓ [Phase 3: 空间映射]
ANTsPy SyNRA 配准 → 病灶映射到标准空间
    ↓ [Phase 3B: AI 融合] (开发中)
Inpainting 模型修复病灶区域纹理
    ↓ [Phase 4: 评估] (待开发)
SSIM/PSNR/Dice 质量评估
    ↓ [Phase 5: 可视化]
PyVista 3D 渲染 + 呼吸动画
```

---

## 2. 项目架构

### 2.1 目录结构
```
DigitalTwinLung_COPD/
├── config.yaml                    # 全局配置文件（关键！）
├── run_phase2_atlas.py            # Phase 2 入口：标准底座构建
├── run_phase3_pipeline.py         # Phase 3 入口：空间映射流水线
├── process_copd_data.py           # COPD 数据预处理脚本
│
├── src/                           # 源代码
│   ├── 01_preprocessing/          # Phase 1: 预处理模块
│   │   ├── clean_background.py    # 背景清洗
│   │   ├── extract_emphysema.py   # LAA-950 病灶提取
│   │   ├── precise_lung_segment.py # 精细肺分割
│   │   └── simple_lung_segment.py  # 简单阈值分割
│   │
│   ├── 02_atlas_build/            # Phase 2: 标准底座构建
│   │   └── build_template_ants.py # ANTs 模板构建 + JLF 肺叶融合
│   │
│   ├── 03_registration/           # Phase 3: 空间配准
│   │   └── register_lesions.py    # ANTsPy SyNRA 病灶配准
│   │
│   ├── 04_texture_synthesis/      # Phase 3B: AI 纹理融合 (开发中)
│   │   ├── dataset.py             # 数据加载器
│   │   ├── network.py             # 网络架构
│   │   ├── losses.py              # 损失函数
│   │   └── train.py               # 训练脚本
│   │
│   ├── 05_visualization/          # Phase 5: 可视化
│   │   ├── static_render.py       # 静态 3D 渲染（三视图 + 5 肺叶着色）
│   │   └── dynamic_breath.py      # 呼吸动画
│   │
│   └── utils/                     # 工具模块
│       ├── io.py                  # NIfTI 读写
│       ├── logger.py              # 日志管理
│       └── metrics.py             # 评估指标
│
├── data/                          # 数据目录
│   ├── 00_raw/                    # 原始数据
│   │   ├── copd/                  # 29 例 COPD CT
│   │   └── normal/                # 正常肺 CT（用于底座构建）
│   │
│   ├── 01_cleaned/                # 预处理后数据
│   │   ├── copd_clean/            # 清洗后 CT
│   │   ├── copd_mask/             # 肺部 mask
│   │   └── copd_emphysema/        # 病灶 mask (LAA-950)
│   │
│   ├── 02_atlas/                  # 标准底座
│   │   ├── standard_template.nii.gz          # 标准 CT 模板
│   │   ├── standard_mask.nii.gz              # 二值肺 mask
│   │   ├── standard_lung_lobes_labeled.nii.gz # 5 肺叶标签 (1-5)
│   │   └── standard_trachea_mask.nii.gz      # 气管树 mask
│   │
│   ├── 03_mapped/                 # 配准结果
│   │   ├── copd_001/              # 患者输出目录
│   │   │   ├── *_warped.nii.gz        # 变形后 CT
│   │   │   ├── *_warped_lesion.nii.gz # 变形后病灶 mask
│   │   │   ├── *_transform_0.nii.gz   # SyN 形变场
│   │   │   └── *_transform_1.mat      # 仿射变换矩阵
│   │   └── visualizations/        # 可视化输出
│   │
│   └── 04_final_viz/              # 最终渲染结果
│
└── logs/                          # 日志文件
```

### 2.2 各 Phase 功能划分

| Phase | 目录/脚本 | 功能 | 状态 |
|-------|-----------|------|------|
| Phase 1 | `src/01_preprocessing/` | 肺分割、背景清洗、病灶提取 | ✅ 完成 |
| Phase 2 | `src/02_atlas_build/` | 标准底座构建、JLF 肺叶融合 | ✅ 完成 |
| Phase 3 | `src/03_registration/` | 病灶空间配准 | ✅ 完成 |
| Phase 3B | `src/04_texture_synthesis/` | AI 纹理融合 Inpainting | ⏳ 待开发 |
| Phase 4 | - | 质量评估 | ⏳ 待开发 |
| Phase 5 | `src/05_visualization/` | 3D 渲染、呼吸动画 | ✅ 基础完成 |

### 2.3 关键配置文件 `config.yaml`
- **路径配置**: 所有数据目录路径
- **预处理参数**: 分割模型、LAA-950 阈值
- **配准参数**: SyNRA 迭代次数、缩放因子
- **训练参数**: 批大小、学习率（Phase 3B 用）
- **可视化参数**: 透明度、颜色、分辨率

---

## 3. 当前开发进度

### 3.1 已完成的 Phase
| Phase | 完成时间 | 验证状态 |
|-------|----------|----------|
| Phase 1 预处理 | 2025-12-30 | ✅ 3 例 COPD 测试通过 |
| Phase 2 底座构建 | 2025-12-24 | ✅ Dice >= 0.85 验证通过 |
| Phase 3 空间映射 | 2025-12-31 | ✅ 3/3 配准成功 |
| Phase 5 可视化 | 2025-12-31 | ✅ 5 肺叶着色修复完成 |

### 3.2 最近完成的工作
**日期**: 2025-12-31

1. **Phase 3 配准流水线完成**:
   - 修复变换文件后缀问题（`.nii.gz` vs `.gz`）
   - 修复维度不匹配时的错误处理
   - 3 例快速测试全部配准成功

2. **可视化 5 肺叶着色修复**:
   - 修复 `keep_largest_n=1` → `2`（显示双侧肺）
   - 添加 `lobes_mask_path` 参数支持 5 肺叶着色
   - 生成三视图渲染：x/y/z 轴

### 3.3 待开发功能
| 功能 | 优先级 | 预计工作量 |
|------|--------|------------|
| 处理全部 29 例 COPD | 高 | 约 8.5 小时（29 × 17.5 分钟） |
| Phase 3B AI 纹理融合 | 高 | 需要设计 Inpainting 网络架构 |
| Phase 4 质量评估 | 中 | SSIM/Dice 指标计算 |
| 呼吸动画增强 | 低 | 需优化帧率和平滑度 |

---

## 4. 关键技术细节

### 4.1 Phase 2 标准底座构建
```python
# 使用 ANTsPy 的 build_template 函数
template = ants.build_template(
    initial_template=initial_image,
    image_list=images,
    type_of_transform="SyN",
    iterations=5,
    gradient_step=0.2
)
```
- **输入**: 5-37 例正常肺 CT
- **输出**: `standard_template.nii.gz`（512×512×364）
- **验收标准**: Dice >= 0.85

**肺叶标签融合 (JLF)**:
```python
# Joint Label Fusion 融合肺叶标签
fused_labels = ants.joint_label_fusion(
    target_image=template,
    target_image_mask=template_mask,
    atlas_list=registered_cts,
    label_list=registered_lobes,
    beta=4, rad=2
)
```

### 4.2 Phase 3 空间映射 (ANTsPy SyNRA)
```python
# 配准算法
registration = ants.registration(
    fixed=template,
    moving=moving_ct,
    type_of_transform="SyNRA",  # 刚性+仿射+SyN
    reg_iterations=(20, 10, 0),  # 优化后的快速参数
    shrink_factors=(4, 2, 1),
    smoothing_sigmas=(2, 1, 0)
)

# 应用变换到病灶 mask
warped_lesion = ants.apply_transforms(
    fixed=template,
    moving=lesion_mask,
    transformlist=registration['fwdtransforms'],
    interpolator='nearestNeighbor'  # 保持二值
)
```

### 4.3 肺叶分割和病灶检测
**肺叶分割** (LungMask LTRCLobes):
```python
import lungmask
model = lungmask.get_model(modelname='LTRCLobes', fillmodel='R231')
segmentation = lungmask.apply(input_image, model)
# 输出: 5 标签 (1=左上叶, 2=左下叶, 3=右上叶, 4=右中叶, 5=右下叶)
```

**病灶检测** (LAA-950 算法):
```python
# 肺气肿区域 = HU < -950 且在肺内
emphysema_mask = (ct_data < -950) & (lung_mask > 0)
```

### 4.4 可视化渲染 (PyVista)
```python
# 5 肺叶着色渲染
lobe_colors = {
    1: (0.4, 0.6, 0.9),   # 左上叶 - 浅蓝
    2: (0.2, 0.4, 0.8),   # 左下叶 - 深蓝
    3: (0.9, 0.6, 0.4),   # 右上叶 - 浅橙
    4: (0.9, 0.8, 0.4),   # 右中叶 - 黄
    5: (0.8, 0.4, 0.4),   # 右下叶 - 红
}

# 调用方式
render_multiview(
    ct_path=template_path,
    lesion_mask_path=warped_lesion_path,
    lung_mask_path=mask_path,
    lobes_mask_path=lobes_path,  # 5 肺叶着色
    lung_opacity=0.25,
    lesion_opacity=0.9
)
```

---

## 5. 最近修复的问题

### 5.1 Phase 3 配准失败问题
| 问题 | 原因 | 修复 |
|------|------|------|
| Transform 文件读取失败 | `Path.suffix` 只返回 `.gz`，不是 `.nii.gz` | 使用 `suffixes` 获取完整后缀 |
| 维度不匹配报错 | 模板 mask 与 warped 形状不一致 | 添加跳过逻辑，不强制约束 |

**代码位置**: `src/03_registration/register_lesions.py` 第 100-111 行

### 5.2 可视化只显示单侧肺叶
| 问题 | 原因 | 修复 |
|------|------|------|
| 只显示一个肺 | `keep_largest_n=1` 只保留最大连通分量 | 改为 `keep_largest_n=2` |
| 没有 5 肺叶颜色 | 使用二值 mask 而非肺叶标签 | 添加 `lobes_mask_path` 参数 |

**代码位置**: 
- `src/05_visualization/static_render.py` 第 514 行
- `run_phase3_pipeline.py` 第 433-446 行

---

## 6. 数据说明

### 6.1 输入数据
| 数据集 | 路径 | 数量 | 说明 |
|--------|------|------|------|
| COPD CT | `data/00_raw/copd/` | 29 例 | NIfTI 格式 |
| 正常肺 CT | `data/00_raw/normal/` | 37 例 | 用于底座构建 |

### 6.2 标准模板
| 文件 | 大小 | 形状 | 说明 |
|------|------|------|------|
| `standard_template.nii.gz` | ~50 MB | 512×512×364 | 标准 CT 模板 |
| `standard_mask.nii.gz` | ~1 MB | 同上 | 二值肺 mask |
| `standard_lung_lobes_labeled.nii.gz` | ~1 MB | 同上 | 5 肺叶标签 (1-5) |
| `standard_trachea_mask.nii.gz` | ~0.5 MB | 同上 | 气管树 mask |

### 6.3 测试数据
已处理的 3 例快速测试数据:
| 患者 ID | LAA% | 严重程度 | 病灶体素数 |
|---------|------|----------|------------|
| copd_001 | 0.5% | 极轻微 (GOLD 0-1) | 69,015 |
| copd_002 | 1.8% | 轻度 (GOLD 1) | 286,540 |
| copd_003 | 35.6% | 重度 (GOLD 3-4) | 5,909,515 |

**注意**: 85 倍的病灶体素差异是临床合理的，反映了 COPD 从早期到晚期的进展。

---

## 7. 下一步工作建议

### 7.1 处理全部 29 例 COPD 数据
```bash
# Step 1: 预处理（约 2.5 小时）
python process_copd_data.py

# Step 2: 空间映射（约 8.5 小时）
python run_phase3_pipeline.py

# Step 3: 生成全部可视化
python run_phase3_pipeline.py --viz-only
```

### 7.2 Phase 3B AI 纹理融合开发计划
1. **数据准备**: 构建配对训练集 (病灶区域 → 正常纹理)
2. **网络设计**: 参考 Partial Conv Inpainting 或 GAN 架构
3. **损失函数**: 重建损失 + 感知损失 + 对抗损失
4. **训练**: 使用 `src/04_texture_synthesis/` 中的框架

### 7.3 Phase 4 & 5 实现路线图
| 阶段 | 任务 | 依赖 |
|------|------|------|
| Phase 4 | 实现 SSIM/PSNR/Dice 评估 | Phase 3 完成 |
| Phase 5a | 优化三视图渲染质量 | Phase 3 完成 |
| Phase 5b | 实现呼吸动画 | Phase 4 完成 |

---

## 8. 重要注意事项

### 8.1 用户偏好
- **语言**: 用户偏好用中文交流
- **交互风格**: 直接给出解决方案，避免过多确认

### 8.2 已知设计决策
1. **气管树缺失**: 预处理使用 LTRCLobes 模型只分割 5 肺叶，不包含气管树
   - **影响**: 配准后 CT 没有气管树
   - **解决**: 可视化时使用标准模板作为背景
   - **不影响**: 病灶配准的准确性

2. **变换文件格式**:
   - SyN 形变场: `.nii.gz` (1-2 MB)
   - 仿射矩阵: `.mat` (132 字节)
   - **注意**: 必须保持正确的后缀

3. **可视化背景**:
   - 使用 `standard_template.nii.gz` 作为背景
   - 传入 `lobes_mask_path` 以启用 5 肺叶着色

### 8.3 常用命令
```bash
# 快速测试（3 例）
python run_phase3_pipeline.py --quick-test

# 仅可视化
python run_phase3_pipeline.py --viz-only

# 限制处理数量
python run_phase3_pipeline.py --limit 5

# 跳过配准
python run_phase3_pipeline.py --skip-registration
```

---

## 9. 联系与资源

- **配置文件**: `config.yaml`（所有参数在此修改）
- **日志目录**: `logs/`（包含详细的执行日志）
- **工程文档**: `Engineering_Edition.md`（原始需求文档）

---

*文档生成时间: 2025-12-31*
*当前开发阶段: Phase 3 完成，Phase 3B 待开发*

