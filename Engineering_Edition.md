项目数据策略与工程实施指南 (v6.3 Engineering Edition)
课题名称： 基于全代码自动化的COPD数字孪生肺构建与3D可视化研究 适用场景： 硕士毕业设计 / 科研项目开发 技术栈： Python, LungMask, TotalSegmentator, ANTsPy, PyTorch, PyVista 更新日期： 2025年12月25日

> **重要更新 (2025-12-25)**：分割方案
> - 肺叶分割：LungMask LTRCLobes_R231（边界清晰，支持病理肺）
> - 气管树分割：TotalSegmentator `--task lung_vessels`（完整支气管树 3-4 级分支）
>   - 默认 `--task total` 仅输出主气管 `trachea.nii.gz`
>   - `--task lung_vessels` 输出 `lung_trachea_bronchia.nii.gz`（完整支气管树）

---

**1. 工程文件结构设计 (Project Directory Structure)**
建议采用以下目录结构。这种结构将“源代码”、“数据”、“配置”和“实验记录”分离，是深度学习与医学影像处理的标准范式。
```
DigitalTwinLung_COPD/
│
├── README.md                     # 项目说明书 (项目背景、安装步骤、运行方法)
├── requirements.txt              # 依赖库列表 (pip install -r requirements.txt)
├── config.yaml                   # [重要] 全局配置文件 (路径、阈值、超参数、数据契约)
├── .gitignore                    # [重要] Git 忽略规则文件
├── run_pipeline.py               # 一键运行全流程的入口脚本
├── Engineering_Edition.md        # 项目数据策略与工程实施指南
├── v5_1_Final.md                 # 研究课题评估与实施方案
│
├── AeroPath/                     # [已弃用] 第三方气管树分割模型
│   # 注意: 已改用 TotalSegmentator --task lung_vessels
│   # 无需克隆此仓库
│
├── data/                         # 【数据层】 (在 .gitignore 中忽略)
│   ├── 00_raw/                   # 原始 NIfTI 数据 (Phase 2 后直接存储转换好的 NIfTI)
│   │   ├── normal/               # 正常肺 CT (normal_001.nii.gz, ...)
│   │   └── copd/                 # COPD 患者 CT (copd_001.nii.gz, ...)
│   │   # 注: prepare_phase2_data.py 已将 DICOM 转换为 NIfTI 存入此目录
│   │   # 不再需要 01_cleaned/*_nifti/ 中间转换目录
│   │
│   ├── 01_cleaned/               # 预处理输出 (分割 + 清理结果)
│   │   ├── normal_mask/          # 正常肺分割 Mask
│   │   │   ├── *_mask.nii.gz             # 肺部二值 mask
│   │   │   ├── *_trachea_mask.nii.gz     # [新增] 气管树 mask
│   │   │   └── *_lung_lobes_labeled.nii.gz  # [新增] 5肺叶标签 mask (值1-5)
│   │   ├── normal_clean/         # 背景清理后的纯净 CT (normal_001_clean.nii.gz)
│   │   ├── copd_mask/            # COPD 肺部 Mask (同上结构)
│   │   ├── copd_clean/           # COPD 背景清理后 (copd_001_clean.nii.gz)
│   │   └── copd_emphysema/       # LAA-950 提取的肺气肿病灶 (copd_001_emphysema.nii.gz)
│   │
│   ├── 02_atlas/                 # 标准数字孪生底座
│   │   ├── standard_template.nii.gz      # 最终生成的平均 CT (Phase 2 输出)
│   │   ├── standard_mask.nii.gz          # 模板肺部 Mask (质量评估用)
│   │   ├── standard_trachea_mask.nii.gz  # [新增] 模板气管树 Mask
│   │   └── temp_template*.nii.gz         # 临时模板文件 (Phase 1 使用)
│   │
│   ├── 03_mapped/                # 配准后的中间结果
│   │   └── copd_001/             # 按病人ID存放
│   │       ├── warped_ct.nii.gz      # 变形到标准空间的 CT
│   │       ├── warped_lesion.nii.gz  # 变形到标准空间的病灶
│   │       └── transform*.mat        # 变形场矩阵
│   │
│   └── 04_final_viz/             # 最终用于可视化的融合文件
│       ├── fused_copd_twin.nii.gz    # AI 融合后的数字孪生
│       └── renders/              # 渲染输出图片/视频
│
├── checkpoints/                  # 【模型层】 训练模型权重 (在 .gitignore 中忽略)
│   ├── inpainting_best.pth       # 最佳模型权重
│   ├── inpainting_latest.pth     # 最新模型权重
│   └── training_log.json         # 训练历史记录
│
├── logs/                         # 【日志层】 运行日志 (在 .gitignore 中忽略)
│   ├── preprocessing_2025xxxx.log
│   ├── training_2025xxxx.log
│   └── pipeline.log              # 主流程日志
│
├── src/                          # 【代码层】 核心逻辑代码
│   ├── __init__.py
│   │
│   ├── utils/                    # 工具包 (复用函数)
│   │   ├── __init__.py
│   │   ├── io.py                 # 读取/保存 NIfTI, DICOM
│   │   ├── math_ops.py           # 归一化、裁剪、矩阵运算
│   │   ├── visualization.py      # PyVista 通用绘图函数
│   │   ├── logger.py             # [新增] 统一日志配置
│   │   ├── metrics.py            # [新增] 评估指标计算
│   │   ├── data_quality.py       # [新增] 数据质量检查（输入数据验证）
│   │   └── validation.py         # [新增] 配准结果验证与诊断
│   │
│   ├── 01_preprocessing/         # 阶段一：清洗与特征提取
│   │   ├── __init__.py
│   │   ├── run_segmentation.py   # 分割模块（LungMask + TotalSegmentator）
│   │   ├── clean_background.py   # 去除骨骼背景
│   │   ├── simple_lung_segment.py # 简单肺分割（阈值法）
│   │   ├── precise_lung_segment.py # [新增] 精确肺分割（纯净度 99.5%）
│   │   └── extract_emphysema.py  # LAA-950 算法提取病灶 Mask（含气道排除）
│   │   # [2025-12-25 更新] 分割模型
│   │   # - segment_lung_lobes_lungmask(): LungMask 肺叶分割（推荐）
│   │   # - segment_airway_totalsegmentator(): TotalSegmentator lung_vessels 气管树分割（推荐）
│   │   # - run_lungmask_batch(): 批量分割入口（推荐）
│   │   # - extract_trachea_mask(): [已弃用] TotalSegmentator total 任务
│   │   # - create_labeled_lung_lobes(): [已弃用] TotalSegmentator 肺叶
│   │
│   ├── 02_atlas_build/           # 阶段二：底座构建
│   │   ├── __init__.py
│   │   └── build_template_ants.py# 调用 ants.build_template + 气管树模板生成
│   │
│   ├── 03_registration/          # 阶段三(上)：空间映射
│   │   ├── __init__.py
│   │   └── register_lesions.py   # 计算变形场并扭曲 Mask
│   │
│   ├── 04_texture_synthesis/     # 阶段三(下)：AI 纹理融合
│   │   ├── __init__.py
│   │   ├── dataset.py            # PyTorch 数据加载器 (提取 Patch)
│   │   ├── network.py            # 定义 GAN / Inpainting 模型结构
│   │   ├── losses.py             # [新增] 损失函数定义
│   │   ├── train.py              # 训练脚本
│   │   └── inference_fuse.py     # 推理脚本：生成融合后的 CT
│   │
│   └── 05_visualization/         # 阶段四：3D 可视化
│       ├── __init__.py
│       ├── static_render.py      # 生成高清截图（含多视角渲染功能）
│       └── dynamic_breath.py     # 生成呼吸动画 (循环正弦波逻辑)
│
├── tests/                        # 【测试层】 单元测试与集成测试
│   ├── __init__.py
│   ├── conftest.py               # pytest 配置与 fixtures
│   ├── test_io.py                # 测试数据读写功能
│   ├── test_preprocessing.py     # 测试预处理流程
│   ├── test_registration.py      # 测试配准功能
│   ├── test_network.py           # 测试网络前向传播
│   ├── test_metrics.py           # 测试评估指标计算
│   └── test_data_quality.py      # 测试数据质量检查
│
├── notebooks/                    # 【实验层】 Jupyter Notebooks (用于探索和调试)
│   ├── 1.0_check_data.ipynb      # 检查数据质量
│   ├── 2.0_test_ants.ipynb       # 测试小样本配准参数
│   ├── 3.0_train_debug.ipynb     # [新增] AI 训练调试
│   └── 4.0_viz_demo.ipynb        # 快速查看可视化效果
│
└── docs/                         # 【文档层】
    ├── method_v5.1.md            # 保存的技术方案文档
    └── research_log.md           # 实验日志
```

**补充说明：新增目录的作用**

| 新增目录/文件 | 作用 | 说明 |
| :--- | :--- | :--- |
| `.gitignore` | Git忽略规则 | 避免将大型数据文件误提交到版本库 |
| `checkpoints/` | 模型权重存储 | 保存训练过程中的最佳模型和最新模型 |
| `logs/` | 运行日志 | 记录长时间任务的运行状态，便于问题追溯 |
| `tests/` | 测试代码 | 保证代码质量，支持回归测试 |
| `src/utils/logger.py` | 日志配置 | 统一的日志格式和输出管理 |
| `src/utils/metrics.py` | 评估指标 | SSIM、Dice等质量评估函数 |
| `src/utils/data_quality.py` | 数据质量检查 | 入库前的数据验证脚本 |
| `src/utils/validation.py` | 配准结果验证 | 验证配准质量、mask 覆盖率、形状一致性 |
| `src/01_preprocessing/precise_lung_segment.py` | 精确肺分割 | 排除骨骼/肌肉/心脏，纯净度 99.5% |
| `run_mvp_pipeline.py` | MVP 流水线 | 6 步骤完整流程（替代 run_phase1_mvp.py） |

**数据文件版本对照表**

| 文件 | 版本 | 说明 |
| :--- | :--- | :--- |
| `copd_001_mask_v2.nii.gz` | v2 | 简单阈值分割，纯净度 44.6% |
| `copd_001_mask_v3.nii.gz` | v3 | 精确肺分割，纯净度 99.5%，排除骨骼/肌肉/心脏 |
| `copd_001_emphysema_v2.nii.gz` | v2 | LAA-950 基础版，含气道误标 |
| `copd_001_emphysema_v4.nii.gz` | v4 | LAA-950 + 气道排除，病灶更准确 |
| `copd_001_render_v3.png` | v3 | 基于 v2 mask 的渲染 |
| `copd_001_render_v5.png` | v5 | 基于 v3 mask + v4 emphysema 的最终渲染 |

**2. 数据集策略 (Data Strategy)**
2.1 数据规模 (Target Numbers)
* 正常对照组 (Normal): 15 - 20 例

  * 用途： 仅用于 data/00_raw/normal -> src/02_atlas_build，生成唯一的底座。

* COPD 患者组 (COPD): 30 - 50 例

  * 用途： 用于提取病理特征、训练 AI 模型、以及最终的 Demo 展示。

2.2 推荐数据源
* LIDC-IDRI: 筛选 Normal (无结节) 和 Emphysema (肺气肿) 病例。

* COPDGene: 如果能申请到，是最佳选择。

**3. 分阶段执行指南 (Phased Execution Guide)**
请按照以下顺序，依次编写和运行 src/ 下的脚本。

🚀 第一阶段：基础设施与最小闭环 (MVP) ✅ 已完成 (2025-12-09)
目标： 用极少数据跑通“清洗 -> 配准 -> 可视化”流程，验证环境。

1.环境配置：

* 安装 Python 3.9+, PyTorch, ANTsPy, PyVista。
* 安装分割模型：`pip install lungmask TotalSegmentator`
* 编写 requirements.txt。

2.数据清洗：

* 准备 3 例正常 + 1 例 COPD 放入 data/00_raw/。

* 运行 src/01_preprocessing/run_segmentation.py (提取 Mask)。

* 运行 src/01_preprocessing/clean_background.py (置换背景为 -1000)。

3.简单配准：

* 暂时跳过 Atlas 构建，直接选 1 例正常肺作为“临时底座”。

* 运行 src/01_preprocessing/extract_emphysema.py (提取 COPD 病灶 Mask)。

* 运行 src/03_registration/register_lesions.py (将病灶映射到临时底座)。

4.可视化验证：

* 运行 src/05_visualization/static_render.py。

* 检查点： 只要能看到一个灰色的肺里面有一团红色的病灶，第一阶段即成功。

**Phase 1 算法优化记录 (2025-12-04)**

| 优化项 | 优化前 | 优化后 | 效果 |
| :--- | :--- | :--- | :--- |
| 肺分割模块 | `simple_lung_segment.py` | `precise_lung_segment.py` | 纯净度 44.6% → 99.5% |
| HU 阈值范围 | -1000 ~ -300 | -950 ~ -200 | 排除纯空气和软组织 |
| 肺选择策略 | 最大连通域 | 最大 2 个连通域（左右肺） | 双肺完整 |
| 气道排除 | 无 | `extract_airway_mask()` | 移除气管/支气管误标 |
| LAA-950 参数 | 仅阈值 | 阈值 + 气道排除 + 连通域过滤 | 病灶更准确 |

**气道排除参数配置**

```python
# 在 extract_emphysema.py 中启用气道排除
laa_percentage, stats = extract_emphysema_mask(
    ct_path=ct_path,
    lung_mask_path=lung_mask_path,
    output_path=output_path,
    threshold=-950,
    # 气道排除参数
    exclude_airway=True,           # 启用气道排除
    airway_hu_threshold=-980,      # 气道 HU 阈值
    min_airway_size=1000,          # 最小气道体素数
    airway_dilation_radius=2       # 气道膨胀半径
)
```

**性能指标对比**

| 指标 | 优化前 | 优化后 | 改善 |
| :--- | :--- | :--- | :--- |
| 肺组织纯净度 | 44.6% | 99.5% | +54.9% |
| 软组织/骨骼占比 | 55.4% | 0.5% | -54.9% |
| LAA-950 百分比 | 0.38% | 0.24% | 更准确 |
| 病灶体素数 | 109,122 | 21,360 | -80% (排除气道) |

**Phase 2 新增功能 (2025-12-25 更新)**

| 功能 | 函数 | 说明 |
| :--- | :--- | :--- |
| 肺叶分割 | `segment_lung_lobes_lungmask()` | LungMask LTRCLobes_R231，边界清晰 |
| 气管树分割 | `segment_airway_totalsegmentator()` | TotalSegmentator lung_vessels，完整支气管树 3-4 级分支 |
| 批量分割 | `run_lungmask_batch()` | 推荐入口，自动调用上述两个函数 |
| 气管树模板生成 | `generate_template_trachea_mask()` | 配准生成标准气管树 mask |
| 气管树连续性验证 | `validate_trachea_continuity()` | 检查气管树的连通性和解剖合理性 |

> **已弃用函数**（保留用于兼容性）：
> - `extract_trachea_mask()`: TotalSegmentator --task total，仅主气管
> - `create_labeled_lung_lobes()`: TotalSegmentator 肺叶，边界碎片化
> - `run_totalsegmentator_batch()`: TotalSegmentator 批量分割

**肺叶标签对照表**

| 标签值 | 解剖结构 | LungMask 输出 |
| :---: | :--- | :--- |
| 1 | 左上叶 (Left Upper) | 直接输出 |
| 2 | 左下叶 (Left Lower) | 直接输出 |
| 3 | 右上叶 (Right Upper) | 直接输出 |
| 4 | 右中叶 (Right Middle) | 直接输出 |
| 5 | 右下叶 (Right Lower) | 直接输出 |

> **注意**：LungMask 的标签值与项目定义完全一致，无需转换！

**分割示例代码 (2025-12-24 更新)**

```python
from src.01_preprocessing.run_segmentation import (
    segment_lung_lobes_lungmask,
    segment_airway_raidionics,
    run_lungmask_batch
)

# 方式 1: 单例肺叶分割 (LungMask)
labeled_lobes, volume_stats, affine = segment_lung_lobes_lungmask(
    input_path="input.nii.gz",
    output_path="output/lung_lobes_labeled.nii.gz",
    use_fusion=True  # 使用 LTRCLobes_R231 融合模型
)
# volume_stats: {1: 1234.5, 2: 987.6, ...}  # 单位: mm³

# 方式 2: 单例气管树分割 (Raidionicsrads)
trachea_mask, affine = segment_airway_raidionics(
    input_path="input.nii.gz",
    output_path="output/trachea_mask.nii.gz"
)

# 方式 3: 批量分割 (推荐)
results = run_lungmask_batch(
    input_dir="data/00_raw/normal",
    mask_output_dir="data/01_cleaned/normal_mask",
    clean_output_dir="data/01_cleaned/normal_clean",
    extract_trachea=True,       # 启用气管树分割
    create_labeled_lobes=True,  # 启用肺叶标签
    use_fusion=True             # 使用融合模型
)
```

**配准性能优化记录 (2025-12-04)**

| 优化项 | 优化前 | 优化后 | 效果 |
| :--- | :--- | :--- | :--- |
| 配准时间 | 21 分钟 | 4.6 分钟 | -78% |
| SyN 迭代次数 | [100, 70, 50, 20] | [20, 10, 0] | 减少 87% |
| 多分辨率级别 | 4 级 | 3 级 | 避免全分辨率迭代 |
| 缩放因子 | [8, 4, 2, 1] | [4, 2, 1] | 减少计算量 |
| 病灶保留率 | 77.4% | 81.0% | +3.6% |

**配准参数配置**

```python
# 优化后的默认参数 (register_lesions.py)
type_of_transform = "SyNRA"
reg_iterations = (20, 10, 0)      # 3 级多分辨率，最高分辨率不迭代
shrink_factors = (4, 2, 1)        # 避免 shrink=8 的过度模糊
smoothing_sigmas = (2, 1, 0)      # 匹配缩放因子

# 原始参数（高精度但耗时）
# reg_iterations = (100, 70, 50, 20)
# shrink_factors = (8, 4, 2, 1)
# smoothing_sigmas = (3, 2, 1, 0)
```

**性能瓶颈分析**

| 阶段 | 原始耗时 | 优化后耗时 | 占比变化 |
| :--- | :--- | :--- | :--- |
| Stage 0 (Rigid) | 61.59 秒 | 61.74 秒 | 无变化 |
| Stage 1 (Affine) | 68.12 秒 | 54.59 秒 | -20% |
| Stage 2 (SyN) | 1000+ 秒 | 93.64 秒 | -91% |
| **总计** | **1281 秒** | **273.5 秒** | **-78%** |

**代码清理与重构记录 (2025-12-09)**

Phase 1 完成后进行了代码整理，清理临时脚本并整合有价值的功能到正式模块：

| 操作 | 文件 | 说明 |
| :--- | :--- | :--- |
| 删除 | `run_phase1_mvp.py` | 单文件测试脚本，功能已在 `run_mvp_pipeline.py` 中 |
| 删除 | `temp_wheels/` | ANTsPy 安装包临时目录 |
| 整合 | `clean_lung_mask()` → `static_render.py` | 肺 mask 连通分量清理函数 |
| 整合 | `render_multiview()` → `static_render.py` | 多视角渲染（X/Y/Z 三轴） |
| 整合 | `render_template_only()` → `static_render.py` | 模板肺部单独渲染 |
| 新建 | `src/utils/validation.py` | 配准结果验证模块 |
| 保留 | `generate_multiview_comparison.py` | 正式的多视角可视化脚本 |
| 保留 | `run_mvp_pipeline.py` | 正式的 MVP 流水线脚本 |

**新增模块功能说明：**

`src/utils/validation.py`:
- `validate_registration_result()` - 验证配准质量（保留率、Z 轴覆盖率）
- `check_mask_coverage()` - 检查 mask 覆盖统计信息
- `compare_ct_shapes()` - 比较配准前后 CT 形状

`src/05_visualization/static_render.py` 新增:
- `clean_lung_mask(mask, keep_largest_n)` - 保留最大 N 个连通分量
- `render_multiview(ct_path, lesion_mask_path, lung_mask_path, ...)` - X/Y/Z 多视角渲染
- `render_template_only(ct_path, lung_mask_path, ...)` - 模板肺部渲染

🏗️ 第二阶段：标准底座构建 (Atlas Construction)
目标： 生成高质量的数字孪生底座。

1.数据扩充： 放入 15-20 例正常数据。

2.构建模版：

* 运行 src/02_atlas_build/build_template_ants.py。

* 注意： 此脚本可能需运行一整夜。

3.结果固化： 将生成的 standard_template.nii.gz 放入 data/02_atlas/，这是项目的基石。

🧠 第三阶段：病理映射与 AI 融合 (The "Deluxe" Part)
目标： 批量处理 COPD 数据并训练 AI。

1.批量映射：

* 对 30-50 例 COPD 数据，批量运行 src/01_preprocessing 和 src/03_registration。

* 产出：所有病人的病灶 Mask 都被扭曲到了标准底座空间。

2.AI 训练 (可选/进阶)：

* 编写 src/04_texture_synthesis/dataset.py，从 COPD 原图中切出病灶 Patch，从正常图中切出健康 Patch。

* 运行 src/04_texture_synthesis/train.py 训练 Inpainting 模型。

3.最终融合：

* 运行 src/04_texture_synthesis/inference_fuse.py。

* 逻辑： 底座 + 映射过来的 Mask -> 挖空 -> AI 填补 -> 最终 CT。

🎬 第四阶段：全代码交互演示 (Final Demo)
目标： 产出论文图片和答辩视频。

1.动态模拟：

* 编写 src/05_visualization/dynamic_breath.py。

* 加入 sin(t) 函数和 COPD 的呼气延迟逻辑。

2.成果输出：

* 录制屏幕或保存 .mp4 文件。

* 截取不同视角的 3D 高清图用于论文。

---

**4. 阶段验收标准 (Acceptance Criteria)**

每个阶段必须达到以下量化标准后才能进入下一阶段：

| 阶段 | 验收项 | 量化标准 | 验证方法 |
| :--- | :--- | :--- | :--- |
| **Phase 1: MVP** | 流程跑通 | 能输出 3D 渲染图 | 运行 `static_render.py` |
| | 病灶可见 | 红色高亮区域可识别 | 目视检查截图 |
| | 无代码报错 | 全流程无 Exception | 检查日志文件 |
| **Phase 2: Atlas** | 模板生成 | `standard_template.nii.gz` 文件存在 | 检查文件大小 > 10MB |
| | 形态合理 | 与任一输入肺的 Dice ≥ 0.85 | 运行 `test_registration.py` |
| | 纹理清晰 | 血管/气管结构可辨识 | 在 3D Slicer 中目视确认 |
| **Phase 3: Fusion** | 配准精度 | 病灶位置偏差 ≤ 5mm | 计算质心距离 |
| | 融合质量 | SSIM ≥ 0.85, 边界无明显断裂 | 运行 `metrics.py` |
| | 模型收敛 | 验证集 Loss 稳定下降 | 检查 `training_log.json` |
| **Phase 4: Demo** | 渲染性能 | 动画帧率 ≥ 15 FPS | PyVista 内置计时 |
| | 论文素材 | ≥ 3 张高清插图 + 1 个演示视频 | 检查 `data/04_final_viz/renders/` |
| | 呼吸模拟 | 呼气/吸气周期可区分 | 目视确认动画效果 |

---

**5. .gitignore 规范 (Git Ignore Rules)**

项目根目录下必须包含 `.gitignore` 文件，内容如下：

```gitignore
# ========================
# 数据文件 (Data Files)
# ========================
data/
*.nii
*.nii.gz
*.dcm
*.dicom

# ========================
# 模型权重 (Model Checkpoints)
# ========================
checkpoints/
*.pth
*.pt
*.onnx
*.h5

# ========================
# 日志文件 (Log Files)
# ========================
logs/
*.log

# ========================
# Python 缓存 (Python Cache)
# ========================
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# ========================
# 虚拟环境 (Virtual Environment)
# ========================
.venv/
venv/
ENV/
env/

# ========================
# IDE 配置 (IDE Settings)
# ========================
.vscode/
.idea/
*.swp
*.swo
*~

# ========================
# Jupyter Notebook 检查点
# ========================
.ipynb_checkpoints/

# ========================
# 系统文件 (System Files)
# ========================
.DS_Store
Thumbs.db
```

---

**6. 日志管理策略 (Logging Strategy)**

6.1 日志配置模块

在 `src/utils/logger.py` 中实现统一的日志配置：

```python
# src/utils/logger.py

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    配置并返回一个 Logger 实例

    Args:
        name: Logger 名称（通常使用模块名）
        log_dir: 日志文件存储目录
        level: 日志级别
        console_output: 是否同时输出到控制台

    Returns:
        配置好的 Logger 实例
    """
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 创建 Logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 防止重复添加 Handler
    if logger.handlers:
        return logger

    # 日志格式
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件 Handler（按日期命名）
    date_str = datetime.now().strftime('%Y%m%d')
    file_handler = logging.FileHandler(
        log_path / f"{name}_{date_str}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台 Handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

# 使用示例
# from src.utils.logger import setup_logger
# logger = setup_logger("preprocessing")
# logger.info("开始处理数据...")
```

6.2 日志使用规范

| 日志级别 | 使用场景 | 示例 |
| :--- | :--- | :--- |
| `DEBUG` | 详细调试信息 | `logger.debug(f"当前 Patch 尺寸: {patch.shape}")` |
| `INFO` | 正常运行信息 | `logger.info("模型训练开始")` |
| `WARNING` | 潜在问题警告 | `logger.warning("数据层厚超过 3mm，可能影响配准精度")` |
| `ERROR` | 错误但程序可继续 | `logger.error(f"文件读取失败: {filepath}")` |
| `CRITICAL` | 严重错误，程序终止 | `logger.critical("GPU 显存不足，训练终止")` |

---

**7. config.yaml 数据契约定义 (Data Contract in config.yaml)**

全局配置文件 `config.yaml` 需包含以下内容：

```yaml
# config.yaml - 全局配置文件
# 更新日期: 2025-12-09

# ========================
# 路径配置 (Path Configuration)
# ========================
# Phase 2 优化后的数据流：
#   1. 00_raw/{normal,copd}/ - 存储 NIfTI 格式的 CT 数据（已由 prepare_phase2_data.py 转换）
#   2. 01_cleaned/{normal,copd}_{mask,clean}/, copd_emphysema/ - 分割和清理结果
#   3. 02_atlas/ - 模板输出
#   4. 03_mapped/copd_xxx/ - 配准结果
# 注意：已移除冗余的 *_nifti/ 中间转换目录
# ========================
paths:
  data_root: "data"
  raw_data: "data/00_raw"           # 原始 NIfTI 数据 (Phase 2 后直接存储转换好的 NIfTI)
  cleaned_data: "data/01_cleaned"   # 预处理输出 (分割 + 清理)
  atlas: "data/02_atlas"            # 模板输出
  mapped: "data/03_mapped"          # 配准结果
  final_viz: "data/04_final_viz"    # 可视化输出
  checkpoints: "checkpoints"
  logs: "logs"

# ========================
# 文件命名规范 (File Naming Convention)
# ========================
naming:
  # 原始数据命名: {type}_{patient_id}.nii.gz
  # 示例: normal_001.nii.gz, copd_023.nii.gz
  raw_pattern: "{type}_{patient_id:03d}.nii.gz"

  # 清洗后数据命名: {type}_clean_{patient_id}.nii.gz
  cleaned_pattern: "{type}_clean_{patient_id:03d}.nii.gz"

  # 病灶 Mask 命名: lesion_mask_{patient_id}.nii.gz
  lesion_mask_pattern: "lesion_mask_{patient_id:03d}.nii.gz"

  # 配准结果命名: warped_{patient_id}.nii.gz
  warped_pattern: "warped_{patient_id:03d}.nii.gz"

  # 融合结果命名: fused_{patient_id}.nii.gz
  fused_pattern: "fused_{patient_id:03d}.nii.gz"

# ========================
# 数据格式约定 (Data Format Contract)
# ========================
data_format:
  # NIfTI 文件规范
  nifti:
    dtype: "float32"              # 数据类型
    orientation: "RAS"            # 解剖方向标准

  # CT 图像规范
  ct:
    hu_range: [-1024, 3000]       # 有效 HU 值范围
    background_value: -1000       # 背景填充值 (空气)

  # Mask 规范
  mask:
    dtype: "uint8"                # 二值 Mask 数据类型
    foreground_value: 1           # 前景值
    background_value: 0           # 背景值

# ========================
# 预处理参数 (Preprocessing Parameters)
# ========================
preprocessing:
  # 分割参数
  segmentation:
    tool: "TotalSegmentator"
    target_organs: ["lung_upper_lobe_left", "lung_lower_lobe_left",
                   "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]

  # 病灶提取阈值
  emphysema:
    laa_threshold: -950           # LAA-950 阈值 (HU)
    min_volume_cc: 0.1            # 最小病灶体积 (立方厘米)

  # 数据质量检查阈值
  quality_check:
    min_slices: 100               # 最小层数
    max_slice_thickness: 2.5      # 最大层厚 (mm)
    min_lung_volume_cc: 2000      # 最小肺体积 (cc)

# ========================
# 配准参数 (Registration Parameters)
# ========================
registration:
  method: "SyNRA"                 # ANTsPy 配准方法 (刚性+仿射+SyN)
  metric: "MI"                    # 相似性度量 (Mutual Information)
  # 优化后的快速配准参数 (2025-12-04)
  iterations: [20, 10, 0]         # 迭代次数 (3 级多分辨率，最高分辨率不迭代)
  shrink_factors: [4, 2, 1]       # 缩放因子 (避免过度模糊)
  smoothing_sigmas: [2, 1, 0]     # 平滑参数
  # 原始高精度参数（耗时约 21 分钟）:
  # iterations: [100, 70, 50, 20]
  # shrink_factors: [8, 4, 2, 1]
  # smoothing_sigmas: [3, 2, 1, 0]

# ========================
# AI 训练参数 (Training Parameters)
# ========================
training:
  # 数据加载
  batch_size: 4
  num_workers: 4
  patch_size: [64, 64, 64]        # 3D Patch 尺寸

  # 优化器
  optimizer: "Adam"
  learning_rate: 0.0001
  betas: [0.9, 0.999]
  weight_decay: 0.0001

  # 训练策略
  epochs: 200
  warmup_epochs: 10
  scheduler: "CosineAnnealing"

  # 损失权重
  loss_weights:
    l1_loss: 1.0
    perceptual_loss: 0.1
    adversarial_loss: 0.01        # 如果使用 GAN

  # Checkpoint
  save_interval: 10               # 每 N 个 epoch 保存一次
  keep_last_n: 3                  # 保留最近 N 个 checkpoint

# ========================
# 评估指标阈值 (Evaluation Thresholds)
# ========================
evaluation:
  ssim_threshold: 0.85
  psnr_threshold: 25.0
  dice_threshold: 0.80
  hd95_threshold: 5.0             # mm

# ========================
# 可视化参数 (Visualization Parameters)
# ========================
visualization:
  # 体渲染
  lung_opacity: 0.3               # 肺实质透明度
  lesion_color: [1.0, 0.2, 0.2]   # 病灶颜色 (RGB)
  lesion_opacity: 0.8             # 病灶透明度

  # 动态呼吸
  breath_amplitude: 0.1           # 呼吸振幅 (相对于肺尺寸)
  breath_frequency: 0.2           # 呼吸频率 (Hz)
  copd_exhale_delay: 0.3          # COPD 呼气延迟 (秒)

  # 输出
  render_resolution: [1920, 1080]
  video_fps: 30
```

---

**8. 测试策略 (Testing Strategy)**

8.1 测试目录结构说明

| 测试文件 | 测试内容 | 关键测试用例 |
| :--- | :--- | :--- |
| `conftest.py` | pytest 配置 | 共享 fixtures（测试数据路径、临时目录等） |
| `test_io.py` | 数据读写 | NIfTI 读取/保存、DICOM 转换 |
| `test_preprocessing.py` | 预处理流程 | 分割结果验证、背景置换检查 |
| `test_registration.py` | 配准功能 | 刚性配准、非线性配准、Dice 计算 |
| `test_network.py` | 网络前向传播 | 输入输出尺寸、梯度检查 |
| `test_metrics.py` | 评估指标 | SSIM/Dice 计算正确性 |
| `test_data_quality.py` | 数据质量 | 质量检查脚本验证 |

8.2 运行测试命令

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_preprocessing.py -v

# 运行带覆盖率报告
pytest tests/ --cov=src --cov-report=html

# 只运行快速测试（跳过耗时测试）
pytest tests/ -v -m "not slow"
```

8.3 测试 Fixture 示例

```python
# tests/conftest.py

import pytest
import numpy as np
import tempfile
from pathlib import Path

@pytest.fixture
def sample_ct_array():
    """生成用于测试的 3D CT 数组"""
    np.random.seed(42)
    # 模拟 CT 图像: 128x128x64, HU 范围 [-1000, 0]
    ct = np.random.uniform(-1000, 0, size=(128, 128, 64)).astype(np.float32)
    return ct

@pytest.fixture
def sample_mask_array():
    """生成用于测试的二值 Mask"""
    mask = np.zeros((128, 128, 64), dtype=np.uint8)
    # 在中心区域创建一个球形 Mask
    center = (64, 64, 32)
    radius = 20
    for x in range(128):
        for y in range(128):
            for z in range(64):
                if (x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2 < radius**2:
                    mask[x, y, z] = 1
    return mask

@pytest.fixture
def temp_output_dir():
    """创建临时输出目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
```

