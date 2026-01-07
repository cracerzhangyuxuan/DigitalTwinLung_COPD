# 数字肺底座 (Digital Lung Base) 设计文档

**版本**: 1.0  
**创建日期**: 2026-01-07  
**适用阶段**: Phase 3A (COPD 病灶配准与可视化)

---

## 1. 概述

数字肺底座是将多个分散的模板文件合并为统一融合标签文件的预处理方案，旨在简化 Phase 3A 流程的输入文件管理。

### 1.1 设计目标

- **简化文件管理**：将肺叶标签和气管树 mask 合并为单一文件
- **保持数据完整性**：通过标签值区分不同解剖结构
- **向后兼容**：支持自动回退到分散文件模式

### 1.2 核心文件

| 文件名 | 用途 | 必需性 |
|--------|------|--------|
| `digital_lung_labels.nii.gz` | 融合标签（肺叶 + 气管树） | 推荐 |
| `standard_template.nii.gz` | CT 模板（配准 Fixed Image） | **必需** |
| `standard_mask.nii.gz` | 肺部二值 mask（配准约束） | **必需** |
| `digital_lung_base.json` | 元数据（标签定义） | 推荐 |

---

## 2. 融合标签文件结构

### 2.1 标签值定义

`digital_lung_labels.nii.gz` 使用以下标签值编码不同的解剖结构：

| 标签值 | 解剖结构 | 英文名称 | 颜色（可视化） |
|--------|---------|---------|---------------|
| 1 | 左上叶 | Left Upper Lobe | 浅蓝 (0.4, 0.6, 0.9) |
| 2 | 左下叶 | Left Lower Lobe | 深蓝 (0.2, 0.4, 0.8) |
| 3 | 右上叶 | Right Upper Lobe | 浅橙 (0.9, 0.6, 0.4) |
| 4 | 右中叶 | Right Middle Lobe | 黄色 (0.9, 0.8, 0.4) |
| 5 | 右下叶 | Right Lower Lobe | 红色 (0.8, 0.4, 0.4) |
| 6 | 气管树 | Trachea/Airway | 橙色 (0.8, 0.4, 0.2) |

### 2.2 空间信息

- **形状**: 512 × 512 × 364 体素
- **体素间距**: 0.754 × 0.754 × 1.0 mm
- **坐标系**: 与 `standard_template.nii.gz` 完全一致

---

## 3. 构建与使用

### 3.1 自动构建

数字肺底座会在首次运行 Phase 3A 流程时自动构建：

```bash
python run_phase3_pipeline.py --viz-only --limit 1
```

如果检测到 `digital_lung_labels.nii.gz` 不存在，流程会自动调用构建函数。

### 3.2 手动构建

如果需要强制重新生成数字肺底座：

```bash
python src/02_atlas_build/build_digital_lung_base.py
```

**输出**：
```
开始构建数字肺底座...
加载源文件...
  template: (512, 512, 364)
  mask: (512, 512, 364)
  lobes: (512, 512, 364)
  trachea: (512, 512, 364)
  空间一致性验证通过 (4 个文件)
融合标签...
  left_upper_lobe (值=1): 2,149,359 体素
  left_lower_lobe (值=2): 683,856 体素
  right_upper_lobe (值=3): 2,881,009 体素
  right_middle_lobe (值=4): 668,823 体素
  right_lower_lobe (值=5): 2,962,673 体素
  trachea (值=6): 57,140 体素
保存融合标签: data/02_atlas/digital_lung_labels.nii.gz
保存元数据: data/02_atlas/digital_lung_base.json
数字肺底座构建完成!
```

### 3.3 源文件要求

构建数字肺底座需要以下源文件（位于 `data/02_atlas/`）：

- `standard_template.nii.gz` - CT 模板
- `standard_mask.nii.gz` - 肺部二值 mask
- `standard_lung_lobes_labeled.nii.gz` - 5 肺叶标签（值 1-5）
- `standard_trachea_mask.nii.gz` - 气管树二值 mask

---

## 4. 元数据文件

`digital_lung_base.json` 记录融合标签的详细信息：

```json
{
  "version": "1.0",
  "created": "2026-01-07T09:20:03.740427",
  "description": "数字肺底座 - 融合标签文件",
  "files": {
    "labels": "digital_lung_labels.nii.gz",
    "template": "standard_template.nii.gz"
  },
  "label_values": {
    "left_upper_lobe": 1,
    "left_lower_lobe": 2,
    "right_upper_lobe": 3,
    "right_middle_lobe": 4,
    "right_lower_lobe": 5,
    "trachea": 6
  },
  "label_stats": {
    "left_upper_lobe": 2149359,
    "left_lower_lobe": 683856,
    "right_upper_lobe": 2881009,
    "right_middle_lobe": 668823,
    "right_lower_lobe": 2962673,
    "trachea": 57140
  },
  "spatial_info": {
    "shape": [512, 512, 364],
    "spacing": [0.75390625, 0.75390625, 1.0]
  },
  "source_files": {
    "template": "standard_template.nii.gz",
    "mask": "standard_mask.nii.gz",
    "lobes": "standard_lung_lobes_labeled.nii.gz",
    "trachea": "standard_trachea_mask.nii.gz"
  }
}
```

---

## 5. 向后兼容性

### 5.1 自动回退机制

如果 `digital_lung_labels.nii.gz` 不存在且无法自动构建，流程会自动回退到分散文件模式：

```python
# 优先使用数字肺底座
if digital_lung_labels_path.exists():
    use_digital_base = True
    lobes_to_render = str(digital_lung_labels_path)
    trachea_to_render = str(digital_lung_labels_path)
else:
    # 回退到分散文件
    use_digital_base = False
    lobes_to_render = str(template_lobes_path) if template_lobes_path.exists() else None
    trachea_to_render = str(template_trachea_path) if template_trachea_path.exists() else None
```

### 5.2 分散文件模式

在分散文件模式下，需要以下文件：
- `standard_lung_lobes_labeled.nii.gz` - 肺叶标签
- `standard_trachea_mask.nii.gz` - 气管树 mask

---

## 6. 技术实现

### 6.1 核心函数

**构建函数**: `src/02_atlas_build/build_digital_lung_base.py`

```python
def build_digital_lung_base(
    atlas_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    force_rebuild: bool = False
) -> Tuple[bool, Dict]:
    """
    构建数字肺底座文件
    
    将肺叶标签和气管树 mask 融合为单一标签文件。
    """
```

**加载函数**: `src/02_atlas_build/build_digital_lung_base.py`

```python
def load_digital_lung_base(
    atlas_dir: Union[str, Path]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    加载数字肺底座，提取肺叶和气管树数据
    
    Returns:
        (lobes_data, trachea_data, meta)
    """
```

### 6.2 可视化集成

`src/05_visualization/static_render.py` 的 `render_multiview()` 函数支持从融合标签中提取数据：

```python
if use_digital_base:
    # 从融合标签中提取肺叶（值 1-5）和气管树（值 6）
    lobes_mask = lobes_data.copy()
    lobes_mask[lobes_mask == 6] = 0  # 移除气管树标签
    trachea_from_digital_base = (lobes_data == 6).astype(np.uint8)
```

---

## 7. 常见问题

### Q1: 为什么 `standard_mask.nii.gz` 仍然需要？

**A**: `standard_mask.nii.gz` 用于两个关键功能：
1. **配准约束**：确保变形后的病灶严格在模板肺内（`warp_mask()` 函数）
2. **可视化边界**：定义肺部渲染的边界（`render_multiview()` 函数）

虽然可以从融合标签中提取肺部 mask（合并标签 1-5），但保留独立的二值 mask 文件可以提高处理效率。

### Q2: 如何验证数字肺底座是否正确？

**A**: 运行可视化测试：

```bash
python run_phase3_pipeline.py --viz-only --limit 1
```

检查日志输出：
```
数字肺底座: digital_lung_labels.nii.gz (融合标签)
肺叶标签: 从数字肺底座提取 (5 肺叶着色)
气管树: 从数字肺底座提取 (橙色渲染)
气管树（从融合标签）: 57,140 体素
使用肺叶标签: 5 个肺叶
气管树网格（数字肺底座）: 33,678 点
```

### Q3: 数字肺底座与分散文件的性能差异？

**A**: 
- **文件 I/O**: 数字肺底座减少了 1 次文件读取（肺叶和气管树合并）
- **内存占用**: 相同（融合标签与分散文件总大小相当）
- **处理速度**: 略快（减少了文件加载开销）

---

## 8. 更新日志

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| 1.0 | 2026-01-07 | 初始版本，完成数字肺底座设计与实现 |

---

## 9. 参考资料

- Phase 3A 流水线: `run_phase3_pipeline.py`
- 构建模块: `src/02_atlas_build/build_digital_lung_base.py`
- 可视化模块: `src/05_visualization/static_render.py`
- Phase 3 实施指南: `docs/Phase3_Guide.md`

