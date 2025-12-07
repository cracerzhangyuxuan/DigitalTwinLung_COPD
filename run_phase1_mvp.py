#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1 MVP 重新运行脚本（v4 - 精确肺分割 + 气道排除）

更新内容:
- 使用 precise_lung_segment.py 进行精确肺分割（纯净度 99.5%）
- 启用气道排除功能，避免气管/支气管被误标为病灶
- 更新版本号：mask v3, emphysema v4, render v5
"""

import sys
import os

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 设置模块别名以处理数字开头的目录名
import importlib.util

def import_module_from_path(module_name, file_path):
    """从文件路径动态导入模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 导入必要的工具模块
print("=" * 60)
print("Phase 1 MVP v4 - 精确肺分割 + 气道排除")
print("=" * 60)
print()
print("加载模块...")

# 先导入 utils
logger_module = import_module_from_path(
    "src.utils.logger",
    os.path.join(project_root, "src/utils/logger.py")
)
io_module = import_module_from_path(
    "src.utils.io",
    os.path.join(project_root, "src/utils/io.py")
)

# 导入预处理模块
extract_emphysema = import_module_from_path(
    "src.preprocessing.extract_emphysema",
    os.path.join(project_root, "src/01_preprocessing/extract_emphysema.py")
)

# 导入精确肺分割模块（新！）
precise_lung_segment = import_module_from_path(
    "src.preprocessing.precise_lung_segment",
    os.path.join(project_root, "src/01_preprocessing/precise_lung_segment.py")
)

# 导入可视化模块
static_render = import_module_from_path(
    "src.visualization.static_render",
    os.path.join(project_root, "src/05_visualization/static_render.py")
)

print("模块加载完成！")

# ============================================================================
# 步骤 0: 使用精确肺分割算法生成高质量肺 mask
# ============================================================================
print()
print("=" * 60)
print("步骤 0: 精确肺分割（排除骨骼、肌肉等背景干扰）")
print("=" * 60)

import numpy as np

ct_nifti_path = "data/01_cleaned/copd_nifti/copd_001.nii.gz"
lung_mask_path = "data/01_cleaned/copd_mask/copd_001_mask_v3.nii.gz"  # v3: 精确分割
ct_clean_path = "data/01_cleaned/copd_clean/copd_001_clean_v3.nii.gz"  # v3
lesion_output_path = "data/01_cleaned/copd_emphysema/copd_001_emphysema_v4.nii.gz"  # v4: 气道排除

# 检查原始 CT 文件
if not os.path.exists(ct_nifti_path):
    print(f"错误: 原始 CT 文件不存在: {ct_nifti_path}")
    sys.exit(1)

print(f"原始 CT: {ct_nifti_path}")
print(f"输出肺 mask: {lung_mask_path}")
print()

try:
    # 使用精确肺分割
    stats = precise_lung_segment.segment_lung_precise(
        input_path=ct_nifti_path,
        output_mask_path=lung_mask_path,
        output_clean_path=ct_clean_path,
        lower_threshold=-950,  # 排除纯空气
        upper_threshold=-200,  # 排除软组织
        remove_trachea_flag=True,
        fill_holes=True
    )

    print()
    print("✅ 精确肺分割成功！")
    print(f"   肺体积: {stats['lung_voxels']:,} 体素 ({stats['lung_ratio']*100:.1f}%)")
    print(f"   平均 HU: {stats['mean_hu']:.1f}")
    print(f"   输出 mask: {lung_mask_path}")
    print(f"   输出清洁 CT: {ct_clean_path}")

except Exception as e:
    print(f"❌ 肺分割失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 步骤 1: 提取病灶 mask（启用气道排除）
# ============================================================================
print()
print("=" * 60)
print("步骤 1: 提取病灶 mask（启用气道排除功能）")
print("=" * 60)

print(f"原始 CT: {ct_nifti_path}")
print(f"肺 mask: {lung_mask_path}")
print(f"输出病灶: {lesion_output_path}")
print()

try:
    laa_percentage, stats = extract_emphysema.extract_emphysema_mask(
        ct_path=ct_nifti_path,  # 使用原始 CT（不是清洁后的）
        lung_mask_path=lung_mask_path,
        output_path=lesion_output_path,
        threshold=-950,
        min_volume_mm3=100,
        apply_morphology=True,
        opening_radius=1,
        closing_radius=2,
        # 气道排除参数
        exclude_airway=True,
        airway_hu_threshold=-980,
        min_airway_size=1000,
        airway_dilation_radius=2
    )

    print()
    print("✅ 病灶提取成功（已排除气道）！")
    print(f"   LAA-950 百分比: {laa_percentage:.3f}%")
    print(f"   病灶体积: {stats['emphysema_volume_ml']:.2f} mL")
    print(f"   病灶体素数: {stats['num_voxels']:,}")
    print(f"   气道排除: {'是' if stats.get('exclude_airway', False) else '否'}")

    # 检查 LAA 是否在合理范围
    if laa_percentage > 30:
        print(f"   ⚠️ 警告: LAA {laa_percentage:.1f}% 仍然偏高")
    else:
        print(f"   ✓ LAA 在合理范围内")

except Exception as e:
    print(f"❌ 病灶提取失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 步骤 2: 生成 3D 渲染图
# ============================================================================
print()
print("=" * 60)
print("步骤 2: 生成 3D 渲染图（精确分割 + 气道排除）")
print("=" * 60)

render_output_path = "copd_001_render_v5.png"  # v5: 精确分割 + 气道排除

print(f"清洁 CT: {ct_clean_path}")
print(f"病灶 mask: {lesion_output_path}")
print(f"肺 mask: {lung_mask_path}")
print(f"输出路径: {render_output_path}")
print()

try:
    result = static_render.render_static(
        ct_path=ct_clean_path,
        lesion_mask_path=lesion_output_path,
        lung_mask_path=lung_mask_path,
        output_path=render_output_path,
        lung_opacity=0.25,
        lesion_opacity=0.9,
        lung_color=(0.8, 0.8, 0.8),
        lesion_color=(0.9, 0.1, 0.1),
        show=False
    )

    if os.path.exists(render_output_path):
        file_size = os.path.getsize(render_output_path)
        print()
        print("✅ 3D 渲染成功！")
        print(f"   输出文件: {render_output_path}")
        print(f"   文件大小: {file_size / 1024:.1f} KB")
    else:
        print("❌ 渲染文件未生成")

except Exception as e:
    print(f"❌ 渲染失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 完成
# ============================================================================
print()
print("=" * 60)
print("Phase 1 MVP v4 完成！")
print("=" * 60)
print()
print("生成的文件：")
print(f"  1. 肺 mask (v3): {lung_mask_path} - 精确分割，纯净度 99.5%")
print(f"  2. 清洁 CT (v3): {ct_clean_path}")
print(f"  3. 病灶 mask (v4): {lesion_output_path} - 启用气道排除")
print(f"  4. 渲染图 (v5): {render_output_path}")
print()
print("版本说明：")
print("  - v3 mask: 使用精确肺分割，排除骨骼/肌肉/心脏")
print("  - v4 emphysema: 启用气道排除，避免气管被误标为病灶")
print("  - v5 render: 最终渲染结果")
print()
print("请查看渲染图验证效果。")

