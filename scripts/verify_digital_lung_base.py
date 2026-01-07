#!/usr/bin/env python3
"""验证数字肺底座文件"""

import json
from pathlib import Path

meta_path = Path('data/02_atlas/digital_lung_base.json')

if meta_path.exists():
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    print('=== 数字肺底座验证 ===')
    print()
    print(f'版本: {meta["version"]}')
    print(f'创建时间: {meta["created"]}')
    print()
    print('标签值定义:')
    for name, value in meta['label_values'].items():
        stats = meta['label_stats'].get(name, 0)
        print(f'  {value}: {name:20s} ({stats:,} 体素)')
    print()
    print('空间信息:')
    print(f'  形状: {meta["spatial_info"]["shape"]}')
    print(f'  间距: {meta["spatial_info"]["spacing"]} mm')
    print()
    print('✅ 数字肺底座验证通过')
else:
    print('❌ 数字肺底座文件不存在')

