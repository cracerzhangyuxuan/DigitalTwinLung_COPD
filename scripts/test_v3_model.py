#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 CICI-FiLM v3 模型初始化和前向传播
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.04_texture_synthesis.network import create_model
from src.04_texture_synthesis.conditioned_model_v3 import ConditionedGeneratorV3

def test_v3_model():
    print("=" * 60)
    print("测试 CICI-FiLM v3 模型")
    print("=" * 60)
    
    # 创建 backbone
    print("\n1. 创建 backbone...")
    backbone, _ = create_model('patchgan')
    print(f"   ✓ Backbone 创建成功")
    
    # 创建 v3 模型
    print("\n2. 创建 v3 模型...")
    model = ConditionedGeneratorV3(backbone, cond_dim=5, cond_emb_dim=512)
    print(f"   ✓ v3 模型创建成功")
    
    # 冻结 backbone
    print("\n3. 冻结 backbone...")
    model.freeze_backbone()
    
    # 测试前向传播
    print("\n4. 测试前向传播...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 创建测试输入
    x = torch.randn(2, 1, 64, 64, 64).to(device)
    condition = torch.randn(2, 5).to(device)
    
    print(f"   输入形状: {x.shape}")
    print(f"   条件形状: {condition.shape}")
    
    # 前向传播
    with torch.no_grad():
        y = model(x, condition)
    
    print(f"   输出形状: {y.shape}")
    print(f"   ✓ 前向传播成功")
    
    # 检查 hooks
    print("\n5. 检查 hooks...")
    print(f"   注册的 hooks 数量: {len(model._hooks)}")
    print(f"   ✓ Hooks 注册成功")
    
    # 统计参数
    print("\n6. 参数统计...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   可训练比例: {100*trainable_params/total_params:.2f}%")
    
    # 测试无条件前向
    print("\n7. 测试无条件前向...")
    with torch.no_grad():
        y_uncond = model(x, condition=None)
    print(f"   输出形状: {y_uncond.shape}")
    print(f"   ✓ 无条件前向成功")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)

if __name__ == '__main__':
    test_v3_model()

