# Phase 2: 服务器 GPU 运行方案

**文档版本**: 1.0  
**更新日期**: 2025-12-09  
**适用阶段**: Phase 2 - Atlas Construction

---

## 一、概述

Phase 2 Atlas 构建是一个计算密集型任务，建议在具备 GPU 的服务器上运行。本文档提供完整的部署、运行和验证方案。

### 预计运行时间

| 数据量 | CPU 运行时间 | GPU 加速时间 |
|--------|-------------|-------------|
| 15-20 例 | 4-8 小时 | 2-4 小时 |
| 40 例 | 8-16 小时 | 4-8 小时 |
| 快速测试 (3 例) | 10-30 分钟 | 5-15 分钟 |

---

## 二、服务器环境要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 8 核 | 16+ 核 |
| 内存 | 32 GB | 64 GB |
| GPU | NVIDIA GTX 1080 | NVIDIA RTX 3090 / A100 |
| 显存 | 8 GB | 24+ GB |
| 存储 | 100 GB SSD | 500 GB NVMe SSD |

### 软件要求

```
- Ubuntu 20.04+ / CentOS 7+
- Python 3.9+
- CUDA 11.0+ (如需 GPU 加速)
- conda 或 pip
```

---

## 三、部署步骤

### 3.1 项目上传

```bash
# 方法 1: 使用 rsync (推荐)
rsync -avz --exclude='data/' --exclude='.venv/' --exclude='__pycache__/' \
    ./DigitalTwinLung_COPD/ user@server:/home/user/DigitalTwinLung_COPD/

# 方法 2: 使用 scp
scp -r DigitalTwinLung_COPD user@server:/home/user/

# 方法 3: 使用 git
ssh user@server
git clone https://github.com/your-repo/DigitalTwinLung_COPD.git
```

### 3.2 环境配置

```bash
# 连接服务器
ssh user@server

# 进入项目目录
cd /home/user/DigitalTwinLung_COPD

# 创建 conda 环境
conda create -n lung_twin python=3.9 -y
conda activate lung_twin

# 安装依赖
pip install -r requirements.txt

# 安装 ANTsPy (关键依赖)
# 方法 1: conda 安装 (推荐)
conda install -c aramislab antspyx -y

# 方法 2: pip 安装 (可能需要编译)
pip install antspyx
```

### 3.3 数据上传

```bash
# 仅上传准备好的数据
rsync -avz --progress \
    ./data/01_cleaned/normal_clean/ \
    user@server:/home/user/DigitalTwinLung_COPD/data/01_cleaned/normal_clean/

rsync -avz --progress \
    ./data/01_cleaned/normal_mask/ \
    user@server:/home/user/DigitalTwinLung_COPD/data/01_cleaned/normal_mask/
```

---

## 四、运行方案

### 4.1 环境检查

```bash
# 激活环境
conda activate lung_twin

# 检查环境
python run_phase2_atlas.py --check-only
```

预期输出：
```
======================================================================
Phase 2: Atlas Construction - 环境检查
======================================================================
✅ ANTsPy 版本: 0.x.x
✅ nibabel 版本: x.x.x
✅ numpy 版本: x.x.x
✅ scipy 版本: x.x.x
✅ 输入数据: N 个 NIfTI 文件
✅ 输出目录: .../data/02_atlas
✅ 配置文件: .../config.yaml

环境检查完成，未执行构建。
```

### 4.2 后台运行 (推荐)

```bash
# 使用 nohup 后台运行
nohup python run_phase2_atlas.py > logs/phase2_atlas.log 2>&1 &

# 记录进程 ID
echo $! > logs/phase2_atlas.pid

# 查看运行状态
tail -f logs/phase2_atlas.log
```

### 4.3 使用 screen 运行

```bash
# 创建新的 screen 会话
screen -S atlas_build

# 在 screen 中运行
conda activate lung_twin
python run_phase2_atlas.py

# 分离会话 (Ctrl+A, D)
# 重新连接
screen -r atlas_build
```

### 4.4 使用 tmux 运行

```bash
# 创建新的 tmux 会话
tmux new -s atlas_build

# 运行命令
conda activate lung_twin
python run_phase2_atlas.py

# 分离会话 (Ctrl+B, D)
# 重新连接
tmux attach -t atlas_build
```

### 4.5 常用运行参数

```bash
# 标准运行 (使用所有数据)
python run_phase2_atlas.py

# 限制使用 20 例数据
python run_phase2_atlas.py --num-images 20

# 快速测试模式 (3 例, 2 次迭代)
python run_phase2_atlas.py --quick-test

# 跳过质量评估
python run_phase2_atlas.py --skip-eval

# 完整后台运行示例
nohup python run_phase2_atlas.py --num-images 20 > logs/phase2.log 2>&1 &
```

---

## 五、监控与调试

### 5.1 实时日志监控

```bash
# 查看最新日志
tail -f logs/phase2_atlas.log

# 查看最后 100 行
tail -100 logs/phase2_atlas.log

# 搜索错误信息
grep -i "error\|exception\|failed" logs/phase2_atlas.log
```

### 5.2 资源监控

```bash
# GPU 使用情况
watch -n 1 nvidia-smi

# CPU 和内存
htop

# 磁盘空间
df -h
```

### 5.3 检查进程状态

```bash
# 查看进程
ps aux | grep python | grep phase2

# 使用保存的 PID
cat logs/phase2_atlas.pid
kill -0 $(cat logs/phase2_atlas.pid) && echo "运行中" || echo "已结束"
```

---

## 六、验证方案

### 6.1 输出文件检查

```bash
# 检查输出文件是否存在
ls -lh data/02_atlas/

# 预期文件:
# - standard_template.nii.gz (>10MB)
# - standard_mask.nii.gz
# - atlas_evaluation_report.json
```

### 6.2 文件大小验证

```bash
# 检查模板文件大小
size=$(stat -f%z data/02_atlas/standard_template.nii.gz 2>/dev/null || \
       stat -c%s data/02_atlas/standard_template.nii.gz)
if [ $size -gt 10485760 ]; then
    echo "✅ 模板文件大小正常: $(echo "scale=2; $size/1048576" | bc) MB"
else
    echo "❌ 模板文件太小"
fi
```

### 6.3 质量报告检查

```bash
# 查看评估报告
cat data/02_atlas/atlas_evaluation_report.json | python -m json.tool

# 检查 Dice 分数
python -c "
import json
with open('data/02_atlas/atlas_evaluation_report.json') as f:
    report = json.load(f)
print(f'平均 Dice: {report[\"mean_dice\"]:.4f}')
print(f'验收状态: {\"通过\" if report[\"passed\"] else \"未通过\"}')
"
```

### 6.4 可视化验证 (可选)

```bash
# 在本地下载结果后用 3D Slicer 检查
scp user@server:/path/to/data/02_atlas/standard_template.nii.gz ./

# 或使用 Python 快速预览
python -c "
import nibabel as nib
import numpy as np
img = nib.load('data/02_atlas/standard_template.nii.gz')
print(f'形状: {img.shape}')
print(f'体素大小: {img.header.get_zooms()}')
print(f'数据范围: [{img.get_fdata().min():.1f}, {img.get_fdata().max():.1f}]')
"
```

---

## 七、故障排除

### 7.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| ANTsPy 导入失败 | 安装不完整 | 使用 conda 重新安装 |
| 内存不足 | 数据量太大 | 减少 `--num-images` 或增加内存 |
| 进程被杀死 | OOM Killer | 检查 `dmesg`，增加内存 |
| 运行中断 | SSH 断开 | 使用 screen/tmux/nohup |

### 7.2 错误日志分析

```bash
# 查看最后的错误
tail -50 logs/phase2_atlas.log

# 检查系统日志
dmesg | tail -50

# 检查 OOM 情况
grep -i "out of memory\|oom" /var/log/syslog
```

---

## 八、下载结果

```bash
# 下载模板文件
scp user@server:/path/to/data/02_atlas/standard_template.nii.gz ./

# 下载所有结果
rsync -avz user@server:/path/to/data/02_atlas/ ./data/02_atlas/

# 下载日志
scp user@server:/path/to/logs/phase2_atlas.log ./logs/
```

---

## 九、清理

```bash
# 运行完成后清理临时文件
rm -f logs/phase2_atlas.pid

# 清理大型中间文件 (可选)
# rm -f data/02_atlas/temp_*.nii.gz
```

---

## 附录：快速参考命令

```bash
# 一键部署脚本
#!/bin/bash
conda activate lung_twin
python run_phase2_atlas.py --check-only && \
nohup python run_phase2_atlas.py > logs/phase2.log 2>&1 &
echo "Atlas 构建已启动，PID: $!"
echo $! > logs/phase2.pid
```

