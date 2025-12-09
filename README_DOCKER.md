# Docker 部署指南 - DigitalTwinLung COPD

本文档提供使用 Docker 容器在 GPU 服务器上运行 Phase 2 流水线的完整指南。

## 📋 目录

1. [环境要求](#环境要求)
2. [快速开始](#快速开始)
3. [构建镜像](#构建镜像)
4. [运行容器](#运行容器)
5. [数据管理](#数据管理)
6. [常用命令](#常用命令)
7. [故障排除](#故障排除)

---

## 🖥️ 环境要求

### 服务器硬件
| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA GPU, 8GB VRAM | RTX 3090/4090, 24GB VRAM |
| CPU | 8 核 | 16+ 核 |
| 内存 | 32 GB | 64+ GB |
| 存储 | 100 GB SSD | 500+ GB NVMe SSD |

### 软件要求
- **Docker**: 20.10+
- **NVIDIA Container Toolkit**: 用于 GPU 支持
- **NVIDIA Driver**: 515+ (支持 CUDA 11.8)

### 安装 NVIDIA Container Toolkit
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 验证安装
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 🚀 快速开始

```bash
# 1. 克隆项目（或上传到服务器）
git clone <your-repo-url> DigitalTwinLung_COPD
cd DigitalTwinLung_COPD

# 2. 上传数据到 data/00_raw/normal/
# 使用 rsync 或 scp（见下方"数据管理"章节）

# 3. 构建 Docker 镜像
docker build -t digitaltwin-lung:phase2 .

# 4. 运行完整流水线
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    digitaltwin-lung:phase2

# 5. 查看结果
ls -la data/02_atlas/
```

---

## 🔨 构建镜像

### 基本构建
```bash
docker build -t digitaltwin-lung:phase2 .
```

### 带缓存构建（加速重复构建）
```bash
docker build --build-arg BUILDKIT_INLINE_CACHE=1 -t digitaltwin-lung:phase2 .
```

### 指定 CUDA 版本
```bash
# 修改 Dockerfile 第一行的基础镜像
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# 改为您需要的版本，如:
# FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
```

### 查看镜像大小
```bash
docker images digitaltwin-lung:phase2
# 预计大小: 15-20 GB（包含 CUDA、PyTorch、TotalSegmentator 模型）
```

---

## 🏃 运行容器

### 完整流水线（推荐）
```bash
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/checkpoints:/app/checkpoints \
    --name phase2-pipeline \
    digitaltwin-lung:phase2
```

### 快速测试（仅处理 3 例）
```bash
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    digitaltwin-lung:phase2 --quick-test
```

### 仅运行分割（跳过 Atlas 构建）
```bash
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    digitaltwin-lung:phase2 --skip-atlas
```

### 跳过分割（使用已有结果）
```bash
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    digitaltwin-lung:phase2 --skip-segmentation
```

### 后台运行
```bash
docker run -d --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    --name phase2-pipeline \
    digitaltwin-lung:phase2

# 查看日志
docker logs -f phase2-pipeline
```

### 指定 GPU
```bash
# 使用第一个 GPU
docker run --gpus '"device=0"' ...

# 使用多个 GPU
docker run --gpus '"device=0,1"' ...

# 使用所有 GPU
docker run --gpus all ...
```

### 交互模式（调试）
```bash
docker run -it --gpus all \
    -v $(pwd)/data:/app/data \
    --entrypoint /bin/bash \
    digitaltwin-lung:phase2

# 在容器内运行
python run_phase2_pipeline.py --check-only
python run_phase2_pipeline.py --quick-test
```

---

## 📁 数据管理

### 目录结构
```
data/
├── 00_raw/                    # 原始数据（需上传）
│   ├── normal/                # 正常肺 NIfTI 文件
│   │   ├── normal_001.nii.gz
│   │   ├── normal_002.nii.gz
│   │   └── ...
│   └── copd/                  # COPD 数据（Phase 3 使用）
├── 01_cleaned/                # 分割输出（自动生成）
│   ├── normal_mask/
│   └── normal_clean/
├── 02_atlas/                  # Atlas 输出（自动生成）
│   ├── standard_template.nii.gz
│   └── template_mask.nii.gz
├── 03_mapped/                 # 配准输出（Phase 3）
└── 04_final_viz/              # 可视化输出
```

### 上传数据到服务器

#### 使用 rsync（推荐，支持断点续传）
```bash
# 从本地上传到服务器
rsync -avzP --progress \
    ./data/00_raw/normal/ \
    user@server:/path/to/DigitalTwinLung_COPD/data/00_raw/normal/

# 仅上传新文件
rsync -avzP --ignore-existing \
    ./data/00_raw/ \
    user@server:/path/to/DigitalTwinLung_COPD/data/00_raw/
```

#### 使用 scp
```bash
scp -r ./data/00_raw/normal/*.nii.gz \
    user@server:/path/to/DigitalTwinLung_COPD/data/00_raw/normal/
```

### 下载结果
```bash
# 下载 Atlas 结果
rsync -avzP \
    user@server:/path/to/DigitalTwinLung_COPD/data/02_atlas/ \
    ./data/02_atlas/

# 下载日志
rsync -avzP \
    user@server:/path/to/DigitalTwinLung_COPD/logs/ \
    ./logs/
```

---

## 📝 常用命令

### 容器管理
```bash
# 查看运行中的容器
docker ps

# 查看所有容器
docker ps -a

# 停止容器
docker stop phase2-pipeline

# 删除容器
docker rm phase2-pipeline

# 查看容器日志
docker logs phase2-pipeline
docker logs -f phase2-pipeline  # 实时跟踪
```

### 镜像管理
```bash
# 查看镜像
docker images

# 删除镜像
docker rmi digitaltwin-lung:phase2

# 清理未使用的镜像
docker image prune
```

### 资源监控
```bash
# 查看 GPU 使用
nvidia-smi -l 1

# 查看容器资源使用
docker stats phase2-pipeline
```

---

## ❓ 故障排除

### 问题 1: GPU 不可用
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 如果失败，重新安装 nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 问题 2: 内存不足
```bash
# 增加交换空间
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 或使用 --quick-test 减少处理数量
docker run --gpus all ... digitaltwin-lung:phase2 --quick-test
```

### 问题 3: TotalSegmentator 模型下载失败
```bash
# 在容器内手动下载模型
docker run -it --gpus all \
    -v $(pwd)/data:/app/data \
    -v totalseg-models:/app/.totalsegmentator \
    --entrypoint /bin/bash \
    digitaltwin-lung:phase2

# 在容器内运行
TotalSegmentator --help
# 首次运行会自动下载模型
```

### 问题 4: 权限问题
```bash
# 修复数据目录权限
sudo chown -R $(id -u):$(id -g) data/
sudo chmod -R 755 data/
```

### 问题 5: 容器异常退出
```bash
# 查看退出日志
docker logs phase2-pipeline

# 查看详细信息
docker inspect phase2-pipeline
```

---

## 📊 预计运行时间

| 步骤 | GPU (RTX 3090) | CPU |
|------|---------------|-----|
| 分割 (37 例) | ~30 分钟 | ~5 小时 |
| Atlas 构建 (5 迭代) | ~4-6 小时 | ~8-12 小时 |
| **总计** | **~5-7 小时** | **~13-17 小时** |

快速测试模式 (3 例, 2 迭代): ~30 分钟

---

## 📧 联系方式

如有问题，请联系项目维护者或提交 Issue。

