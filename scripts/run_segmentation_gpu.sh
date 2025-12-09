#!/bin/bash
# =============================================================================
# GPU 加速肺部分割运行脚本 (Phase 2 预处理)
#
# 使用方法：
#   ./scripts/run_segmentation_gpu.sh                    # 默认运行
#   ./scripts/run_segmentation_gpu.sh --test             # 测试模式
#   ./scripts/run_segmentation_gpu.sh --threshold        # 使用阈值方法
#   ./scripts/run_segmentation_gpu.sh --background       # 后台运行
#
# 环境要求：
#   - Conda 环境: lung_atlas (或通过 CONDA_ENV 变量指定)
#   - Python 3.9+
#   - TotalSegmentator (GPU 模式)
#   - CUDA 11.x+ (GPU 模式)
#
# 作者: DigitalTwinLung_COPD Team
# 日期: 2025-12-09
# =============================================================================

set -e  # 遇到错误立即退出

# =============================================================================
# 配置参数
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="${CONDA_ENV:-lung_atlas}"
LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/segmentation_${TIMESTAMP}.log"

# GPU 设置
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# 性能设置
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

# Python 设置
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# =============================================================================
# 解析命令行参数
# =============================================================================
METHOD="auto"
DEVICE="auto"
DATA_TYPE="all"
TEST_RUN=""
BACKGROUND=false
FAST=false
FORCE=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_RUN="--test-run 2"
            shift
            ;;
        --threshold)
            METHOD="threshold"
            shift
            ;;
        --totalsegmentator)
            METHOD="totalsegmentator"
            shift
            ;;
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --gpu)
            DEVICE="cuda:0"
            shift
            ;;
        --normal)
            DATA_TYPE="normal"
            shift
            ;;
        --copd)
            DATA_TYPE="copd"
            shift
            ;;
        --background|-bg)
            BACKGROUND=true
            shift
            ;;
        --fast)
            FAST=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --help|-h)
            echo "GPU 加速肺部分割运行脚本"
            echo ""
            echo "使用方法:"
            echo "  $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --test          测试模式 (仅处理 2 个文件)"
            echo "  --threshold     使用阈值方法 (CPU)"
            echo "  --totalsegmentator  使用 TotalSegmentator (GPU)"
            echo "  --cpu           强制使用 CPU"
            echo "  --gpu           强制使用 GPU"
            echo "  --normal        仅处理正常肺数据"
            echo "  --copd          仅处理 COPD 数据"
            echo "  --background    后台运行"
            echo "  --fast          快速模式 (TotalSegmentator)"
            echo "  --force         强制重新处理已存在的文件"
            echo "  --check         仅检查环境"
            echo "  --help          显示此帮助"
            echo ""
            echo "环境变量:"
            echo "  CONDA_ENV       Conda 环境名称 (默认: lung_atlas)"
            echo "  CUDA_VISIBLE_DEVICES  GPU 设备 ID (默认: 0)"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# 函数定义
# =============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

check_environment() {
    log_info "检查运行环境..."
    
    # 检查 Conda
    if ! command -v conda &> /dev/null; then
        log_error "Conda 未安装或未在 PATH 中"
        return 1
    fi
    
    # 检查 Conda 环境
    if ! conda env list | grep -q "^${CONDA_ENV} "; then
        log_error "Conda 环境 '${CONDA_ENV}' 不存在"
        log_info "请运行: conda create -n ${CONDA_ENV} python=3.10"
        return 1
    fi
    
    # 检查项目目录
    if [[ ! -f "${PROJECT_ROOT}/config.yaml" ]]; then
        log_error "配置文件不存在: ${PROJECT_ROOT}/config.yaml"
        return 1
    fi
    
    # 检查输入数据
    local normal_count=$(find "${PROJECT_ROOT}/data/00_raw/normal" -name "*.nii.gz" 2>/dev/null | wc -l)
    local copd_count=$(find "${PROJECT_ROOT}/data/00_raw/copd" -name "*.nii.gz" 2>/dev/null | wc -l)
    
    if [[ $normal_count -eq 0 && $copd_count -eq 0 ]]; then
        log_error "未找到输入数据"
        return 1
    fi
    
    log_info "输入数据: ${normal_count} 正常肺, ${copd_count} COPD"
    
    # 检查 GPU
    if [[ "$DEVICE" != "cpu" ]]; then
        if command -v nvidia-smi &> /dev/null; then
            log_info "GPU 信息:"
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
        else
            log_info "nvidia-smi 不可用，将在 Python 中检查 GPU"
        fi
    fi
    
    return 0
}

activate_conda() {
    log_info "激活 Conda 环境: ${CONDA_ENV}"
    
    # 尝试多种激活方式
    if [[ -f "${CONDA_PREFIX}/etc/profile.d/conda.sh" ]]; then
        source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    else
        log_error "无法找到 conda.sh，请确保 Conda 正确安装"
        return 1
    fi
    
    conda activate "${CONDA_ENV}"
    log_info "Python 路径: $(which python)"
    log_info "Python 版本: $(python --version)"
}

run_segmentation() {
    log_info "开始肺部分割..."
    
    cd "${PROJECT_ROOT}"
    
    # 构建命令
    local cmd="python run_segmentation_gpu.py"
    cmd+=" --method ${METHOD}"
    cmd+=" --device ${DEVICE}"
    cmd+=" --type ${DATA_TYPE}"
    
    [[ -n "$TEST_RUN" ]] && cmd+=" ${TEST_RUN}"
    [[ "$FAST" == true ]] && cmd+=" --fast"
    [[ "$FORCE" == true ]] && cmd+=" --force"
    [[ "$CHECK_ONLY" == true ]] && cmd+=" --check-only"
    
    log_info "执行命令: ${cmd}"
    
    # 执行
    eval "${cmd}"
}

# =============================================================================
# 主流程
# =============================================================================

main() {
    log_info "=========================================="
    log_info "GPU 加速肺部分割"
    log_info "=========================================="
    log_info "项目目录: ${PROJECT_ROOT}"
    log_info "日志文件: ${LOG_FILE}"
    log_info "方法: ${METHOD}"
    log_info "设备: ${DEVICE}"
    log_info "数据类型: ${DATA_TYPE}"
    
    # 创建日志目录
    mkdir -p "${LOG_DIR}"
    
    # 环境检查
    if ! check_environment; then
        log_error "环境检查失败"
        exit 1
    fi
    
    # 激活 Conda
    if ! activate_conda; then
        log_error "激活 Conda 环境失败"
        exit 1
    fi
    
    # 运行分割
    run_segmentation
}

# =============================================================================
# 执行
# =============================================================================

if [[ "$BACKGROUND" == true ]]; then
    log_info "后台运行模式，日志输出到: ${LOG_FILE}"
    nohup bash -c "$(declare -f log_info log_error check_environment activate_conda run_segmentation main); main" > "${LOG_FILE}" 2>&1 &
    echo "进程已启动，PID: $!"
    echo "查看日志: tail -f ${LOG_FILE}"
else
    main 2>&1 | tee "${LOG_FILE}"
fi

