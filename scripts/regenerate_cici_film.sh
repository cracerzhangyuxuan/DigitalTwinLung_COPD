#!/bin/bash
# ============================================================
#  CICI-FiLM Exp-0 / Exp-1 / Exp-2(v3) 重新推理 + 诊断图表重新生成
#  修复 HU 校准尖峰后的完整重跑脚本 (Linux/服务器 版)
#
#  用法:
#    bash scripts/regenerate_cici_film.sh              # 默认 cuda, 全部 6 例
#    bash scripts/regenerate_cici_film.sh --device cpu  # 用 CPU
#    bash scripts/regenerate_cici_film.sh --skip-exp2   # 跳过 Exp-2
# ============================================================
set -e

DEVICE="cuda"
SKIP_EXP2=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)  DEVICE="$2"; shift 2 ;;
        --skip-exp2) SKIP_EXP2=true; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# 项目根目录 = 脚本所在目录的上一级
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PATIENTS=(copd_024 copd_025 copd_026 copd_027 copd_028 copd_029)

echo ""
echo "============================================================"
echo "  CICI-FiLM 重新推理 (修复 HU 校准尖峰)"
echo "  项目根目录: $ROOT"
echo "  设备: $DEVICE | 患者: ${#PATIENTS[@]} 例"
echo "============================================================"

# ---- Step 1: Exp-0 (固定 HU 校准, 零训练) ----
echo ""
echo "[Step 1/4] Exp-0 (固定 HU 校准) 推理..."
for pid in "${PATIENTS[@]}"; do
    echo "  [$pid] Exp-0 ..."
    python scripts/inference_cici_film.py \
        --mode exp0 \
        --backbone-checkpoint checkpoints/patchgan/best.pth \
        --template data/02_atlas/standard_template.nii.gz \
        --mask "data/03_mapped/${pid}/${pid}_warped_lesion.nii.gz" \
        --patient-features data/patient_features.json \
        --patient-id "$pid" \
        --output "results/cici_film/exp0/${pid}.nii.gz" \
        --device "$DEVICE"
    echo "  [$pid] Exp-0 done."
done

# ---- Step 2: Exp-1 (自适应 HU 校准, 零训练) ----
echo ""
echo "[Step 2/4] Exp-1 (自适应 HU 校准) 推理..."
for pid in "${PATIENTS[@]}"; do
    echo "  [$pid] Exp-1 ..."
    python scripts/inference_cici_film.py \
        --mode exp1 \
        --backbone-checkpoint checkpoints/patchgan/best.pth \
        --template data/02_atlas/standard_template.nii.gz \
        --mask "data/03_mapped/${pid}/${pid}_warped_lesion.nii.gz" \
        --patient-features data/patient_features.json \
        --patient-id "$pid" \
        --output "results/cici_film/exp1/${pid}.nii.gz" \
        --device "$DEVICE"
    echo "  [$pid] Exp-1 done."
done

# ---- Step 3: Exp-2 v3 (CICI-FiLM SPADE, 需要训练好的 checkpoint) ----
if [ "$SKIP_EXP2" = false ]; then
    EXP2_CKPT="checkpoints/cici_film_v3/best.pth"
    if [ -f "$EXP2_CKPT" ]; then
        echo ""
        echo "[Step 3/4] Exp-2 v3 (CICI-FiLM SPADE) 推理..."
        for pid in "${PATIENTS[@]}"; do
            echo "  [$pid] Exp-2 v3 ..."
            python scripts/inference_cici_film.py \
                --mode exp2 \
                --film-version v3 \
                --film-checkpoint "$EXP2_CKPT" \
                --template data/02_atlas/standard_template.nii.gz \
                --mask "data/03_mapped/${pid}/${pid}_warped_lesion.nii.gz" \
                --patient-features data/patient_features.json \
                --patient-id "$pid" \
                --output "results/cici_film/exp2_v3/${pid}.nii.gz" \
                --device "$DEVICE"
            echo "  [$pid] Exp-2 v3 done."
        done
    else
        echo ""
        echo "[Step 3/4] 跳过 Exp-2: checkpoint 不存在 ($EXP2_CKPT)"
    fi
else
    echo ""
    echo "[Step 3/4] 跳过 Exp-2 (--skip-exp2)"
fi

# ---- Step 4: 诊断图表 ----
echo ""
echo "[Step 4/4] 重新生成诊断直方图..."

# 判断 exp2_v3 目录是否有结果
EXP2_ARG=""
if [ -d "results/cici_film/exp2_v3" ] && ls results/cici_film/exp2_v3/*.nii.gz &>/dev/null; then
    EXP2_ARG="--exp2-dir results/cici_film/exp2_v3"
fi

python scripts/generate_cici_film_diagnostics.py $EXP2_ARG

echo ""
echo "============================================================"
echo "  全部完成!"
echo "  输出:"
echo "    推理结果:  results/cici_film/exp0/  exp1/  exp2_v3/"
echo "    诊断图表:  results/cici_film/diagnostics/"
echo "============================================================"
