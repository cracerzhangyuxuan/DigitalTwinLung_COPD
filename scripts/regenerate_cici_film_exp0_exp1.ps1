<#
.SYNOPSIS
    CICI-FiLM Exp-0 / Exp-1 重新推理 + 诊断图表重新生成
    修复 HU 校准尖峰后的完整重跑脚本 (PowerShell 版)

.USAGE
    # 默认使用 cpu 推理全部 6 例验证集
    .\scripts\regenerate_cici_film_exp0_exp1.ps1

    # 使用 GPU 推理（速度更快）
    .\scripts\regenerate_cici_film_exp0_exp1.ps1 -Device cuda

    # 只跑某几例
    .\scripts\regenerate_cici_film_exp0_exp1.ps1 -Patients copd_024,copd_025
#>
param(
    [string]$Device = "cpu",
    [string]$Patients = "copd_024,copd_025,copd_026,copd_027,copd_028,copd_029"
)

$ErrorActionPreference = "Continue"
$ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $ROOT) { $ROOT = (Get-Location).Path }
# 如果 $ROOT 不包含 DigitalTwinLung_COPD，fallback
if (-not (Test-Path "$ROOT\scripts\inference_cici_film.py")) {
    $ROOT = "D:\DigitalTwinLung_COPD"
}

$PYTHON = "python"
$patientList = $Patients -split ","

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  CICI-FiLM 重新推理 (修复 HU 校准尖峰)" -ForegroundColor Cyan
Write-Host "  设备: $Device | 患者: $($patientList.Count) 例" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# ---- Step 1: Exp-0 ----
Write-Host "`n[Step 1/3] Exp-0 (固定 HU 校准) 推理..." -ForegroundColor Yellow
foreach ($pid in $patientList) {
    $pid = $pid.Trim()
    Write-Host "  [$pid] Exp-0 ..." -NoNewline
    & $PYTHON "$ROOT\scripts\inference_cici_film.py" `
        --mode exp0 `
        --backbone-checkpoint "$ROOT\checkpoints\patchgan\best.pth" `
        --template "$ROOT\data\02_atlas\standard_template.nii.gz" `
        --mask "$ROOT\data\03_mapped\$pid\${pid}_warped_lesion.nii.gz" `
        --patient-features "$ROOT\data\patient_features.json" `
        --patient-id $pid `
        --output "$ROOT\results\cici_film\exp0\$pid.nii.gz" `
        --device $Device
    if ($LASTEXITCODE -eq 0) { Write-Host " OK" -ForegroundColor Green }
    else { Write-Host " FAIL" -ForegroundColor Red }
}

# ---- Step 2: Exp-1 ----
Write-Host "`n[Step 2/3] Exp-1 (自适应 HU 校准) 推理..." -ForegroundColor Yellow
foreach ($pid in $patientList) {
    $pid = $pid.Trim()
    Write-Host "  [$pid] Exp-1 ..." -NoNewline
    & $PYTHON "$ROOT\scripts\inference_cici_film.py" `
        --mode exp1 `
        --backbone-checkpoint "$ROOT\checkpoints\patchgan\best.pth" `
        --template "$ROOT\data\02_atlas\standard_template.nii.gz" `
        --mask "$ROOT\data\03_mapped\$pid\${pid}_warped_lesion.nii.gz" `
        --patient-features "$ROOT\data\patient_features.json" `
        --patient-id $pid `
        --output "$ROOT\results\cici_film\exp1\$pid.nii.gz" `
        --device $Device
    if ($LASTEXITCODE -eq 0) { Write-Host " OK" -ForegroundColor Green }
    else { Write-Host " FAIL" -ForegroundColor Red }
}

# ---- Step 3: 诊断图表 ----
Write-Host "`n[Step 3/3] 重新生成诊断直方图..." -ForegroundColor Yellow
& $PYTHON "$ROOT\scripts\generate_cici_film_diagnostics.py" `
    --exp2-dir "results/cici_film/exp2_v3"

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "  完成! 检查 results/cici_film/diagnostics/" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
