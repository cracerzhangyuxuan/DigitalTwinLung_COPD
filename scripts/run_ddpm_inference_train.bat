@echo off
echo ============================================================
echo  DDPM Training Set Inference (copd_002, copd_003)
echo  DDIM 10 步加速 + 进度日志
echo  copd_001 已完成，跳过
echo ============================================================

echo.
echo [1/2] ddpm copd_002...
python run_phase3_pipeline.py --inference --model-type ddpm --device cuda:1 --start-patient-id copd_002 --limit 1
echo [1/2] Done.

echo.
echo [2/2] ddpm copd_003...
python run_phase3_pipeline.py --inference --model-type ddpm --device cuda:1 --start-patient-id copd_003 --limit 1
echo [2/2] Done.

echo.
echo ============================================================
echo  DDPM Training Set Inference Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. python run_phase3_pipeline.py --evaluate --model-type ddpm --limit 3
echo   2. python run_phase3_pipeline.py --visualize --model-type ddpm --limit 3
echo   3. python scripts/evaluate_l1_l4_metrics.py
echo   4. python scripts/generate_report_charts.py
echo.
pause

