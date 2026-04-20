@echo off
echo ============================================================
echo  DDPM Remaining Inference (2 patients: copd_025, copd_026)
echo  copd_024 already complete.
echo ============================================================

echo.
echo [1/2] ddpm copd_025...
python run_phase3_pipeline.py --inference --model-type ddpm --device cuda:1 --start-patient-id copd_025 --limit 1
echo [1/2] Done.

echo.
echo [2/2] ddpm copd_026...
python run_phase3_pipeline.py --inference --model-type ddpm --device cuda:1 --start-patient-id copd_026 --limit 1
echo [2/2] Done.

echo.
echo ============================================================
echo  DDPM Inference Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. python run_phase3_pipeline.py --evaluate --model-type ddpm --start-patient-id copd_024 --limit 3
echo   2. python run_phase3_pipeline.py --visualize --model-type ddpm --start-patient-id copd_024 --limit 3
echo   3. python scripts/evaluate_validation_l1_l4.py
echo   4. python scripts/generate_validation_charts.py
echo.
pause

