@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   6 Models x copd_027-029  Full Pipeline
echo   Inference - Evaluate - Visualize
echo ============================================================
echo.
echo Start: %date% %time%
echo.

REM ============================
REM  Phase 1: Inference
REM ============================
echo [Phase 1/3] Inference =============================
echo.

REM --- 5 standard models (batch limit 3) ---
for %%M in (unet partial_conv patchgan attgan mae_patchgan) do (
    echo [Inference] %%M  copd_027-029 ...
    python run_phase3_pipeline.py --inference --model-type %%M --device cuda:1 --start-patient-id copd_027 --limit 3
    if errorlevel 1 (
        echo [FAIL] %%M inference failed
    ) else (
        echo [OK] %%M inference done
    )
    echo.
)

REM --- DDPM (per-patient, avoid timeout) ---
for %%P in (copd_027 copd_028 copd_029) do (
    echo [Inference] ddpm  %%P ...
    python run_phase3_pipeline.py --inference --model-type ddpm --device cuda:1 --start-patient-id %%P --limit 1
    if errorlevel 1 (
        echo [FAIL] ddpm %%P inference failed
    ) else (
        echo [OK] ddpm %%P inference done
    )
    echo.
)

echo.
echo [Phase 1/3] Inference complete
echo ================================================
echo.

REM ============================
REM  Phase 2: Evaluate
REM ============================
echo [Phase 2/3] Evaluate ==============================
echo.

for %%M in (unet partial_conv patchgan attgan mae_patchgan ddpm) do (
    echo [Evaluate] %%M  copd_027-029 ...
    python run_phase3_pipeline.py --evaluate --model-type %%M --start-patient-id copd_027 --limit 3
    if errorlevel 1 (
        echo [FAIL] %%M evaluate failed
    ) else (
        echo [OK] %%M evaluate done
    )
    echo.
)

echo.
echo [Phase 2/3] Evaluate complete
echo ================================================
echo.

REM ============================
REM  Phase 3: Visualize
REM ============================
echo [Phase 3/3] Visualize =============================
echo.

for %%M in (unet partial_conv patchgan attgan mae_patchgan ddpm) do (
    echo [Visualize] %%M  copd_027-029 ...
    python run_phase3_pipeline.py --visualize --model-type %%M --start-patient-id copd_027 --limit 3
    if errorlevel 1 (
        echo [FAIL] %%M visualize failed
    ) else (
        echo [OK] %%M visualize done
    )
    echo.
)

echo.
echo ============================================================
echo   All Done!
echo   End: %date% %time%
echo ============================================================
echo.

pause

