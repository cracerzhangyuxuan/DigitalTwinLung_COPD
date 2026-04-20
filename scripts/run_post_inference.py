"""
一键完成：三个新模型的 评估 + 可视化 + 数据汇总 + 图表生成

前提：mae_patchgan 和 ddpm 的推理结果已存在于 data/04_final_viz/{model}/

使用方法：
    python scripts/run_post_inference.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_NEW = ["attgan", "mae_patchgan", "ddpm"]
PATIENTS = ["copd_024", "copd_025", "copd_026"]

def check_inference_complete():
    """检查所有推理结果是否就绪"""
    all_ok = True
    for model in MODELS_NEW:
        for pid in PATIENTS:
            fused = ROOT / "data" / "04_final_viz" / model / f"{pid}_fused.nii.gz"
            if not fused.exists():
                print(f"  [MISSING] {model}/{pid}_fused.nii.gz")
                all_ok = False
            else:
                size_mb = fused.stat().st_size / 1024 / 1024
                print(f"  [OK] {model}/{pid}_fused.nii.gz ({size_mb:.1f} MB)")
    return all_ok

def run_cmd(desc, cmd_list):
    """运行命令"""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd_list)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd_list, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  [WARN] 返回码: {result.returncode}")
    return result.returncode == 0

def main():
    print("="*60)
    print("  后推理全流程：评估 + 可视化 + 数据汇总 + 图表")
    print("="*60)

    # 1. 检查推理结果
    print("\n[Step 1] 检查推理结果完整性...")
    if not check_inference_complete():
        print("\n[ERROR] 部分推理结果缺失！请先完成推理：")
        print("  python scripts/run_remaining_inference.bat")
        sys.exit(1)
    print("[OK] 所有推理结果就绪")

    # 2. 运行评估（3个新模型）
    print("\n[Step 2] 执行新模型评估...")
    for model in MODELS_NEW:
        run_cmd(f"评估 {model}",
                [sys.executable, "run_phase3_pipeline.py",
                 "--evaluate", "--model-type", model,
                 "--start-patient-id", "copd_024", "--limit", "3"])

    # 3. 运行可视化（3个新模型）
    print("\n[Step 3] 执行新模型可视化...")
    for model in MODELS_NEW:
        run_cmd(f"可视化 {model}",
                [sys.executable, "run_phase3_pipeline.py",
                 "--visualize", "--model-type", model,
                 "--start-patient-id", "copd_024", "--limit", "3"])

    # 4. 汇总 L1-L4 验证集指标（6模型）
    print("\n[Step 4] 汇总验证集 L1-L4 指标（6模型）...")
    run_cmd("验证集 L1-L4 评估",
            [sys.executable, "scripts/evaluate_validation_l1_l4.py"])

    # 5. 生成图表（6模型）
    print("\n[Step 5] 生成验证集图表（6模型）...")
    run_cmd("图表生成",
            [sys.executable, "scripts/generate_validation_charts.py"])

    print("\n" + "="*60)
    print("  全部完成！")
    print("="*60)
    print("  输出文件：")
    print("    - results/validation_metrics.json  (6模型完整数据)")
    print("    - results/validation_metrics.csv")
    print("    - results/chart_radar_l2l4_val.png")
    print("    - results/chart_glcm_contrast_val.png")
    print("    - results/chart_delta_ei_val.png")
    print("    - results/chart_train_vs_val_val.png")
    print("    - results/chart_composite_score_val.png")

if __name__ == "__main__":
    main()

