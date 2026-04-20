"""一次性执行所有剩余推理任务（mae_patchgan copd_026 + ddpm 全部3例）"""
import subprocess
import sys
from pathlib import Path

def check_output(model_type, expected):
    """检查模型已有输出"""
    d = Path(f'data/04_final_viz/{model_type}')
    if not d.exists():
        return []
    return sorted([f.name for f in d.glob('*_fused.nii.gz')])

def run_inference(model_type, start_pid, limit):
    """执行推理"""
    cmd = [
        sys.executable, 'run_phase3_pipeline.py',
        '--inference', '--model-type', model_type,
        '--device', 'cuda:1',
        '--start-patient-id', start_pid,
        '--limit', str(limit)
    ]
    print(f"\n{'='*60}")
    print(f"开始推理: {model_type} (start={start_pid}, limit={limit})")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd='.', timeout=1800)  # 30分钟超时
    return result.returncode == 0

# --- 检查 mae_patchgan copd_026 ---
mae_files = check_output('mae_patchgan', ['copd_024', 'copd_025', 'copd_026'])
print(f"mae_patchgan 已有: {mae_files}")
if 'copd_026_fused.nii.gz' not in mae_files:
    print("需要补推 mae_patchgan copd_026")
    run_inference('mae_patchgan', 'copd_026', 1)
else:
    print("mae_patchgan copd_026 已完成, 跳过")

# --- 检查 ddpm ---
ddpm_files = check_output('ddpm', ['copd_024', 'copd_025', 'copd_026'])
print(f"\nddpm 已有: {ddpm_files}")

missing_ddpm = []
for pid in ['copd_024', 'copd_025', 'copd_026']:
    if f'{pid}_fused.nii.gz' not in ddpm_files:
        missing_ddpm.append(pid)

if missing_ddpm:
    start_pid = missing_ddpm[0]
    limit = len(missing_ddpm)
    print(f"需要推理 ddpm: {missing_ddpm}")
    run_inference('ddpm', start_pid, limit)
else:
    print("ddpm 全部完成, 跳过")

# --- 最终验证 ---
print(f"\n{'='*60}")
print("最终检查:")
print(f"{'='*60}")
for model in ['attgan', 'mae_patchgan', 'ddpm']:
    files = check_output(model, [])
    status = "✓" if len(files) >= 3 else "✗"
    print(f"  [{status}] {model}: {len(files)} files -> {files}")
print("\n推理全部完成!" if all(len(check_output(m, [])) >= 3 for m in ['attgan', 'mae_patchgan', 'ddpm']) else "\n有推理未完成，请手动检查!")

