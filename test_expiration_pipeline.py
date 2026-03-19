#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_expiration_pipeline.py — 呼气相流水线端到端测试脚本

测试流程：
  Step 1: prepare_phase2_data.py  → DICOM 转 NIfTI（呼气相，2 例）
  Step 2: run_phase2_pipeline.py  → 肺部分割 + Atlas 构建（呼气相）
  Step 3: run_phase3_pipeline.py  → COPD 配准 + 数字孪生底座生成（呼气相）

核心约束：
  · 所有生成文件名含 _exp 中缀
  · 不修改/覆盖任何吸气相文件（文件名 stem 中不含 _exp）
  · 每步 stdout/stderr 实时流式到控制台 + 日志文件

注意事项：
  · Atlas 构建（步骤 2）需要至少 5 例数据；仅有 2 例时该子步骤
    预期失败（returncode != 0），脚本会记录并继续步骤 3。
    如需完整验证 Atlas，请先准备 ≥5 例 DICOM 数据再运行本脚本。

用法：
  python test_expiration_pipeline.py
"""

import sys
import os
import subprocess
import time
import logging
import glob as _glob
from pathlib import Path
from datetime import datetime

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT    = PROJECT_ROOT / "data"
LOG_DIR      = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE  = LOG_DIR / f"test_expiration_pipeline_{TIMESTAMP}.log"

# ============================================================
# 步骤定义
# ============================================================
STEPS = [
    {
        "id": 1,
        "name": "DICOM → NIfTI（呼气相，2 例）",
        "cmd": [
            sys.executable, "prepare_phase2_data.py",
            "--full", "--expiration", "--max-normal", "2", "--max-copd", "2",
        ],
        # 精确路径验证（文件必须存在）
        "expected_files": [
            DATA_ROOT / "00_raw" / "normal" / "normal_001_exp.nii.gz",
            DATA_ROOT / "00_raw" / "normal" / "normal_002_exp.nii.gz",
            DATA_ROOT / "00_raw" / "copd"   / "copd_001_exp.nii.gz",
            DATA_ROOT / "00_raw" / "copd"   / "copd_002_exp.nii.gz",
        ],
        # glob 模式验证（至少一个匹配即通过）
        "expected_globs": [],
        # 步骤失败时是否终止后续步骤
        "abort_on_fail": True,
    },
    {
        "id": 2,
        "name": "肺部分割 + Atlas 构建（呼气相，快速测试）",
        "cmd": [
            sys.executable, "run_phase2_pipeline.py",
            "--expiration", "--quick-test",
        ],
        "expected_files": [
            DATA_ROOT / "01_cleaned" / "normal_clean" / "normal_001_exp_clean.nii.gz",
        ],
        "expected_globs": [
            # Atlas 模板（2 例数据时可能不生成，记为警告）
            str(DATA_ROOT / "02_atlas" / "standard_template.nii.gz"),
        ],
        # Atlas 阶段因数据不足预期失败；允许继续步骤 3
        "abort_on_fail": False,
    },
    {
        "id": 3,
        "name": "COPD 配准 + 数字孪生底座生成（呼气相，快速测试）",
        "cmd": [
            sys.executable, "run_phase3_pipeline.py",
            "--expiration", "--quick-test",
        ],
        "expected_files": [],
        "expected_globs": [
            str(DATA_ROOT / "03_mapped" / "*_exp*.nii.gz"),
            str(DATA_ROOT / "04_final_viz" / "**" / "*_exp*.nii.gz"),
        ],
        "abort_on_fail": False,
    },
]

# ============================================================
# 日志初始化
# ============================================================
def setup_logger() -> logging.Logger:
    """创建同时输出到控制台和日志文件的 logger"""
    logger = logging.getLogger("ExpPipelineTest")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ============================================================
# 吸气相文件快照（隔离验证）
# ============================================================
def snapshot_insp_files() -> dict:
    """记录 data/ 下所有不含 _exp 的 NIfTI 文件及其修改时间"""
    snap = {}
    for p_str in _glob.glob(str(DATA_ROOT / "**" / "*.nii.gz"), recursive=True):
        p = Path(p_str)
        if "_exp" not in p.stem:
            snap[str(p)] = p.stat().st_mtime
    return snap


def verify_isolation(snap_before: dict, logger: logging.Logger) -> bool:
    """对比快照，确认无吸气相文件（不含 _exp）被修改或新增"""
    violations = []

    # 已有文件被修改
    for fpath, mt_before in snap_before.items():
        p = Path(fpath)
        if p.exists() and p.stat().st_mtime != mt_before:
            violations.append(f"  [已修改] {Path(fpath).relative_to(PROJECT_ROOT)}")

    # 新增的吸气相文件
    for p_str in _glob.glob(str(DATA_ROOT / "**" / "*.nii.gz"), recursive=True):
        p = Path(p_str)
        if "_exp" not in p.stem and str(p) not in snap_before:
            violations.append(f"  [新增吸气相] {p.relative_to(PROJECT_ROOT)}")

    if violations:
        logger.error("❌ 隔离违规！以下吸气相文件被修改或新增：")
        for v in violations:
            logger.error(v)
        return False

    logger.info("✅ 隔离验证通过：无吸气相文件被修改")
    return True


# ============================================================
# 运行单个步骤（实时流式输出）
# ============================================================
def run_step(step: dict, logger: logging.Logger) -> bool:
    """
    执行流水线步骤，将 stdout/stderr 实时输出到控制台并追加到日志文件。

    Returns:
        True  — returncode == 0
        False — returncode != 0 或启动失败
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"▶  步骤 {step['id']}: {step['name']}")
    logger.info(f"   命令: {' '.join(str(c) for c in step['cmd'])}")
    logger.info("=" * 70)

    # 构造子进程环境：强制 UTF-8 I/O，避免 Windows CP936/GBK 乱码
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"]        = "1"          # PEP 540：Python 3.7+ UTF-8 模式
    child_env["PYTHONIOENCODING"]  = "utf-8"      # 兼容旧版本
    child_env["PYTHONLEGACYWINDOWSSTDIO"] = "0"   # 禁用遗留 Windows stdio

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            step["cmd"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(PROJECT_ROOT),
            env=child_env,
        )
        # 实时读取并同步写入日志文件（追加模式）
        with open(LOG_FILE, "a", encoding="utf-8") as log_fh:
            for line in proc.stdout:
                stripped = line.rstrip("\n")
                print(stripped)
                log_fh.write(stripped + "\n")
        proc.wait()
    except FileNotFoundError as exc:
        logger.error(f"无法启动进程: {exc}")
        return False

    elapsed = time.time() - t0
    rc = proc.returncode
    if rc == 0:
        logger.info(
            f"✅ 步骤 {step['id']} 完成（returncode=0，耗时 {elapsed:.1f}s）"
        )
    else:
        logger.error(
            f"❌ 步骤 {step['id']} 失败（returncode={rc}，耗时 {elapsed:.1f}s）"
        )
    return rc == 0


# ============================================================
# 中间产物验证
# ============================================================
def verify_step_outputs(step: dict, logger: logging.Logger) -> tuple:
    """
    检查步骤的预期输出文件是否存在。

    Returns:
        (passed_count, total_count)
    """
    checks = []

    # 精确路径验证
    for p in step.get("expected_files", []):
        checks.append((str(p), Path(p).exists(), "精确"))

    # glob 模式验证：至少一个含 _exp 的匹配即通过
    for pattern in step.get("expected_globs", []):
        matches    = _glob.glob(pattern, recursive=True)
        exp_hits   = [m for m in matches if "_exp" in Path(m).stem]
        # 对于 standard_template.nii.gz 这类不含 _exp 的特殊模板文件，
        # 有任意匹配即算通过
        found = len(exp_hits) > 0 or len(matches) > 0
        checks.append((pattern, found, "glob"))

    passed = sum(1 for _, ok, _ in checks if ok)
    total  = len(checks)

    if total == 0:
        logger.info(f"  （步骤 {step['id']} 无预设文件验证项）")
        return 0, 0

    logger.info(f"  文件验证（{passed}/{total} 通过）：")
    for path_str, ok, kind in checks:
        icon  = "✅" if ok else "⚠️ "
        label = Path(path_str).name if kind == "精确" else path_str
        logger.info(f"    {icon} [{kind}] {label}")

    return passed, total


# ============================================================
# 主函数
# ============================================================
def main():
    logger = setup_logger()

    logger.info("=" * 70)
    logger.info("呼气相流水线端到端测试  (test_expiration_pipeline.py)")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"日志文件: {LOG_FILE}")
    logger.info("=" * 70)

    # ── 测试前：快照吸气相文件 ──────────────────────────────────────────
    logger.info("")
    logger.info("📸 记录吸气相文件快照（用于测试后隔离验证）...")
    insp_snap = snapshot_insp_files()
    logger.info(f"   共记录 {len(insp_snap)} 个吸气相 NIfTI 文件")

    # ── 逐步执行流水线 ──────────────────────────────────────────────────
    step_results = []   # [(id, name, run_ok, passed, total)]

    for step in STEPS:
        run_ok = run_step(step, logger)

        logger.info("")
        logger.info(f"📂 验证步骤 {step['id']} 输出文件...")
        passed, total = verify_step_outputs(step, logger)
        step_results.append((step["id"], step["name"], run_ok, passed, total))

        if not run_ok and step.get("abort_on_fail", True):
            logger.error("")
            logger.error(
                f"💥 步骤 {step['id']} 失败且设置了 abort_on_fail=True，终止测试。"
            )
            logger.error("   请检查上方日志排查具体原因。")
            break

        if not run_ok:
            logger.warning(
                f"⚠️  步骤 {step['id']} 失败（abort_on_fail=False），继续后续步骤..."
            )

    # ── 隔离验证 ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("🔒 执行吸气相隔离验证...")
    isolation_ok = verify_isolation(insp_snap, logger)

    # ── 摘要报告 ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 测试摘要报告")
    logger.info("=" * 70)

    all_ok = True
    for sid, sname, run_ok, passed, total in step_results:
        icon  = "✅" if run_ok else "❌"
        finfo = f"文件验证 {passed}/{total}" if total > 0 else "无文件验证项"
        logger.info(f"  {icon} 步骤 {sid}: {sname}")
        logger.info(f"       {finfo}")
        if not run_ok:
            all_ok = False

    iso_icon = "✅" if isolation_ok else "❌"
    logger.info(f"  {iso_icon} 吸气相隔离验证")
    if not isolation_ok:
        all_ok = False

    # 统计本次生成的 _exp NIfTI 文件
    exp_files = sorted(
        _glob.glob(str(DATA_ROOT / "**" / "*_exp*.nii.gz"), recursive=True)
    )
    logger.info("")
    logger.info(f"  共生成含 _exp 中缀的 NIfTI 文件: {len(exp_files)} 个")
    for f in exp_files[:12]:
        logger.info(f"    · {Path(f).relative_to(PROJECT_ROOT)}")
    if len(exp_files) > 12:
        logger.info(f"    · ... 还有 {len(exp_files) - 12} 个（查看完整日志）")

    logger.info("")
    if all_ok:
        logger.info("🎉 所有步骤通过！呼气相流水线验证成功。")
    else:
        logger.error("❗ 存在失败或违规项，请检查日志排查问题。")

    logger.info("")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"完整日志: {LOG_FILE}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

