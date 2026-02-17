#!/usr/bin/env python3
"""
Calibration quality gate — replay RMSE 기준 통과/실패 판정.

replay_in_sim.py 또는 tune_sim_dynamics.py 출력 JSON을 읽어서,
global RMSE가 임계값 이하인지 확인합니다.
통과 시 exit 0, 실패 시 exit 1.

Usage:
    # replay_in_sim.py 리포트 검증 (wheel + arm 각각)
    python check_calibration_gate.py \
        --reports calibration/replay_command_report.json \
                  calibration/replay_arm_report.json

    # tune_sim_dynamics.py 출력 검증
    python check_calibration_gate.py \
        --reports calibration/tuned_dynamics.json

    # 임계값 커스텀
    python check_calibration_gate.py \
        --reports calibration/replay_command_report.json \
        --wheel_rmse_threshold 0.15 \
        --arm_rmse_threshold 0.10

    # 파이프라인 자동화 (실패 시 후속 명령 중단)
    python check_calibration_gate.py --reports calibration/replay_command_report.json && \
        python train_lekiwi.py --num_envs 2048 --headless
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_report(path: str) -> dict:
    p = Path(path).expanduser()
    if not p.is_file():
        print(f"  FAIL: 파일 없음 — {p}")
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def check_replay_report(data: dict, path: str, thresholds: dict) -> list[str]:
    """replay_in_sim.py 출력 (mode: command | arm_command | position)."""
    failures = []
    mode = data.get("mode", "unknown")
    rmse = data.get("global_rmse_rad")
    mae = data.get("global_mae_rad")

    if rmse is None and mae is None:
        if data.get("error"):
            failures.append(f"  {path}: 에러 — {data['error']}")
            return failures
        # 데이터 없음
        failures.append(f"  {path}: global_rmse_rad/mae_rad 없음 (빈 리포트?)")
        return failures

    if mode in ("command", "position"):
        thresh = thresholds["wheel"]
        label = "wheel"
    elif mode == "arm_command":
        thresh = thresholds["arm"]
        label = "arm"
    else:
        thresh = thresholds["wheel"]
        label = mode

    status = "PASS" if (rmse is not None and rmse <= thresh) else "FAIL"
    rmse_str = f"{rmse:.6f}" if rmse is not None else "N/A"
    mae_str = f"{mae:.6f}" if mae is not None else "N/A"

    msg = f"  {path} [{mode}] {label}_rmse={rmse_str} (threshold={thresh:.4f}) mae={mae_str} → {status}"
    print(msg)

    if status == "FAIL":
        failures.append(msg)

    # per-sequence detail
    for seq in data.get("sequences", []):
        seq_rmse = seq.get("mean_rmse_rad")
        if seq_rmse is not None:
            print(f"    seq={seq.get('name', '?')}: rmse={seq_rmse:.6f}")
    for test in data.get("tests", []):
        test_rmse = test.get("rmse_rad")
        if test_rmse is not None:
            print(f"    test={test.get('name', '?')}: rmse={test_rmse:.6f}")

    return failures


def check_tuning_report(data: dict, path: str, thresholds: dict) -> list[str]:
    """tune_sim_dynamics.py 출력 (best_params + best_eval)."""
    failures = []
    best_eval = data.get("best_eval", {})
    wheel_rmse = best_eval.get("mean_rmse_rad")
    arm_rmse = best_eval.get("arm_mean_rmse_rad")

    # wheel
    if wheel_rmse is not None:
        status = "PASS" if wheel_rmse <= thresholds["wheel"] else "FAIL"
        msg = f"  {path} [tuning] wheel_rmse={wheel_rmse:.6f} (threshold={thresholds['wheel']:.4f}) → {status}"
        print(msg)
        if status == "FAIL":
            failures.append(msg)
    else:
        print(f"  {path} [tuning] wheel_rmse=N/A (스킵)")

    # arm
    if arm_rmse is not None:
        status = "PASS" if arm_rmse <= thresholds["arm"] else "FAIL"
        msg = f"  {path} [tuning] arm_rmse={arm_rmse:.6f} (threshold={thresholds['arm']:.4f}) → {status}"
        print(msg)
        if status == "FAIL":
            failures.append(msg)
    else:
        print(f"  {path} [tuning] arm_rmse=N/A (스킵)")

    return failures


def classify_report(data: dict) -> str:
    """리포트 종류 판별."""
    if "best_params" in data or "best_eval" in data:
        return "tuning"
    if "mode" in data:
        return "replay"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="캘리브레이션 품질 게이트 — replay/tuning RMSE 임계값 검사"
    )
    parser.add_argument(
        "--reports", nargs="+", required=True,
        help="replay_in_sim.py 또는 tune_sim_dynamics.py 출력 JSON 경로 (복수 가능)",
    )
    parser.add_argument(
        "--wheel_rmse_threshold", type=float, default=0.20,
        help="wheel/base RMSE 통과 임계값 (rad, 기본 0.20)",
    )
    parser.add_argument(
        "--arm_rmse_threshold", type=float, default=0.15,
        help="arm RMSE 통과 임계값 (rad, 기본 0.15)",
    )
    args = parser.parse_args()

    thresholds = {
        "wheel": args.wheel_rmse_threshold,
        "arm": args.arm_rmse_threshold,
    }

    print("=" * 60)
    print("  Calibration Quality Gate")
    print(f"  thresholds: wheel_rmse ≤ {thresholds['wheel']:.4f} rad, "
          f"arm_rmse ≤ {thresholds['arm']:.4f} rad")
    print("=" * 60)

    all_failures: list[str] = []

    for path in args.reports:
        data = load_report(path)
        kind = classify_report(data)

        if kind == "replay":
            failures = check_replay_report(data, path, thresholds)
        elif kind == "tuning":
            failures = check_tuning_report(data, path, thresholds)
        else:
            print(f"  {path}: 알 수 없는 리포트 형식 (skip)")
            continue

        all_failures.extend(failures)

    print("=" * 60)
    if all_failures:
        print(f"  GATE FAILED — {len(all_failures)}개 항목 임계값 초과")
        print("  RL 학습/롤아웃 수집 전에 캘리브레이션을 재수행하세요.")
        print("=" * 60)
        sys.exit(1)
    else:
        print("  GATE PASSED — 모든 항목 임계값 이내")
        print("  RL 학습/롤아웃 수집 진행 가능")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()