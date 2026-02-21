#!/usr/bin/env python3
"""Tucked Pose 측정 — 리더암 TCP 입력으로 접힌 자세 기록.

사용법:
  (Home PC에서)
  python calibrate_tucked_pose.py [--port 15002]

  (Windows에서)
  python isaac_teleop.py

워크플로:
  1. TCP 연결 대기
  2. 팔을 최대한 접은 상태(tucked)로 만듦
  3. Enter → 현재 관절 값을 tucked pose로 저장

목적:
  - Self-collision 방지: USD에 self-collision이 없어서 관절이 몸체를 관통함.
    tucked pose 값이 관절별 한계가 되어 관통을 방지.
  - Navigate skill: 이동 시 팔을 tucked pose로 고정.
  - RL 에피소드 초기 arm 자세.

주의:
  isaac_teleop.py가 leader→sim 좌표 변환을 하므로 (SIGNS 포함),
  측정값은 sim 좌표계 기준. 리더암만 측정하면 됨.

출력: calibration/tucked_pose.json
"""

import argparse
import json
import math
import os
import socket
import sys
import threading
import time

import numpy as np

ARM_JOINT_NAMES = [
    "STS3215_03a_v1_Revolute_45",         # 0: shoulder_pan
    "STS3215_03a_v1_1_Revolute_49",       # 1: shoulder_lift
    "STS3215_03a_v1_2_Revolute_51",       # 2: elbow_flex
    "STS3215_03a_v1_3_Revolute_53",       # 3: wrist_flex
    "STS3215_03a_Wrist_Roll_v1_Revolute_55",  # 4: wrist_roll
    "STS3215_03a_v1_4_Revolute_57",       # 5: gripper
]
SHORT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# 기존 tucked pose (비교용)
OLD_TUCKED = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.2154,
    "elbow_flex": 0.1889,
    "wrist_flex": 0.1251,
    "wrist_roll": 0.032,
    "gripper": -0.2015,
}


def _infer_unit(arr: np.ndarray) -> str:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "rad"
    p95 = float(np.percentile(np.abs(finite), 95))
    if 20.0 <= p95 <= 120.0:
        return "m100"
    if p95 > 7.0:
        return "deg"
    return "rad"


def _to_rad(arr: np.ndarray, unit: str) -> np.ndarray:
    if unit == "deg":
        return np.deg2rad(arr)
    if unit == "m100":
        return arr * (math.pi / 100.0)
    return arr


def main():
    parser = argparse.ArgumentParser(description="Tucked pose calibration via TCP")
    parser.add_argument("--port", type=int, default=15002)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--unit", type=str, default="auto", choices=["auto", "rad", "deg", "m100"])
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: calibration/tucked_pose.json)")
    cli = parser.parse_args()

    out_path = cli.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "calibration", "tucked_pose.json",
    )

    n_joints = len(ARM_JOINT_NAMES)
    current = np.zeros(n_joints)
    n_samples = 0
    resolved_unit = None
    connected = False
    lock = threading.Lock()
    stop_event = threading.Event()

    def handle_line(line: str):
        nonlocal n_samples, resolved_unit, connected
        if not line:
            return
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return

        names = msg.get("name", [])
        positions = msg.get("position", [])
        if not (isinstance(names, list) and isinstance(positions, list) and len(names) == len(positions)):
            return

        name_to_pos = dict(zip(names, positions))
        raw = np.zeros(n_joints)
        found = False
        for i, jn in enumerate(ARM_JOINT_NAMES):
            if jn in name_to_pos:
                raw[i] = float(name_to_pos[jn])
                found = True

        if not found:
            return

        unit = cli.unit
        if unit == "auto":
            if resolved_unit is None:
                resolved_unit = _infer_unit(raw)
            unit = resolved_unit

        rad = _to_rad(raw, unit)

        with lock:
            for i in range(n_joints):
                current[i] = rad[i]
            n_samples += 1
            connected = True

    def tcp_server():
        nonlocal connected
        while not stop_event.is_set():
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((cli.host, cli.port))
                server.listen(1)
                server.settimeout(1.0)
                print(f"  [TCP] {cli.host}:{cli.port} 대기 중...")

                while not stop_event.is_set():
                    try:
                        conn, addr = server.accept()
                    except socket.timeout:
                        continue

                    print(f"  [TCP] 연결됨: {addr[0]}:{addr[1]}")
                    conn.settimeout(1.0)
                    buf = ""

                    with conn:
                        while not stop_event.is_set():
                            try:
                                pkt = conn.recv(4096)
                            except socket.timeout:
                                continue
                            except OSError:
                                break
                            if not pkt:
                                break
                            buf += pkt.decode("utf-8", errors="ignore")
                            while "\n" in buf:
                                line, buf = buf.split("\n", 1)
                                handle_line(line.strip())

                    with lock:
                        connected = False
                    print("  [TCP] 연결 끊김")
            except OSError as ex:
                print(f"  [TCP] 소켓 에러: {ex}")
                time.sleep(1.0)
            finally:
                try:
                    server.close()
                except Exception:
                    pass

    # TCP 서버 시작
    t = threading.Thread(target=tcp_server, daemon=True)
    t.start()

    print("\n=== Tucked Pose 측정 ===")
    print("isaac_teleop.py 를 실행해서 리더암을 연결하세요.\n")

    # 연결 대기
    while not connected:
        time.sleep(0.2)
    if resolved_unit:
        print(f"  단위: {resolved_unit}")
    print("  연결 확인!\n")
    time.sleep(0.5)

    print("  팔을 최대한 접어주세요 (tucked pose).")
    print("  shoulder_pan은 0 유지, 나머지 관절을 몸체에 닿기 직전까지 접습니다.")
    print("  그리퍼도 완전히 닫아주세요.")
    print("")
    print("  자세가 완성되면 Enter 를 누르세요.\n")

    # 실시간 표시
    import select

    try:
        while True:
            time.sleep(0.2)
            with lock:
                if n_samples == 0:
                    continue

                lines = []
                lines.append(f"  {'관절':<16s}  {'현재(rad)':>10s}  {'현재(deg)':>10s}  {'기존값(rad)':>10s}  {'차이(deg)':>10s}")
                lines.append("  " + "-" * 70)

                for i in range(n_joints):
                    old_val = OLD_TUCKED.get(SHORT_NAMES[i], float('nan'))
                    diff_deg = math.degrees(current[i] - old_val) if np.isfinite(old_val) else float('nan')

                    lines.append(
                        f"  {SHORT_NAMES[i]:<16s}"
                        f"  {current[i]:>+10.4f}"
                        f"  {math.degrees(current[i]):>+10.1f}"
                        f"  {old_val:>+10.4f}"
                        f"  {diff_deg:>+10.1f}"
                    )

                # ANSI 커서로 덮어쓰기
                sys.stdout.write(f"\033[{n_joints + 3}A\033[J")
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()

            # Enter 체크
            if select.select([sys.stdin], [], [], 0.0)[0]:
                sys.stdin.readline()
                break

    except KeyboardInterrupt:
        print("\n\n  측정 취소!")
        stop_event.set()
        return

    # ── 결과 저장 ──
    stop_event.set()

    with lock:
        if n_samples == 0:
            print("  데이터 없음 — 저장하지 않습니다.")
            return

        tucked = {}
        tucked_sim = {}
        for i in range(n_joints):
            tucked[SHORT_NAMES[i]] = round(float(current[i]), 6)
            tucked_sim[ARM_JOINT_NAMES[i]] = round(float(current[i]), 6)

    result = {
        "description": "LeKiwi tucked pose - self-collision 방지용 접힌 자세 (measured via isaac_teleop.py TCP)",
        "unit": "radians",
        "joints": tucked,
        "joints_sim_names": tucked_sim,
    }

    print(f"\n\n{'='*60}")
    print("  Tucked Pose 측정 완료")
    print(f"{'='*60}\n")

    print(f"  {'관절':<16s}  {'값(rad)':>10s}  {'값(deg)':>10s}")
    print("  " + "-" * 40)
    for name in SHORT_NAMES:
        val = tucked[name]
        print(f"  {name:<16s}  {val:>+10.4f}  {math.degrees(val):>+10.1f}")

    # 기존값과 비교
    print(f"\n  기존 tucked_pose.json 대비 변화:")
    for name in SHORT_NAMES:
        old = OLD_TUCKED.get(name, float('nan'))
        new = tucked[name]
        if np.isfinite(old):
            diff = math.degrees(new - old)
            if abs(diff) > 0.5:
                print(f"    {name}: {diff:+.1f}°")

    # 저장
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  저장: {out_path}")
    print(f"  이 값은 self-collision 방지 관절 한계 및 Navigate arm target으로 사용됩니다.")


if __name__ == "__main__":
    main()
