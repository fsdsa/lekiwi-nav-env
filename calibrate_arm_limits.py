#!/usr/bin/env python3
"""관절 리밋 캘리브레이션 — 리더암 TCP 입력으로 min/max 측정.

사용법:
  (Home PC에서)
  python calibrate_arm_limits.py [--port 15002] [--unit auto]

  (Windows에서)
  python isaac_teleop.py

워크플로:
  1. 실행 후 TCP 연결 대기
  2. 관절 하나씩 가이드 (Enter로 다음 관절)
  3. 해당 관절을 양방향 끝까지 밀고 Enter
  4. 6개 관절 완료 후 JSON 저장

주의:
  isaac_teleop.py가 leader→sim 변환을 하므로, 받는 값은 sim 좌표계.
  리더암 리밋 = 텔레옵에서 보낼 수 있는 최대 범위.
  팔로워 암이 로봇 본체에 장착되어 간섭이 다를 수 있으므로,
  리더로 측정 후 팔로워로도 검증 권장.

출력 JSON은 env_cfg.arm_limit_json 으로 바로 사용 가능:
  {"joint_limits_rad": {"<joint_name>": {"min": ..., "max": ...}, ...}}
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

# ── 조인트 이름 (lekiwi_robot_cfg와 동일) ──
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

BAKED_LIMITS_RAD = {
    "STS3215_03a_v1_Revolute_45": (-1.7453, 1.7208),
    "STS3215_03a_v1_1_Revolute_49": (-1.7453, 1.7453),
    "STS3215_03a_v1_2_Revolute_51": (-1.7173, 1.7344),
    "STS3215_03a_v1_3_Revolute_53": (-1.7424, 1.7069),
    "STS3215_03a_Wrist_Roll_v1_Revolute_55": (-1.6805, 1.5680),
    "STS3215_03a_v1_4_Revolute_57": (0.006566, 1.7453),
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
    parser = argparse.ArgumentParser(description="Arm joint limit calibration via TCP")
    parser.add_argument("--port", type=int, default=15002)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--unit", type=str, default="auto", choices=["auto", "rad", "deg", "m100"])
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: calibration/arm_limits_measured.json)")
    cli = parser.parse_args()

    out_path = cli.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "calibration", "arm_limits_measured.json",
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

    print("\n=== 관절 리밋 캘리브레이션 (한 관절씩) ===")
    print("isaac_teleop.py 를 실행해서 리더암을 연결하세요.\n")

    # 연결 대기
    while not connected:
        time.sleep(0.2)
    if resolved_unit:
        print(f"  단위: {resolved_unit}")
    print("  연결 확인!\n")
    time.sleep(0.5)

    # ── 관절별 캘리브레이션 ──
    results = {}  # joint_name → (min_rad, max_rad)

    for idx in range(n_joints):
        jname = ARM_JOINT_NAMES[idx]
        sname = SHORT_NAMES[idx]
        baked = BAKED_LIMITS_RAD.get(jname, (float('nan'), float('nan')))

        print(f"\n{'='*60}")
        print(f"  [{idx+1}/{n_joints}] {sname}")
        print(f"  기존 baked: [{baked[0]:+.4f}, {baked[1]:+.4f}] rad"
              f"  = [{math.degrees(baked[0]):+.1f}, {math.degrees(baked[1]):+.1f}] deg")
        print(f"{'='*60}")
        print(f"  이 관절을 양방향 끝까지 천천히 밀어주세요.")
        print(f"  다 밀었으면 Enter 를 누르세요.\n")

        # 이 관절의 min/max 추적
        j_min = np.inf
        j_max = -np.inf

        try:
            while True:
                # 0.2초마다 업데이트 표시
                time.sleep(0.2)
                with lock:
                    val = current[idx]

                if val < j_min:
                    j_min = val
                if val > j_max:
                    j_max = val

                rng = math.degrees(j_max - j_min) if np.isfinite(j_min) and np.isfinite(j_max) else 0.0

                # 실시간 한 줄 표시
                sys.stdout.write(
                    f"\r  현재: {val:+.4f} rad ({math.degrees(val):+.1f}°)"
                    f"  |  min: {j_min:+.4f}  max: {j_max:+.4f}"
                    f"  |  범위: {rng:.1f}°"
                    f"     "
                )
                sys.stdout.flush()

                # stdin에 Enter 입력이 있는지 확인 (non-blocking)
                import select
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    sys.stdin.readline()
                    break

        except KeyboardInterrupt:
            # Ctrl+C → 전체 종료
            print("\n\n  캘리브레이션 중단!")
            stop_event.set()
            return

        # 결과 저장
        results[jname] = (float(j_min), float(j_max))
        print(f"\n  >> {sname} 확정: [{j_min:+.4f}, {j_max:+.4f}] rad"
              f"  = [{math.degrees(j_min):+.1f}, {math.degrees(j_max):+.1f}] deg"
              f"  (범위 {math.degrees(j_max - j_min):.1f}°)")

        # baked와 비교
        diff_lo = j_min - baked[0]
        diff_hi = j_max - baked[1]
        if abs(diff_lo) > 0.01 or abs(diff_hi) > 0.01:
            print(f"     baked 대비: min {math.degrees(diff_lo):+.1f}°, max {math.degrees(diff_hi):+.1f}°")

    # ── 전체 결과 출력 & 저장 ──
    stop_event.set()

    print(f"\n\n{'='*60}")
    print("  최종 결과")
    print(f"{'='*60}\n")

    print(f"  {'관절':<16s}  {'min(rad)':>10s}  {'max(rad)':>10s}  {'min(deg)':>10s}  {'max(deg)':>10s}  {'범위(deg)':>10s}")
    print("  " + "-" * 80)

    output_json = {"joint_limits_rad": {}}
    for idx in range(n_joints):
        jname = ARM_JOINT_NAMES[idx]
        sname = SHORT_NAMES[idx]
        if jname not in results:
            print(f"  {sname:<16s}  (미측정)")
            continue

        lo, hi = results[jname]
        print(f"  {sname:<16s}  {lo:>+10.4f}  {hi:>+10.4f}"
              f"  {math.degrees(lo):>+10.1f}  {math.degrees(hi):>+10.1f}"
              f"  {math.degrees(hi - lo):>10.1f}")

        output_json["joint_limits_rad"][jname] = {
            "min": round(lo, 6),
            "max": round(hi, 6),
        }

    # 저장
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    print(f"\n  저장: {out_path}")
    print(f"  사용: record_teleop.py --arm-limit-json {out_path}")
    print(f"    또는 train_lekiwi.py에서 env_cfg.arm_limit_json = \"{out_path}\"")


if __name__ == "__main__":
    main()
