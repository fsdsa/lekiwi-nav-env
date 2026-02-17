#!/usr/bin/env python3
"""
LeKiwi dual teleop logger (Laptop side).

This script runs on the laptop where the SO100 leader arm and keyboard are connected.
It forwards teleop actions to:
  1) real LeKiwi robot (via LeKiwiClient -> Raspberry Pi host)
  2) desktop Isaac Sim receiver (TCP JSONL stream)

It also logs timestamped packets locally for later replay/alignment.

Usage (Windows PowerShell):
  python .\teleop_dual_logger.py `
    --robot_ip 192.168.1.46 `
    --leader_port COM6 `
    --desktop_host 100.91.14.65 `
    --desktop_port 15002 `
    --fps 30 `
    --no_rerun
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


WR_LEADER_KEY = "wrist_roll.pos"
WR_FOLLOWER_KEY = "arm_wrist_roll.pos"
REZERO_KEY = "c"
QUIT_KEY = "q"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeKiwi dual teleop logger (laptop)")
    parser.add_argument("--robot_ip", type=str, default="192.168.1.46")
    parser.add_argument("--robot_id", type=str, default="my_lekiwi")
    parser.add_argument("--leader_port", type=str, default="COM6")
    parser.add_argument("--leader_id", type=str, default="my_awesome_leader_arm")
    parser.add_argument("--keyboard_id", type=str, default="my_laptop_keyboard")

    parser.add_argument("--desktop_host", type=str, default="127.0.0.1", help="Desktop IP running sim receiver")
    parser.add_argument("--desktop_port", type=int, default=15002)
    parser.add_argument("--reconnect_interval_s", type=float, default=2.0)
    parser.add_argument("--connect_timeout_s", type=float, default=0.05,
                        help="TCP connect timeout for desktop forwarder (low for teleop latency)")
    parser.add_argument("--send_timeout_s", type=float, default=0.01,
                        help="TCP send timeout for desktop forwarder")
    parser.add_argument("--no_forward", action="store_true",
                        help="Do not forward to desktop (real robot teleop only)")

    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--log_dir", type=str, default="logs/dual_teleop")
    parser.add_argument("--session_name", type=str, default="lekiwi_dual_teleop")
    parser.add_argument("--no_rerun", action="store_true")
    return parser.parse_args()


def _print_controls() -> None:
    print("[teleop] controls")
    print("  arm: move SO100 leader arm (follower arm tracks)")
    print("  base: w/s=forward/back, a/d=left/right strafe, z/x=rotate")
    print("  speed: r=faster, f=slower")
    print(f"  {REZERO_KEY}=re-zero wrist_roll, {QUIT_KEY}=quit")
    print("  keep terminal focused and keyboard layout in EN")


def _print_collection_plan() -> None:
    print("[teleop] what to do now (recommended 90s)")
    print("  0-30s  : arm only (big back-and-forth on each joint)")
    print("  30-60s : base only (w/s, a/d, z/x, include r/f speed changes)")
    print("  60-90s : arm + base together (mixed motions)")
    print("  finish : press 'q' to stop and save logs")


def _wrap_m100_100(v: float) -> float:
    return ((v + 100.0) % 200.0) - 100.0


def _circular_offset_m100_100(follower: float, leader: float) -> float:
    return ((follower - leader + 100.0) % 200.0) - 100.0


def _to_numeric_dict(payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in payload.items():
        if isinstance(val, (int, float, np.integer, np.floating)):
            out[str(key)] = float(val)
    return out


class TcpForwarder:
    def __init__(self, host: str, port: int, reconnect_interval_s: float, connect_timeout_s: float, send_timeout_s: float):
        self.host = host
        self.port = port
        self.reconnect_interval_s = reconnect_interval_s
        self.connect_timeout_s = connect_timeout_s
        self.send_timeout_s = send_timeout_s
        self._sock: socket.socket | None = None
        self._last_retry_ts = 0.0

    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _maybe_connect(self):
        now = time.time()
        if now - self._last_retry_ts < self.reconnect_interval_s:
            return
        self._last_retry_ts = now

        self.close()
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout_s)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(self.send_timeout_s)
            self._sock = sock
            print(f"[forwarder] connected -> {self.host}:{self.port}")
        except OSError as exc:
            print(f"[forwarder] connect failed: {exc}")
            self._sock = None

    def send_packet(self, packet: dict[str, Any]) -> bool:
        if self._sock is None:
            self._maybe_connect()
            if self._sock is None:
                return False

        try:
            line = (json.dumps(packet, ensure_ascii=False) + "\n").encode("utf-8")
            self._sock.sendall(line)
            return True
        except OSError as exc:
            print(f"[forwarder] send failed: {exc}")
            self.close()
            return False


def main() -> None:
    args = parse_args()

    robot_cfg = LeKiwiClientConfig(remote_ip=args.robot_ip, id=args.robot_id)
    leader_cfg = SO100LeaderConfig(port=args.leader_port, id=args.leader_id)
    keyboard_cfg = KeyboardTeleopConfig(id=args.keyboard_id)

    robot = LeKiwiClient(robot_cfg)
    leader_arm = SO100Leader(leader_cfg)
    keyboard = KeyboardTeleop(keyboard_cfg)

    print("[setup] connecting robot/leader/keyboard...")
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise RuntimeError("Robot or teleop is not connected.")

    if not args.no_rerun:
        init_rerun(session_name=args.session_name)

    out_dir = Path(args.log_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"dual_teleop_{ts}.jsonl"
    print(f"[log] {log_path}")

    forwarder = None if args.no_forward else TcpForwarder(
        host=args.desktop_host,
        port=args.desktop_port,
        reconnect_interval_s=args.reconnect_interval_s,
        connect_timeout_s=args.connect_timeout_s,
        send_timeout_s=args.send_timeout_s,
    )
    if args.no_forward:
        print("[forwarder] disabled (--no_forward)")

    wr_offset = 0.0
    first_obs = robot.get_observation()
    first_leader = leader_arm.get_action()
    if WR_FOLLOWER_KEY in first_obs and WR_LEADER_KEY in first_leader:
        f = float(first_obs[WR_FOLLOWER_KEY])
        l = float(first_leader[WR_LEADER_KEY])
        if -120.0 <= f <= 120.0 and -120.0 <= l <= 120.0:
            wr_offset = _circular_offset_m100_100(f, l)
        else:
            wr_offset = f - l

    print(f"[teleop] wrist_roll offset = {wr_offset:.3f} (re-zero key: '{REZERO_KEY}')")
    print(f"[teleop] quit key: '{QUIT_KEY}'")
    _print_controls()
    _print_collection_plan()

    seq = 0
    dropped = 0

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            while True:
                t0 = time.perf_counter()
                t_wall = time.time()
                t_mono_ns = time.monotonic_ns()

                observation = robot.get_observation()
                observation_state = _to_numeric_dict(observation)

                leader_action = leader_arm.get_action()
                keyboard_keys = keyboard.get_action()
                base_action = robot._from_keyboard_to_base_action(keyboard_keys)

                if QUIT_KEY in keyboard_keys:
                    print("[teleop] quit requested")
                    break

                if REZERO_KEY in keyboard_keys and WR_FOLLOWER_KEY in observation and WR_LEADER_KEY in leader_action:
                    f = float(observation[WR_FOLLOWER_KEY])
                    l = float(leader_action[WR_LEADER_KEY])
                    if -120.0 <= f <= 120.0 and -120.0 <= l <= 120.0:
                        wr_offset = _circular_offset_m100_100(f, l)
                    else:
                        wr_offset = f - l
                    print(f"[teleop] re-zero wrist_roll offset = {wr_offset:.3f}")

                arm_action = {f"arm_{k}": float(v) for k, v in leader_action.items()}
                if WR_LEADER_KEY in leader_action:
                    desired = float(leader_action[WR_LEADER_KEY]) + wr_offset
                    if -120.0 <= desired <= 120.0:
                        desired = _wrap_m100_100(desired)
                    arm_action[WR_FOLLOWER_KEY] = desired

                action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
                action_sent = robot.send_action(action)

                action_state = _to_numeric_dict(action_sent)
                packet = {
                    "version": 1,
                    "seq": seq,
                    "t_wall_s": t_wall,
                    "t_mono_ns": int(t_mono_ns),
                    "action": action_state,
                    "observation": observation_state,
                }

                if forwarder is not None:
                    ok = forwarder.send_packet(packet)
                    if not ok:
                        dropped += 1

                log_file.write(json.dumps(packet, ensure_ascii=False) + "\n")
                if seq % 20 == 0:
                    log_file.flush()

                if not args.no_rerun:
                    log_rerun_data(observation=observation, action=action)

                if seq % int(max(1, args.fps)) == 0:
                    if forwarder is None:
                        status = "disabled"
                    else:
                        status = "connected" if forwarder.is_connected else "disconnected"
                    print(f"[teleop] seq={seq} forwarder={status} dropped={dropped}")

                seq += 1
                precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\n[teleop] interrupted")
    finally:
        if forwarder is not None:
            forwarder.close()
        try:
            robot.disconnect()
        except Exception:
            pass
        try:
            leader_arm.disconnect()
        except Exception:
            pass
        try:
            keyboard.disconnect()
        except Exception:
            pass
        print("[teleop] disconnected")


if __name__ == "__main__":
    main()
