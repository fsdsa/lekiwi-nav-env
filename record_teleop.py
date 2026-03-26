#!/usr/bin/env python3
"""
LeKiwi Navigation — 텔레옵 데모 녹화.

기존 텔레옵 시스템(리더암 + 키보드 → TCP → ROS2)을 그대로 사용하거나,
ROS2를 쓰지 않고 TCP JSON을 직접 수신해 Isaac Lab 형식으로 데모를 녹화.

데이터 흐름:
    Windows (리더암 + 키보드)
        → TCP JSON
    Home Ubuntu
        A) tcp_joint_state_reader.py 사용:
            → ROS2 /leader_joint_states (arm positions)
            → ROS2 /wheel_cmds (wheel velocities)
            → 이 스크립트는 ROS2 구독
        B) ROS2 미사용:
            → 이 스크립트가 TCP JSON 직접 수신
        → action 9D 변환
        → Isaac Lab env step
        → (obs N-D, action 9D) HDF5 저장

Action 변환:
    base (vx, vy, wz) 또는 wheel(rad/s) 입력을 action[0:3]로 정규화,
    arm(6 rad) 입력을 action[3:9]로 정규화.

Goal:
    Isaac Lab 환경이 매 에피소드 랜덤 목표 생성 → GUI에 표시
    사용자가 목표까지 텔레옵 → 도달하면 자동 저장 + 새 목표

전제 조건:
    - tcp_joint_state_reader.py 실행 중 (ROS2 토픽 발행)
    - conda activate env_isaaclab && source ~/isaacsim/setup_conda_env.sh

Usage:
    cd ~/IsaacLab/scripts/lekiwi_nav_env

    # 기본 (10 에피소드, ROS2 가능하면 ROS2 우선, 아니면 TCP fallback)
    python record_teleop.py --num_demos 10

    # Skill-2 (ApproachAndGrasp) 텔레옵 데모 (30D obs, v6 action)
    python record_teleop.py --num_demos 20 --skill approach_and_grasp \
      --multi_object_json object_catalog.json \
      --gripper_contact_prim_path "/World/envs/env_.*/Robot/<gripper_body_prim>" \
      --dynamics_json calibration/tuned_dynamics.json \
      --arm_limit_json calibration/arm_limits_real2sim.json

    # Legacy (v8 FSM) 텔레옵 데모
    python record_teleop.py --num_demos 20 \
      --multi_object_json object_catalog.json \
      --gripper_contact_prim_path "/World/envs/env_.*/Robot/<gripper_body_prim>"

    # TCP 직접 수신 강제 (Windows sender를 이 스크립트 포트로 직접 연결)
    python record_teleop.py --teleop_source tcp --listen_port 15002

    # 더 많이
    python record_teleop.py --num_demos 30 --output demos/session_02.hdf5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import select
import socket
import sys
import termios
import time
import threading
import tty

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# —— AppLauncher 먼저 ——
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="LeKiwi Nav — ROS2 텔레옵 데모 녹화")
parser.add_argument("--num_demos", type=int, default=10,
                    help="수집할 성공 에피소드 수")
parser.add_argument("--output", type=str, default=None,
                    help="출력 HDF5 경로 (기본: demos/teleop_TIMESTAMP.hdf5)")
parser.add_argument("--resume", action="store_true",
                    help="기존 HDF5에 이어서 녹화 (--output 필수)")
parser.add_argument("--arm_topic", type=str, default="/leader_joint_states",
                    help="팔 관절 ROS2 토픽")
parser.add_argument("--wheel_topic", type=str, default="/wheel_cmds",
                    help="바퀴 명령 ROS2 토픽")
parser.add_argument("--teleop_source", type=str, default="auto", choices=["auto", "ros2", "tcp"],
                    help="텔레옵 입력 소스: auto(ROS2 우선, 실패 시 TCP), ros2, tcp")
parser.add_argument("--listen_host", type=str, default="0.0.0.0",
                    help="TCP 직접 수신 모드 listen host")
parser.add_argument("--listen_port", type=int, default=15002,
                    help="TCP 직접 수신 모드 listen port")
parser.add_argument("--calibration_json", type=str, default=None,
                    help="calibration JSON 경로 (wheel/base geometry override)")
parser.add_argument("--dynamics_json", type=str, default=None,
                    help="tune_sim_dynamics.py 출력 JSON 경로")
parser.add_argument("--arm_limit_json", type=str, default=None,
                    help="arm limit JSON 경로")
parser.add_argument("--arm_limit_margin_rad", type=float, default=0.0,
                    help="arm limit margin (rad)")
parser.add_argument("--object_usd", type=str, default="",
                    help="physics grasp object USD path (empty = legacy proximity grasp)")
parser.add_argument("--dest_object_usd", type=str, default="",
                    help="destination object USD path (배경 스폰, combined 모드에서 place 기준점)")
parser.add_argument("--dest_object_scale", type=float, default=0.56,
                    help="destination object scale")
parser.add_argument("--multi_object_json", type=str, default="",
                    help="multi-object catalog JSON path (37D obs)")
parser.add_argument("--object_mass", type=float, default=0.3,
                    help="physics grasp object mass (kg)")
# S2 expert 자동 실행 (combined 모드 Phase 1)
parser.add_argument("--s2_bc_checkpoint", type=str, default="",
                    help="combined: S2 BC checkpoint → Phase 1 자동 실행")
parser.add_argument("--s2_resip_checkpoint", type=str, default="",
                    help="combined: S2 ResiP checkpoint (optional)")
parser.add_argument("--object_scale_phys", type=float, default=1.0,
                    help="physics grasp object uniform scale")
parser.add_argument("--gripper_contact_prim_path", type=str, default="",
                    help="contact sensor prim path for gripper body (required in multi-object mode)")
parser.add_argument("--grasp_gripper_threshold", type=float, default=0.7,
                    help="gripper joint position threshold for closed state")
parser.add_argument("--grasp_contact_threshold", type=float, default=0.65,
                    help="minimum contact force magnitude for grasp success")
parser.add_argument("--grasp_max_object_dist", type=float, default=0.25,
                    help="max object distance for contact-based grasp success")
parser.add_argument("--grasp_attach_height", type=float, default=0.15,
                    help="attached object z-height after grasp success")
parser.add_argument(
    "--arm_input_unit",
    type=str,
    default="auto",
    choices=["auto", "rad", "deg", "m100"],
    help="teleop arm position unit (auto/rad/deg/m100)",
)
parser.add_argument(
    "--skill",
    type=str,
    default="legacy",
    choices=["navigate", "approach_and_grasp", "carry", "carry_and_place", "combined", "legacy"],
    help="환경 모드: navigate(Skill-1), approach_and_grasp(Skill-2), carry(Skill-3 carry only), carry_and_place(Skill-3), combined(Skill-2→3 연속), legacy(v8 FSM)",
)
parser.add_argument(
    "--grasp_hold_steps",
    type=int,
    default=450,
    help="combined mode: 파지 유지 스텝 수 (450 60Hz)",
)
parser.add_argument("--carry_interp_steps", type=int, default=600,
                    help="carry: arm 보간 완료까지 예상 스텝 수")
parser.add_argument(
    "--home_dist_thresh",
    type=float,
    default=0.7,
    help="combined mode: Phase 2→3 전환 거리 (home이 이 거리 이내 + FOV 내일 때 Skill-3 기록 시작)",
)
parser.add_argument(
    "--home_fov_thresh",
    type=float,
    default=0.76,
    help="combined mode: Phase 2→3 전환 FOV 각도 (rad, spawn_heading_max_rad과 동일)",
)
# GUI 필수 (텔레옵)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.skill != "carry":
    args.headless = False  # 텔레옵은 GUI 필수 (carry는 완전 자동이므로 headless 허용)
args.num_envs = 1

launcher = AppLauncher(args)
sim_app = launcher.app

# PhysX tensor 경고/에러 suppress (num_envs=1에서 매 step 스팸)
import sys, io

class _PhysxFilter(io.TextIOBase):
    """stderr에서 PhysX tensor 스팸 필터링."""
    def __init__(self, stream):
        self._stream = stream
    def write(self, msg):
        if "omni.physx.tensors.plugin" in msg or "Incompatible device" in msg:
            return len(msg)
        return self._stream.write(msg)
    def flush(self):
        self._stream.flush()

sys.stderr = _PhysxFilter(sys.stderr)

# —— 나머지 import ——
import h5py
import numpy as np
import torch

# S3 carry: arm 보간 시작/끝 자세 (실험 데이터에서 측정)
S3_ARM_START = np.array([-0.040, -0.193, +0.275, -1.280, -0.035], dtype=np.float64)  # S2 lift 후
S3_GRIP_START = 0.276
S3_ARM_END = np.array([+0.002, -0.193, +0.295, -1.306, +0.006], dtype=np.float64)    # S4 시작
S3_GRIP_END = 0.15  # 목표를 낮게 → 물체 저항으로 실제 grip ≈ 0.27~0.30 (S4 시작과 일치)

ROS2_AVAILABLE = False
ROS2_IMPORT_ERROR: Exception | None = None
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except Exception as ex:  # noqa: BLE001 - ABI mismatch 등도 잡아 fallback한다.
    ROS2_IMPORT_ERROR = ex

from lekiwi_robot_cfg import (
    ARM_JOINT_NAMES, WHEEL_JOINT_NAMES,
    WHEEL_ANGLES_RAD,
)


# ═══════════════════════════════════════════════════════════════════════
#  Teleop Input (ROS2 / TCP)
# ═══════════════════════════════════════════════════════════════════════

class TeleopInputBase:
    """텔레옵 입력 공통 인터페이스."""

    def __init__(self):
        # 다중 상속 경로에서도 공통 초기화 훅을 유지한다.
        super().__init__()

    def get_latest(self) -> tuple[np.ndarray, np.ndarray, bool]:
        raise NotImplementedError

    def shutdown(self):
        pass


if ROS2_AVAILABLE:
    class Ros2TeleopSubscriber(Node, TeleopInputBase):
        """ROS2에서 텔레옵 명령 수신."""

        def __init__(self, arm_topic: str, wheel_topic: str, M_inv: np.ndarray, wheel_radius: float):
            Node.__init__(self, "teleop_recorder")
            TeleopInputBase.__init__(self)

            self._lock = threading.Lock()
            self._M_inv = M_inv
            self._wheel_radius = float(wheel_radius)

            # 최신 데이터
            self._arm_positions = np.zeros(6)   # rad
            self._wheel_velocities = np.zeros(3)  # rad/s
            self._arm_stamp = 0.0
            self._wheel_stamp = 0.0

            # 구독
            self.create_subscription(JointState, arm_topic, self._arm_cb, 10)
            self.create_subscription(JointState, wheel_topic, self._wheel_cb, 10)
            self.get_logger().info(f"Subscribing: {arm_topic}, {wheel_topic}")

        def _arm_cb(self, msg: JointState):
            """팔 관절 위치 수신."""
            with self._lock:
                name_to_pos = dict(zip(msg.name, msg.position))
                for i, jn in enumerate(ARM_JOINT_NAMES):
                    if jn in name_to_pos:
                        self._arm_positions[i] = name_to_pos[jn]
                self._arm_stamp = time.time()

        def _wheel_cb(self, msg: JointState):
            """바퀴 속도 수신."""
            with self._lock:
                name_to_vel = dict(zip(msg.name, msg.velocity))
                for i, jn in enumerate(WHEEL_JOINT_NAMES):
                    if jn in name_to_vel:
                        self._wheel_velocities[i] = name_to_vel[jn]
                self._wheel_stamp = time.time()

        def get_latest(self) -> tuple[np.ndarray, np.ndarray, bool]:
            """
            최신 텔레옵 데이터 반환.
            Returns: (arm_positions (6,), body_cmd (3,), is_active)
            """
            with self._lock:
                arm = self._arm_positions.copy()
                wheel = self._wheel_velocities.copy()
                now = time.time()
                active = (now - self._arm_stamp < 1.0) or (now - self._wheel_stamp < 1.0)
            body_cmd = wheel_to_body_vel(wheel, self._M_inv, self._wheel_radius)
            return arm, body_cmd, active


class TcpTeleopSubscriber(TeleopInputBase):
    """TCP JSON lines에서 텔레옵 명령 수신."""

    def __init__(self, host: str, port: int):
        super().__init__()
        self._host = host
        self._port = port
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._arm_positions = np.zeros(6, dtype=np.float64)
        self._base_cmd = np.zeros(3, dtype=np.float64)  # [vx, vy, wz]
        self._stamp = 0.0

        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def _serve_loop(self):
        while not self._stop.is_set():
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((self._host, self._port))
                server.listen(1)
                server.settimeout(1.0)
                print(f"  [TCP] Listening on {self._host}:{self._port}")

                while not self._stop.is_set():
                    try:
                        conn, addr = server.accept()
                    except socket.timeout:
                        continue

                    print(f"  [TCP] Client connected: {addr[0]}:{addr[1]}")
                    conn.settimeout(1.0)
                    buffer = ""

                    with conn:
                        while not self._stop.is_set():
                            try:
                                packet = conn.recv(4096)
                            except socket.timeout:
                                continue
                            except OSError:
                                break

                            if not packet:
                                break

                            buffer += packet.decode("utf-8", errors="ignore")
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                self._handle_line(line.strip())

                    print("  [TCP] Client disconnected")
            except OSError as ex:
                print(f"  [TCP] Socket error: {ex}")
                time.sleep(1.0)
            finally:
                try:
                    server.close()
                except Exception:  # noqa: BLE001
                    pass

    def _handle_line(self, line: str):
        if not line:
            return
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return

        with self._lock:
            payload = msg.get("action", msg) if isinstance(msg, dict) else {}
            if not isinstance(payload, dict):
                payload = {}

            # Legacy packet: {"name":[...], "position":[...], "base":{...}}
            names = msg.get("name", []) if isinstance(msg, dict) else []
            positions = msg.get("position", []) if isinstance(msg, dict) else []
            if isinstance(names, list) and isinstance(positions, list) and len(names) == len(positions):
                name_to_pos = dict(zip(names, positions))
                for i, jn in enumerate(ARM_JOINT_NAMES):
                    if jn in name_to_pos:
                        self._arm_positions[i] = float(name_to_pos[jn])

            # New packet compatibility: teleop_dual_logger.py forwards {"action": {...}}
            arm_fallback_keys = [
                "arm_shoulder_pan.pos",
                "arm_shoulder_lift.pos",
                "arm_elbow_flex.pos",
                "arm_wrist_flex.pos",
                "arm_wrist_roll.pos",
                "arm_gripper.pos",
            ]
            for i, key in enumerate(arm_fallback_keys):
                if key in payload:
                    self._arm_positions[i] = float(payload[key])
            for i, jn in enumerate(ARM_JOINT_NAMES):
                if jn in payload:
                    self._arm_positions[i] = float(payload[jn])

            # Base command parsing: support both {"base":{vx,vy,wz}} and x.vel/y.vel/theta.vel.
            base = msg.get("base", {}) if isinstance(msg, dict) else {}
            if isinstance(base, dict):
                self._base_cmd[0] = float(base.get("vx", self._base_cmd[0]))
                self._base_cmd[1] = float(base.get("vy", self._base_cmd[1]))
                self._base_cmd[2] = float(base.get("wz", self._base_cmd[2]))
            elif isinstance(base, (list, tuple)) and len(base) >= 3:
                self._base_cmd[0] = float(base[0])
                self._base_cmd[1] = float(base[1])
                self._base_cmd[2] = float(base[2])

            self._base_cmd[0] = float(payload.get("x.vel", payload.get("base.vx", self._base_cmd[0])))
            self._base_cmd[1] = float(payload.get("y.vel", payload.get("base.vy", self._base_cmd[1])))
            self._base_cmd[2] = float(payload.get("theta.vel", payload.get("base.wz", self._base_cmd[2])))

            self._stamp = time.time()

    def get_latest(self) -> tuple[np.ndarray, np.ndarray, bool]:
        with self._lock:
            arm = self._arm_positions.copy()
            # isaac_teleop sends IK frame (vx=forward, vy=left)
            # Convert to body frame (vx=right, vy=forward) for _apply_action()
            ik = self._base_cmd.copy()
            body_cmd = np.array([-ik[1], ik[0], ik[2]])
            active = (time.time() - self._stamp) < 1.0
        return arm, body_cmd, active

    def shutdown(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


# ═══════════════════════════════════════════════════════════════════════
#  Kiwi 역 IK (wheel rad/s → body velocity, ROS2 path 전용)
# ═══════════════════════════════════════════════════════════════════════

def build_kiwi_M_inv(base_radius: float):
    """역 Kiwi IK 행렬: [vx, vy, wz] = r * M_inv @ wheel_radps"""
    angles = np.array(WHEEL_ANGLES_RAD)
    M = np.array([
        [math.cos(a), math.sin(a), float(base_radius)] for a in angles
    ])
    M_inv = np.linalg.inv(M)
    return M_inv


def wheel_to_body_vel(wheel_radps: np.ndarray, M_inv: np.ndarray, wheel_radius: float) -> np.ndarray:
    """바퀴 각속도 → 몸체 속도 (vx, vy, wz)."""
    return float(wheel_radius) * M_inv @ wheel_radps


# ═══════════════════════════════════════════════════════════════════════
#  텔레옵 → Action 변환
# ═══════════════════════════════════════════════════════════════════════

def teleop_to_action(
    arm_pos: np.ndarray,
    body_cmd: np.ndarray,
    max_lin_vel: float,
    max_ang_vel: float,
    arm_action_scale: float,
    arm_action_to_limits: bool = False,
    arm_center: np.ndarray | None = None,
    arm_half_range: np.ndarray | None = None,
    use_v6: bool = False,
) -> np.ndarray:
    """
    텔레옵 데이터 → Isaac Lab 환경 action (9D, [-1, 1]).

    legacy: action[0:3] = base(vx,vy,wz), action[3:9] = arm(6)
    v6:     action[0:6] = arm(5)+grip(1), action[6:9] = base(vx,vy,wz)
    """
    # Base 정규화
    base_norm = np.array([
        np.clip(body_cmd[0] / max_lin_vel, -1.0, 1.0),
        np.clip(body_cmd[1] / max_lin_vel, -1.0, 1.0),
        np.clip(body_cmd[2] / max_ang_vel, -1.0, 1.0),
    ])

    # Arm 정규화
    if arm_action_to_limits and arm_center is not None and arm_half_range is not None:
        safe_half = np.where(np.abs(arm_half_range) > 1e-6, arm_half_range, 1.0)
        arm_norm = np.clip((arm_pos - arm_center) / safe_half, -1.0, 1.0)
        # 그리퍼(idx 5): 하한 클립 해제 — 리밋 아래 타겟으로 PD가 강하게 닫도록
        grip_raw = (arm_pos[5] - arm_center[5]) / safe_half[5]
        arm_norm[5] = np.clip(grip_raw, -1.5, 1.0)
    else:
        arm_norm = np.clip(arm_pos / arm_action_scale, -1.0, 1.0)

    action = np.zeros(9)
    if use_v6:
        # v6: [arm5, grip1, base3]
        action[0:6] = arm_norm
        action[6:9] = base_norm
    else:
        # legacy: [base3, arm6]
        action[0:3] = base_norm
        action[3:9] = arm_norm

    return action


def _infer_arm_unit(arm_pos: np.ndarray) -> str:
    arr = np.asarray(arm_pos, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "rad"
    p95_abs = float(np.percentile(np.abs(finite), 95))
    if 20.0 <= p95_abs <= 120.0:
        return "m100"
    if p95_abs > 7.0:
        return "deg"
    return "rad"


def normalize_arm_positions_to_rad(arm_pos: np.ndarray, unit: str) -> tuple[np.ndarray, str]:
    unit_l = str(unit).strip().lower()
    arr = np.asarray(arm_pos, dtype=np.float64)
    if unit_l == "auto":
        unit_l = _infer_arm_unit(arr)
    if unit_l == "deg":
        return np.deg2rad(arr), unit_l
    if unit_l == "m100":
        return arr * (np.pi / 100.0), unit_l
    return arr, "rad"


# ═══════════════════════════════════════════════════════════════════════
#  Non-blocking keyboard input (arrow key detection)
# ═══════════════════════════════════════════════════════════════════════

_old_term_settings = None
_keyboard_available = False


def _setup_keyboard():
    """Terminal을 cbreak 모드로 설정 (비차단 키 입력, Ctrl+C 유지)."""
    global _old_term_settings, _keyboard_available
    if not sys.stdin.isatty():
        print("  [WARN] stdin is not a terminal — 화살표 키 비활성화")
        return
    try:
        fd = sys.stdin.fileno()
        _old_term_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        _keyboard_available = True
    except (termios.error, OSError):
        print("  [WARN] termios setup 실패 — 화살표 키 비활성화")


def _restore_keyboard():
    """Terminal 원래 설정 복원."""
    global _old_term_settings, _keyboard_available
    if _old_term_settings is not None:
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _old_term_settings)
        except (termios.error, OSError):
            pass
        _old_term_settings = None
    _keyboard_available = False


def _check_arrow_key():
    """비차단 화살표 키 확인. 'right'/'left' 또는 None 반환."""
    if not _keyboard_available:
        return None
    try:
        if not select.select([sys.stdin], [], [], 0.0)[0]:
            return None
        ch = sys.stdin.read(1)
        if ch == '\x1b':  # ESC sequence (arrow keys)
            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch2 = sys.stdin.read(1)
                if ch2 == '[' and select.select([sys.stdin], [], [], 0.05)[0]:
                    ch3 = sys.stdin.read(1)
                    return {'A': 'up', 'B': 'down', 'C': 'right', 'D': 'left'}.get(ch3)
    except (OSError, ValueError):
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    # —— 출력 경로 ——
    if args.resume and not args.output and args.skill != "combined":
        print("  ERROR: --resume 사용 시 --output으로 기존 파일을 지정해야 합니다. (combined 모드는 자동 검색)")
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        os.makedirs("demos", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"demos/teleop_{timestamp}.hdf5"

    print("\n" + "=" * 60)
    print("  LeKiwi Nav — ROS2 텔레옵 데모 녹화")
    print("=" * 60)
    print(f"  목표: {args.num_demos} 에피소드")
    print(f"  저장: {output_path}")
    if args.resume:
        print(f"  모드: RESUME (기존 파일에 이어서 녹화)")
    print(f"  teleop source: {args.teleop_source}")
    print(f"  arm input unit: {args.arm_input_unit}")
    print(f"  ROS2 토픽: {args.arm_topic}, {args.wheel_topic}")
    print(f"  TCP 수신: {args.listen_host}:{args.listen_port}")
    print()
    print("  ── 사용법 ──")
    print("  1. ROS2 모드: tcp_joint_state_reader.py 실행 후 ROS2 토픽 확인")
    print("     TCP 모드: Windows sender를 본 스크립트 listen 포트로 직접 연결")
    print("  2. 리더암 + 키보드로 로봇을 목표(빨간점)까지 이동")
    print("  3. 목표 도달 시 자동 저장 + 새 목표 생성")
    print("  4. 도중 정지 시 Ctrl+C")
    print("=" * 60 + "\n")

    # —— Isaac Lab 환경 ——
    use_v6 = args.skill not in ("legacy",)
    is_navigate = (args.skill == "navigate")
    is_carry = (args.skill == "carry")

    if args.skill == "navigate":
        from lekiwi_skill1_env import Skill1Env, Skill1EnvCfg
        env_cfg = Skill1EnvCfg()
    elif args.skill == "approach_and_grasp":
        from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg
        env_cfg = Skill2EnvCfg()
    elif args.skill == "carry":
        from lekiwi_skill2_eval import Skill2Env as CarryEnvBase, Skill2EnvCfg as CarryEnvCfgBase
        env_cfg = CarryEnvCfgBase()
    elif args.skill in ("carry_and_place", "combined"):
        from lekiwi_skill3_env import Skill3Env, Skill3EnvCfg
        env_cfg = Skill3EnvCfg()
    else:
        from lekiwi_nav_env import LeKiwiNavEnv, LeKiwiNavEnvCfg
        env_cfg = LeKiwiNavEnvCfg()

    env_cfg.scene.num_envs = 1
    # 텔레옵은 에피소드 길이 충분히 확보 (수동 종료 사용, 1시간)
    env_cfg.episode_length_s = 3600.0
    # 텔레옵: oob 제한 크게 (carry 중 자유 이동)
    env_cfg.max_dist_from_origin = 50.0
    # 텔레옵 시 DR 비활성화 (action delay, obs noise가 제어를 방해)
    env_cfg.enable_domain_randomization = False
    # 텔레옵 시 PhysX에 baked limits 미기록 → USD 기본 리밋 사용 (test.py와 동일)
    env_cfg.arm_limit_write_to_sim = False
    if args.calibration_json is not None:
        raw = str(args.calibration_json).strip()
        env_cfg.calibration_json = os.path.expanduser(raw) if raw else ""
    if args.dynamics_json:
        env_cfg.dynamics_json = os.path.expanduser(args.dynamics_json)
    if args.arm_limit_json:
        env_cfg.arm_limit_json = os.path.expanduser(args.arm_limit_json)
        env_cfg.arm_limit_margin_rad = float(args.arm_limit_margin_rad)
    physics_grasp_mode = bool(str(args.object_usd).strip()) or bool(str(args.multi_object_json).strip())
    multi_object_mode = bool(str(args.multi_object_json).strip())
    if multi_object_mode:
        env_cfg.multi_object_json = os.path.expanduser(args.multi_object_json)
    if str(args.dest_object_usd).strip():
        env_cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
        env_cfg.dest_object_scale = float(args.dest_object_scale)
    if physics_grasp_mode:
        env_cfg.object_usd = os.path.expanduser(args.object_usd)
        env_cfg.object_mass = float(args.object_mass)
        env_cfg.object_scale = float(args.object_scale_phys)
        env_cfg.gripper_contact_prim_path = str(args.gripper_contact_prim_path)
        env_cfg.grasp_gripper_threshold = float(args.grasp_gripper_threshold)
        env_cfg.grasp_contact_threshold = float(args.grasp_contact_threshold)
        env_cfg.grasp_max_object_dist = float(args.grasp_max_object_dist)
        env_cfg.grasp_attach_height = float(args.grasp_attach_height)
    # 텔레옵: break_force (legacy env에서만 사용, Skill2/3Env는 순수 마찰)
    env_cfg.grasp_joint_break_force = 1e8
    env_cfg.grasp_joint_break_torque = 1e8
    if multi_object_mode and not str(args.gripper_contact_prim_path).strip():
        raise ValueError(
            "multi-object(37D) 텔레옵 데모에는 --gripper_contact_prim_path가 필요합니다."
        )
    is_combined = (args.skill == "combined")
    if is_combined:
        env_cfg.grasp_gripper_threshold = 0.65
        env_cfg.dest_object_fixed = False
        env_cfg.dest_object_mass = 50.0
    if is_carry:
        # train_resip combined_s2_s3와 동일한 config
        env_cfg.grasp_contact_threshold = 0.55
        env_cfg.grasp_gripper_threshold = 0.65
        env_cfg.grasp_max_object_dist = 0.50
        env_cfg.grasp_success_height = 100.0  # S2 자동종료 비활성
        env_cfg.lift_hold_steps = 0
        env_cfg.dest_object_fixed = False
        env_cfg.dest_object_scale = 0.56
        env_cfg.dest_object_mass = 50.0
        env_cfg.spawn_heading_noise_std = 0.3
        env_cfg.spawn_heading_max_rad = 0.5
    if is_carry and not args.s2_bc_checkpoint:
        print("  ERROR: --skill carry 에는 --s2_bc_checkpoint 가 필수입니다.")
        sys.exit(1)
    if args.skill == "navigate":
        env_cfg.force_tucked_pose = True  # env가 매 step arm 강제
        env = Skill1Env(cfg=env_cfg)
        # VIVA S1: TUCKED_POSE 대신 all-zero(스폰 자세) 강제
        env._tucked_pose = torch.zeros(5, dtype=torch.float32, device=env.device)
        # gripper도 0 (open) — 모듈 상수 패치
        import lekiwi_skill1_env as _s1mod
        _s1mod._TUCKED_GRIPPER_RAD = 0.0
    elif args.skill == "approach_and_grasp":
        env = Skill2Env(cfg=env_cfg)
    elif args.skill == "carry":
        env = CarryEnvBase(cfg=env_cfg)
        env._combined_mode = True
    elif args.skill in ("carry_and_place", "combined"):
        env = Skill3Env(cfg=env_cfg)
        if is_combined:
            env._combined_mode = True
    else:
        env = LeKiwiNavEnv(cfg=env_cfg)

    # Teleop: grasp_max_object_dist 강제 적용 (configclass 복사 이슈 대비)
    if physics_grasp_mode:
        env.cfg.grasp_max_object_dist = float(args.grasp_max_object_dist)
        env.cfg.grasp_contact_threshold = float(args.grasp_contact_threshold)

    # Teleop: 환경 자동 종료 비활성화 (물체 충돌/out_of_bounds로 리셋 방지)
    # carry Phase 1(S2 expert)에서는 env terminated 활용 (topple detection: objZ < 0.03)
    _original_get_dones = env._get_dones
    _teleop_allow_terminate = [False]  # carry Phase 1에서만 True로 변경

    def _teleop_get_dones():
        terminated, truncated = _original_get_dones()
        if _teleop_allow_terminate[0]:
            # Skill3Env._get_dones에 object_toppled 없으므로 직접 추가
            env_z = env.scene.env_origins[:, 2] if hasattr(env.scene, "env_origins") else 0.0
            objZ = env.object_pos_w[:, 2] - env_z
            toppled = (objZ < 0.03) & (env.episode_length_buf > 20)
            terminated = terminated | toppled
        else:
            terminated[:] = False
        truncated[:] = False
        return terminated, truncated

    env._get_dones = _teleop_get_dones

    # 목적지 마커: dest object USD가 있으면 마커 불필요 (빨간 컵 자체가 보임)
    #              dest object USD가 없으면 초록 구체로 위치 표시
    _dest_marker = None
    _dest_attr = "dest_object_pos_w" if hasattr(env, "dest_object_pos_w") else "home_pos_w"
    _has_dest_rigid = getattr(env, "_dest_object_rigid", None) is not None
    if hasattr(env, _dest_attr) and not _has_dest_rigid:
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils
        _dest_marker = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/World/Visuals/dest_marker",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.08,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
            },
        ))
        _dm = getattr(env, _dest_attr)[:1].clone(); _dm[:, 2] = 0.08
        _dest_marker.visualize(translations=_dm)

    base_radius = float(env.base_radius)
    wheel_radius = float(env.wheel_radius)
    print(f"  geometry: wheel_radius={wheel_radius:.6f}, base_radius={base_radius:.6f}")
    print(f"  skill: {args.skill} (action format: {'v6' if use_v6 else 'legacy'})")
    print(f"  obs_dim: {int(env.observation_space.shape[0])}")
    if multi_object_mode:
        print(f"  multi_object_json: {os.path.expanduser(args.multi_object_json)}")
    if physics_grasp_mode:
        print(f"  gripper_contact_prim_path: {args.gripper_contact_prim_path}")

    # Kiwi 역 IK (ROS2 wheel->body path에서 사용)
    M_inv = build_kiwi_M_inv(base_radius)

    # —— 텔레옵 입력 소스 초기화 ——
    selected_source = args.teleop_source
    if selected_source == "auto":
        selected_source = "ros2" if ROS2_AVAILABLE else "tcp"

    teleop_input: TeleopInputBase
    ros_executor = None

    if selected_source == "ros2":
        if not ROS2_AVAILABLE:
            raise RuntimeError(
                "teleop_source=ros2 이지만 rclpy import에 실패했습니다. "
                f"오류: {ROS2_IMPORT_ERROR}\n"
                "해결: --teleop_source tcp 로 실행하거나, Python/ROS ABI를 맞추세요."
            )
        rclpy.init()
        teleop_sub = Ros2TeleopSubscriber(args.arm_topic, args.wheel_topic, M_inv, wheel_radius)
        ros_executor = SingleThreadedExecutor()
        ros_executor.add_node(teleop_sub)

        def ros_spin():
            while rclpy.ok():
                ros_executor.spin_once(timeout_sec=0.01)

        ros_thread = threading.Thread(target=ros_spin, daemon=True)
        ros_thread.start()
        teleop_input = teleop_sub
        print("  ✅ Teleop source: ROS2")
    elif selected_source == "tcp":
        teleop_input = TcpTeleopSubscriber(args.listen_host, args.listen_port)
        print("  ✅ Teleop source: TCP direct")
    else:
        raise ValueError(f"Unsupported teleop source: {selected_source}")

    # 환경 파라미터
    max_lin_vel = float(env.cfg.max_lin_vel)
    max_ang_vel = float(env.cfg.max_ang_vel)
    arm_action_scale = float(env.cfg.arm_action_scale)
    arm_action_to_limits = bool(env.cfg.arm_action_to_limits)
    arm_center = None
    arm_half_range = None
    if arm_action_to_limits:
        # env._apply_action()과 동일한 리밋 소스 사용 (sim 리밋과 다를 수 있음)
        override = getattr(env, "_arm_action_limits_override", None)
        if override is not None:
            lim = override[0].detach().cpu().numpy()
        else:
            lim = env.robot.data.soft_joint_pos_limits[0, env.arm_idx].detach().cpu().numpy()
        arm_center = 0.5 * (lim[:, 0] + lim[:, 1])
        arm_half_range = 0.5 * (lim[:, 1] - lim[:, 0])
        arm_half_range = np.where(np.abs(arm_half_range) > 1e-6, arm_half_range, 1.0)
        print("  arm mapping: action [-1,1] -> joint limits (center/half-range)")
        print(f"    gripper limit: [{lim[5,0]:.4f}, {lim[5,1]:.4f}] center={arm_center[5]:.4f}")
    else:
        print(f"  arm mapping: action * arm_action_scale ({arm_action_scale:.4f})")
    goal_thresh = float(getattr(env.cfg, "goal_reached_thresh", 0.30))

    # wz 부호 보정: test.py(원본 텔레옵)에서 wz=-wz로 반전.
    # dynamics_json의 command_transform.wz_sign이 있으면 사용, 없으면 -1.0 (test.py 기본 동작).
    wz_sign = -1.0
    ct = getattr(env, "_dynamics_command_transform", None)
    if ct is not None and isinstance(ct, dict):
        wz_sign = float(ct.get("wz_sign", -1.0))
    print(f"  wz_sign: {wz_sign}")

    # —— 녹화 루프 ——
    obs, info = env.reset()
    if _dest_marker is not None:
        _hm = getattr(env, _dest_attr)[:1].clone(); _hm[:, 2] = 0.08
        _dest_marker.visualize(translations=_hm)

    # Navigate: direction command 라벨 헬퍼 (early define — 아래에서 사용)
    _NAV_DIR_LABELS = {
        (0, 1, 0): "FORWARD",
        (0, -1, 0): "BACKWARD",
        (-1, 0, 0): "STRAFE LEFT",
        (1, 0, 0): "STRAFE RIGHT",
        (0, 0, 1): "TURN LEFT (CCW)",
        (0, 0, -1): "TURN RIGHT (CW)",
    }

    def _nav_dir_label(cmd_tensor):
        """direction_cmd 텐서 → 사람이 읽을 수 있는 라벨."""
        c = cmd_tensor.cpu().tolist()
        best_label = "UNKNOWN"
        best_dot = -1.0
        for key, label in _NAV_DIR_LABELS.items():
            dot = sum(a * b for a, b in zip(c, key))
            if dot > best_dot:
                best_dot = dot
                best_label = label
        return best_label

    # 디버그: 물체 스폰 확인
    if hasattr(env, 'object_rigid') and env.object_rigid is not None:
        obj_pos = env.object_rigid.data.root_pos_w[0].cpu().numpy()
        print(f"  [DEBUG] object_rigid pos: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")
        print(f"  [DEBUG] object_pos_w: {env.object_pos_w[0].cpu().numpy()}")
        print(f"  [DEBUG] object_rigid num_instances: {env.object_rigid.num_instances}")
    else:
        print(f"  [DEBUG] object_rigid: None (physics_grasp={getattr(env, '_physics_grasp', '?')})")

    # Navigate/Carry: 방향별 순서 스케줄 (6방향 × N회씩, 순서대로)
    _nav_dir_schedule = None
    _nav_dir_idx = 0
    _CARRY_RECORD_STEPS = 600
    _CARRY_REST_STEPS = 0  # carry는 S2 expert가 매번 재실행되므로 rest 불필요
    # Carry: 방향→body frame base 명령 매핑 (LeKiwi: +Y=forward, +X=right, +wz=CCW)
    _CARRY_BASE_SPEED = 0.25   # m/s (navigate와 동일)
    _CARRY_ANG_SPEED = 1.7     # rad/s (turn은 현재 속도 유지)
    _CARRY_DIR_TO_CMD = {
        "FORWARD":      (0.0, _CARRY_BASE_SPEED, 0.0),
        "BACKWARD":     (0.0, -_CARRY_BASE_SPEED, 0.0),
        "STRAFE LEFT":  (-_CARRY_BASE_SPEED, 0.0, 0.0),
        "STRAFE RIGHT": (_CARRY_BASE_SPEED, 0.0, 0.0),
        "TURN LEFT":    (0.0, 0.0, _CARRY_ANG_SPEED),
        "TURN RIGHT":   (0.0, 0.0, -_CARRY_ANG_SPEED),
    }
    if is_carry:
        _nav_all_dirs = list(_CARRY_DIR_TO_CMD.keys())
        reps = max(1, args.num_demos // 6)
        _nav_dir_schedule = []
        for label in _nav_all_dirs:
            _nav_dir_schedule.extend([label] * reps)
        print(f"  [Carry] 방향 스케줄: {len(_nav_dir_schedule)}개 (6방향 × {reps}회)")
        print(f"  [Carry] 각 방향 {_CARRY_RECORD_STEPS} steps 자동 저장 (headless 자동)")
        print(f"  [Carry] 첫 번째 방향: {_nav_dir_schedule[0]}")
    if is_navigate:
        _nav_all_dirs = [
            ([0, 1, 0],  "FORWARD"),
            ([0, -1, 0], "BACKWARD"),
            ([-1, 0, 0], "STRAFE LEFT"),
            ([1, 0, 0],  "STRAFE RIGHT"),
            ([0, 0, 1],  "TURN LEFT (CCW)"),
            ([0, 0, -1], "TURN RIGHT (CW)"),
        ]
        reps = max(1, args.num_demos // 6)
        _nav_dir_schedule = []
        for cmd, label in _nav_all_dirs:
            _nav_dir_schedule.extend([(cmd, label)] * reps)
        # 첫 에피소드 방향 강제 지정
        cmd0, label0 = _nav_dir_schedule[0]
        env._direction_cmd[0] = torch.tensor(cmd0, dtype=torch.float32, device=env.device)
        print(f"  [Navigate] direction_cmd: {label0} (schedule 1/{len(_nav_dir_schedule)})")

    episode_obs = []
    episode_actions = []
    episode_active = []
    episode_robot_state = []
    episode_object_pos_w = []   # 물체 절대 world 좌표 (3D)
    episode_object_quat_w = []  # 물체 절대 world 방향 (4D wxyz)
    episode_robot_pos_w = []    # 로봇 절대 world 좌표 (3D)
    episode_robot_quat_w = []   # 로봇 절대 world 방향 (4D wxyz)
    saved_count = 0
    step_count = 0

    # HDF5 공통 attrs 헬퍼
    def _write_hdf5_attrs(hf, obs_dim, skill_name):
        hf.attrs["obs_dim"] = obs_dim
        hf.attrs["action_dim"] = 9
        hf.attrs["max_lin_vel"] = float(max_lin_vel)
        hf.attrs["max_ang_vel"] = float(max_ang_vel)
        hf.attrs["arm_action_scale"] = float(arm_action_scale)
        hf.attrs["arm_action_to_limits"] = bool(arm_action_to_limits)
        if args.dynamics_json:
            hf.attrs["dynamics_json"] = str(os.path.expanduser(args.dynamics_json))
        if args.arm_limit_json:
            hf.attrs["arm_limit_json"] = str(os.path.expanduser(args.arm_limit_json))
            hf.attrs["arm_limit_margin_rad"] = float(args.arm_limit_margin_rad)
        hf.attrs["skill"] = skill_name
        hf.attrs["action_format"] = "v6" if use_v6 else "legacy"
        hf.attrs["physics_grasp_mode"] = bool(physics_grasp_mode)
        hf.attrs["multi_object_mode"] = bool(multi_object_mode)
        if args.object_usd:
            hf.attrs["object_usd"] = str(os.path.expanduser(args.object_usd))
        if args.multi_object_json:
            hf.attrs["multi_object_json"] = str(os.path.expanduser(args.multi_object_json))
        hf.attrs["object_mass"] = float(args.object_mass)
        hf.attrs["object_scale_phys"] = float(args.object_scale_phys)
        if args.gripper_contact_prim_path:
            hf.attrs["gripper_contact_prim_path"] = str(args.gripper_contact_prim_path)

    if is_combined or is_carry:
        os.makedirs("demos", exist_ok=True)

        if is_combined and args.resume:
            # demos/에서 가장 최근 combined_skill2_*.hdf5 찾기
            import glob as _glob
            s2_files = sorted(_glob.glob("demos/combined_skill2_*.hdf5"))
            if not s2_files:
                print("  ERROR: --resume인데 demos/combined_skill2_*.hdf5 파일이 없습니다.")
                sys.exit(1)
            skill2_path = s2_files[-1]  # 가장 최근 (타임스탬프 정렬)
            skill3_path = skill2_path.replace("skill2", "skill3")
            if not os.path.isfile(skill3_path):
                print(f"  ERROR: {skill3_path} 파일이 없습니다.")
                sys.exit(1)
            hdf5_skill2 = h5py.File(skill2_path, "a")
            hdf5_skill3 = h5py.File(skill3_path, "a")
            skill2_saved = sum(1 for k in hdf5_skill2.keys() if k.startswith("episode_"))
            skill3_saved = sum(1 for k in hdf5_skill3.keys() if k.startswith("episode_"))
            print(f"  Resume: {skill2_path}")
            print(f"          Skill-2 {skill2_saved}개, Skill-3 {skill3_saved}개 에피소드에서 이어서 녹화")
        elif is_combined:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            skill2_path = f"demos/combined_skill2_{timestamp}.hdf5"
            skill3_path = f"demos/combined_skill3_{timestamp}.hdf5"
            hdf5_skill2 = h5py.File(skill2_path, "w")
            hdf5_skill3 = h5py.File(skill3_path, "w")
            # ⚠️ Skill3Env combined mode에서 obs는 항상 29D (Skill3Env._get_observations).
            # Phase 1(S2 구간)도 29D로 기록됨. train_resip.py main_combined()에서는
            # Skill2Env(30D)를 사용하므로, S2 학습에 이 데모를 쓰려면 obs 변환 필요:
            #   - S2 30D: [0:21] arm+vel, [21:24] object_rel, [24:26] contact_LR, [26:29] bbox, [29:30] cat
            #   - S3 29D: [0:21] arm+vel, [21:24] dest_rel, [24:25] contact(연속), [25:28] bbox, [28:29] cat
            # S3 데모(combined_skill3_*.hdf5)는 그대로 BC 학습에 사용 가능.
            # S2 데모는 별도 env(Skill2Env)에서 수집하거나, 기존 S2 데모 사용 권장.
            _write_hdf5_attrs(hdf5_skill2, 29, "approach_and_grasp")  # 실제 29D 기록
            _write_hdf5_attrs(hdf5_skill3, 29, "carry_and_place")
            skill2_saved = 0
            skill3_saved = 0

        if is_carry:
            # Carry mode: 단일 HDF5 파일 (carry 에피소드만 기록)
            if args.resume and os.path.isfile(output_path):
                hdf5_file = h5py.File(output_path, "a")
                existing = sum(1 for k in hdf5_file.keys() if k.startswith("episode_"))
                saved_count = existing
                print(f"  Resume: 기존 {existing}개 에피소드 발견, episode_{existing}부터 이어서 녹화")
            else:
                hdf5_file = h5py.File(output_path, "w")
                _write_hdf5_attrs(hdf5_file, int(obs["policy"].shape[-1]), "carry")
        else:
            hdf5_file = None  # combined: 단일 파일 미사용

        # S2 expert 로드 (Phase 1 자동 실행)
        s2_expert_mode = bool(str(args.s2_bc_checkpoint).strip())
        s2_dp, s2_rpol, s2_scale = None, None, None
        if s2_expert_mode:
            from diffusion_policy import DiffusionPolicyAgent, ResidualPolicy
            _dev = env.device
            # S2 BC
            _ck = torch.load(args.s2_bc_checkpoint, map_location=_dev, weights_only=False)
            _c = _ck["config"]
            s2_dp = DiffusionPolicyAgent(
                obs_dim=_c["obs_dim"], act_dim=_c["act_dim"],
                pred_horizon=_c["pred_horizon"], action_horizon=_c["action_horizon"],
                num_diffusion_iters=_c["num_diffusion_iters"],
                inference_steps=_c.get("inference_steps", 16),
                down_dims=_c.get("down_dims", [256, 512, 1024]),
            ).to(_dev)
            _sd = _ck["model_state_dict"]
            s2_dp.model.load_state_dict({k[6:]: v for k, v in _sd.items() if k.startswith("model.")})
            s2_dp.normalizer.load_state_dict({k[11:]: v for k, v in _sd.items() if k.startswith("normalizer.")})
            s2_dp.eval(); s2_dp.inference_steps = 4
            for p in s2_dp.parameters(): p.requires_grad = False
            print(f"  [S2 Expert] BC loaded: {args.s2_bc_checkpoint}")
            # S2 ResiP (optional)
            if str(args.s2_resip_checkpoint).strip() and os.path.isfile(args.s2_resip_checkpoint):
                _rck = torch.load(args.s2_resip_checkpoint, map_location=_dev, weights_only=False)
                s2_rpol = ResidualPolicy(
                    obs_dim=_c["obs_dim"], action_dim=_c["act_dim"],
                    actor_hidden_size=256, actor_num_layers=2,
                    init_logstd=-1.0, action_head_std=0.0,
                    action_scale=0.1, learn_std=True,
                ).to(_dev)
                s2_rpol.load_state_dict(_rck["residual_policy_state_dict"])
                s2_rpol.eval()
                for p in s2_rpol.parameters(): p.requires_grad = False
                print(f"  [S2 Expert] ResiP loaded: {args.s2_resip_checkpoint}")
            s2_scale = torch.zeros(_c["act_dim"], device=_dev)
            s2_scale[0:5] = 0.20; s2_scale[5] = 0.25; s2_scale[6:9] = 0.35
            # S2 lift 감지용
            s2_lift_counter = 0
            S2_LIFT_HOLD = 400
            print(f"  [S2 Expert] Phase 1 자동 실행 (lift hold {S2_LIFT_HOLD} steps)")

        # Phase tracking: 1=Skill-2(기록/carry:미기록), 2=Transit(미기록)/carry:carry텔레옵, 3=Skill-3(기록)
        import random as _rnd
        current_phase = 1
        grasp_hold_counter = 0; s2_lift_counter = 0
        carry_total_steps = 0
        carry_arm_start = S3_ARM_START.copy()
        carry_grip_start = S3_GRIP_START
        env._s3_transition_dist = _rnd.uniform(0.6, 0.9)
        phase1_obs, phase1_actions, phase1_active, phase1_robot_state = [], [], [], []
        phase1_object_pos_w, phase1_object_quat_w, phase1_robot_pos_w, phase1_robot_quat_w = [], [], [], []
        phase3_obs, phase3_actions, phase3_active, phase3_robot_state = [], [], [], []
        phase3_object_pos_w, phase3_object_quat_w, phase3_robot_pos_w, phase3_robot_quat_w = [], [], [], []
        phase3_dest_pos_w = []
        if is_combined:
            print(f"  Combined mode: Skill-2 -> Transit -> Skill-3 연속 레코딩")
            print(f"    grasp_hold_steps: {args.grasp_hold_steps} ({args.grasp_hold_steps/60:.1f}s)")
            print(f"    home_dist_thresh: {args.home_dist_thresh}m (Phase 2->3 전환)")
            print(f"    home_fov_thresh: {args.home_fov_thresh:.2f}rad (Phase 2->3 전환)")
            print(f"    grasp_gripper_threshold: {env.cfg.grasp_gripper_threshold}")
            print(f"    Skill-2 output: {skill2_path}")
            print(f"    Skill-3 output: {skill3_path}")
            print(f"    → (오른쪽 화살표): 현재 Phase 저장/진행")
            print(f"    ← (왼쪽 화살표): 현재 Phase 폐기, 리셋")
        elif is_carry:
            print(f"  Carry mode: S2 expert → lift → carry 텔레옵 (base만)")
            print(f"    carry_interp_steps: {args.carry_interp_steps}")
            print(f"    grasp_gripper_threshold: {env.cfg.grasp_gripper_threshold}")
            print(f"    output: {output_path}")
            print(f"    → (오른쪽 화살표): carry 에피소드 저장")
            print(f"    ← (왼쪽 화살표): carry 에피소드 폐기, 리셋")
    else:
        if args.resume and os.path.isfile(output_path):
            hdf5_file = h5py.File(output_path, "a")
            # 기존 에피소드 수 카운트
            existing = sum(1 for k in hdf5_file.keys() if k.startswith("episode_"))
            saved_count = existing
            print(f"  Resume: 기존 {existing}개 에피소드 발견, episode_{existing}부터 이어서 녹화")
        else:
            hdf5_file = h5py.File(output_path, "w")
            _write_hdf5_attrs(hdf5_file, int(obs["policy"].shape[-1]), str(args.skill))

    # 키보드 비차단 입력 설정
    _setup_keyboard()

    print("  ⏳ 텔레옵 입력 연결 대기 중...")
    resolved_arm_unit: str | None = None

    # robot_state 9D 헬퍼
    def _read_robot_state_9d():
        arm_ps = env.robot.data.joint_pos[0, env.arm_idx].cpu().numpy()
        vx = env.robot.data.root_lin_vel_b[0, 0].item()
        vy = env.robot.data.root_lin_vel_b[0, 1].item()
        wz = env.robot.data.root_ang_vel_b[0, 2].item()
        return np.concatenate([arm_ps, np.array([vx, vy, wz], dtype=np.float32)])

    # action 저장 헬퍼 — gripper 연속값 그대로 유지
    def _save_action(action_np_in):
        return action_np_in.copy()

    # HDF5 에피소드 저장 헬퍼
    def _save_episode(hf, ep_idx, ep_obs, ep_actions, ep_active, ep_rs,
                      ep_object_pos_w=None, ep_object_quat_w=None,
                      ep_robot_pos_w=None, ep_robot_quat_w=None,
                      ep_dest_pos_w=None,
                      direction_cmd=None):
        grp = hf.create_group(f"episode_{ep_idx}")
        grp.create_dataset("obs", data=np.array(ep_obs))
        grp.create_dataset("actions", data=np.array(ep_actions))
        grp.create_dataset("robot_state", data=np.array(ep_rs, dtype=np.float32))
        grp.create_dataset("teleop_active", data=np.array(ep_active, dtype=np.int8))
        # per-step world state (replay + 분석용)
        if ep_object_pos_w and len(ep_object_pos_w) > 0:
            grp.create_dataset("object_pos_w", data=np.array(ep_object_pos_w, dtype=np.float32))
        if ep_object_quat_w and len(ep_object_quat_w) > 0:
            grp.create_dataset("object_quat_w", data=np.array(ep_object_quat_w, dtype=np.float32))
        if ep_robot_pos_w and len(ep_robot_pos_w) > 0:
            grp.create_dataset("robot_pos_w", data=np.array(ep_robot_pos_w, dtype=np.float32))
        if ep_robot_quat_w and len(ep_robot_quat_w) > 0:
            grp.create_dataset("robot_quat_w", data=np.array(ep_robot_quat_w, dtype=np.float32))
        if ep_dest_pos_w and len(ep_dest_pos_w) > 0:
            grp.create_dataset("dest_pos_w", data=np.array(ep_dest_pos_w, dtype=np.float32))
        # 초기 환경 상태 (리스트 첫 원소 = 에피소드 시작 시점)
        if ep_robot_pos_w and len(ep_robot_pos_w) > 0:
            grp.attrs["robot_init_pos"] = ep_robot_pos_w[0]
        if ep_robot_quat_w and len(ep_robot_quat_w) > 0:
            grp.attrs["robot_init_quat"] = ep_robot_quat_w[0]
        if ep_object_pos_w and len(ep_object_pos_w) > 0:
            grp.attrs["object_init_pos"] = ep_object_pos_w[0]
        if ep_object_quat_w and len(ep_object_quat_w) > 0:
            grp.attrs["object_init_quat"] = ep_object_quat_w[0]
        grp.attrs["num_steps"] = len(ep_obs)
        grp.attrs["num_active_steps"] = int(np.sum(np.asarray(ep_active, dtype=np.int32)))
        grp.attrs["success"] = True
        if direction_cmd is not None:
            grp.attrs["direction_cmd"] = direction_cmd
        hf.flush()

    try:
        max_demos = args.num_demos
        while sim_app.is_running() and saved_count < max_demos:
            # 텔레옵 입력 읽기
            arm_pos, body_cmd, is_active = teleop_input.get_latest()
            arm_pos_rad, unit_used = normalize_arm_positions_to_rad(arm_pos, args.arm_input_unit)
            if resolved_arm_unit is None and is_active:
                resolved_arm_unit = unit_used
                print(f"  arm unit resolved: {resolved_arm_unit}")

            # wz 부호 보정 (test.py 호환: wz=-wz)
            body_cmd[2] *= wz_sign

            # 텔레옵 → action 변환
            if is_active:
                action_np = teleop_to_action(
                    arm_pos_rad, body_cmd,
                    max_lin_vel, max_ang_vel, arm_action_scale,
                    arm_action_to_limits=arm_action_to_limits,
                    arm_center=arm_center,
                    arm_half_range=arm_half_range,
                    use_v6=use_v6,
                )
            else:
                action_np = np.zeros(9)

            # Navigate: arm/gripper를 all-zero(스폰 자세)로 고정 (base만 텔레옵)
            if is_navigate:
                # VIVA S1: arm all-zero (스폰 자세) 유지. gripper도 0 (open).
                zero_arm_6 = np.zeros(6, dtype=np.float64)  # [arm0..arm4, grip] = 0
                if arm_action_to_limits and arm_center is not None:
                    action_np[0:6] = (zero_arm_6 - arm_center) / arm_half_range
                else:
                    action_np[0:6] = zero_arm_6 / arm_action_scale

            # Carry: arm 보간 + base 자동 명령 (완전 자동, 텔레옵 불필요)
            if is_carry and current_phase == 2:
                # arm 보간
                t = min(carry_total_steps / args.carry_interp_steps, 1.0)
                arm_target_5 = carry_arm_start  # arm 고정 (보간 없음, navigate처럼)
                grip_target = carry_grip_start  # grip 고정 (보간 없음)
                arm_target_6 = np.concatenate([arm_target_5, [grip_target]])
                if arm_action_to_limits and arm_center is not None:
                    action_np[0:6] = (arm_target_6 - arm_center) / arm_half_range
                else:
                    action_np[0:6] = arm_target_6 / arm_action_scale
                # base 자동 명령 (방향 스케줄)
                _cur_dir = _nav_dir_schedule[saved_count] if saved_count < len(_nav_dir_schedule) else "FORWARD"
                _bvx, _bvy, _bwz = _CARRY_DIR_TO_CMD[_cur_dir]
                # 미세 노이즈 (PID 없음, navigate와 동일하게 고정 명령)
                _noise_lin = np.random.normal(0, 0.005, 2)
                _noise_wz = np.random.normal(0, 0.005)
                action_np[6] = (_bvx + _noise_lin[0]) / max_lin_vel
                action_np[7] = (_bvy + _noise_lin[1]) / max_lin_vel
                action_np[8] = ((_bwz * wz_sign) + _noise_wz) / max_ang_vel

            action = torch.tensor(action_np, dtype=torch.float32, device=env.device).unsqueeze(0)

            # Carry: env terminated 사용 안 함 (Skill2EvalEnv에 topple 없음, 직접 체크)
            # Combined: Phase 1에서만 env terminated 활용
            if is_combined and current_phase == 1:
                _teleop_allow_terminate[0] = True
            else:
                _teleop_allow_terminate[0] = False

            # Combined/Carry Phase 1: S2 expert 자동 실행
            s2_obs_pre = None
            if (is_combined or is_carry) and current_phase == 1:
                if is_carry:
                    # Skill2EvalEnv: obs["policy"]가 곧 30D actor obs
                    s2_obs_pre = obs["policy"]
                else:
                    # Skill3Env: skill2 actor obs를 별도 계산
                    s2_obs_pre = env._compute_skill2_actor_obs()
                if s2_expert_mode and s2_dp is not None:
                    with torch.no_grad():
                        _s2_obs = s2_obs_pre
                        _s2_ba = s2_dp.base_action_normalized(_s2_obs)
                        if s2_rpol is not None:
                            _s2_no = torch.nan_to_num(torch.clamp(
                                s2_dp.normalizer(_s2_obs, "obs", forward=True), -3, 3), nan=0.0)
                            _s2_ro = torch.cat([_s2_no, _s2_ba], dim=-1)
                            _, _, _, _, _s2_ram = s2_rpol.get_action_and_value(_s2_ro)
                            action = s2_dp.normalizer(
                                _s2_ba + torch.clamp(_s2_ram, -1, 1) * s2_scale,
                                "action", forward=False)
                        else:
                            action = s2_dp.normalizer(_s2_ba, "action", forward=False)
                    action_np = action[0].cpu().numpy()

            # 환경 step
            next_obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            # ══════════════════════════════════════════════════
            #  Carry mode — Phase 1(S2 expert) + Phase 2(carry teleop)
            # ══════════════════════════════════════════════════
            if is_carry:
                action_s = _save_action(action_np)
                rs = _read_robot_state_9d()

                if current_phase == 1:
                    # Phase 1: S2 expert 자동 실행 (기록 안 함)
                    _grip_pos = env.robot.data.joint_pos[0, env.gripper_idx].item()
                    _grip_closed = _grip_pos < float(env.cfg.grasp_gripper_threshold)
                    _has_contact = False
                    if env.contact_sensor is not None:
                        _cf = env._contact_force_per_env()[0].item()
                        _has_contact = _cf > float(env.cfg.grasp_contact_threshold)
                    _objZ = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()

                    _eg = bool(env.object_grasped[0].item()) if hasattr(env, 'object_grasped') else (_grip_closed and _has_contact)
                    if _eg and _objZ > 0.05:
                        s2_lift_counter += 1
                    else:
                        s2_lift_counter = 0

                    # S2 Phase 1 실패 감지 (train_resip 동일): topple/nolift 직접 체크
                    _s2_failed = False
                    if _objZ < 0.026 and step_count > 20:
                        print(f"\n  [TOPPLE] S2 phase objZ={_objZ:.3f} step={step_count}")
                        _s2_failed = True
                    elif step_count > 700 and _objZ < 0.04:
                        print(f"\n  [NOLIFT] S2 phase objZ={_objZ:.3f} step={step_count}")
                        _s2_failed = True
                    if _s2_failed:
                        print(f"\n  [S2 FAIL] objZ={_objZ:.3f} step={step_count} — env terminated, 자동 리셋")
                        s2_lift_counter = 0; current_phase = 1; carry_total_steps = 0
                        if s2_expert_mode and s2_dp is not None:
                            s2_dp.reset()
                        obs, info = env.reset()
                        step_count = 0
                        continue

                    if step_count % 25 == 0:
                        print(f"\r  [S2 Expert] step={step_count} grip={_grip_pos:.3f} objZ={_objZ:.3f} lift={s2_lift_counter}/{S2_LIFT_HOLD}", end="", flush=True)

                    if s2_lift_counter >= S2_LIFT_HOLD:
                        # 전환 시점의 실제 arm pose 캡처
                        carry_arm_start = env.robot.data.joint_pos[0, env.arm_idx][:5].cpu().numpy().astype(np.float64)
                        carry_grip_start = env.robot.data.joint_pos[0, env.gripper_idx].item()

                        # grip 범위 필터: 서버 분포 mean=0.276 std=0.038 → 95% [0.20, 0.35]
                        # OOD grip으로 BC 학습하면 전환 시 고장
                        if carry_grip_start < 0.20 or carry_grip_start > 0.35:
                            print(f"\n  [GRIP OOD] grip={carry_grip_start:.3f} not in [0.20, 0.35] — 리셋")
                            s2_lift_counter = 0
                            if s2_expert_mode and s2_dp is not None:
                                s2_dp.reset()
                            obs, info = env.reset()
                            step_count = 0
                            continue

                        _dir_label = _nav_dir_schedule[saved_count] if _nav_dir_schedule and saved_count < len(_nav_dir_schedule) else "?"
                        print(f"\n  >>> Carry: S2 lift 완료, carry 텔레옵 시작 — 방향: {_dir_label}")
                        print(f"      arm_start={[f'{v:+.3f}' for v in carry_arm_start]} grip={carry_grip_start:.3f}")
                        current_phase = 2
                        carry_total_steps = 0
                        s2_lift_counter = 0
                        if s2_expert_mode and s2_dp is not None:
                            s2_dp.reset()
                        env.episode_length_buf[0] = 0

                elif current_phase == 2:
                    # Phase 2: Carry 텔레옵 (기록)
                    carry_total_steps += 1
                    episode_obs.append(obs["policy"][0].cpu().numpy())
                    episode_actions.append(action_s)
                    episode_active.append(bool(is_active))
                    episode_robot_state.append(rs)
                    episode_object_pos_w.append(env.object_rigid.data.root_pos_w[0].cpu().numpy())
                    episode_object_quat_w.append(env.object_rigid.data.root_quat_w[0].cpu().numpy())
                    episode_robot_pos_w.append(env.robot.data.root_pos_w[0].cpu().numpy())
                    episode_robot_quat_w.append(env.robot.data.root_quat_w[0].cpu().numpy())

                    # Drop detection: objZ < 0.05 → 물체 떨어트림 → 자동 폐기+리셋
                    _gp = env.robot.data.joint_pos[0, env.gripper_idx].item()
                    _objZ = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
                    if _objZ < 0.05 and carry_total_steps > 10:
                        print(f"\n  [DROP] objZ={_objZ:.3f} — 물체 낙하, 자동 리셋")
                        episode_obs.clear(); episode_actions.clear()
                        episode_active.clear(); episode_robot_state.clear()
                        episode_object_pos_w.clear(); episode_object_quat_w.clear()
                        episode_robot_pos_w.clear(); episode_robot_quat_w.clear()
                        s2_lift_counter = 0; current_phase = 1; carry_total_steps = 0
                        obs, info = env.reset()
                        step_count = 0
                        continue

                    if step_count % 25 == 0:
                        t_interp = min(carry_total_steps / args.carry_interp_steps, 1.0)
                        conn_str = "ON" if is_active else "OFF"
                        _ap = env.robot.data.joint_pos[0, env.arm_idx][:5].cpu().tolist()
                        _dir_label = _nav_dir_schedule[saved_count] if _nav_dir_schedule and saved_count < len(_nav_dir_schedule) else "?"
                        print(f"  [{conn_str}] Carry [{_dir_label}] | steps={carry_total_steps}/{_CARRY_RECORD_STEPS} interp={t_interp:.2f} | "
                              f"arm3={_ap[3]:+.3f} grip={_gp:.3f} objZ={_objZ:.3f} | saved={saved_count}/{max_demos}")

                    # 600 step 자동 저장
                    if carry_total_steps >= _CARRY_RECORD_STEPS:
                        _save_episode(hdf5_file, saved_count,
                                      episode_obs, episode_actions, episode_active, episode_robot_state,
                                      episode_object_pos_w, episode_object_quat_w,
                                      episode_robot_pos_w, episode_robot_quat_w)
                        saved_count += 1
                        _dir_done = _nav_dir_schedule[saved_count - 1] if _nav_dir_schedule and saved_count - 1 < len(_nav_dir_schedule) else "?"
                        _dir_next = _nav_dir_schedule[saved_count] if _nav_dir_schedule and saved_count < len(_nav_dir_schedule) else "DONE"
                        print(f"\n  [AUTO] Carry 저장 ({carry_total_steps} steps, {_dir_done}) — {saved_count}/{max_demos}")
                        episode_obs.clear(); episode_actions.clear()
                        episode_active.clear(); episode_robot_state.clear()
                        episode_object_pos_w.clear(); episode_object_quat_w.clear()
                        episode_robot_pos_w.clear(); episode_robot_quat_w.clear()
                        s2_lift_counter = 0; current_phase = 1; carry_total_steps = 0
                        if saved_count >= max_demos:
                            break
                        print(f"  [Carry] 다음 방향: {_dir_next}")
                        obs, info = env.reset()
                        step_count = 0
                        continue

                # 수동 종료 (화살표 키)
                key = _check_arrow_key()
                if key == 'left':
                    print(f"\n  [←] Carry 폐기, 리셋")
                    episode_obs.clear(); episode_actions.clear()
                    episode_active.clear(); episode_robot_state.clear()
                    episode_object_pos_w.clear(); episode_object_quat_w.clear()
                    episode_robot_pos_w.clear(); episode_robot_quat_w.clear()
                    s2_lift_counter = 0; current_phase = 1; carry_total_steps = 0
                    obs, info = env.reset()
                    step_count = 0
                elif key == 'right':
                    if current_phase == 2 and len(episode_obs) > 10:
                        _save_episode(hdf5_file, saved_count,
                                      episode_obs, episode_actions, episode_active, episode_robot_state,
                                      episode_object_pos_w, episode_object_quat_w,
                                      episode_robot_pos_w, episode_robot_quat_w)
                        saved_count += 1
                        print(f"\n  [→] Carry 저장 ({len(episode_obs)} steps)")
                        episode_obs.clear(); episode_actions.clear()
                        episode_active.clear(); episode_robot_state.clear()
                        episode_object_pos_w.clear(); episode_object_quat_w.clear()
                        episode_robot_pos_w.clear(); episode_robot_quat_w.clear()
                        s2_lift_counter = 0; current_phase = 1; carry_total_steps = 0
                        obs, info = env.reset()
                        step_count = 0
                        if saved_count >= max_demos:
                            break
                    else:
                        print(f"\n  [→] Carry: Phase {current_phase}, steps={len(episode_obs)} — 아직 기록 부족")
                else:
                    obs = next_obs

            # ══════════════════════════════════════════════════
            #  Combined mode — phase-aware recording
            # ══════════════════════════════════════════════════
            elif is_combined:
                action_s = _save_action(action_np)
                rs = _read_robot_state_9d()

                # 목적지(dest_object/home) 거리/각도 계산 (Phase 2/3 전환 + 상태 출력용)
                from isaaclab.utils.math import quat_apply_inverse
                _dest_pos = getattr(env, _dest_attr)
                dest_delta_w = _dest_pos[0:1] - env.robot.data.root_pos_w[0:1]
                dest_rel_b = quat_apply_inverse(env.robot.data.root_quat_w[0:1], dest_delta_w)[0]
                home_dist = torch.norm(dest_rel_b[:2]).item()
                heading_to_home = math.atan2(dest_rel_b[0].item(), dest_rel_b[1].item())  # +y forward

                if current_phase == 1:
                    # Phase 1: Skill-2 (30D obs) 레코딩 — PRE-step obs 사용
                    phase1_obs.append(s2_obs_pre[0].cpu().numpy())
                    phase1_actions.append(action_s)
                    phase1_active.append(bool(is_active))
                    phase1_robot_state.append(rs)
                    phase1_object_pos_w.append(env.object_pos_w[0].cpu().numpy())
                    phase1_object_quat_w.append(env.object_rigid.data.root_quat_w[0].cpu().numpy())
                    phase1_robot_pos_w.append(env.robot.data.root_pos_w[0].cpu().numpy())
                    phase1_robot_quat_w.append(env.robot.data.root_quat_w[0].cpu().numpy())

                    # 파지 유지 카운터
                    _grip_pos = env.robot.data.joint_pos[0, env.gripper_idx].item()
                    _grip_closed = _grip_pos < float(env.cfg.grasp_gripper_threshold)
                    _has_contact = False
                    if env.contact_sensor is not None:
                        _cf = env._contact_force_per_env()[0].item()
                        _has_contact = _cf > float(env.cfg.grasp_contact_threshold)
                    _objZ = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()

                    if s2_expert_mode:
                        # Expert 모드: lift 감지 (grasped + objZ > 0.05)
                        _eg = bool(env.object_grasped[0].item()) if hasattr(env, 'object_grasped') else (_grip_closed and _has_contact)
                        if _eg and _objZ > 0.05:
                            s2_lift_counter += 1
                        else:
                            s2_lift_counter = 0
                        if step_count % 25 == 0:
                            print(f"\r  [S2 Expert] step={step_count} grip={_grip_pos:.3f} objZ={_objZ:.3f} lift={s2_lift_counter}/{S2_LIFT_HOLD}", end="", flush=True)
                        _phase1_done = s2_lift_counter >= S2_LIFT_HOLD
                    else:
                        # 텔레옵 모드: contact + gripper 유지
                        if _grip_closed and _has_contact:
                            grasp_hold_counter += 1
                        else:
                            grasp_hold_counter = 0; s2_lift_counter = 0
                        _phase1_done = grasp_hold_counter >= args.grasp_hold_steps

                    # Phase 1→2 전환
                    if _phase1_done:
                        _save_episode(hdf5_skill2, skill2_saved,
                                      phase1_obs, phase1_actions, phase1_active, phase1_robot_state,
                                      phase1_object_pos_w, phase1_object_quat_w,
                                      phase1_robot_pos_w, phase1_robot_quat_w)
                        skill2_saved += 1
                        print(f"\n  >>> Phase 1->2: Skill-2 저장 ({len(phase1_obs)} steps), Transit 시작")
                        phase1_obs.clear(); phase1_actions.clear()
                        phase1_active.clear(); phase1_robot_state.clear()
                        phase1_object_pos_w.clear(); phase1_object_quat_w.clear()
                        phase1_robot_pos_w.clear(); phase1_robot_quat_w.clear()
                        current_phase = 2
                        grasp_hold_counter = 0; s2_lift_counter = 0
                        if s2_expert_mode and s2_dp is not None:
                            s2_dp.reset()
                            # S2 마지막 arm action 저장 (Phase 2에서 arm 고정용)
                            env._s2_last_arm_action = action[0, :6].clone()
                        env.episode_length_buf[0] = 0  # Transit용 타이머 리셋

                elif current_phase == 2:
                    # Phase 2: Transit (미기록) — home 근처로 이동
                    env.episode_length_buf[0] = 0  # timeout 방지
                    # Expert 모드: arm을 S2 마지막 action으로 고정, base만 텔레옵
                    if s2_expert_mode and hasattr(env, '_s2_last_arm_action'):
                        action[0, :6] = env._s2_last_arm_action
                    # 실시간 arm delta 표시 (10 step마다)
                    if step_count % 10 == 0:
                        _S2M = [-0.040, -0.193, 0.275, -1.280, -0.035]
                        _ap = env.robot.data.joint_pos[0, env.arm_idx][:5].cpu().tolist()
                        _gp = env.robot.data.joint_pos[0, env.gripper_idx].item()
                        _d = [a - m for a, m in zip(_ap, _S2M)]
                        print(f"\r  [Transit] dist={home_dist:.2f}m arm3={_ap[3]:+.3f}(Δ{_d[3]:+.3f}) grip={_gp:.3f}", end="", flush=True)

                    # Phase 2→3 전환: home 근접 + FOV 내 (에피소드별 랜덤 거리)
                    if not hasattr(env, '_s3_transition_dist'):
                        import random as _rnd
                        env._s3_transition_dist = _rnd.uniform(0.6, 0.9)
                    close_enough = home_dist < env._s3_transition_dist
                    in_fov = abs(heading_to_home) < args.home_fov_thresh
                    if close_enough and in_fov:
                        # S3 전환 시 arm 상태 출력 (S2 전환 mean과 비교)
                        _S2_MEAN = [-0.040, -0.193, 0.275, -1.280, -0.035]  # 서버 89K 샘플 평균
                        _S2_GRIP_MEAN = 0.276
                        _arm_pos = env.robot.data.joint_pos[0, env.arm_idx][:5].cpu().tolist()
                        _grip_pos = env.robot.data.joint_pos[0, env.gripper_idx].item()
                        _delta = [a - m for a, m in zip(_arm_pos, _S2_MEAN)]
                        print(f"\n  >>> Phase 2->3: Transit 완료 "
                              f"(home={home_dist:.2f}m, heading={heading_to_home:+.2f}rad, "
                              f"transition_dist={env._s3_transition_dist:.2f}m)")
                        print(f"      arm_pos={[f'{v:+.3f}' for v in _arm_pos]} grip={_grip_pos:.3f}")
                        print(f"      delta  ={[f'{v:+.3f}' for v in _delta]} grip_delta={_grip_pos - _S2_GRIP_MEAN:+.3f}")
                        print(f"      Skill-3 기록 시작")
                        current_phase = 3
                        env.episode_length_buf[0] = 0  # Skill-3용 타이머 리셋

                elif current_phase == 3:
                    # Phase 3: Skill-3 (29D obs) 레코딩
                    phase3_obs.append(obs["policy"][0].cpu().numpy())
                    phase3_actions.append(action_s)
                    phase3_active.append(bool(is_active))
                    phase3_robot_state.append(rs)
                    # 약병: 실제 물리 위치 (object_pos_w는 gripper 추종 텔레포트 값)
                    phase3_object_pos_w.append(env.object_rigid.data.root_pos_w[0].cpu().numpy())
                    phase3_object_quat_w.append(env.object_rigid.data.root_quat_w[0].cpu().numpy())
                    phase3_robot_pos_w.append(env.robot.data.root_pos_w[0].cpu().numpy())
                    phase3_robot_quat_w.append(env.robot.data.root_quat_w[0].cpu().numpy())
                    # 컵: dest object 위치 (물리 body, 50kg)
                    phase3_dest_pos_w.append(env.dest_object_pos_w[0].cpu().numpy())

                # 상태 출력 (+ grasp 디버그 정보)
                if step_count % 25 == 0:
                    grasped = bool(env.object_grasped[0].item())
                    grip_sim = env.robot.data.joint_pos[0, env.gripper_idx].item()
                    conn = "ON" if is_active else "OFF"
                    phase_names = {1: "Approach+Grasp", 2: "Transit", 3: "Carry+Place"}
                    extra = ""
                    if current_phase == 1:
                        hold_pct = grasp_hold_counter / args.grasp_hold_steps * 100
                        # Grasp 조건 디버그
                        g_closed = grip_sim < float(env.cfg.grasp_gripper_threshold)
                        cf = env._contact_force_per_env()[0].item()
                        g_contact = cf > float(env.cfg.grasp_contact_threshold)
                        # object_dist 직접 계산 (base→object XY)
                        from isaaclab.utils.math import quat_apply_inverse as _qai
                        _od_w = env.object_pos_w[0:1] - env.robot.data.root_pos_w[0:1]
                        _od_b = _qai(env.robot.data.root_quat_w[0:1], _od_w)[0]
                        od = float(torch.norm(_od_b[:2]).item())
                        bbox_max = env.object_bbox.max(dim=-1).values[0].item()
                        ad = min(max(float(env.cfg.grasp_max_object_dist) + bbox_max * 0.5, 0.10), 0.60)
                        g_close = od < ad
                        extra = (
                            f"hold={grasp_hold_counter}/{args.grasp_hold_steps}({hold_pct:.0f}%)\n"
                            f"    grasp: grip={grip_sim:.3f}({'O' if g_closed else 'X'}) "
                            f"contact={cf:.2f}({'O' if g_contact else 'X'}) "
                            f"dist={od:.3f}/{ad:.3f}({'O' if g_close else 'X'})"
                        )
                    elif current_phase == 2:
                        extra = f"heading={heading_to_home:+.2f}rad"
                    elif current_phase == 3:
                        extra = f"steps={len(phase3_obs)}"
                    obj_z = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
                    print(
                        f"  [{conn}] Phase-{current_phase}({phase_names[current_phase]}) | "
                        f"home={home_dist:.2f}m | "
                        f"grip={grip_sim:+.3f} {'GRASPED' if grasped else ''} | "
                        f"objZ={obj_z:.3f} | "
                        f"{extra} | "
                        f"s2={skill2_saved} s3={skill3_saved}/{max_demos}"
                    )

                # 수동 종료 (화살표 키)
                key = _check_arrow_key()

                if key == 'left':
                    # ← : 현재 phase 폐기, 리셋
                    print(f"\n  [←] Phase {current_phase} 폐기, 리셋")
                    phase1_obs.clear(); phase1_actions.clear()
                    phase1_active.clear(); phase1_robot_state.clear()
                    phase1_object_pos_w.clear(); phase1_object_quat_w.clear()
                    phase1_robot_pos_w.clear(); phase1_robot_quat_w.clear()
                    phase3_obs.clear(); phase3_actions.clear()
                    phase3_active.clear(); phase3_robot_state.clear()
                    phase3_object_pos_w.clear(); phase3_object_quat_w.clear(); phase3_dest_pos_w.clear()
                    phase3_robot_pos_w.clear(); phase3_robot_quat_w.clear()
                    grasp_hold_counter = 0; s2_lift_counter = 0
                    current_phase = 1
                    env._s3_transition_dist = _rnd.uniform(0.6, 0.9)  # 새 에피소드 랜덤 거리
                    obs, info = env.reset()
                    if _dest_marker is not None:
                        _hm = getattr(env, _dest_attr)[:1].clone(); _hm[:, 2] = 0.08
                        _dest_marker.visualize(translations=_hm)
                    step_count = 0
                elif key == 'right':
                    if current_phase == 1:
                        grasped_now = bool(env.object_grasped[0].item())
                        if grasped_now and len(phase1_obs) > 10:
                            _save_episode(hdf5_skill2, skill2_saved,
                                          phase1_obs, phase1_actions, phase1_active, phase1_robot_state,
                                          phase1_object_pos_w, phase1_object_quat_w,
                                          phase1_robot_pos_w, phase1_robot_quat_w)
                            skill2_saved += 1
                            print(f"\n  [→] Phase 1 수동 완료: Skill-2 저장 ({len(phase1_obs)} steps), Transit 시작")
                            phase1_obs.clear(); phase1_actions.clear()
                            phase1_active.clear(); phase1_robot_state.clear()
                            phase1_object_pos_w.clear(); phase1_object_quat_w.clear()
                            phase1_robot_pos_w.clear(); phase1_robot_quat_w.clear()
                            current_phase = 2
                            grasp_hold_counter = 0; s2_lift_counter = 0
                            env.episode_length_buf[0] = 0
                        else:
                            print(f"\n  [→] Phase 1: 파지 미완료 (grasped={grasped_now}, steps={len(phase1_obs)}) — 폐기, 리셋")
                            phase1_obs.clear(); phase1_actions.clear()
                            phase1_active.clear(); phase1_robot_state.clear()
                            phase1_object_pos_w.clear(); phase1_object_quat_w.clear()
                            phase1_robot_pos_w.clear(); phase1_robot_quat_w.clear()
                            grasp_hold_counter = 0; s2_lift_counter = 0
                            current_phase = 1
                            env._s3_transition_dist = _rnd.uniform(0.6, 0.9)
                            obs, info = env.reset()
                            if _dest_marker is not None:
                                _hm = getattr(env, _dest_attr)[:1].clone(); _hm[:, 2] = 0.08
                                _dest_marker.visualize(translations=_hm)
                            step_count = 0
                    elif current_phase == 2:
                        print(f"\n  [→] Phase 2 수동 완료: Transit 건너뜀, Skill-3 기록 시작")
                        current_phase = 3
                        env.episode_length_buf[0] = 0
                    elif current_phase == 3:
                        active_s = int(np.sum(np.asarray(phase3_active, dtype=np.int32)))
                        if len(phase3_obs) > 10 and active_s > 10:
                            _save_episode(hdf5_skill3, skill3_saved,
                                          phase3_obs, phase3_actions, phase3_active, phase3_robot_state,
                                          phase3_object_pos_w, phase3_object_quat_w,
                                          phase3_robot_pos_w, phase3_robot_quat_w,
                                          ep_dest_pos_w=phase3_dest_pos_w)
                            skill3_saved += 1
                            print(f"\n  [→] Phase 3 수동 완료: Skill-3 저장 ({len(phase3_obs)} steps)")
                        else:
                            print(f"\n  [→] Phase 3: steps 부족 ({len(phase3_obs)}, active={active_s}) — 폐기")
                        phase3_obs.clear(); phase3_actions.clear()
                        phase3_active.clear(); phase3_robot_state.clear()
                        phase3_object_pos_w.clear(); phase3_object_quat_w.clear(); phase3_dest_pos_w.clear()
                        phase3_robot_pos_w.clear(); phase3_robot_quat_w.clear()
                        current_phase = 1
                        grasp_hold_counter = 0; s2_lift_counter = 0
                        env._s3_transition_dist = _rnd.uniform(0.6, 0.9)
                        obs, info = env.reset()
                        if _dest_marker is not None:
                            _hm = getattr(env, _dest_attr)[:1].clone(); _hm[:, 2] = 0.08
                            _dest_marker.visualize(translations=_hm)
                        step_count = 0
                        saved_count = min(skill2_saved, skill3_saved)
                        if saved_count >= max_demos:
                            break
                else:
                    obs = next_obs

            # ══════════════════════════════════════════════════
            #  Single-skill: Navigate 자동 수집
            # ══════════════════════════════════════════════════
            elif is_navigate:
                _NAV_RECORD_STEPS = 600
                _NAV_REST_STEPS = 100

                episode_obs.append(obs["policy"][0].cpu().numpy())
                episode_actions.append(_save_action(action_np))
                episode_active.append(bool(is_active))
                episode_robot_state.append(_read_robot_state_9d())
                episode_robot_pos_w.append(env.robot.data.root_pos_w[0].cpu().numpy())
                episode_robot_quat_w.append(env.robot.data.root_quat_w[0].cpu().numpy())

                ep_steps = len(episode_obs)

                # 상태 출력
                if step_count % 25 == 0:
                    dir_label = _nav_dir_label(env._direction_cmd[0])
                    conn_str = "ON" if is_active else "OFF"
                    root_pos = env.robot.data.root_pos_w[0, :2].cpu().numpy()
                    vx = env.robot.data.root_lin_vel_b[0, 0].item()
                    vy = env.robot.data.root_lin_vel_b[0, 1].item()
                    wz = env.robot.data.root_ang_vel_b[0, 2].item()
                    print(
                        f"  [{conn_str}] | "
                        f"pos=({root_pos[0]:+.2f},{root_pos[1]:+.2f}) | "
                        f"dir={dir_label} | "
                        f"vel=(vx={vx:+.2f},vy={vy:+.2f},wz={wz:+.2f}) | "
                        f"steps={ep_steps}/{_NAV_RECORD_STEPS} | "
                        f"saved={saved_count}/{max_demos}"
                    )

                # 600 step 도달 → 자동 저장 + 100 step 쉬기 + 다음 방향
                if ep_steps >= _NAV_RECORD_STEPS:
                    nav_cmd = env._direction_cmd[0].cpu().numpy()
                    _save_episode(hdf5_file, saved_count,
                                  episode_obs, episode_actions, episode_active, episode_robot_state,
                                  episode_object_pos_w, episode_object_quat_w,
                                  episode_robot_pos_w, episode_robot_quat_w,
                                  direction_cmd=nav_cmd)
                    saved_count += 1
                    dir_label = _nav_dir_label(env._direction_cmd[0])
                    print(f"\n  [AUTO] Episode {saved_count}/{max_demos} 저장 ({ep_steps} steps, {dir_label})")

                    episode_obs.clear(); episode_actions.clear()
                    episode_active.clear(); episode_robot_state.clear()
                    episode_object_pos_w.clear(); episode_object_quat_w.clear()
                    episode_robot_pos_w.clear(); episode_robot_quat_w.clear()

                    if saved_count >= max_demos:
                        break

                    # 100 step 쉬기 (env step만 진행, 기록 안 함)
                    print(f"  [REST] {_NAV_REST_STEPS} steps 대기중...")
                    for _ in range(_NAV_REST_STEPS):
                        zero_action = torch.zeros(1, env.action_space.shape[-1], device=env.device)
                        env.step(zero_action)

                    # 리셋 + 다음 방향
                    obs, info = env.reset()
                    step_count = 0
                    if _nav_dir_schedule is not None and saved_count < len(_nav_dir_schedule):
                        cmd_next, label_next = _nav_dir_schedule[saved_count]
                        env._direction_cmd[0] = torch.tensor(cmd_next, dtype=torch.float32, device=env.device)
                        print(f"  [Navigate] direction_cmd: {label_next} (schedule {saved_count+1}/{len(_nav_dir_schedule)})")
                    else:
                        print(f"  [Navigate] direction_cmd: {_nav_dir_label(env._direction_cmd[0])}")
                else:
                    obs = next_obs

            # ══════════════════════════════════════════════════
            #  Single-skill mode — 기존 로직 (navigate 이외)
            # ══════════════════════════════════════════════════
            else:
                episode_obs.append(obs["policy"][0].cpu().numpy())
                episode_actions.append(_save_action(action_np))
                episode_active.append(bool(is_active))
                episode_robot_state.append(_read_robot_state_9d())
                episode_object_pos_w.append(env.object_pos_w[0].cpu().numpy())
                episode_object_quat_w.append(env.object_rigid.data.root_quat_w[0].cpu().numpy())
                episode_robot_pos_w.append(env.robot.data.root_pos_w[0].cpu().numpy())
                episode_robot_quat_w.append(env.robot.data.root_quat_w[0].cpu().numpy())

                # 상태 출력
                if step_count % 25 == 0:
                    conn_str = "ON" if is_active else "OFF"
                    root_pos = env.robot.data.root_pos_w[0, :2].cpu().numpy()
                    target_pos = env.object_pos_w[0, :2].cpu().numpy()
                    dist = np.linalg.norm(root_pos - target_pos)
                    grip_raw = arm_pos_rad[5] if arm_pos_rad is not None else float('nan')
                    grip_action = action_np[5]
                    grip_sim = env.robot.data.joint_pos[0, env.gripper_idx].item()
                    wr_raw = arm_pos_rad[4] if arm_pos_rad is not None else float('nan')
                    wr_sim = env.robot.data.joint_pos[0, env.arm_idx[4]].item()
                    obj_z = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
                    print(
                        f"  [{conn_str}] | "
                        f"pos=({root_pos[0]:+.2f},{root_pos[1]:+.2f}) | "
                        f"obj=({target_pos[0]:+.2f},{target_pos[1]:+.2f}) | "
                        f"dist={dist:.2f}m | objZ={obj_z:.3f} | "
                        f"steps={len(episode_obs)} | "
                        f"saved={saved_count}/{max_demos}\n"
                        f"    grip: raw={grip_raw:+.4f} action={grip_action:+.4f} sim={grip_sim:+.4f}\n"
                        f"    wrist_roll: raw={wr_raw:+.4f} sim={wr_sim:+.4f}"
                    )

                # 수동 종료 (화살표 키)
                key = _check_arrow_key()
                if key == 'right':
                    active_steps = int(np.sum(np.asarray(episode_active, dtype=np.int32)))
                    if len(episode_obs) > 10 and active_steps > 10:
                        _save_episode(hdf5_file, saved_count,
                                      episode_obs, episode_actions, episode_active, episode_robot_state,
                                      episode_object_pos_w, episode_object_quat_w,
                                      episode_robot_pos_w, episode_robot_quat_w)
                        saved_count += 1
                        print(f"\n  [→] Episode {saved_count} 저장 ({len(episode_obs)} steps)")
                    else:
                        print(f"\n  [→] steps 부족 ({len(episode_obs)}, active={active_steps}) — 폐기")
                    episode_obs.clear(); episode_actions.clear()
                    episode_active.clear(); episode_robot_state.clear()
                    episode_object_pos_w.clear(); episode_object_quat_w.clear()
                    episode_robot_pos_w.clear(); episode_robot_quat_w.clear()
                    obs, info = env.reset()
                    if _dest_marker is not None:
                        _hm = getattr(env, _dest_attr)[:1].clone(); _hm[:, 2] = 0.08
                        _dest_marker.visualize(translations=_hm)
                    step_count = 0
                    if saved_count >= max_demos:
                        break
                elif key == 'left':
                    print(f"\n  [←] 폐기, 리셋")
                    episode_obs.clear(); episode_actions.clear()
                    episode_active.clear(); episode_robot_state.clear()
                    episode_object_pos_w.clear(); episode_object_quat_w.clear()
                    episode_robot_pos_w.clear(); episode_robot_quat_w.clear()
                    obs, info = env.reset()
                    if _dest_marker is not None:
                        _hm = getattr(env, _dest_attr)[:1].clone(); _hm[:, 2] = 0.08
                        _dest_marker.visualize(translations=_hm)
                    step_count = 0
                else:
                    obs = next_obs

    except KeyboardInterrupt:
        print("\n\n  중단됨 (Ctrl+C)")
    finally:
        _restore_keyboard()

    # —— 마무리 ——
    if is_combined:
        hdf5_skill2.close()
        hdf5_skill3.close()
        print(f"\n" + "=" * 60)
        print(f"  Combined 녹화 완료")
        print(f"  Skill-2 에피소드: {skill2_saved} -> {skill2_path}")
        print(f"  Skill-3 에피소드: {skill3_saved} -> {skill3_path}")
        print("=" * 60)
    elif is_carry:
        hdf5_file.close()
        print(f"\n" + "=" * 60)
        print(f"  Carry 녹화 완료")
        print(f"  저장된 에피소드: {saved_count}")
        print(f"  파일: {output_path}")
        print("=" * 60)
    else:
        hdf5_file.close()
        print(f"\n" + "=" * 60)
        print(f"  녹화 완료")
        print(f"  저장된 에피소드: {saved_count}")
        print(f"  파일: {output_path}")
        print(f"\n  다음 단계:")
        print(f"    python train_bc.py --demo_dir demos/ --epochs 200")
        print("=" * 60)

    teleop_input.shutdown()
    if selected_source == "ros2":
        teleop_input.destroy_node()
        rclpy.shutdown()

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
