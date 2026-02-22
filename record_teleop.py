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
parser.add_argument("--multi_object_json", type=str, default="",
                    help="multi-object catalog JSON path (37D obs)")
parser.add_argument("--object_mass", type=float, default=0.3,
                    help="physics grasp object mass (kg)")
parser.add_argument("--object_scale_phys", type=float, default=1.0,
                    help="physics grasp object uniform scale")
parser.add_argument("--gripper_contact_prim_path", type=str, default="",
                    help="contact sensor prim path for gripper body (required in multi-object mode)")
parser.add_argument("--grasp_gripper_threshold", type=float, default=0.7,
                    help="gripper joint position threshold for closed state")
parser.add_argument("--grasp_contact_threshold", type=float, default=0.5,
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
    choices=["approach_and_grasp", "carry_and_place", "combined", "legacy"],
    help="환경 모드: approach_and_grasp(Skill-2), carry_and_place(Skill-3), combined(Skill-2→3 연속), legacy(v8 FSM)",
)
parser.add_argument(
    "--grasp_hold_steps",
    type=int,
    default=600,
    help="combined mode: 파지 유지 스텝 수 (600 = 10s @ 60Hz)",
)
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
args.headless = False  # 텔레옵은 GUI 필수
args.num_envs = 1

launcher = AppLauncher(args)
sim_app = launcher.app

# —— 나머지 import ——
import h5py
import numpy as np
import torch

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
            body_cmd = self._base_cmd.copy()
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
    use_v6 = args.skill != "legacy"

    if args.skill == "approach_and_grasp":
        from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg
        env_cfg = Skill2EnvCfg()
    elif args.skill in ("carry_and_place", "combined"):
        from lekiwi_skill3_env import Skill3Env, Skill3EnvCfg
        env_cfg = Skill3EnvCfg()
    else:
        from lekiwi_nav_env import LeKiwiNavEnv, LeKiwiNavEnvCfg
        env_cfg = LeKiwiNavEnvCfg()

    env_cfg.scene.num_envs = 1
    # 텔레옵은 에피소드 길이 충분히 확보 (수동 종료 사용, 1시간)
    env_cfg.episode_length_s = 3600.0
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
    if physics_grasp_mode:
        env_cfg.object_usd = os.path.expanduser(args.object_usd)
        env_cfg.object_mass = float(args.object_mass)
        env_cfg.object_scale = float(args.object_scale_phys)
        env_cfg.gripper_contact_prim_path = str(args.gripper_contact_prim_path)
        env_cfg.grasp_gripper_threshold = float(args.grasp_gripper_threshold)
        env_cfg.grasp_contact_threshold = float(args.grasp_contact_threshold)
        env_cfg.grasp_max_object_dist = float(args.grasp_max_object_dist)
        env_cfg.grasp_attach_height = float(args.grasp_attach_height)
    if multi_object_mode and not str(args.gripper_contact_prim_path).strip():
        raise ValueError(
            "multi-object(37D) 텔레옵 데모에는 --gripper_contact_prim_path가 필요합니다."
        )
    is_combined = (args.skill == "combined")
    if is_combined:
        env_cfg.grasp_gripper_threshold = 0.5  # combined: 확실한 파지만 인정
    if args.skill == "approach_and_grasp":
        env = Skill2Env(cfg=env_cfg)
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
    # grasp state 업데이트는 유지하되, terminated/truncated는 항상 False
    _original_get_dones = env._get_dones

    def _teleop_get_dones():
        terminated, truncated = _original_get_dones()
        terminated[:] = False
        truncated[:] = False
        return terminated, truncated

    env._get_dones = _teleop_get_dones

    # Home 위치 마커 (초록 구체)
    _home_marker = None
    if hasattr(env, "home_pos_w"):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils
        _home_marker = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/World/Visuals/home_marker",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.08,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
            },
        ))
        _hm = env.home_pos_w[:1].clone(); _hm[:, 2] = 0.08
        _home_marker.visualize(translations=_hm)

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
    if _home_marker is not None:
        _hm = env.home_pos_w[:1].clone(); _hm[:, 2] = 0.08
        _home_marker.visualize(translations=_hm)

    # 디버그: 물체 스폰 확인
    if hasattr(env, 'object_rigid') and env.object_rigid is not None:
        obj_pos = env.object_rigid.data.root_pos_w[0].cpu().numpy()
        print(f"  [DEBUG] object_rigid pos: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")
        print(f"  [DEBUG] object_pos_w: {env.object_pos_w[0].cpu().numpy()}")
        print(f"  [DEBUG] object_rigid num_instances: {env.object_rigid.num_instances}")
    else:
        print(f"  [DEBUG] object_rigid: None (physics_grasp={getattr(env, '_physics_grasp', '?')})")

    episode_obs = []
    episode_actions = []
    episode_active = []
    episode_robot_state = []
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

    if is_combined:
        os.makedirs("demos", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        skill2_path = f"demos/combined_skill2_{timestamp}.hdf5"
        skill3_path = f"demos/combined_skill3_{timestamp}.hdf5"
        hdf5_skill2 = h5py.File(skill2_path, "w")
        hdf5_skill3 = h5py.File(skill3_path, "w")
        _write_hdf5_attrs(hdf5_skill2, 30, "approach_and_grasp")
        _write_hdf5_attrs(hdf5_skill3, 29, "carry_and_place")
        hdf5_file = None  # 단일 파일 미사용

        # Phase tracking: 1=Skill-2(기록), 2=Transit(미기록), 3=Skill-3(기록)
        current_phase = 1
        grasp_hold_counter = 0
        skill2_saved = 0
        skill3_saved = 0
        phase1_obs, phase1_actions, phase1_active, phase1_robot_state = [], [], [], []
        phase3_obs, phase3_actions, phase3_active, phase3_robot_state = [], [], [], []
        print(f"  Combined mode: Skill-2 -> Transit -> Skill-3 연속 레코딩")
        print(f"    grasp_hold_steps: {args.grasp_hold_steps} ({args.grasp_hold_steps/60:.1f}s)")
        print(f"    home_dist_thresh: {args.home_dist_thresh}m (Phase 2->3 전환)")
        print(f"    home_fov_thresh: {args.home_fov_thresh:.2f}rad (Phase 2->3 전환)")
        print(f"    grasp_gripper_threshold: {env.cfg.grasp_gripper_threshold}")
        print(f"    Skill-2 output: {skill2_path}")
        print(f"    Skill-3 output: {skill3_path}")
        print(f"    → (오른쪽 화살표): 현재 Phase 저장/진행")
        print(f"    ← (왼쪽 화살표): 현재 Phase 폐기, 리셋")
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

    # action 저장 헬퍼
    def _save_action(action_np_in):
        a = action_np_in.copy()
        if use_v6:
            a[5] = 1.0 if a[5] > 0.5 else 0.0
        return a

    # HDF5 에피소드 저장 헬퍼
    def _save_episode(hf, ep_idx, ep_obs, ep_actions, ep_active, ep_rs):
        grp = hf.create_group(f"episode_{ep_idx}")
        grp.create_dataset("obs", data=np.array(ep_obs))
        grp.create_dataset("actions", data=np.array(ep_actions))
        grp.create_dataset("robot_state", data=np.array(ep_rs, dtype=np.float32))
        grp.create_dataset("teleop_active", data=np.array(ep_active, dtype=np.int8))
        grp.attrs["num_steps"] = len(ep_obs)
        grp.attrs["num_active_steps"] = int(np.sum(np.asarray(ep_active, dtype=np.int32)))
        grp.attrs["success"] = True
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

            action = torch.tensor(action_np, dtype=torch.float32, device=env.device).unsqueeze(0)

            # 환경 step
            next_obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            # ══════════════════════════════════════════════════
            #  Combined mode — phase-aware recording
            # ══════════════════════════════════════════════════
            if is_combined:
                action_s = _save_action(action_np)
                rs = _read_robot_state_9d()

                # home 거리/각도 계산 (Phase 2/3 전환 + 상태 출력용)
                from isaaclab.utils.math import quat_apply_inverse
                home_delta_w = env.home_pos_w[0:1] - env.robot.data.root_pos_w[0:1]
                home_rel_b = quat_apply_inverse(env.robot.data.root_quat_w[0:1], home_delta_w)[0]
                home_dist = torch.norm(home_rel_b[:2]).item()
                heading_to_home = math.atan2(home_rel_b[0].item(), home_rel_b[1].item())  # +y forward

                if current_phase == 1:
                    # Phase 1: Skill-2 (30D obs) 레코딩
                    s2_obs = env._compute_skill2_actor_obs()
                    phase1_obs.append(s2_obs[0].cpu().numpy())
                    phase1_actions.append(action_s)
                    phase1_active.append(bool(is_active))
                    phase1_robot_state.append(rs)

                    # 파지 유지 카운터
                    if bool(env.object_grasped[0].item()):
                        grasp_hold_counter += 1
                    else:
                        grasp_hold_counter = 0

                    # Phase 1→2 전환: grasp 유지 충분
                    if grasp_hold_counter >= args.grasp_hold_steps:
                        _save_episode(hdf5_skill2, skill2_saved,
                                      phase1_obs, phase1_actions, phase1_active, phase1_robot_state)
                        skill2_saved += 1
                        print(f"\n  >>> Phase 1->2: Skill-2 저장 ({len(phase1_obs)} steps), Transit 시작")
                        phase1_obs.clear(); phase1_actions.clear()
                        phase1_active.clear(); phase1_robot_state.clear()
                        current_phase = 2
                        grasp_hold_counter = 0
                        env.episode_length_buf[0] = 0  # Transit용 타이머 리셋

                elif current_phase == 2:
                    # Phase 2: Transit (미기록) — home 근처로 이동
                    env.episode_length_buf[0] = 0  # timeout 방지

                    # Phase 2→3 전환: home 근접 + FOV 내
                    close_enough = home_dist < args.home_dist_thresh
                    in_fov = abs(heading_to_home) < args.home_fov_thresh
                    if close_enough and in_fov:
                        print(f"\n  >>> Phase 2->3: Transit 완료 "
                              f"(home={home_dist:.2f}m, heading={heading_to_home:+.2f}rad), Skill-3 기록 시작")
                        current_phase = 3
                        env.episode_length_buf[0] = 0  # Skill-3용 타이머 리셋

                elif current_phase == 3:
                    # Phase 3: Skill-3 (29D obs) 레코딩
                    phase3_obs.append(obs["policy"][0].cpu().numpy())
                    phase3_actions.append(action_s)
                    phase3_active.append(bool(is_active))
                    phase3_robot_state.append(rs)

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
                    print(
                        f"  [{conn}] Phase-{current_phase}({phase_names[current_phase]}) | "
                        f"home={home_dist:.2f}m | "
                        f"grip={grip_sim:+.3f} {'GRASPED' if grasped else ''} | "
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
                    phase3_obs.clear(); phase3_actions.clear()
                    phase3_active.clear(); phase3_robot_state.clear()
                    grasp_hold_counter = 0
                    current_phase = 1
                    obs, info = env.reset()
    if _home_marker is not None:
        _hm = env.home_pos_w[:1].clone(); _hm[:, 2] = 0.08
        _home_marker.visualize(translations=_hm)
                    step_count = 0
                elif key == 'right':
                    if current_phase == 1:
                        grasped_now = bool(env.object_grasped[0].item())
                        if grasped_now and len(phase1_obs) > 10:
                            _save_episode(hdf5_skill2, skill2_saved,
                                          phase1_obs, phase1_actions, phase1_active, phase1_robot_state)
                            skill2_saved += 1
                            print(f"\n  [→] Phase 1 수동 완료: Skill-2 저장 ({len(phase1_obs)} steps), Transit 시작")
                            phase1_obs.clear(); phase1_actions.clear()
                            phase1_active.clear(); phase1_robot_state.clear()
                            current_phase = 2
                            grasp_hold_counter = 0
                            env.episode_length_buf[0] = 0
                        else:
                            print(f"\n  [→] Phase 1: 파지 미완료 (grasped={grasped_now}, steps={len(phase1_obs)}) — 폐기, 리셋")
                            phase1_obs.clear(); phase1_actions.clear()
                            phase1_active.clear(); phase1_robot_state.clear()
                            grasp_hold_counter = 0
                            current_phase = 1
                            obs, info = env.reset()
    if _home_marker is not None:
        _hm = env.home_pos_w[:1].clone(); _hm[:, 2] = 0.08
        _home_marker.visualize(translations=_hm)
                            step_count = 0
                    elif current_phase == 2:
                        print(f"\n  [→] Phase 2 수동 완료: Transit 건너뜀, Skill-3 기록 시작")
                        current_phase = 3
                        env.episode_length_buf[0] = 0
                    elif current_phase == 3:
                        active_s = int(np.sum(np.asarray(phase3_active, dtype=np.int32)))
                        if len(phase3_obs) > 10 and active_s > 10:
                            _save_episode(hdf5_skill3, skill3_saved,
                                          phase3_obs, phase3_actions, phase3_active, phase3_robot_state)
                            skill3_saved += 1
                            print(f"\n  [→] Phase 3 수동 완료: Skill-3 저장 ({len(phase3_obs)} steps)")
                        else:
                            print(f"\n  [→] Phase 3: steps 부족 ({len(phase3_obs)}, active={active_s}) — 폐기")
                        phase3_obs.clear(); phase3_actions.clear()
                        phase3_active.clear(); phase3_robot_state.clear()
                        current_phase = 1
                        grasp_hold_counter = 0
                        obs, info = env.reset()
    if _home_marker is not None:
        _hm = env.home_pos_w[:1].clone(); _hm[:, 2] = 0.08
        _home_marker.visualize(translations=_hm)
                        step_count = 0
                        saved_count = min(skill2_saved, skill3_saved)
                        if saved_count >= max_demos:
                            break
                else:
                    obs = next_obs

            # ══════════════════════════════════════════════════
            #  Single-skill mode — 기존 로직
            # ══════════════════════════════════════════════════
            else:
                episode_obs.append(obs["policy"][0].cpu().numpy())
                episode_actions.append(_save_action(action_np))
                episode_active.append(bool(is_active))
                episode_robot_state.append(_read_robot_state_9d())

                # 상태 출력
                if step_count % 25 == 0:
                    root_pos = env.robot.data.root_pos_w[0, :2].cpu().numpy()
                    target_pos = env.object_pos_w[0, :2].cpu().numpy()
                    dist = np.linalg.norm(root_pos - target_pos)
                    conn_str = "ON" if is_active else "OFF"
                    grip_raw = arm_pos_rad[5] if arm_pos_rad is not None else float('nan')
                    grip_action = action_np[5]
                    grip_sim = env.robot.data.joint_pos[0, env.gripper_idx].item()
                    wr_raw = arm_pos_rad[4] if arm_pos_rad is not None else float('nan')
                    wr_sim = env.robot.data.joint_pos[0, env.arm_idx[4]].item()
                    print(
                        f"  [{conn_str}] | "
                        f"pos=({root_pos[0]:+.2f},{root_pos[1]:+.2f}) | "
                        f"obj=({target_pos[0]:+.2f},{target_pos[1]:+.2f}) | "
                        f"dist={dist:.2f}m | "
                        f"steps={len(episode_obs)} | "
                        f"saved={saved_count}/{max_demos}\n"
                        f"    grip: raw={grip_raw:+.4f} action={grip_action:+.4f} sim={grip_sim:+.4f}\n"
                        f"    wrist_roll: raw={wr_raw:+.4f} sim={wr_sim:+.4f}"
                    )

                # 수동 종료 (화살표 키)
                key = _check_arrow_key()
                if key == 'right':
                    # → : 저장 후 리셋
                    active_steps = int(np.sum(np.asarray(episode_active, dtype=np.int32)))
                    if len(episode_obs) > 10 and active_steps > 10:
                        _save_episode(hdf5_file, saved_count,
                                      episode_obs, episode_actions, episode_active, episode_robot_state)
                        saved_count += 1
                        print(f"\n  [→] Episode {saved_count} 저장 ({len(episode_obs)} steps)")
                    else:
                        print(f"\n  [→] steps 부족 ({len(episode_obs)}, active={active_steps}) — 폐기")
                    episode_obs.clear(); episode_actions.clear()
                    episode_active.clear(); episode_robot_state.clear()
                    obs, info = env.reset()
    if _home_marker is not None:
        _hm = env.home_pos_w[:1].clone(); _hm[:, 2] = 0.08
        _home_marker.visualize(translations=_hm)
                    step_count = 0
                    if saved_count >= max_demos:
                        break
                elif key == 'left':
                    # ← : 폐기 후 리셋
                    print(f"\n  [←] 폐기, 리셋")
                    episode_obs.clear(); episode_actions.clear()
                    episode_active.clear(); episode_robot_state.clear()
                    obs, info = env.reset()
    if _home_marker is not None:
        _hm = env.home_pos_w[:1].clone(); _hm[:, 2] = 0.08
        _home_marker.visualize(translations=_hm)
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
