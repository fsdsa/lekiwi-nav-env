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
import socket
import sys
import time
import threading

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
parser.add_argument(
    "--arm_input_unit",
    type=str,
    default="auto",
    choices=["auto", "rad", "deg", "m100"],
    help="teleop arm position unit (auto/rad/deg/m100)",
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

from lekiwi_nav_env import LeKiwiNavEnv, LeKiwiNavEnvCfg
from lekiwi_robot_cfg import (
    ARM_JOINT_NAMES, WHEEL_JOINT_NAMES,
    WHEEL_ANGLES_RAD,
)


# ═══════════════════════════════════════════════════════════════════════
#  Teleop Input (ROS2 / TCP)
# ═══════════════════════════════════════════════════════════════════════

class TeleopInputBase:
    """텔레옵 입력 공통 인터페이스."""

    def get_latest(self) -> tuple[np.ndarray, np.ndarray, bool]:
        raise NotImplementedError

    def shutdown(self):
        pass


if ROS2_AVAILABLE:
    class Ros2TeleopSubscriber(Node, TeleopInputBase):
        """ROS2에서 텔레옵 명령 수신."""

        def __init__(self, arm_topic: str, wheel_topic: str, M_inv: np.ndarray, wheel_radius: float):
            super().__init__("teleop_recorder")

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
) -> np.ndarray:
    """
    텔레옵 데이터 → Isaac Lab 환경 action (9D, [-1, 1]).

    action[0:3] = (vx, vy, wz) / (max_lin_vel, max_lin_vel, max_ang_vel)
    action[3:9] = arm_pos / arm_action_scale
    """
    vx, vy, wz = body_cmd

    # 정규화
    action = np.zeros(9)
    action[0] = np.clip(vx / max_lin_vel, -1.0, 1.0)
    action[1] = np.clip(vy / max_lin_vel, -1.0, 1.0)
    action[2] = np.clip(wz / max_ang_vel, -1.0, 1.0)
    if arm_action_to_limits and arm_center is not None and arm_half_range is not None:
        safe_half = np.where(np.abs(arm_half_range) > 1e-6, arm_half_range, 1.0)
        action[3:9] = np.clip((arm_pos - arm_center) / safe_half, -1.0, 1.0)
    else:
        action[3:9] = np.clip(arm_pos / arm_action_scale, -1.0, 1.0)

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
    env_cfg = LeKiwiNavEnvCfg()
    env_cfg.scene.num_envs = 1
    if args.calibration_json is not None:
        raw = str(args.calibration_json).strip()
        env_cfg.calibration_json = os.path.expanduser(raw) if raw else ""
    if args.dynamics_json:
        env_cfg.dynamics_json = os.path.expanduser(args.dynamics_json)
    if args.arm_limit_json:
        env_cfg.arm_limit_json = os.path.expanduser(args.arm_limit_json)
        env_cfg.arm_limit_margin_rad = float(args.arm_limit_margin_rad)
    env = LeKiwiNavEnv(cfg=env_cfg)

    base_radius = float(env.base_radius)
    wheel_radius = float(env.wheel_radius)
    print(f"  geometry: wheel_radius={wheel_radius:.6f}, base_radius={base_radius:.6f}")

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
        lim = env.robot.data.soft_joint_pos_limits[0, env.arm_idx].detach().cpu().numpy()
        arm_center = 0.5 * (lim[:, 0] + lim[:, 1])
        arm_half_range = 0.5 * (lim[:, 1] - lim[:, 0])
        arm_half_range = np.where(np.abs(arm_half_range) > 1e-6, arm_half_range, 1.0)
        print("  arm mapping: action [-1,1] -> joint limits (center/half-range)")
    else:
        print(f"  arm mapping: action * arm_action_scale ({arm_action_scale:.4f})")
    goal_thresh = float(getattr(env.cfg, "goal_reached_thresh", 0.30))

    # —— 녹화 루프 ——
    obs, info = env.reset()

    episode_obs = []
    episode_actions = []
    episode_active = []
    saved_count = 0
    step_count = 0

    hdf5_file = h5py.File(output_path, "w")
    hdf5_file.attrs["obs_dim"] = int(obs["policy"].shape[-1])
    hdf5_file.attrs["action_dim"] = 9
    hdf5_file.attrs["max_lin_vel"] = float(max_lin_vel)
    hdf5_file.attrs["max_ang_vel"] = float(max_ang_vel)
    hdf5_file.attrs["arm_action_scale"] = float(arm_action_scale)
    hdf5_file.attrs["arm_action_to_limits"] = bool(arm_action_to_limits)
    if args.dynamics_json:
        hdf5_file.attrs["dynamics_json"] = str(os.path.expanduser(args.dynamics_json))
    if args.arm_limit_json:
        hdf5_file.attrs["arm_limit_json"] = str(os.path.expanduser(args.arm_limit_json))
        hdf5_file.attrs["arm_limit_margin_rad"] = float(args.arm_limit_margin_rad)

    print("  ⏳ 텔레옵 입력 연결 대기 중...")
    resolved_arm_unit: str | None = None

    try:
        while sim_app.is_running() and saved_count < args.num_demos:
            # 텔레옵 입력 읽기
            arm_pos, body_cmd, is_active = teleop_input.get_latest()
            arm_pos_rad, unit_used = normalize_arm_positions_to_rad(arm_pos, args.arm_input_unit)
            if resolved_arm_unit is None and is_active:
                resolved_arm_unit = unit_used
                print(f"  arm unit resolved: {resolved_arm_unit}")

            # 텔레옵 → action 변환
            if is_active:
                action_np = teleop_to_action(
                    arm_pos_rad, body_cmd,
                    max_lin_vel, max_ang_vel, arm_action_scale,
                    arm_action_to_limits=arm_action_to_limits,
                    arm_center=arm_center,
                    arm_half_range=arm_half_range,
                )
            else:
                action_np = np.zeros(9)  # 연결 끊겼으면 정지

            action = torch.tensor(action_np, dtype=torch.float32, device=env.device).unsqueeze(0)

            # 환경 step
            next_obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            # 데이터 기록 (항상): 시계열 간격을 일정하게 유지한다.
            episode_obs.append(obs["policy"][0].cpu().numpy())
            episode_actions.append(action_np)
            episode_active.append(bool(is_active))

            # 상태 출력
            if step_count % 25 == 0:  # 25Hz control → 매초
                root_pos = env.robot.data.root_pos_w[0, :2].cpu().numpy()
                goal_pos = env.goal_pos_w[0, :2].cpu().numpy()
                dist = np.linalg.norm(root_pos - goal_pos)
                conn_str = "🟢 연결" if is_active else "🔴 끊김"
                print(
                    f"  {conn_str} | "
                    f"pos=({root_pos[0]:+.2f},{root_pos[1]:+.2f}) | "
                    f"goal=({goal_pos[0]:+.2f},{goal_pos[1]:+.2f}) | "
                    f"dist={dist:.2f}m | "
                    f"steps={len(episode_obs)} | "
                    f"saved={saved_count}/{args.num_demos}"
                )

            # 목표 도달 확인 (truncated)
            done = terminated.any() or truncated.any()

            if done:
                root_pos = env.robot.data.root_pos_w[0, :2].cpu().numpy()
                goal_pos = env.goal_pos_w[0, :2].cpu().numpy()
                final_dist = float(np.linalg.norm(root_pos - goal_pos))

                # 성공: task_success가 있으면 그것을 우선 사용, 없으면 기존 distance 기반 fallback
                active_steps = int(np.sum(np.asarray(episode_active, dtype=np.int32)))
                if hasattr(env, "task_success"):
                    success = bool(env.task_success[0].item()) and active_steps > 10
                else:
                    success = bool(truncated.any() and final_dist < goal_thresh * 2 and active_steps > 10)

                if success:
                    ep_name = f"episode_{saved_count}"
                    grp = hdf5_file.create_group(ep_name)
                    grp.create_dataset("obs", data=np.array(episode_obs))
                    grp.create_dataset("actions", data=np.array(episode_actions))
                    grp.create_dataset("teleop_active", data=np.array(episode_active, dtype=np.int8))
                    grp.attrs["num_steps"] = len(episode_obs)
                    grp.attrs["num_active_steps"] = active_steps
                    grp.attrs["final_dist"] = final_dist
                    grp.attrs["success"] = True
                    hdf5_file.flush()

                    saved_count += 1
                    print(f"\n  ✅ Episode {saved_count} 저장! "
                          f"({len(episode_obs)} steps, dist={final_dist:.3f}m)")
                elif terminated.any():
                    print(f"\n  ❌ 실패 (벗어남/전도) — 폐기, 리셋")
                else:
                    print(f"\n  ⚠ 시간 초과 또는 불완전 — 폐기, 리셋")

                # 리셋
                episode_obs.clear()
                episode_actions.clear()
                episode_active.clear()
                obs, info = env.reset()
                step_count = 0

                if saved_count >= args.num_demos:
                    break
            else:
                obs = next_obs

    except KeyboardInterrupt:
        print("\n\n  중단됨 (Ctrl+C)")

    # —— 마무리 ——
    hdf5_file.close()

    print(f"\n" + "=" * 60)
    print(f"  녹화 완료")
    print(f"  저장된 에피소드: {saved_count}")
    print(f"  파일: {output_path}")
    print(f"\n  다음 단계:")
    print(f"    python train_bc.py --demo_dir demos/ --epochs 200")
    print("=" * 60)

    # 텔레옵 입력 정리
    teleop_input.shutdown()
    if selected_source == "ros2":
        teleop_input.destroy_node()
        rclpy.shutdown()

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
