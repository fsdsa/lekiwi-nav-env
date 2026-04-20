#!/usr/bin/env python3
"""
ProcTHOR Scene 데모 녹화 — VLA 파인튜닝용.

두 가지 모드 지원:
  1. 텔레옵 모드 (기본): 리더암+키보드로 조종, → 저장 / ← 폐기
  2. Expert 모드 (--bc_checkpoint): RL Expert 정책이 자동 실행, 성공 에피소드 자동 저장

데이터 구조 (HDF5, convert_hdf5_to_lerobot_v3.py 호환):
    episode_0/
        actions:      (T, 9) float32   — [arm5, grip1, base_vx, base_vy, base_wz]
        robot_state:  (T, 9) float32   — [arm5, grip1, base_vx, base_vy, base_wz]
        images/base_rgb:  (T, H, W, 3) uint8
        images/wrist_rgb: (T, H, W, 3) uint8
        attrs["instruction"]: str

Usage (텔레옵):
    PYTHONUNBUFFERED=1 python vllm/record_teleop_scene.py \\
      --skill approach_and_grasp \\
      --instruction "pick up the medicine bottle" \\
      --object_usd .../5_HTP/model_clean.usd \\
      --dest_object_usd .../ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd \\
      --scene_idx 1302 --scene_scale 0.6 \\
      --num_demos 10

Usage (Expert):
    PYTHONUNBUFFERED=1 python vllm/record_teleop_scene.py \\
      --skill approach_and_grasp \\
      --instruction "pick up the medicine bottle" \\
      --bc_checkpoint checkpoints/dp_bc_small/dp_bc_epoch150.pt \\
      --resip_checkpoint checkpoints/resip/resip_best.pt \\
      --object_usd .../5_HTP/model_clean.usd \\
      --dest_object_usd .../ACE_Coffee_Mug_Kristen_16_oz_cup/model_clean.usd \\
      --scene_idx 1302 --scene_scale 0.6 \\
      --num_demos 100 --only_success --headless
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ProcTHOR Scene 데모 녹화 (텔레옵/Expert)")

# Mode & Skill
parser.add_argument("--skill", type=str, default="approach_and_grasp",
                    choices=["approach_and_grasp", "carry_and_place", "navigate", "combined_s2_s3", "full"],
                    help="스킬 선택")
parser.add_argument("--direction", type=str, default=None,
                    choices=["forward", "backward", "left", "right", "turn_left", "turn_right"],
                    help="Navigate: 방향 지정 (없으면 6방향 순환)")
parser.add_argument("--instruction", type=str,
                    default="find the medicine bottle and place it next to the red cup",
                    help="VLA instruction 텍스트 (legacy fallback)")
parser.add_argument("--source_object_name", type=str, default="medicine bottle",
                    help="S2/S3 instruction에 사용될 source object 이름 (VLM classify 결과와 일치해야 함)")

# Demos
parser.add_argument("--num_demos", type=int, default=10)
parser.add_argument("--output", type=str, default=None,
                    help="출력 HDF5 경로 (기본: demos/scene_{mode}_{skill}_TIMESTAMP.hdf5)")
parser.add_argument("--resume", action="store_true")

# Expert mode
parser.add_argument("--bc_checkpoint", type=str, default="",
                    help="DP BC checkpoint 경로 (제공 시 expert 모드 활성화)")
parser.add_argument("--resip_checkpoint", type=str, default="",
                    help="ResiP checkpoint 경로 (expert 모드)")
parser.add_argument("--max_episode_steps", type=int, default=2000,
                    help="Expert 모드: 에피소드 최대 스텝")
parser.add_argument("--s3_max_steps", type=int, default=150,
                    help="combined_s2_s3: S3 phase 최대 스텝")
parser.add_argument("--only_success", action="store_true",
                    help="Expert 모드: 성공 에피소드만 저장")
parser.add_argument("--eval_only", action="store_true",
                    help="Expert 모드: 녹화 없이 성공률만 측정 (num_demos = 시도 횟수)")
parser.add_argument("--handoff_buffer", type=str, default="",
                    help="Skill-3 carry_and_place: handoff buffer pickle 경로")
parser.add_argument("--demo", type=str, default="",
                    help="Skill-3 carry_and_place: 초기 상태 복원용 HDF5 (eval_dp_bc.py와 동일)")
parser.add_argument("--bc_checkpoint_s3", type=str, default="",
                    help="combined_s2_s3: Skill-3 BC 체크포인트")
parser.add_argument("--resip_checkpoint_s3", type=str, default="",
                    help="combined_s2_s3: Skill-3 ResiP 체크포인트")
parser.add_argument("--sim_device", type=str, default="",
                    help="sim device override (cpu/cuda:0). 비어있으면 자동 (scene→cpu, 나머지→cuda:0)")

# Teleop
parser.add_argument("--teleop_source", type=str, default="auto", choices=["auto", "ros2", "tcp"])
parser.add_argument("--listen_host", type=str, default="0.0.0.0")
parser.add_argument("--listen_port", type=int, default=15002)
parser.add_argument("--arm_topic", type=str, default="/leader_joint_states")
parser.add_argument("--wheel_topic", type=str, default="/wheel_cmds")
parser.add_argument("--arm_input_unit", type=str, default="auto",
                    choices=["auto", "rad", "deg", "m100"])

# Explore mode: 키보드로 scene 탐색, 좌표/yaw 출력 (데이터 저장 안 함)
parser.add_argument("--explore", action="store_true",
                    help="Scene 탐색 모드: WASD+ZX 이동, P=좌표 출력, Q=종료")
parser.add_argument("--difficulty", type=str, default="easy",
                    choices=["easy", "hard"],
                    help="full 모드: easy=같은 방, hard=다른 방")

# Camera
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=400)

# Env
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--arm_limit_json", type=str, default="calibration/arm_limits_measured.json")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")

# Scene
parser.add_argument("--scene_idx", type=int, default=1302)
parser.add_argument("--scene_usd", type=str, default="")
parser.add_argument("--scene_install_dir", type=str, default="~/molmospaces/assets/usd")
parser.add_argument("--scene_scale", type=float, default=1.0)
parser.add_argument("--scene_floor_z", type=float, default=None)
parser.add_argument("--scene_object_rest_z", type=float, default=0.033)
parser.add_argument("--scene_settle_steps", type=int, default=60)
parser.add_argument("--scene_robot_x", type=float, default=None)
parser.add_argument("--scene_robot_y", type=float, default=None)
parser.add_argument("--scene_robot_yaw_deg", type=float, default=None)

# VLM (instruction 생성용, 선택)
parser.add_argument("--vlm_server", type=str, default="")
parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 모드 결정
EXPERT_MODE = bool(args.bc_checkpoint) or (args.skill == "navigate") or args.explore

# 텔레옵: GUI 필수. Expert: headless 허용
if not EXPERT_MODE:
    args.headless = False

args.num_envs = 1
args.enable_cameras = True  # always enable — physics sync depends on render pipeline

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import h5py
import json
import numpy as np
import select
import socket
import termios
import threading
import tty
import torch
from PIL import Image

EVAL_ONLY_GLOBAL = getattr(args, "eval_only", False)
import omni.replicator.core as rep

from lekiwi_skill2_eval import Skill2Env, Skill2EnvCfg
from lekiwi_robot_cfg import ARM_JOINT_NAMES, WHEEL_JOINT_NAMES, WHEEL_ANGLES_RAD

# Navigate tucked pose (raw rad) — env._nav_apply_action에서도 동일 사용
NAV_TUCKED_RAW_LIST = [-0.02966, -0.213839, 0.09066, -0.4, 0.058418, -0.201554]  # arm5 + grip1
from procthor_scene import (
    SceneSpawnCfg,
    apply_scene_task_layout,
    estimate_spawn_clearance,
    resolve_scene_usd,
    sample_scene_task_layout,
)


# ═══════════════════════════════════════════════════════════════════════
#  텔레옵 입력 (record_teleop.py에서 가져옴)
# ═══════════════════════════════════════════════════════════════════════

class TeleopInputBase:
    def get_latest(self) -> tuple[np.ndarray, np.ndarray, bool]:
        raise NotImplementedError
    def shutdown(self):
        pass


class TcpTeleopSubscriber(TeleopInputBase):
    """TCP JSON lines에서 텔레옵 명령 수신."""

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._arm_positions = np.zeros(6, dtype=np.float64)
        self._base_cmd = np.zeros(3, dtype=np.float64)
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
                except Exception:
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
            # Arm positions
            names = msg.get("name", []) if isinstance(msg, dict) else []
            positions = msg.get("position", []) if isinstance(msg, dict) else []
            if isinstance(names, list) and isinstance(positions, list) and len(names) == len(positions):
                name_to_pos = dict(zip(names, positions))
                for i, jn in enumerate(ARM_JOINT_NAMES):
                    if jn in name_to_pos:
                        self._arm_positions[i] = float(name_to_pos[jn])
            arm_keys = [
                "arm_shoulder_pan.pos", "arm_shoulder_lift.pos", "arm_elbow_flex.pos",
                "arm_wrist_flex.pos", "arm_wrist_roll.pos", "arm_gripper.pos",
            ]
            for i, key in enumerate(arm_keys):
                if key in payload:
                    self._arm_positions[i] = float(payload[key])
            for i, jn in enumerate(ARM_JOINT_NAMES):
                if jn in payload:
                    self._arm_positions[i] = float(payload[jn])
            # Base command
            base = msg.get("base", {}) if isinstance(msg, dict) else {}
            if isinstance(base, dict):
                self._base_cmd[0] = float(base.get("vx", self._base_cmd[0]))
                self._base_cmd[1] = float(base.get("vy", self._base_cmd[1]))
                self._base_cmd[2] = float(base.get("wz", self._base_cmd[2]))
            self._base_cmd[0] = float(payload.get("x.vel", payload.get("base.vx", self._base_cmd[0])))
            self._base_cmd[1] = float(payload.get("y.vel", payload.get("base.vy", self._base_cmd[1])))
            self._base_cmd[2] = float(payload.get("theta.vel", payload.get("base.wz", self._base_cmd[2])))
            self._stamp = time.time()

    def get_latest(self) -> tuple[np.ndarray, np.ndarray, bool]:
        with self._lock:
            arm = self._arm_positions.copy()
            ik = self._base_cmd.copy()
            body_cmd = np.array([-ik[1], ik[0], ik[2]])
            active = (time.time() - self._stamp) < 1.0
        return arm, body_cmd, active

    def shutdown(self):
        self._stop.set()


def teleop_to_action(
    arm_pos: np.ndarray, body_cmd: np.ndarray,
    max_lin_vel: float, max_ang_vel: float, arm_action_scale: float,
    arm_action_to_limits: bool = False,
    arm_center: np.ndarray | None = None,
    arm_half_range: np.ndarray | None = None,
) -> np.ndarray:
    """텔레옵 → v6 action 9D."""
    base_norm = np.array([
        np.clip(body_cmd[0] / max_lin_vel, -1.0, 1.0),
        np.clip(body_cmd[1] / max_lin_vel, -1.0, 1.0),
        np.clip(body_cmd[2] / max_ang_vel, -1.0, 1.0),
    ])
    if arm_action_to_limits and arm_center is not None and arm_half_range is not None:
        safe_half = np.where(np.abs(arm_half_range) > 1e-6, arm_half_range, 1.0)
        arm_norm = np.clip((arm_pos - arm_center) / safe_half, -1.0, 1.0)
        grip_raw = (arm_pos[5] - arm_center[5]) / safe_half[5]
        arm_norm[5] = np.clip(grip_raw, -1.5, 1.0)
    else:
        arm_norm = np.clip(arm_pos / arm_action_scale, -1.0, 1.0)
    action = np.zeros(9)
    action[0:6] = arm_norm
    action[6:9] = base_norm
    return action


def normalize_arm_positions_to_rad(arm_pos: np.ndarray, unit: str) -> tuple[np.ndarray, str]:
    arr = np.asarray(arm_pos, dtype=np.float64)
    if unit == "auto":
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return arr, "rad"
        mx = np.max(np.abs(finite))
        if mx > 200:
            unit = "m100"
        elif mx > 6.3:
            unit = "deg"
        else:
            unit = "rad"
    if unit == "deg":
        return np.radians(arr), unit
    elif unit == "m100":
        return arr / 100.0, unit
    return arr, unit


# ═══════════════════════════════════════════════════════════════════════
#  키보드 입력 (비차단)
# ═══════════════════════════════════════════════════════════════════════

_old_settings = None

def _setup_keyboard():
    global _old_settings
    fd = sys.stdin.fileno()
    _old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

def _restore_keyboard():
    if _old_settings is not None:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _old_settings)

def _check_arrow_key():
    """비차단 화살표 키 체크. 오른쪽=1, 왼쪽=-1, 없음=0."""
    fd = sys.stdin.fileno()
    if select.select([sys.stdin], [], [], 0.0)[0]:
        ch = os.read(fd, 3)
        if ch == b'\x1b[C':
            return 1   # 오른쪽
        elif ch == b'\x1b[D':
            return -1  # 왼쪽
    return 0


# ═══════════════════════════════════════════════════════════════════════
#  Expert 정책 로딩
# ═══════════════════════════════════════════════════════════════════════

def load_expert_policy(bc_path: str, resip_path: str, device: torch.device):
    """DP BC + (선택) ResiP residual 정책 로드."""
    from diffusion_policy import DiffusionPolicyAgent, ResidualPolicy

    bc_ckpt = torch.load(bc_path, map_location=device, weights_only=False)
    dp_cfg = bc_ckpt["config"]

    dp_agent = DiffusionPolicyAgent(
        obs_dim=dp_cfg["obs_dim"],
        act_dim=dp_cfg["act_dim"],
        pred_horizon=dp_cfg["pred_horizon"],
        action_horizon=dp_cfg["action_horizon"],
        num_diffusion_iters=dp_cfg["num_diffusion_iters"],
        inference_steps=dp_cfg.get("inference_steps", 4),
        down_dims=dp_cfg.get("down_dims", [256, 512, 1024]),
    ).to(device)

    state_dict = bc_ckpt["model_state_dict"]
    model_state = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    norm_state = {k[len("normalizer."):]: v for k, v in state_dict.items() if k.startswith("normalizer.")}
    dp_agent.model.load_state_dict(model_state)
    dp_agent.normalizer.load_state_dict(norm_state, device=device)
    dp_agent.eval()
    for p in dp_agent.parameters():
        p.requires_grad = False
    dp_agent.inference_steps = 4

    residual_policy = None
    if resip_path and os.path.isfile(resip_path):
        resip_ckpt = torch.load(resip_path, map_location=device, weights_only=False)
        resip_args = resip_ckpt.get("args", {})
        residual_policy = ResidualPolicy(
            obs_dim=dp_cfg["obs_dim"],
            action_dim=dp_cfg["act_dim"],
            actor_hidden_size=resip_args.get("actor_hidden_size", 256),
            actor_num_layers=resip_args.get("actor_num_layers", 2),
            critic_hidden_size=resip_args.get("critic_hidden_size", 256),
            critic_num_layers=resip_args.get("critic_num_layers", 2),
            action_scale=resip_args.get("action_scale", None) or 1.0,
            action_head_std=resip_args.get("action_head_std", 0.0),
            init_logstd=resip_args.get("init_logstd", -1.0),
        ).to(device)
        residual_policy.load_state_dict(resip_ckpt["residual_policy_state_dict"])
        residual_policy.eval()
        best_sr = resip_ckpt.get("success_rate", "N/A")
        print(f"  [Expert] ResiP loaded (best SR: {best_sr})")
    else:
        print(f"  [Expert] DP BC only (no residual)")

    # per_dim_action_scale — must match training
    act_dim = dp_cfg["act_dim"]
    per_dim_scale = torch.zeros(act_dim, device=device)
    per_dim_scale[0:5] = 0.20   # arm
    per_dim_scale[5]   = 0.25   # gripper
    per_dim_scale[6:9] = 0.35   # base

    print(f"  [Expert] DP: {bc_path}")
    print(f"  [Expert] obs_dim={dp_cfg['obs_dim']}, act_dim={act_dim}")

    return dp_agent, residual_policy, dp_cfg, per_dim_scale


# ═══════════════════════════════════════════════════════════════════════
#  환경 + 카메라 셋업
# ═══════════════════════════════════════════════════════════════════════

def setup_env(args):
    # Skill 선택
    if args.skill == "navigate":
        cfg = Skill2EnvCfg()
    elif args.skill == "carry_and_place":
        from lekiwi_skill3_env import Skill3Env, Skill3EnvCfg
        cfg = Skill3EnvCfg()
    else:
        cfg = Skill2EnvCfg()

    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 1.0
    cfg.enable_domain_randomization = False
    cfg.arm_limit_write_to_sim = False

    if args.skill == "navigate":
        cfg.episode_length_s = 30.0
        cfg.max_dist_from_origin = 50.0
        cfg.dr_action_delay_steps = 0
        cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
        if args.object_usd:
            cfg.object_usd = os.path.expanduser(args.object_usd)
    else:
        cfg.object_scale = 0.7
        cfg.dest_object_scale = 0.56
        cfg.dest_object_fixed = False
        cfg.dest_object_mass = 50.0
        cfg.grasp_contact_threshold = 0.55
        cfg.grasp_gripper_threshold = 0.65
        cfg.max_dist_from_origin = 50.0
        if args.object_usd:
            cfg.object_usd = os.path.expanduser(args.object_usd)
        if args.skill == "carry_and_place":
            # carry_and_place: eval_dp_bc.py와 동일 설정
            cfg.grasp_success_height = 1.00
            cfg.lift_hold_steps = 0
            cfg.grasp_contact_threshold = 0.55
            cfg.grasp_gripper_threshold = 0.65
            cfg.dest_spawn_dist_min = 0.6
            cfg.dest_spawn_dist_max = 0.9
            cfg.dest_spawn_min_separation = 0.3
            cfg.dest_object_fixed = False
            cfg.place_radius = 0.172
            cfg.spawn_heading_noise_std = 0.1
            cfg.spawn_heading_max_rad = 0.26
            cfg.episode_length_s = 100.0
        elif args.skill == "combined_s2_s3":
            # approach_and_grasp와 동일 + auto-terminate 방지
            cfg.grasp_success_height = 100.0  # env가 lift success로 종료하지 않도록
            cfg.lift_hold_steps = 0
            cfg.episode_length_s = 200.0
        elif args.skill == "full":
            # full end-to-end 텔레옵: lift success로 종료하지 않도록
            cfg.grasp_success_height = 100.0
            cfg.lift_hold_steps = 0
        else:
            # approach_and_grasp
            cfg.grasp_success_height = 0.05
            cfg.lift_hold_steps = 200
        if args.dest_object_usd:
            cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
        cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
    if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
        cfg.arm_limit_json = args.arm_limit_json

    if EXPERT_MODE:
        cfg.episode_length_s = args.max_episode_steps * cfg.sim.dt * cfg.decimation + 1.0
    else:
        cfg.episode_length_s = 3600.0  # 텔레옵: 1시간

    # Skill-3: handoff buffer
    if args.skill == "carry_and_place" and args.handoff_buffer:
        cfg.handoff_buffer_path = args.handoff_buffer

    # Scene
    scene_path = resolve_scene_usd(args.scene_idx, args.scene_usd, args.scene_install_dir)
    use_scene = scene_path is not None
    if use_scene:
        cfg.scene_reference_usd = str(scene_path)
        cfg.scene_scale = args.scene_scale
        cfg.use_builtin_ground = True
        from procthor_scene import _load_support_floor_z, SCENE_PRESETS
        preset = SCENE_PRESETS.get(args.scene_idx)
        if preset:
            floor_z = _load_support_floor_z(str(scene_path.resolve()), preset.support_floor_prim_path)
        else:
            floor_z = 0.0
        # cube를 room floor mesh보다 0.1m 아래로 내림 (Z-fighting 방지, 물리 backup만)
        cfg.builtin_ground_z = floor_z * args.scene_scale - 0.1
        cfg.sim.device = args.sim_device if args.sim_device else "cpu"
        print(f"  [Scene] {scene_path}, floor_z={floor_z:.4f}, "
              f"scene_scale={args.scene_scale}, device={cfg.sim.device}")
    else:
        cfg.use_builtin_ground = True
        cfg.sim.device = args.sim_device if args.sim_device else "cuda:0"

    # 텔레옵: 자동 종료 비활성화
    if not EXPERT_MODE:
        cfg.grasp_joint_break_force = 1e8
        cfg.grasp_joint_break_torque = 1e8

    if args.skill == "navigate":
        # Skill2Env 사용 (scene_reference_usd 지원)
        env = Skill2Env(cfg=cfg)
        # Navigate: arm을 그룹 B carry pose로 강제 (카메라 가림 방지)
        _NAV_TUCKED_ARM_T = torch.tensor(
            [-0.02966, -0.213839, 0.09066, -0.4, 0.058418], device=env.device)
        _NAV_TUCKED_GRIP_V = -0.201554
        _original_apply_nav = env._apply_action
        def _nav_apply_action():
            # Skill1Env 방식: base→IK→wheel 직접 처리 + arm 강제
            body_vx = env.actions[:, 6] * env.cfg.max_lin_vel
            body_vy = env.actions[:, 7] * env.cfg.max_lin_vel
            body_wz = env.actions[:, 8] * env.cfg.max_ang_vel
            ik_vx = body_vy
            ik_vy = -body_vx
            ik_wz = body_wz
            ik_cmd = torch.stack([ik_vx, ik_vy, ik_wz], dim=-1)
            wheel_radps = ik_cmd @ env.kiwi_M.T / env.wheel_radius
            vel_target = torch.zeros(1, env.robot.num_joints, device=env.device)
            vel_target[:, env.wheel_idx] = wheel_radps
            env.robot.set_joint_velocity_target(vel_target)
            pos_target = torch.zeros(1, env.robot.num_joints, device=env.device)
            pos_target[0, env.arm_idx[:5]] = _NAV_TUCKED_ARM_T
            pos_target[0, env.arm_idx[5]] = _NAV_TUCKED_GRIP_V
            env.robot.set_joint_position_target(pos_target)
        env._apply_action = _nav_apply_action
    elif args.skill == "carry_and_place":
        from lekiwi_skill3_env import Skill3Env
        env = Skill3Env(cfg=cfg)
    else:
        env = Skill2Env(cfg=cfg)

    # 텔레옵: 자동 종료 비활성화
    if not EXPERT_MODE:
        _original_get_dones = env._get_dones
        def _teleop_get_dones():
            terminated, truncated = _original_get_dones()
            terminated[:] = False
            truncated[:] = False
            return terminated, truncated
        env._get_dones = _teleop_get_dones

    # Ground cuboid를 검정색으로 변경 (visual only, physics 건드리지 않음)
    import omni.usd as _ousd
    from pxr import UsdShade, Sdf, Gf
    _stage = _ousd.get_context().get_stage()
    _cube_vis = _stage.GetPrimAtPath("/World/ground/geometry/mesh")
    if not _cube_vis.IsValid():
        _cube_vis = _stage.GetPrimAtPath("/World/ground")
    if _cube_vis.IsValid():
        _mtl_path = "/World/Looks/BlackMatte"
        UsdShade.Material.Define(_stage, _mtl_path)
        _shader = UsdShade.Shader.Define(_stage, _mtl_path + "/Shader")
        _shader.CreateIdAttr("UsdPreviewSurface")
        _shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.02, 0.02))
        _shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        _shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        _mtl = UsdShade.Material.Get(_stage, _mtl_path)
        _mtl.CreateSurfaceOutput().ConnectToSource(_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(_cube_vis).Bind(_mtl)
        print("  [Ground] Black matte material applied")

    # 카메라 — eval_only에서도 동일하게 setup (physics sync 일치)
    base_cam_path = (
        "/World/envs/env_0/Robot/LeKiwi/base_plate_layer1_v5"
        "/Realsense/RSD455/Camera_OmniVision_OV9782_Color"
    )
    wrist_cam_path = (
        "/World/envs/env_0/Robot/LeKiwi"
        "/Wrist_Roll_08c_v1/visuals/mesh_002_3/wrist_camera"
    )
    base_rp = rep.create.render_product(base_cam_path, (args.camera_width, args.camera_height))
    wrist_rp = rep.create.render_product(wrist_cam_path, (args.camera_width, args.camera_height))

    base_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    base_rgb_annot.attach([base_rp])
    wrist_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    wrist_rgb_annot.attach([wrist_rp])

    print(f"  [Camera] Warming up (30 frames)...")
    for _ in range(30):
        env.sim.render()

    EVAL_ONLY = getattr(args, "eval_only", False)
    if not EVAL_ONLY:
        cams = {"base_rgb": base_rgb_annot, "wrist_rgb": wrist_rgb_annot}
    else:
        cams = None
        print(f"  [Eval Only] 카메라 캡처만 비활성화 (render pipeline 동일)")

    # ── Navigate tucked pose의 normalized [-1,1] 변환값 사전 계산 ──
    # env._apply_action은 action[0:6]을 normalized [-1, 1]로 해석:
    #   arm_target = center + action × half  (joint limits 기반)
    # 따라서 데이터에 저장하는 navigate arm action은 raw rad가 아니라
    # tucked pose에 해당하는 normalized 값이어야 inference 시 동일 자세 재현됨.
    arm_lim = env.robot.data.soft_joint_pos_limits[0, env.arm_idx[:6]]  # (6, 2)
    arm_lo = arm_lim[..., 0]
    arm_hi = arm_lim[..., 1]
    arm_center = 0.5 * (arm_lo + arm_hi)
    arm_half = 0.5 * (arm_hi - arm_lo)
    nav_tucked_raw = torch.tensor(NAV_TUCKED_RAW_LIST, dtype=torch.float32, device=env.device)
    nav_tucked_normalized = ((nav_tucked_raw - arm_center) / arm_half.clamp(min=1e-6)).clamp(-1.0, 1.0)
    cams["_nav_arm_normalized"] = nav_tucked_normalized
    print(f"  [Navigate] tucked pose raw: {[f'{v:+.3f}' for v in nav_tucked_raw.tolist()]}")
    print(f"  [Navigate] tucked pose norm: {[f'{v:+.3f}' for v in nav_tucked_normalized.tolist()]}")

    return env, cams, scene_path


def restore_carry_init_state(env, demo_ep_data):
    """carry_and_place: demo HDF5 에피소드에서 grasp 초기 상태 복원.
    eval_dp_bc.py의 _restore_init_state carry_and_place 부분을 이식."""
    from isaaclab.utils.math import quat_apply, quat_mul

    device = env.device
    env_id = torch.tensor([0], device=device)
    ea = demo_ep_data["ep_attrs"]

    # 1. 로봇 base 위치+방향 (eval_dp_bc.py와 동일)
    if "robot_init_pos" in ea and "robot_init_quat" in ea:
        rs = env.robot.data.root_state_w.clone()
        rs[0, 0:3] = torch.tensor(ea["robot_init_pos"], dtype=torch.float32, device=device)
        rs[0, 3:7] = torch.tensor(ea["robot_init_quat"], dtype=torch.float32, device=device)
        rs[0, 7:] = 0.0
        env.robot.write_root_state_to_sim(rs, env_id)
        env.home_pos_w[0] = rs[0, :3]

    init_joints = torch.tensor(demo_ep_data["obs"][0, 0:6], dtype=torch.float32, device=device)
    target_grip = init_joints[5].item()

    # 2. arm 자세 + gripper 열린 상태
    jp = env.robot.data.default_joint_pos[0:1].clone()
    jp[0, env.arm_idx[:5]] = init_joints[:5]
    jp[0, env.gripper_idx] = 1.4
    jp[0, env.wheel_idx] = 0.0
    jv = torch.zeros_like(jp)
    env.robot.write_joint_state_to_sim(jp, jv, env_ids=env_id)
    env.robot.set_joint_position_target(jp, env_ids=env_id)
    vel_target = torch.zeros(1, env.robot.num_joints, device=device)
    env.robot.set_joint_velocity_target(vel_target, env_ids=env_id)
    for _ in range(10):
        env.robot.write_data_to_sim()
        env.sim.step()
    env.robot.update(env.sim.cfg.dt)

    # 3. EE 위치 + 물체 배치
    wrist_pos = env.robot.data.body_pos_w[0, env._fixed_jaw_body_idx, :]
    wrist_quat = env.robot.data.body_quat_w[0, env._fixed_jaw_body_idx, :]
    rot90_local = torch.tensor([0.8192, -0.5736, 0.0, 0.0], dtype=torch.float32, device=device)
    obj_quat = quat_mul(wrist_quat.unsqueeze(0), rot90_local.unsqueeze(0))[0]
    ee_pos = wrist_pos + quat_apply(wrist_quat.unsqueeze(0), env._ee_local_offset)[0]
    obj_bbox = env.object_bbox[0]
    bbox_center_local = torch.tensor([0.0, 0.0, obj_bbox[2].item() / 2.0],
                                     dtype=torch.float32, device=device)
    bbox_center_world = quat_apply(obj_quat.unsqueeze(0), bbox_center_local.unsqueeze(0))[0]
    obj_root_pos = ee_pos - bbox_center_world

    obj_state = env.object_rigid.data.root_state_w.clone()
    obj_state[0, 0:3] = obj_root_pos
    obj_state[0, 3:7] = obj_quat
    obj_state[0, 7:] = 0.0
    env.object_rigid.write_root_state_to_sim(obj_state, env_id)
    env.object_pos_w[0] = ee_pos

    # 4. gripper 닫기 + 물체 EE에 텔레포트
    grasp_grip = 0.25  # CPU PhysX에서 마찰 부족 → 더 강하게 닫기
    n_close = 300
    for i in range(n_close):
        t_frac = (i + 1) / n_close
        grip_val = target_grip + (grasp_grip - target_grip) * t_frac
        grip_jp = env.robot.data.joint_pos_target[0:1].clone()
        grip_jp[0, env.gripper_idx] = grip_val
        env.robot.set_joint_position_target(grip_jp, env_ids=env_id)
        env.robot.write_data_to_sim()
        w_pos = env.robot.data.body_pos_w[0, env._fixed_jaw_body_idx, :]
        w_quat = env.robot.data.body_quat_w[0, env._fixed_jaw_body_idx, :]
        cur_ee = w_pos + quat_apply(w_quat.unsqueeze(0), env._ee_local_offset)[0]
        cur_obj_quat = quat_mul(w_quat.unsqueeze(0), rot90_local.unsqueeze(0))[0]
        cur_bbox_w = quat_apply(cur_obj_quat.unsqueeze(0), bbox_center_local.unsqueeze(0))[0]
        obj_st = env.object_rigid.data.root_state_w.clone()
        obj_st[0, 0:3] = cur_ee - cur_bbox_w
        obj_st[0, 3:7] = cur_obj_quat
        obj_st[0, 7:] = 0.0
        env.object_rigid.write_root_state_to_sim(obj_st, env_id)
        env.sim.step()
        env.robot.update(env.sim.cfg.dt)
        env.object_rigid.update(env.sim.cfg.dt)

    # 5. 자유 settle (CPU PhysX에서 더 오래)
    for _ in range(120):
        env.robot.write_data_to_sim()
        env.sim.step()
    env.robot.update(env.sim.cfg.dt)
    env.object_rigid.update(env.sim.cfg.dt)

    # 내부 상태
    env.object_grasped[0] = True
    env.just_dropped[0] = False
    if hasattr(env, 'intentional_placed'):
        env.intentional_placed[0] = False
    if hasattr(env, '_fallback_teleport_carry'):
        env._fallback_teleport_carry[0] = False
    env.object_pos_w[0] = env.object_rigid.data.root_pos_w[0]

    # dest object 스폰
    env._spawn_dest_object(env_id)

    obj_z = env.object_rigid.data.root_pos_w[0, 2].item()
    grip_sim = env.robot.data.joint_pos[0, env.gripper_idx].item()
    print(f"    [restore] grip={grip_sim:.3f} obj_z={obj_z:.3f}")


def capture(env, cams):
    env.sim.render()
    b = cams["base_rgb"].get_data()
    w = cams["wrist_rgb"].get_data()
    base_rgb = np.array(b)[..., :3] if b is not None else None
    wrist_rgb = np.array(w)[..., :3] if w is not None else None
    return base_rgb, wrist_rgb


def get_state_9d(env) -> np.ndarray:
    jp = env.robot.data.joint_pos[0]
    arm = jp[env.arm_idx[:5]].cpu().numpy()
    grip = jp[env.gripper_idx].item()
    bv = env.robot.data.root_lin_vel_b[0].cpu().numpy()
    wz = env.robot.data.root_ang_vel_b[0, 2].item()
    state = np.concatenate([arm, [grip, bv[0], bv[1], wz]]).astype(np.float32)
    # NaN 방어: physics settle 부족 시 velocity가 NaN 될 수 있음
    if np.any(np.isnan(state)):
        print(f"  [WARN] get_state_9d NaN detected, replacing with zeros: {state}")
        state = np.nan_to_num(state, nan=0.0)
    return state


# ⚠️ 레거시: direction-based navigate 스케줄. 목적지 기반 navigate로 전환 시 재설계 필요.
# Navigate 시작 위치 프리셋 (scene_scale 적용 후 world 좌표)
_NAV_POSITIONS = [
    (2.412, 10.021, -2.5),
    (2.883, 9.072, -53.3),
    (6.054, 8.870, -51.8),
    (6.533, 8.490, -126.0),
    (8.881, 12.685, -167.5),
    (3.819, 15.470, -93.1),
    (3.239, 15.149, 123.3),
    (7.297, 18.191, 284.7),
    (6.639, 9.290, 119.0),
    (6.464, 7.704, 223.4),
    (8.169, 6.737, 260.6),
    (8.625, 5.616, 129.5),
    (7.282, 3.274, 99.6),
    (4.036, 4.528, 37.4),
    (1.781, 5.427, 136.7),
    (6.116, 7.046, -327.8),
    (4.697, 9.719, -267.1),
    (9.001, 8.618, -171.6),
    (7.366, 3.109, -7.0),
    (6.402, 2.394, 136.1),
]
_NAV_DIRECTIONS = ["forward", "backward", "strafe left", "strafe right", "turn left", "turn right"]
_NAV_POS_NOISE_STD = 0.1  # m
_NAV_YAW_NOISE_STD = 5.0  # degrees
_NAV_ACTION_NOISE_STD = 0.02  # action noise

import random as _nav_rng
import random

# 전체 스케줄: 4 라운드 × 20 위치 × 6 방향 = 480 에피소드
# 라운드 1: 원본 좌표 (노이즈 없음)
# 라운드 2~4: 노이즈 세트 A, B, C
_NAV_SCHEDULE = []
for _round in range(4):
    for _pos in _NAV_POSITIONS:
        for _dir in _NAV_DIRECTIONS:
            if _round == 0:
                # 라운드 1: 원본
                _spos = (_pos[0], _pos[1], _pos[2])
            else:
                # 라운드 2,3: 노이즈
                _nx = _nav_rng.gauss(0, _NAV_POS_NOISE_STD)
                _ny = _nav_rng.gauss(0, _NAV_POS_NOISE_STD)
                _nyaw = _nav_rng.gauss(0, _NAV_YAW_NOISE_STD)
                _spos = (_pos[0] + _nx, _pos[1] + _ny, _pos[2] + _nyaw)
            _NAV_SCHEDULE.append((_spos, "tucked", _dir))
_nav_schedule_idx = 0

# Combined S2→S3 전용 좌표 (6개) × 6방향 × 4라운드 = 144 에피소드
_COMBINED_POSITIONS = [
    (2.754, 16.149, -193.9),
    (6.392, 8.964, -186.7),
    (1.988, 11.784, -287.6),
    (6.688, 7.998, -7.1),
    (3.540, 5.160, -274.1),
    (6.667, 2.623, -208.4),
]
_COMBINED_SCHEDULE = []
for _round in range(4):
    for _pos in _COMBINED_POSITIONS:
        for _dir in _NAV_DIRECTIONS:
            if _round == 0:
                _spos = (_pos[0], _pos[1], _pos[2])
            else:
                _nx = _nav_rng.gauss(0, _NAV_POS_NOISE_STD)
                _ny = _nav_rng.gauss(0, _NAV_POS_NOISE_STD)
                _nyaw = _nav_rng.gauss(0, _NAV_YAW_NOISE_STD)
                _spos = (_pos[0] + _nx, _pos[1] + _ny, _pos[2] + _nyaw)
            _COMBINED_SCHEDULE.append((_spos, "tucked", _dir))
_combined_schedule_idx = 0

# Full end-to-end: 로봇 고정 위치+헤딩 스케줄 (explore에서 측정, scale=1.0)
# (room_id, x, y, yaw_deg)
_FULL_ROOM_SCHEDULE = [
    ("room_7",  0.347,  8.090,  -95.4),
    ("room_9",  1.086, 33.694,  214.4),
    ("room_6",  7.657,  0.638,  354.9),
    ("room_8", 16.483, 28.820,   57.1),
]
_full_room_idx = 0


def reset_with_scene_layout(env, args, scene_path):
    global _nav_pos_idx, _nav_schedule_idx, _combined_schedule_idx, _full_room_idx
    obs, info = env.reset()
    layout = None
    if scene_path is not None:
        robot_xy = None
        if args.scene_robot_x is not None and args.scene_robot_y is not None:
            robot_xy = (float(args.scene_robot_x), float(args.scene_robot_y))
        robot_yaw_rad = (None if args.scene_robot_yaw_deg is None
                         else math.radians(float(args.scene_robot_yaw_deg)))

        # Skill-specific spawn
        source_override = None
        robot_faces = False
        randomize_robot = False
        skill = getattr(args, "skill", None)

        # Navigate / combined_s2_s3: 스케줄에서 위치/방향 가져오기
        _nav_arm_mode = "tucked"
        _use_schedule = (skill == "navigate" and _NAV_SCHEDULE) or \
                        (skill == "combined_s2_s3" and _COMBINED_SCHEDULE)
        if _use_schedule:
            global _combined_schedule_idx
            from procthor_scene import SceneTaskLayout, _load_support_floor_z, SCENE_PRESETS
            if skill == "combined_s2_s3":
                entry = _COMBINED_SCHEDULE[_combined_schedule_idx % len(_COMBINED_SCHEDULE)]
                _combined_schedule_idx += 1
            else:
                entry = _NAV_SCHEDULE[_nav_schedule_idx % len(_NAV_SCHEDULE)]
                _nav_schedule_idx += 1
            pos, _nav_arm_mode, _ = entry
            preset = SCENE_PRESETS.get(args.scene_idx)
            floor_z = 0.0
            if preset and args.scene_floor_z is None:
                floor_z = _load_support_floor_z(
                    str(scene_path.resolve()), preset.support_floor_prim_path
                ) * args.scene_scale
            elif args.scene_floor_z is not None:
                floor_z = float(args.scene_floor_z)

            if skill == "navigate":
                layout = SceneTaskLayout(
                    robot_xy=(pos[0], pos[1]),
                    robot_yaw_rad=math.radians(pos[2]),
                    source_xy=(pos[0] + 1.0, pos[1]),  # dummy
                    source_yaw_rad=0.0,
                    dest_xy=(pos[0] - 1.0, pos[1]),  # dummy
                    dest_yaw_rad=0.0,
                    floor_z=floor_z,
                )
            else:
                # combined_s2_s3: 프리셋 좌표 + 물체는 로봇 전방 1.0m에 직접 배치
                robot_yaw = math.radians(pos[2])
                obj_dist = 0.8 + random.random() * 0.4  # 0.8~1.2m
                # LeKiwi forward = +Y body → world: fwd_x=-sin(yaw), fwd_y=cos(yaw)
                src_x = pos[0] + (-math.sin(robot_yaw)) * obj_dist
                src_y = pos[1] + math.cos(robot_yaw) * obj_dist
                layout = SceneTaskLayout(
                    robot_xy=(pos[0], pos[1]),
                    robot_yaw_rad=robot_yaw,
                    source_xy=(src_x, src_y),
                    source_yaw_rad=random.uniform(-math.pi, math.pi),
                    dest_xy=(pos[0] - 1.0, pos[1]),  # dummy
                    dest_yaw_rad=0.0,
                    floor_z=floor_z,
                )
        else:
            if skill in ("approach_and_grasp", "carry_and_place", "full"):
                ss = float(args.scene_scale) if args.scene_scale > 0 else 1.0
                source_override = SceneSpawnCfg(
                    min_robot_dist=float(getattr(env.cfg, "object_dist_min", 0.8)) / ss,
                    max_robot_dist=float(getattr(env.cfg, "object_dist_max", 1.2)) / ss,
                    clearance_radius=0.14,
                )
                robot_faces = True
                randomize_robot = True

            if skill == "full" and scene_path is not None:
                from procthor_scene import (
                    _load_floor_regions, _load_support_floor_z,
                    _load_scene_obstacles, _find_robot_region,
                    _load_floor_triangles, sample_on_floor_mesh,
                    SCENE_PRESETS as _PRESETS_F, SceneTaskLayout,
                )

                def _room_id(fp):
                    name = fp.path.split("/")[-1]
                    idx = name.find("_visual_")
                    return name[:idx] if idx >= 0 else name

                _scene_str = str(scene_path.resolve())
                _preset_f = _PRESETS_F.get(args.scene_idx)
                _sfz_f = (_load_support_floor_z(
                    _scene_str, _preset_f.support_floor_prim_path
                ) if _preset_f else 0.0)
                _regions_f = _load_floor_regions(_scene_str, support_floor_z=_sfz_f)
                _obstacles_f = _load_scene_obstacles(_scene_str)
                _floor_tris = _load_floor_triangles(_scene_str)
                _difficulty = getattr(args, "difficulty", "easy")

                # room 그룹핑
                _room_groups: dict[str, list] = {}
                for _reg in _regions_f:
                    _rid = _room_id(_reg)
                    _room_groups.setdefault(_rid, []).append(_reg)

                # 로봇: 고정 위치+헤딩 스케줄 (num_demos를 방 수로 나눠 한 방씩)
                _eps_per_room = max(1, args.num_demos // len(_FULL_ROOM_SCHEDULE))
                _room_slot = (_full_room_idx // _eps_per_room) % len(_FULL_ROOM_SCHEDULE)
                _entry = _FULL_ROOM_SCHEDULE[_room_slot]
                _target_room, _rx, _ry, _ryaw_deg = _entry

                # 물체 스폰용 mesh 삼각형 — easy: 로봇 방 / hard: 거실
                _LIVING = {"room_2", "room_3"}
                if _difficulty == "easy":
                    _obj_tris = _floor_tris.get(_target_room, [])
                else:
                    _obj_tris = []
                    for _lr in _LIVING:
                        _obj_tris.extend(_floor_tris.get(_lr, []))

                _s_scale = float(args.scene_scale)
                _fz = _sfz_f * _s_scale

                _r_room = _s_room = _d_room = None
                for _retry in range(200):
                    _rng = random.Random()
                    try:
                        _sxy = sample_on_floor_mesh(
                            _obj_tris, _obstacles_f, 0.3, _rng)
                        _dxy = sample_on_floor_mesh(
                            _obj_tris, _obstacles_f, 0.3, _rng)
                    except RuntimeError:
                        continue
                    if math.dist(_sxy, _dxy) < 1.5:
                        continue
                    # easy: 로봇과 물체 최소 1.5m 이격
                    if _difficulty == "easy":
                        if math.dist((_rx, _ry), _sxy) < 1.5 or math.dist((_rx, _ry), _dxy) < 1.5:
                            continue
                    layout = SceneTaskLayout(
                        robot_xy=(_rx * _s_scale, _ry * _s_scale),
                        robot_yaw_rad=math.radians(_ryaw_deg),
                        source_xy=(_sxy[0] * _s_scale, _sxy[1] * _s_scale),
                        source_yaw_rad=random.uniform(-math.pi, math.pi),
                        dest_xy=(_dxy[0] * _s_scale, _dxy[1] * _s_scale),
                        dest_yaw_rad=random.uniform(-math.pi, math.pi),
                        floor_z=_fz,
                        source_rest_z=float(args.scene_object_rest_z),
                    )
                    _r_room = _find_robot_region((_rx, _ry), _regions_f)
                    _s_room = _find_robot_region(_sxy, _regions_f)
                    _d_room = _find_robot_region(_dxy, _regions_f)
                    break
                else:
                    print(f"  [WARN] {_difficulty} 스폰 실패 200회")

                _r_str = _room_id(_r_room) if _r_room else "?"
                _s_str = _room_id(_s_room) if _s_room else "?"
                _d_str = _room_id(_d_room) if _d_room else "?"
                print(f"  [Spawn] {_difficulty} ep{_full_room_idx+1} | "
                      f"robot={_target_room}({_rx:.1f},{_ry:.1f},{_ryaw_deg:.0f}°) "
                      f"| src={_s_str} | dest={_d_str}")
            else:
                for _retry in range(20):
                    try:
                        layout = sample_scene_task_layout(
                            args.scene_idx, scene_usd=scene_path,
                            robot_xy=robot_xy, robot_yaw_rad=robot_yaw_rad,
                            source_rest_z=args.scene_object_rest_z,
                            floor_z=args.scene_floor_z,
                            scene_scale=args.scene_scale,
                            source_spawn_override=source_override,
                            robot_faces_source=robot_faces,
                            randomize_robot_xy=randomize_robot,
                        )
                        break
                    except RuntimeError:
                        if _retry < 19:
                            continue
                        raise
        if layout is None:
            print(f"  [WARN] layout=None → env.reset() 기본 상태 사용")
            return obs, info, layout

        # 1) 모든 관절 정지 + 텔레포트
        _env_id = torch.tensor([0], device=env.device)
        _zero_vel = torch.zeros(1, env.robot.num_joints, device=env.device)
        env.robot.set_joint_velocity_target(_zero_vel)
        _jp = env.robot.data.default_joint_pos[0:1].clone()
        # approach_and_grasp / full: navigate tucked pose에서 시작 (S1→S2 전환 상태)
        if skill in ("approach_and_grasp", "full"):
            _jp[0, env.arm_idx[:5]] = torch.tensor(
                NAV_TUCKED_RAW_LIST[:5], dtype=torch.float32, device=env.device)
            _jp[0, env.arm_idx[5]] = NAV_TUCKED_RAW_LIST[5]
        _jv = torch.zeros_like(_jp)
        env.robot.write_joint_state_to_sim(_jp, _jv, env_ids=_env_id)
        apply_scene_task_layout(env, layout)
        env.sim.step()
        env.robot.update(env.sim.cfg.dt)

        # 2) settle (정지 상태 유지)
        env.robot.set_joint_velocity_target(_zero_vel)
        for _ in range(max(args.scene_settle_steps, 0)):
            env.sim.step()
            env.sim.render()

        # 3) settle 후 다시 텔레포트 (drift 보정)
        apply_scene_task_layout(env, layout)
        # arm도 다시 쓰기 (settle 중 drift 보정)
        if skill in ("approach_and_grasp", "full"):
            _jp2 = env.robot.data.joint_pos[0:1].clone()
            _jp2[0, env.arm_idx[:5]] = torch.tensor(
                NAV_TUCKED_RAW_LIST[:5], dtype=torch.float32, device=env.device)
            _jp2[0, env.arm_idx[5]] = NAV_TUCKED_RAW_LIST[5]
            _jv2 = torch.zeros_like(_jp2)
            env.robot.write_joint_state_to_sim(_jp2, _jv2, env_ids=_env_id)
            env.robot.set_joint_position_target(_jp2, env_ids=_env_id)
        env.robot.set_joint_velocity_target(_zero_vel)
        env.sim.step()
        env.robot.update(env.sim.cfg.dt)

        if skill == "navigate":
            print(f"  [Layout] robot=({layout.robot_xy[0]:.2f}, {layout.robot_xy[1]:.2f}) "
                  f"arm={_nav_arm_mode} [{_nav_schedule_idx}/{len(_NAV_SCHEDULE)}]")
        elif skill == "combined_s2_s3" and _COMBINED_SCHEDULE:
            _ci = (_combined_schedule_idx - 1) % len(_COMBINED_SCHEDULE)
            _, _, _cd = _COMBINED_SCHEDULE[_ci]
            obj_dist = math.dist(layout.robot_xy, layout.source_xy)
            print(f"  [Layout] robot=({layout.robot_xy[0]:.2f}, {layout.robot_xy[1]:.2f}) "
                  f"src=({layout.source_xy[0]:.2f}, {layout.source_xy[1]:.2f}) "
                  f"carry={_cd} [{_nav_schedule_idx}/{len(_NAV_SCHEDULE)}] obj_dist={obj_dist:.2f}m")
        elif skill == "full":
            obj_dist = math.dist(layout.robot_xy, layout.source_xy)
            dest_dist = math.dist(layout.robot_xy, layout.dest_xy) if layout.dest_xy else 0
            print(f"  [Layout] {getattr(args, 'difficulty', 'easy')} | "
                  f"robot=({layout.robot_xy[0]:.2f}, {layout.robot_xy[1]:.2f}) "
                  f"src=({layout.source_xy[0]:.2f}, {layout.source_xy[1]:.2f}) "
                  f"dest=({layout.dest_xy[0]:.2f}, {layout.dest_xy[1]:.2f}) "
                  f"src_dist={obj_dist:.2f}m dest_dist={dest_dist:.2f}m")
            # 실제 sim 좌표 검증
            _rp = env.robot.data.root_pos_w[0].tolist()
            _sp = env.object_rigid.data.root_pos_w[0].tolist() if getattr(env, 'object_rigid', None) else [0,0,0]
            _dp = env._dest_object_rigid.data.root_pos_w[0].tolist() if getattr(env, '_dest_object_rigid', None) else [0,0,0]
            print(f"  [Actual] robot=({_rp[0]:.2f},{_rp[1]:.2f}) "
                  f"src=({_sp[0]:.2f},{_sp[1]:.2f}) "
                  f"dest=({_dp[0]:.2f},{_dp[1]:.2f})")
        else:
            obj_dist = math.dist(layout.robot_xy, layout.source_xy)
            obj_dist_unscaled = obj_dist / args.scene_scale if args.scene_scale > 0 else obj_dist
            spawn_min = float(getattr(env.cfg, "object_dist_min", 0.8))
            spawn_max = float(getattr(env.cfg, "object_dist_max", 1.2))
            print(f"  [Layout] robot=({layout.robot_xy[0]:.2f}, {layout.robot_xy[1]:.2f}) "
                  f"src=({layout.source_xy[0]:.2f}, {layout.source_xy[1]:.2f}) "
                  f"obj_dist={obj_dist:.2f}m (unscaled={obj_dist_unscaled:.2f}m, "
                  f"cfg=[{spawn_min:.1f}~{spawn_max:.1f}m])")
    return obs, info, layout


def new_episode_buffer(instruction: str) -> dict:
    return {
        "base_rgb": [], "wrist_rgb": [],
        "state": [], "action": [],
        "instruction": instruction,
        "robot_pos_w": [], "object_pos_w": [], "dest_pos_w": [],
    }


def record_step(ep_data: dict, base_rgb, wrist_rgb, state_9d, action_np: np.ndarray, env=None):
    if base_rgb is None or wrist_rgb is None:
        return
    ep_data["base_rgb"].append(base_rgb)
    # Navigate: wrist cam → 검정 이미지 (VLA 학습 시 navigate = wrist cam 미사용)
    if args.skill == "navigate":
        wrist_rgb = np.zeros_like(base_rgb)
    ep_data["wrist_rgb"].append(wrist_rgb)
    ep_data["state"].append(state_9d)
    ep_data["action"].append(action_np.astype(np.float32))
    if env is not None:
        ep_data["robot_pos_w"].append(env.robot.data.root_pos_w[0].cpu().numpy())
        if hasattr(env, "object_pos_w"):
            ep_data["object_pos_w"].append(env.object_pos_w[0].cpu().numpy())
        if hasattr(env, "dest_object_pos_w"):
            ep_data["dest_pos_w"].append(env.dest_object_pos_w[0].cpu().numpy())


# ═══════════════════════════════════════════════════════════════════════
#  HDF5 저장
# ═══════════════════════════════════════════════════════════════════════

def save_episode(hf, ep_idx, data: dict):
    # NaN 검증: state 또는 action에 NaN이 있으면 저장 거부
    state_arr = np.array(data["state"], dtype=np.float32)
    action_arr = np.array(data["action"], dtype=np.float32)
    if np.any(np.isnan(state_arr)):
        nan_count = np.sum(np.any(np.isnan(state_arr), axis=1))
        print(f"  [REJECT] episode_{ep_idx}: robot_state에 NaN {nan_count}/{len(state_arr)} rows — 저장 스킵")
        return False
    if np.any(np.isnan(action_arr)):
        nan_count = np.sum(np.any(np.isnan(action_arr), axis=1))
        print(f"  [REJECT] episode_{ep_idx}: action에 NaN {nan_count}/{len(action_arr)} rows — 저장 스킵")
        return False

    grp = hf.create_group(f"episode_{ep_idx}")
    grp.create_dataset("actions", data=action_arr)
    grp.create_dataset("robot_state", data=state_arr)
    img_grp = grp.create_group("images")
    img_grp.create_dataset("base_rgb", data=np.stack(data["base_rgb"]),
                           compression="gzip", compression_opts=4)
    img_grp.create_dataset("wrist_rgb", data=np.stack(data["wrist_rgb"]),
                           compression="gzip", compression_opts=4)
    grp.attrs["instruction"] = data["instruction"]
    grp.attrs["num_steps"] = len(data["state"])
    if data.get("robot_pos_w"):
        grp.attrs["robot_init_pos"] = data["robot_pos_w"][0]
    if data.get("object_pos_w"):
        grp.attrs["object_init_pos"] = data["object_pos_w"][0]
    if data.get("dest_pos_w"):
        grp.attrs["dest_init_pos"] = data["dest_pos_w"][0]
    hf.flush()
    print(f"  [Saved] episode_{ep_idx}: {len(data['state'])} steps")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════════════════

def make_hdf5(output_path: str) -> tuple:
    """HDF5 파일 열기/이어쓰기. (hdf5_file, saved_count) 반환."""
    global _nav_schedule_idx, _full_room_idx
    if args.resume and os.path.isfile(output_path):
        hdf5_file = h5py.File(output_path, "a")
        saved_count = sum(1 for k in hdf5_file.keys() if k.startswith("episode_"))
        _nav_schedule_idx = saved_count  # resume 시 스케줄도 이어서
        _full_room_idx = saved_count
        print(f"  [Resume] 기존 {saved_count}개 에피소드에서 이어서 녹화 (schedule idx={_nav_schedule_idx})")
    else:
        hdf5_file = h5py.File(output_path, "w")
        hdf5_file.attrs["camera_width"] = args.camera_width
        hdf5_file.attrs["camera_height"] = args.camera_height
        hdf5_file.attrs["scene_idx"] = args.scene_idx
        hdf5_file.attrs["scene_scale"] = args.scene_scale
        hdf5_file.attrs["object_usd"] = str(args.object_usd)
        hdf5_file.attrs["dest_object_usd"] = str(args.dest_object_usd)
        hdf5_file.attrs["skill"] = args.skill
        hdf5_file.attrs["instruction"] = args.instruction
        if args.skill == "full":
            hdf5_file.attrs["difficulty"] = args.difficulty
        saved_count = 0
    return hdf5_file, saved_count


def main():
    mode_str = "Expert" if EXPERT_MODE else "Teleop"
    if args.output:
        output_path = args.output
    else:
        os.makedirs("demos", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        tag = "expert" if EXPERT_MODE else "teleop"
        if args.skill == "full":
            output_path = f"demos/scene_{tag}_{args.skill}_{args.difficulty}_{timestamp}.hdf5"
        else:
            output_path = f"demos/scene_{tag}_{args.skill}_{timestamp}.hdf5"

    print(f"\n{'='*60}")
    print(f"  ProcTHOR Scene 데모 녹화 [{mode_str}] [{args.skill}]")
    print(f"{'='*60}")
    print(f"  목표: {args.num_demos} 에피소드")
    print(f"  저장: {output_path}")
    print(f"  scene: idx={args.scene_idx}, scale={args.scene_scale}")
    if args.skill == "full":
        print(f"  difficulty: {args.difficulty} ({'같은 방' if args.difficulty == 'easy' else '다른 방'})")
    print(f"  camera: {args.camera_width}x{args.camera_height}")
    print(f"  instruction: \"{args.instruction}\"")
    if EXPERT_MODE:
        print(f"  bc: {args.bc_checkpoint}")
        print(f"  resip: {args.resip_checkpoint or '(없음)'}")
        print(f"  max_steps: {args.max_episode_steps}")
        print(f"  only_success: {args.only_success}")
    print(f"{'='*60}\n")

    # 환경 + 카메라
    env, cams, scene_path = setup_env(args)

    if args.explore:
        _run_explore(env, cams, scene_path)
    elif EXPERT_MODE:
        _run_expert(env, cams, scene_path, output_path)
    else:
        _run_teleop(env, cams, scene_path, output_path)

    env.close()
    simulation_app.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _run_explore(env, cams, scene_path):
    """Explore 모드: 키보드로 scene 탐색, P키로 좌표/yaw 출력."""
    from pynput import keyboard as pynput_kb

    _pressed = {}
    _commands = []

    def _on_press(k):
        try:
            s = k.char
        except AttributeError:
            s = {pynput_kb.Key.up: 'w', pynput_kb.Key.down: 's',
                 pynput_kb.Key.left: 'a', pynput_kb.Key.right: 'd',
                 pynput_kb.Key.space: 'space'}.get(k)
        if s:
            _pressed[s] = True
            if s in ('q', 'p'):
                _commands.append(s)

    def _on_release(k):
        try:
            s = k.char
        except AttributeError:
            s = {pynput_kb.Key.up: 'w', pynput_kb.Key.down: 's',
                 pynput_kb.Key.left: 'a', pynput_kb.Key.right: 'd',
                 pynput_kb.Key.space: 'space'}.get(k)
        if s:
            _pressed.pop(s, None)

    listener = pynput_kb.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    # terminate/timeout 비활성화
    _orig_dones = env._get_dones
    def _no_dones():
        t, tr = _orig_dones()
        t[:] = False
        tr[:] = False
        return t, tr
    env._get_dones = _no_dones

    obs, info = env.reset()
    if scene_path is not None:
        from procthor_scene import (
            _load_floor_regions, _load_support_floor_z, _sample_random_point_in_regions,
            _load_scene_obstacles, SCENE_PRESETS, SceneTaskLayout, apply_scene_task_layout,
        )
        import random as _rng
        preset = SCENE_PRESETS.get(args.scene_idx)
        sfz = _load_support_floor_z(str(scene_path.resolve()), preset.support_floor_prim_path) if preset else 0.0
        regions = _load_floor_regions(str(scene_path.resolve()), support_floor_z=sfz)
        obstacles = _load_scene_obstacles(str(scene_path.resolve()))
        ss = float(args.scene_scale) if args.scene_scale > 0 else 1.0
        robot_xy = _sample_random_point_in_regions(regions, obstacles, 0.2, _rng.Random())
        robot_yaw = _rng.Random().uniform(-math.pi, math.pi)
        layout = SceneTaskLayout(
            robot_xy=(robot_xy[0] * ss, robot_xy[1] * ss),
            robot_yaw_rad=robot_yaw,
            source_xy=(robot_xy[0] * ss + 1.0, robot_xy[1] * ss),
            source_yaw_rad=0.0,
            dest_xy=(robot_xy[0] * ss - 1.0, robot_xy[1] * ss),
            dest_yaw_rad=0.0,
            floor_z=sfz * ss,
        )
        apply_scene_task_layout(env, layout)
        for _ in range(30):
            env.sim.step()
            env.sim.render()

    BASE_LIN = 0.5
    BASE_ANG = 1.0
    MAX_LIN = 0.5
    MAX_ANG = 3.0
    coord_list = []

    print("\n  [Explore] Scene 탐색 모드")
    print("  W/↑=forward  S/↓=backward  A/←=strafe left  D/→=strafe right")
    print("  Z=turn left  X=turn right  SPACE=stop  P=좌표 출력  Q=종료\n")

    step = 0
    try:
        while simulation_app.is_running():
            while _commands:
                cmd = _commands.pop(0)
                if cmd == 'q':
                    raise KeyboardInterrupt
                elif cmd == 'p':
                    _rpos = env.robot.data.root_pos_w[0]
                    _rquat = env.robot.data.root_quat_w[0]
                    _ryaw = 2.0 * torch.atan2(_rquat[3], _rquat[0]).item()
                    _ryaw_deg = math.degrees(_ryaw)
                    coord_list.append((_rpos[0].item(), _rpos[1].item(), _ryaw_deg))
                    print(f"  >>> [{len(coord_list)}] ({_rpos[0].item():.3f}, {_rpos[1].item():.3f}, {_ryaw_deg:.1f})")

            kb_vx = kb_vy = kb_wz = 0.0
            if not _pressed.get('space'):
                if _pressed.get('w'): kb_vy += BASE_LIN
                if _pressed.get('s'): kb_vy -= BASE_LIN
                if _pressed.get('a'): kb_vx -= BASE_LIN
                if _pressed.get('d'): kb_vx += BASE_LIN
                if _pressed.get('z'): kb_wz -= BASE_ANG
                if _pressed.get('x'): kb_wz += BASE_ANG

            action = torch.zeros(1, 9, dtype=torch.float32, device=env.device)
            action[0, 6] = kb_vx / MAX_LIN
            action[0, 7] = kb_vy / MAX_LIN
            action[0, 8] = kb_wz / MAX_ANG
            action = action.clamp(-1, 1)

            obs, rew, term, trunc, info = env.step(action)
            env.sim.render()
            step += 1

            if step % 50 == 0:
                _rpos = env.robot.data.root_pos_w[0]
                _rquat = env.robot.data.root_quat_w[0]
                _ryaw = 2.0 * torch.atan2(_rquat[3], _rquat[0]).item()
                print(f"  [step={step}] pos=({_rpos[0].item():.2f}, {_rpos[1].item():.2f}) yaw={math.degrees(_ryaw):.1f}")

    except KeyboardInterrupt:
        pass

    listener.stop()
    print(f"\n  === 저장된 좌표 ({len(coord_list)}개) ===")
    for i, (x, y, yaw) in enumerate(coord_list):
        print(f"  ({x:.3f}, {y:.3f}, {yaw:.1f}),")
    print()


@torch.no_grad()
def _run_expert(env, cams, scene_path, output_path: str):
    """Expert 모드: RL 정책 자동 rollout + 자동 저장."""
    global _combined_schedule_idx, _nav_schedule_idx
    policy_device = env.device
    if args.bc_checkpoint:
        dp_agent, residual_policy, dp_cfg, per_dim_scale = load_expert_policy(
            args.bc_checkpoint, args.resip_checkpoint, policy_device,
        )
    else:
        # Navigate lookup table: policy 불필요
        dp_agent, residual_policy, dp_cfg, per_dim_scale = None, None, None, None

    EVAL_ONLY = getattr(args, "eval_only", False)
    if not EVAL_ONLY:
        hdf5_file, saved_count = make_hdf5(output_path)
    else:
        hdf5_file = None
        saved_count = 0
    total_attempted = 0
    total_success = 0
    start_time = time.time()

    # Navigate: direction map + lookup table action
    _NAV_DIR_MAP = {
        "forward": ([0, 1, 0], "FORWARD"),
        "backward": ([0, -1, 0], "BACKWARD"),
        "strafe left": ([-1, 0, 0], "STRAFE LEFT"),
        "strafe right": ([1, 0, 0], "STRAFE RIGHT"),
        "turn left": ([0, 0, 1], "TURN LEFT"),
        "turn right": ([0, 0, -1], "TURN RIGHT"),
    }
    _NAV_ACTION_MAP = {
        "FORWARD":      [0.0, 0.5, 0.0],
        "BACKWARD":     [0.0, -0.5, 0.0],
        "STRAFE LEFT":  [-0.5, 0.0, 0.0],
        "STRAFE RIGHT": [0.5, 0.0, 0.0],
        "TURN LEFT":    [0.0, 0.0, -0.33],
        "TURN RIGHT":   [0.0, 0.0, 0.33],
    }
    is_navigate = (args.skill == "navigate")
    is_carry = (args.skill == "carry_and_place")
    is_combined = (args.skill == "combined_s2_s3")

    # combined_s2_s3: S3 BC 로드
    s3_dp_agent = None
    if is_combined and args.bc_checkpoint_s3:
        from diffusion_policy import DiffusionPolicyAgent as _DPA
        s3_ckpt = torch.load(args.bc_checkpoint_s3, map_location=policy_device, weights_only=False)
        s3_cfg = s3_ckpt["config"]
        s3_dp_agent = _DPA(
            obs_dim=s3_cfg["obs_dim"], act_dim=s3_cfg["act_dim"],
            pred_horizon=s3_cfg["pred_horizon"], action_horizon=s3_cfg["action_horizon"],
            num_diffusion_iters=s3_cfg["num_diffusion_iters"],
            inference_steps=s3_cfg.get("inference_steps", 4),
            down_dims=s3_cfg.get("down_dims", [256, 512, 1024]),
        ).to(policy_device)
        s3_state = s3_ckpt["model_state_dict"]
        s3_model = {k[len("model."):]: v for k, v in s3_state.items() if k.startswith("model.")}
        s3_norm = {k[len("normalizer."):]: v for k, v in s3_state.items() if k.startswith("normalizer.")}
        s3_dp_agent.model.load_state_dict(s3_model)
        s3_dp_agent.normalizer.load_state_dict(s3_norm, device=policy_device)
        s3_dp_agent.eval()
        s3_dp_agent.inference_steps = 4
        print(f"  [Expert] S3 BC loaded: {args.bc_checkpoint_s3} (obs={s3_cfg['obs_dim']}D)")

    # combined_s2_s3: S3 ResiP 로드
    s3_resip = None
    s3_per_dim = None
    if is_combined and args.resip_checkpoint_s3 and s3_dp_agent is not None:
        from diffusion_policy import ResidualPolicy as _RP
        s3_rp_ckpt = torch.load(args.resip_checkpoint_s3, map_location=policy_device, weights_only=False)
        s3_resip = _RP(
            obs_dim=s3_cfg["obs_dim"], action_dim=s3_cfg["act_dim"],
            action_scale=0.1, learn_std=True,
        ).to(policy_device)
        s3_resip.load_state_dict(s3_rp_ckpt["residual_policy_state_dict"])
        s3_resip.eval()
        for p in s3_resip.parameters():
            p.requires_grad = False
        s3_per_dim = torch.zeros(s3_cfg["act_dim"], device=policy_device)
        s3_per_dim[0:5] = 0.05; s3_per_dim[5] = 0.05  # arm only, base=0
        print(f"  [Expert] S3 ResiP loaded: {args.resip_checkpoint_s3}")

    # Navigate: per_dim override + init_arm_pose 버퍼
    nav_per_dim = None
    _nav_init_arm_pose = None
    if is_navigate and dp_agent is not None:
        _act_dim = dp_cfg["act_dim"]
        nav_per_dim = torch.zeros(_act_dim, device=policy_device)
        nav_per_dim[0:5] = 0.05   # arm: pose drift 보정
        nav_per_dim[5] = 0.05     # gripper
        nav_per_dim[6:9] = 0.0    # base: BC 그대로
        _nav_init_arm_pose = torch.zeros(1, 6, device=policy_device)
        print(f"  [Expert] Navigate BC+ResiP mode (per_dim: arm=0.05, base=0)")

    # Combined S3: direction cmd + init_arm_pose 버퍼
    _combined_dir_cmd = torch.tensor([0.0, 1.0, 0.0], device=policy_device)  # FORWARD (dest는 전방 스폰)
    _combined_init_arm_pose = torch.zeros(6, device=policy_device)

    # carry_and_place: demo HDF5에서 초기 상태 로드
    demo_episodes = []
    if is_carry and args.demo and os.path.isfile(args.demo):
        import h5py as _h5
        _demo_f = _h5.File(args.demo, "r")
        _demo_keys = sorted([k for k in _demo_f.keys() if k.startswith("episode_")],
                            key=lambda k: int(k.split("_")[-1]))
        for _dk in _demo_keys:
            _grp = _demo_f[_dk]
            _ep = {
                "obs": np.array(_grp["obs"]) if "obs" in _grp else np.array(_grp.get("robot_state", [])),
                "ep_attrs": dict(_grp.attrs),
            }
            demo_episodes.append(_ep)
        print(f"  [Demo] {len(demo_episodes)} episodes loaded from {args.demo}")
    _demo_idx = 0

    try:
        _stop_cond = lambda: (total_attempted < args.num_demos) if EVAL_ONLY else (saved_count < args.num_demos)
        while _stop_cond() and simulation_app.is_running():
            # 에피소드 시작
            # 에피소드 시작
            if is_carry and demo_episodes:
                # carry_and_place: eval_dp_bc.py와 동일 — demo에서 전체 초기화
                # scene은 env config으로 이미 로드됨 (배경 렌더링용)
                obs, _ = env.reset()
                ep_data = demo_episodes[_demo_idx % len(demo_episodes)]
                _demo_idx += 1
                restore_carry_init_state(env, ep_data)
                layout = None
            else:
                obs, info, layout = reset_with_scene_layout(env, args, scene_path)
                if layout is None:
                    continue
            if dp_agent is not None:
                dp_agent.reset()
            _stuck_count = 0

            # 같은 방 체크: 로봇과 물체가 다른 방이면 skip (카운트 안 올림)
            if layout is not None and not is_navigate and scene_path is not None:
                from procthor_scene import _load_floor_regions, _load_support_floor_z, SCENE_PRESETS, _find_robot_region
                _preset = SCENE_PRESETS.get(args.scene_idx)
                if _preset:
                    _fz = _load_support_floor_z(str(scene_path.resolve()), _preset.support_floor_prim_path)
                    _regions = _load_floor_regions(str(scene_path.resolve()), support_floor_z=_fz)
                    ss = float(args.scene_scale) if args.scene_scale > 0 else 1.0
                    _robot_xy = (layout.robot_xy[0] / ss, layout.robot_xy[1] / ss)
                    _src_xy = (layout.source_xy[0] / ss, layout.source_xy[1] / ss)
                    _robot_room = _find_robot_region(_robot_xy, _regions)
                    _src_room = _find_robot_region(_src_xy, _regions)
                    if _robot_room is not None and _src_room is not None and _robot_room.path != _src_room.path:
                        print(f"  [Skip] 다른 방 스폰: robot={_robot_room.path} obj={_src_room.path}")
                        continue

            # Navigate: direction command 설정 (스케줄에서 가져옴, reset_with_scene_layout과 동일 idx)
            if is_navigate:
                # _nav_schedule_idx는 reset_with_scene_layout에서 이미 증가됨
                # 현재 에피소드의 entry는 idx-1
                cur_idx = (_nav_schedule_idx - 1) % len(_NAV_SCHEDULE) if _NAV_SCHEDULE else 0
                nav_entry = _NAV_SCHEDULE[cur_idx] if _NAV_SCHEDULE else None
                if args.direction:
                    dir_key = args.direction
                elif nav_entry:
                    dir_key = nav_entry[2]  # (pos, arm, direction)
                else:
                    dir_key = "forward"
                cmd_vec, cmd_label = _NAV_DIR_MAP[dir_key]
                _nav_direction_cmd = torch.tensor(cmd_vec, dtype=torch.float32, device=env.device)
                if hasattr(env, '_direction_cmd'):
                    env._direction_cmd[0] = _nav_direction_cmd
                instruction = f"navigate {cmd_label.lower()}"
                _ep_idx = (_nav_schedule_idx - 1) % len(_NAV_SCHEDULE)
                print(f"    [Navigate] {cmd_label} [{_ep_idx+1}/{len(_NAV_SCHEDULE)}]")
                # Navigate BC+ResiP: init_arm_pose 캡처
                if _nav_init_arm_pose is not None:
                    jp = env.robot.data.joint_pos[0:1]
                    _nav_init_arm_pose[:, :5] = jp[:, env.arm_idx[:5]]
                    _nav_init_arm_pose[:, 5:] = jp[:, env.arm_idx[5:6]]
            elif args.skill == "approach_and_grasp":
                # S2: VLM의 VIVA_APPROACH_LIFT instruction 형식과 일치
                instruction = f"approach and lift the {args.source_object_name}"
            elif is_combined:
                # combined_s2_s3 시작 = S2 phase. instruction은 S2용으로 시작.
                # S3 전환 시 carry direction에 맞춰 다시 설정됨.
                instruction = f"approach and lift the {args.source_object_name}"
            else:
                instruction = args.instruction

            ep_data = new_episode_buffer(instruction)
            success = False
            _combined_phase = "S2" if is_combined else None
            _combined_lift_count = 0
            _active_agent = dp_agent  # navigate: None (lookup table 사용)

            for step in range(args.max_episode_steps):
                # combined_s2_s3: lift 성공 감지 → 컵 스폰 → S3 전환
                if is_combined and _combined_phase == "S2":
                    obj_z = env.object_pos_w[0, 2].item() if hasattr(env, "object_pos_w") else 0
                    grasped = env.object_grasped[0].item() if hasattr(env, "object_grasped") else False
                    if grasped and obj_z > 0.05:
                        _combined_lift_count += 1
                    else:
                        _combined_lift_count = 0
                    if _combined_lift_count >= 200:  # 200 step 유지 → S3 전환
                        print(f"    [S2→S3] Lift success at step {step}!")
                        # S3 전환: init_arm_pose 캡처
                        jp_s3 = env.robot.data.joint_pos[0]
                        _combined_init_arm_pose[:5] = jp_s3[env.arm_idx[:5]]
                        _combined_init_arm_pose[5] = jp_s3[env.arm_idx[5]]
                        # 스케줄에서 direction 가져오기
                        _sched_idx = (_combined_schedule_idx - 1) % len(_COMBINED_SCHEDULE)
                        _, _, _sched_dir = _COMBINED_SCHEDULE[_sched_idx]
                        _dir_vec, _dir_label = _NAV_DIR_MAP[_sched_dir]
                        _combined_dir_cmd[:] = torch.tensor(_dir_vec, dtype=torch.float32, device=env.device)
                        if s3_dp_agent is not None:
                            _active_agent = s3_dp_agent
                            _active_agent.reset()
                        _combined_phase = "S3"
                        _combined_s3_step = 0
                        # S3 instruction은 VLM의 VIVA_CARRY_COMMANDS와 정확히 일치 ("carry forward" 등)
                        # ──────────────────────────────────────────────────────────────────
                        # Carry turn 라벨 swap (실측 env wz 컨벤션 기준)
                        #   env 컨벤션: action[8] > 0 → 우회전(CW), action[8] < 0 → 좌회전(CCW)
                        #   옛 텔레옵 데이터로 학습된 S3 BC checkpoint는 부호가 뒤집혀 있어
                        #   direction_cmd [0,0,+1] (turn left 의도) 입력 시 +0.33 wz 출력
                        #   → 실제 모션은 우회전 → 라벨도 "carry turn right"가 맞다.
                        #   forward/backward/strafe는 carry/navigate 부호가 일치하므로 그대로.
                        # ──────────────────────────────────────────────────────────────────
                        _S3_TURN_LABEL_FIX = {
                            "turn left": "turn right",
                            "turn right": "turn left",
                        }
                        labeled_dir = _S3_TURN_LABEL_FIX.get(_sched_dir, _sched_dir)
                        instruction = f"carry {labeled_dir}"
                        # S3 에피소드 버퍼 새로 시작 (S2 phase 데이터 폐기, S3만 저장)
                        ep_data = new_episode_buffer(instruction)
                        s3_mode = "BC+ResiP" if s3_resip is not None else "BC only"
                        print(f"    [S3] {s3_mode} \"{instruction}\" [{_sched_idx+1}/{len(_COMBINED_SCHEDULE)}]")

                # Actor obs
                actor_obs = obs["policy"]  # (1, obs_dim)

                # Navigate action
                if is_navigate:
                    if dp_agent is not None and _nav_init_arm_pose is not None:
                        # BC+ResiP navigate: 26D obs 수동 구성
                        jp = env.robot.data.joint_pos[0:1]
                        _arm = jp[:, env.arm_idx[:5]]
                        _grip = jp[:, env.arm_idx[5:6]]
                        _bv = env.robot.data.root_lin_vel_b[0:1, :2]
                        _wz = env.robot.data.root_ang_vel_b[0:1, 2:3]
                        _base_vel = torch.cat([_bv, _wz], dim=-1)
                        _lidar = torch.ones(1, 8, device=env.device)
                        nav_obs = torch.cat([
                            _arm, _grip, _base_vel,
                            _nav_direction_cmd.unsqueeze(0),
                            _lidar, _nav_init_arm_pose,
                        ], dim=-1)  # (1, 26D)

                        base_naction = dp_agent.base_action_normalized(nav_obs)
                        if residual_policy is not None:
                            nobs = dp_agent.normalizer(nav_obs, "obs", forward=True).clamp(-3, 3)
                            nobs = torch.nan_to_num(nobs, nan=0.0)
                            ri = torch.cat([nobs, base_naction], dim=-1)
                            _, _, _, _, ra_mean = residual_policy.get_action_and_value(ri)
                            nact = base_naction + ra_mean * nav_per_dim
                        else:
                            nact = base_naction
                        action = dp_agent.normalizer(nact, "action", forward=False).clamp(-1, 1)
                    else:
                        # Lookup table fallback (BC 미제공 시)
                        _base_cmd = _NAV_ACTION_MAP.get(cmd_label, [0.0, 0.5, 0.0])
                        action = torch.zeros(1, 9, dtype=torch.float32, device=env.device)
                        action[0, 6] = _base_cmd[0] + random.gauss(0, _NAV_ACTION_NOISE_STD)
                        action[0, 7] = _base_cmd[1] + random.gauss(0, _NAV_ACTION_NOISE_STD)
                        action[0, 8] = _base_cmd[2] + random.gauss(0, _NAV_ACTION_NOISE_STD)
                        action = action.clamp(-1, 1)
                else:
                    # combined_s2_s3 S3: carry 39D obs (env 30D + dir_cmd 3D + init_arm_pose 6D)
                    if is_combined and _combined_phase == "S3" and s3_dp_agent is not None:
                        actor_obs = torch.cat([
                            actor_obs,
                            _combined_dir_cmd.unsqueeze(0),
                            _combined_init_arm_pose.unsqueeze(0),
                        ], dim=-1)  # (1, 39D)

                    # DP base action (normalized)
                    base_naction = _active_agent.base_action_normalized(actor_obs)

                    # Residual
                    if _combined_phase == "S3" and s3_resip is not None:
                        # S3 carry ResiP
                        nobs = _active_agent.normalizer(actor_obs, "obs", forward=True).clamp(-3, 3)
                        nobs = torch.nan_to_num(nobs, nan=0.0)
                        ri = torch.cat([nobs, base_naction], dim=-1)
                        _, _, _, _, ra_mean = s3_resip.get_action_and_value(ri)
                        naction = base_naction + ra_mean * s3_per_dim
                    elif residual_policy is not None and _combined_phase != "S3":
                        # S2 approach ResiP
                        nobs = _active_agent.normalizer(actor_obs, "obs", forward=True)
                        nobs = torch.clamp(nobs, -3, 3)
                        residual_nobs = torch.cat([nobs, base_naction], dim=-1)
                        residual_naction = residual_policy.get_action(residual_nobs)
                        naction = base_naction + residual_naction * per_dim_scale
                    else:
                        naction = base_naction

                    # Denormalize → raw action
                    action = _active_agent.normalizer(naction, "action", forward=False)

                # 카메라 캡처 (eval_only일 때 캡처만 스킵, render는 유지)
                if not EVAL_ONLY:
                    base_rgb, wrist_rgb = capture(env, cams)
                else:
                    env.sim.render()

                # Navigate: arm action을 normalized tucked pose로 채움 (작은 노이즈 추가)
                # env._nav_apply_action은 arm action을 무시하고 tucked pose 강제하지만,
                # 저장되는 action[0:6]은 inference 시 env._apply_action이 동일 자세를
                # 재현할 수 있도록 normalized [-1,1] 값이어야 함.
                if is_navigate:
                    nav_arm_norm = cams.get("_nav_arm_normalized")
                    if nav_arm_norm is not None:
                        arm_action = nav_arm_norm + torch.randn_like(nav_arm_norm) * 0.01
                        action[0, :6] = arm_action.clamp(-1.0, 1.0)
                    else:
                        action[0, :6] = 0.0

                # Action → numpy (저장용)
                action_np = action[0].cpu().numpy().astype(np.float32)

                # 기록: image + state를 env.step 전에 캡처 (같은 시점)
                if not EVAL_ONLY:
                    _pre_state = get_state_9d(env)
                    record_step(ep_data, base_rgb, wrist_rgb, _pre_state, action_np, env)

                # Env step
                obs, reward, terminated, truncated, info = env.step(action)

                # combined S3 phase 스텝 카운트 + drop 감지 + 제한
                _is_s3_phase = (is_combined and _combined_phase == "S3")
                if _is_s3_phase:
                    _combined_s3_step += 1
                    _s3_objZ = (env.object_pos_w[0, 2] - env.scene.env_origins[0, 2]).item()
                    if _s3_objZ < 0.04 and _combined_s3_step > 5:
                        print(f"    [S3 DROP] objZ={_s3_objZ:.3f} at s3_step={_combined_s3_step} → S2부터 재시도")
                        # 에피소드 버퍼 초기화, S2부터 다시
                        _combined_phase = "S2"
                        _combined_lift_count = 0
                        _active_agent = dp_agent
                        obs, _ = env.reset()
                        # 스케줄 인덱스 되돌리기 (같은 위치/방향에서 재시도)
                        if _COMBINED_SCHEDULE:
                            _combined_schedule_idx -= 1
                        obs, info, layout = reset_with_scene_layout(env, args, scene_path)
                        if dp_agent is not None:
                            dp_agent.reset()
                        ep_data = new_episode_buffer(instruction)
                        continue
                    if _combined_s3_step >= args.s3_max_steps:
                        print(f"    [S3] {args.s3_max_steps} steps 완료 → SUCCESS")
                        success = True
                        break

                # 물체 넘어짐 감지 → 즉시 에피소드 종료 (approach_and_grasp S2 phase만)
                if not is_navigate and not is_carry and not _is_s3_phase:
                    obj_z = env.object_pos_w[0, 2].item() if hasattr(env, "object_pos_w") else 999
                    if obj_z < 0.026:
                        print(f"    [Topple] objZ={obj_z:.3f} < 0.026 → skip")
                        break
                    # 700step까지 물체를 못 들었으면 fail
                    if step == 700 and obj_z < 0.04:
                        print(f"    [NoLift] step=700, objZ={obj_z:.3f} < 0.04 → fail")
                        break

                # Navigate: 속도 기반 충돌 감지 (명령 vs 실제 속도)
                if is_navigate and step > 30:
                    cmd_speed = abs(action_np[6]) + abs(action_np[7])
                    actual_state = get_state_9d(env)
                    actual_speed = float(np.linalg.norm(actual_state[6:8]))
                    if cmd_speed > 0.05 and actual_speed < 0.02:
                        _stuck_count += 1
                        if _stuck_count > 20:
                            print(f"    [Stuck] cmd={cmd_speed:.2f} actual={actual_speed:.3f} → skip")
                            break
                    else:
                        _stuck_count = 0

                # 주기적 상태
                if (step + 1) % 60 == 0:
                    n = len(ep_data["state"])
                    rpos = env.robot.data.root_pos_w[0].tolist()
                    if is_navigate:
                        arm_str = " ".join(f"{v:.2f}" for v in action_np[:6])
                        print(f"    [step={step+1}] ep_steps={n} "
                              f"pos=({rpos[0]:.2f},{rpos[1]:.2f}) arm=[{arm_str}]")
                    else:
                        _oz = env.object_pos_w[0, 2].item() if hasattr(env, "object_pos_w") else 0
                        print(f"    [step={step+1}] ep_steps={n} "
                              f"pos=({rpos[0]:.2f},{rpos[1]:.2f}) objZ={_oz:.3f}")

                # 종료 체크
                if terminated[0] or truncated[0]:
                    ts_mask = info.get("task_success_mask")
                    if ts_mask is not None:
                        success = bool(ts_mask[0].item())
                    else:
                        success = bool(env.task_success[0].item())
                    break

            total_attempted += 1
            if success:
                total_success += 1

            n = len(ep_data["state"]) if not EVAL_ONLY else step + 1

            if EVAL_ONLY:
                sr = total_success / total_attempted if total_attempted > 0 else 0
                tag = "SUCCESS" if success else "FAIL"
                print(f"  [{total_attempted}/{args.num_demos}] {tag} ({n} steps) | "
                      f"SR={sr:.1%} ({total_success}/{total_attempted})")
                continue

            if n < 5:
                print(f"  [Skip] 에피소드 너무 짧음 ({n} steps)")
                continue

            if args.only_success and not success:
                # 실패 시 스케줄 인덱스 되돌리기 (같은 위치/방향에서 재시도)
                if args.skill == "combined_s2_s3" and _COMBINED_SCHEDULE:
                    _combined_schedule_idx -= 1
                elif args.skill == "navigate" and _NAV_SCHEDULE:
                    _nav_schedule_idx -= 1
                print(f"  [Skip] FAIL ({n} steps) — "
                      f"attempted={total_attempted} success={total_success} — retry same schedule")
                continue

            if save_episode(hdf5_file, saved_count, ep_data):
                saved_count += 1

                sr = total_success / total_attempted if total_attempted > 0 else 0
                elapsed = time.time() - start_time
                tag = "SUCCESS" if success else "FAIL"
                print(f"  [{saved_count}/{args.num_demos}] {tag} ({n} steps) | "
                      f"SR={sr:.1%} ({total_success}/{total_attempted}) | "
                      f"{elapsed:.0f}s")

    except KeyboardInterrupt:
        print("\n  중단 (Ctrl+C)")

    if hdf5_file is not None:
        hdf5_file.close()
    elapsed = time.time() - start_time
    sr = total_success / total_attempted if total_attempted > 0 else 0
    if EVAL_ONLY:
        print(f"\n  === EVAL RESULT: SR={sr:.1%} ({total_success}/{total_attempted}), {elapsed:.0f}s ===")
    else:
        print(f"\n  완료: {saved_count} 에피소드 저장 → {output_path}")
        print(f"  SR={sr:.1%} ({total_success}/{total_attempted}), {elapsed:.0f}s")


def _run_teleop(env, cams, scene_path, output_path: str):
    """텔레옵 모드: 리더암+키보드 조작, → 저장 / ← 폐기."""
    global _full_room_idx
    # 텔레옵 입력
    ROS2_AVAILABLE = False
    try:
        import rclpy
        ROS2_AVAILABLE = True
    except Exception:
        pass

    selected_source = args.teleop_source
    if selected_source == "auto":
        selected_source = "ros2" if ROS2_AVAILABLE else "tcp"

    if selected_source == "ros2" and ROS2_AVAILABLE:
        print("  [Teleop] ROS2 not fully implemented here, falling back to TCP")
        selected_source = "tcp"

    if selected_source == "tcp":
        teleop_input = TcpTeleopSubscriber(args.listen_host, args.listen_port)
        print(f"  [Teleop] TCP direct: {args.listen_host}:{args.listen_port}")

    # Action 변환 파라미터
    max_lin_vel = float(env.cfg.max_lin_vel)
    max_ang_vel = float(env.cfg.max_ang_vel)
    arm_action_scale = float(env.cfg.arm_action_scale)
    arm_action_to_limits = bool(env.cfg.arm_action_to_limits)
    arm_center = arm_half_range = None
    if arm_action_to_limits:
        override = getattr(env, "_arm_action_limits_override", None)
        if override is not None:
            lim = override[0].detach().cpu().numpy()
        else:
            lim = env.robot.data.soft_joint_pos_limits[0, env.arm_idx].detach().cpu().numpy()
        arm_center = 0.5 * (lim[:, 0] + lim[:, 1])
        arm_half_range = 0.5 * (lim[:, 1] - lim[:, 0])
        arm_half_range = np.where(np.abs(arm_half_range) > 1e-6, arm_half_range, 1.0)

    # wz 부호 보정
    wz_sign = -1.0
    ct = getattr(env, "_dynamics_command_transform", None)
    if ct is not None and isinstance(ct, dict):
        wz_sign = float(ct.get("wz_sign", -1.0))

    hdf5_file, saved_count = make_hdf5(output_path)

    # 첫 에피소드
    obs, info, layout = reset_with_scene_layout(env, args, scene_path)
    ep_data = new_episode_buffer(args.instruction)

    _setup_keyboard()
    resolved_arm_unit = None

    print(f"  조작:")
    print(f"    → (오른쪽 화살표): 에피소드 저장 + 새 에피소드")
    print(f"    ← (왼쪽 화살표): 에피소드 폐기 + 리셋")
    print(f"    Ctrl+C: 종료")
    print(f"  [Ready] 텔레옵 입력 대기 중...\n")

    try:
        step_count = 0
        while simulation_app.is_running() and saved_count < args.num_demos:
            # 텔레옵 입력
            arm_pos, body_cmd, is_active = teleop_input.get_latest()
            arm_pos_rad, unit_used = normalize_arm_positions_to_rad(arm_pos, args.arm_input_unit)
            if resolved_arm_unit is None and is_active:
                resolved_arm_unit = unit_used
                print(f"  [Teleop] arm unit: {resolved_arm_unit}")

            body_cmd[2] *= wz_sign

            # Action 변환
            if is_active:
                action_np = teleop_to_action(
                    arm_pos_rad, body_cmd,
                    max_lin_vel, max_ang_vel, arm_action_scale,
                    arm_action_to_limits=arm_action_to_limits,
                    arm_center=arm_center, arm_half_range=arm_half_range,
                )
            else:
                action_np = np.zeros(9)

            action_t = torch.tensor(action_np, dtype=torch.float32, device=env.device).unsqueeze(0)

            # 카메라 + state 캡처 (env.step 전, image와 같은 시점)
            base_rgb, wrist_rgb = capture(env, cams)
            _pre_state = get_state_9d(env)

            # Env step
            next_obs, reward, terminated, truncated, info = env.step(action_t)
            step_count += 1

            # 기록
            record_step(ep_data, base_rgb, wrist_rgb, _pre_state, action_np, env)

            # 주기적 상태 출력
            if step_count % 60 == 0:
                n = len(ep_data["state"])
                active_str = "ACTIVE" if is_active else "idle"
                obj_z = env.object_pos_w[0, 2].item() if hasattr(env, "object_pos_w") else 0
                rpos = env.robot.data.root_pos_w[0].tolist()
                quat = env.robot.data.root_quat_w[0].tolist()  # (w,x,y,z)
                yaw_rad = math.atan2(2*(quat[0]*quat[3]+quat[1]*quat[2]), 1-2*(quat[2]**2+quat[3]**2))
                state_9d = get_state_9d(env)
                arm_str = " ".join(f"{v:.2f}" for v in state_9d[:6])
                print(f"    [step={step_count}] {active_str} "
                      f"--scene_robot_x {rpos[0]:.3f} --scene_robot_y {rpos[1]:.3f} "
                      f"--scene_robot_yaw_deg {math.degrees(yaw_rad):.1f} "
                      f"arm=[{arm_str}] objZ={obj_z:.3f}")

            # 키보드: → 저장, ← 폐기
            key = _check_arrow_key()
            if key == 1:  # 오른쪽 → 저장
                n = len(ep_data["state"])
                if n > 10:
                    if save_episode(hdf5_file, saved_count, ep_data):
                        saved_count += 1
                        if args.skill == "full":
                            _full_room_idx += 1  # 저장 성공 시에만 다음 방으로
                        print(f"  [{saved_count}/{args.num_demos}] 저장 완료 ({n} steps)")
                    else:
                        print(f"  [Skip] NaN 데이터 → 저장 거부")
                else:
                    print(f"  [Skip] 너무 짧음 ({n} steps)")
                ep_data = new_episode_buffer(args.instruction)
                obs, info, layout = reset_with_scene_layout(env, args, scene_path)
                step_count = 0

            elif key == -1:  # 왼쪽 → 폐기
                n = len(ep_data["state"])
                print(f"  [Discard] {n} steps 폐기, 리셋")
                ep_data = new_episode_buffer(args.instruction)
                obs, info, layout = reset_with_scene_layout(env, args, scene_path)
                step_count = 0

    except KeyboardInterrupt:
        print("\n  중단 (Ctrl+C)")

    _restore_keyboard()
    hdf5_file.close()
    teleop_input.shutdown()
    print(f"\n  완료: {saved_count} 에피소드 저장 → {output_path}")


if __name__ == "__main__":
    main()