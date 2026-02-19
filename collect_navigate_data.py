#!/usr/bin/env python3
"""
Navigate skill — 스크립트 정책(proportional controller)으로 VLA 학습 데이터 자동 생성.

Navigate는 RL 없이 proportional controller로 "목표 방향 이동" 또는 "탐색 회전"을 수행.
base만 움직이고 arm은 rest pose, gripper는 항상 open.

두 가지 모드:
  - Directed Navigation (70%): 목표 방향 proportional control
  - Search Rotation (30%): 제자리 회전 또는 전진+회전

노이즈 주입:
  (1) 조향 흔들림 σ=0.05 rad
  (2) 속도 양자화 3~5단계
  (3) 5% 확률로 이전 action 반복 (1~3프레임)

Action: [arm_target 5D, gripper 1D (open), base_cmd 3D] = 9D (lekiwi_v6 순서)

Usage:
    python collect_navigate_data.py \\
      --num_envs 4 --num_demos 1000 --headless \\
      --multi_object_json object_catalog.json \\
      --gripper_contact_prim_path "..." \\
      --dynamics_json calibration/tuned_dynamics.json

결과: outputs/navigate_demos/navigate_script_YYYYMMDD_HHMMSS.hdf5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Navigate skill — 스크립트 정책 데이터 수집")
parser.add_argument("--num_demos", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=4,
                    help="카메라 사용 시 1~8 권장 (VRAM)")
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--dynamics_json", type=str, default=None)
parser.add_argument("--calibration_json", type=str, default=None)
parser.add_argument("--arm_limit_json", type=str, default=None)
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str, default="")
parser.add_argument("--no_camera", action="store_true", help="카메라 없이 수집 (state-only)")
parser.add_argument("--base_cam_width", type=int, default=1280)
parser.add_argument("--base_cam_height", type=int, default=720)
parser.add_argument("--wrist_cam_width", type=int, default=640)
parser.add_argument("--wrist_cam_height", type=int, default=480)
parser.add_argument("--min_steps", type=int, default=20, help="최소 에피소드 길이")
parser.add_argument("--directed_ratio", type=float, default=0.7,
                    help="Directed Navigation 비율 (나머지 = Search Rotation)")
parser.add_argument("--search_duration_min", type=int, default=150,
                    help="Search Rotation 최소 길이 (steps)")
parser.add_argument("--search_duration_max", type=int, default=375,
                    help="Search Rotation 최대 길이 (steps)")
parser.add_argument("--nav_arrive_dist", type=float, default=0.5,
                    help="Directed Navigation 도착 판정 거리 (m)")
parser.add_argument("--noise_std", type=float, default=0.05,
                    help="조향 노이즈 σ (rad)")
parser.add_argument("--action_repeat_prob", type=float, default=0.05,
                    help="이전 action 반복 확률")
parser.add_argument("--speed_quantize_levels", type=int, default=5,
                    help="속도 양자화 단계 수 (0=비활성)")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
sim_app = launcher.app

import h5py
import numpy as np
import torch

from isaaclab.sensors import Camera, CameraCfg

from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg

# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

ENV_PRIM = "/World/envs/env_.*/Robot"

BASE_RGB_CAM_PRIM = (
    f"{ENV_PRIM}/base_plate_layer1_v5/Realsense/RSD455"
    f"/Camera_OmniVision_OV9782_Color"
)
WRIST_CAM_PRIM = (
    f"{ENV_PRIM}/Wrist_Roll_08c_v1/visuals/mesh_002_3"
    f"/wrist_camera"
)

# Proportional controller gains
K_LIN = 0.8   # linear velocity gain
K_ANG = 1.5   # angular velocity gain

# Camera approximate FOV half-angle (RealSense D455 ≈ 87° HFOV → ~43.5° half)
CAM_FOV_HALF_RAD = math.radians(43.5)


# ═══════════════════════════════════════════════════════════════════════
#  Camera subclass
# ═══════════════════════════════════════════════════════════════════════

class Skill2EnvWithCam(Skill2Env):
    """Skill2Env + base_rgb/wrist RGB 카메라."""

    def __init__(self, cfg, base_cam_w=1280, base_cam_h=720,
                 wrist_cam_w=640, wrist_cam_h=480, render_mode=None, **kwargs):
        self._base_cam_w = base_cam_w
        self._base_cam_h = base_cam_h
        self._wrist_cam_w = wrist_cam_w
        self._wrist_cam_h = wrist_cam_h
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        super()._setup_scene()
        base_cam_cfg = CameraCfg(
            prim_path=BASE_RGB_CAM_PRIM, spawn=None, update_period=0.0,
            height=self._base_cam_h, width=self._base_cam_w, data_types=["rgb"],
        )
        self.base_cam = Camera(base_cam_cfg)
        self.scene.sensors["base_cam"] = self.base_cam

        wrist_cam_cfg = CameraCfg(
            prim_path=WRIST_CAM_PRIM, spawn=None, update_period=0.0,
            height=self._wrist_cam_h, width=self._wrist_cam_w, data_types=["rgb"],
        )
        self.wrist_cam = Camera(wrist_cam_cfg)
        self.scene.sensors["wrist_cam"] = self.wrist_cam

    def _extract_rgb(self, camera: Camera) -> torch.Tensor | None:
        rgb = camera.data.output.get("rgb")
        if rgb is None:
            return None
        if rgb.dtype == torch.float32:
            rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
        return rgb[:, :, :, :3]

    def get_base_rgb(self) -> torch.Tensor | None:
        return self._extract_rgb(self.base_cam)

    def get_wrist_rgb(self) -> torch.Tensor | None:
        return self._extract_rgb(self.wrist_cam)


# ═══════════════════════════════════════════════════════════════════════
#  Script policy helpers
# ═══════════════════════════════════════════════════════════════════════

def compute_rest_arm_action(env: Skill2Env) -> torch.Tensor:
    """
    Arm rest pose를 [-1,1] action 공간으로 역매핑.
    arm_action_to_limits: target = center + action * half_range
    → action = (target - center) / half_range
    """
    if env._arm_action_limits_override is not None:
        arm_limits = env._arm_action_limits_override
    else:
        arm_limits = env.robot.data.soft_joint_pos_limits[:, env.arm_idx]
    arm_lo = arm_limits[..., 0]  # (N, 6)
    arm_hi = arm_limits[..., 1]

    center = 0.5 * (arm_lo + arm_hi)
    half = 0.5 * (arm_hi - arm_lo)

    # Rest target = default joint positions
    rest_target = env.robot.data.default_joint_pos[:, env.arm_idx]  # (N, 6)

    finite = torch.isfinite(arm_lo) & torch.isfinite(arm_hi) & ((arm_hi - arm_lo) > 1e-6)
    action = torch.where(
        finite,
        (rest_target - center) / (half + 1e-8),
        torch.zeros_like(rest_target),
    )
    return action.clamp(-1.0, 1.0)  # (N, 6)


def compute_robot_yaw(quat_w: torch.Tensor) -> torch.Tensor:
    """Quaternion [w,x,y,z] → yaw angle."""
    return 2.0 * torch.atan2(quat_w[:, 3], quat_w[:, 0])


def quantize_speed(val: torch.Tensor, levels: int) -> torch.Tensor:
    """속도를 levels 단계로 양자화."""
    if levels <= 0:
        return val
    step = 2.0 / levels  # [-1, 1] 범위
    return (val / step).round() * step


def compute_navigate_action(
    env: Skill2Env,
    ep_mode: list[str],
    ep_search_wz: torch.Tensor,
    ep_search_vx: torch.Tensor,
    arm_rest_action: torch.Tensor,
    prev_action: torch.Tensor,
    noise_std: float = 0.05,
    repeat_prob: float = 0.05,
    speed_levels: int = 5,
) -> torch.Tensor:
    """
    Navigate 스크립트 정책 action 계산.

    Returns:
        (N, 9) action in [-1, 1], lekiwi_v6 order [arm5, grip1, base3]
    """
    N = env.num_envs
    device = env.device

    robot_xy = env.robot.data.root_pos_w[:, :2]       # (N, 2)
    robot_yaw = compute_robot_yaw(env.robot.data.root_quat_w)  # (N,)
    target_xy = env.object_pos_w[:, :2]                # (N, 2)

    direction = target_xy - robot_xy
    dist = direction.norm(dim=-1)                      # (N,)
    target_angle = torch.atan2(direction[:, 1], direction[:, 0])
    angle_to_target = torch.atan2(
        torch.sin(target_angle - robot_yaw),
        torch.cos(target_angle - robot_yaw),
    )  # wrapped to [-π, π]

    max_lin = float(env.cfg.max_lin_vel)
    max_ang = float(env.cfg.max_ang_vel)

    # Directed navigation: proportional control toward object
    directed_vx = (K_LIN * torch.cos(angle_to_target)).clamp(-max_lin, max_lin) / max_lin
    directed_vy = (K_LIN * torch.sin(angle_to_target)).clamp(-max_lin, max_lin) / max_lin
    directed_wz = (K_ANG * angle_to_target).clamp(-max_ang, max_ang) / max_ang

    # Slow down when close
    speed_scale = (dist / 1.0).clamp(0.2, 1.0)
    directed_vx *= speed_scale
    directed_vy *= speed_scale

    # Build base action per env
    base_vx = torch.zeros(N, device=device)
    base_vy = torch.zeros(N, device=device)
    base_wz = torch.zeros(N, device=device)

    for i in range(N):
        if ep_mode[i] == "directed":
            base_vx[i] = directed_vx[i]
            base_vy[i] = directed_vy[i]
            base_wz[i] = directed_wz[i]
        else:  # search
            base_vx[i] = ep_search_vx[i]
            base_vy[i] = 0.0
            base_wz[i] = ep_search_wz[i]

    # Noise injection: steering jitter
    if noise_std > 0:
        yaw_noise = torch.randn(N, device=device) * noise_std
        base_wz += yaw_noise / max_ang

    # Speed quantization
    base_vx = quantize_speed(base_vx, speed_levels)
    base_vy = quantize_speed(base_vy, speed_levels)
    base_wz = quantize_speed(base_wz, speed_levels)

    # Clamp
    base_vx = base_vx.clamp(-1.0, 1.0)
    base_vy = base_vy.clamp(-1.0, 1.0)
    base_wz = base_wz.clamp(-1.0, 1.0)

    # Assemble 9D action: [arm5, grip1, base3]
    action = torch.zeros(N, 9, device=device)
    action[:, 0:6] = arm_rest_action[:, 0:6]  # arm5 + gripper
    # Override gripper to fully open (+1.0 in normalized space)
    action[:, 5] = 1.0
    action[:, 6] = base_vx
    action[:, 7] = base_vy
    action[:, 8] = base_wz

    # Action repeat: 5% probability of reusing previous action
    if prev_action is not None:
        repeat_mask = torch.rand(N, device=device) < repeat_prob
        action[repeat_mask] = prev_action[repeat_mask]

    return action


def extract_robot_state_9d(env: Skill2Env) -> torch.Tensor:
    """VLA용 robot_state 9D: [arm_pos(5), gripper(1), base_body_vel(3)].

    단위: arm=rad, gripper=rad, base=m/s(x, y), rad/s(theta)
    단위 변환 불필요: sim과 real 모두 m/s, rad/s
    """
    arm_pos = env.robot.data.joint_pos[:, env.arm_idx]   # 6D
    # body-frame velocity 직접 읽기
    vx_body = env.robot.data.root_lin_vel_b[:, 0:1]   # x.vel (m/s)
    vy_body = env.robot.data.root_lin_vel_b[:, 1:2]   # y.vel (m/s)
    wz_body = env.robot.data.root_ang_vel_b[:, 2:3]   # theta.vel (rad/s)
    base_body_vel = torch.cat([vx_body, vy_body, wz_body], dim=-1)  # (N, 3)
    return torch.cat([arm_pos, base_body_vel], dim=-1)  # 9D


def _get_object_name(env: Skill2Env, env_id: int) -> str:
    """환경의 활성 물체 이름 추출."""
    if not hasattr(env, "_object_catalog"):
        return "target object"
    catalog = list(getattr(env, "_object_catalog", []))
    active_idx = int(env.active_object_idx[env_id].item())
    if active_idx < 0 or active_idx >= len(catalog):
        return "target object"
    entry = catalog[active_idx]
    if not isinstance(entry, dict):
        return "target object"
    for k in ("instruction_name", "display_name", "name", "category_name", "label"):
        raw = str(entry.get(k, "")).strip()
        if raw:
            return raw.replace("_", " ").replace("-", " ")
    return "target object"


def _gen_instruction(mode: str, object_name: str, angle_to_obj: float) -> str:
    """Navigate instruction 생성 (가시성 기반)."""
    name = object_name.lower()
    if mode == "directed":
        if abs(angle_to_obj) < CAM_FOV_HALF_RAD:
            return f"navigate toward the {name}"
        else:
            return f"navigate toward the {name}"
    else:  # search
        return f"turn to search for the {name}"


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    use_camera = not args.no_camera
    num_envs = args.num_envs

    if args.output:
        output_path = args.output
    else:
        os.makedirs("outputs/navigate_demos", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = "_cam" if use_camera else ""
        output_path = f"outputs/navigate_demos/navigate_script{suffix}_{timestamp}.hdf5"

    print("\n" + "=" * 60)
    print("  Navigate Skill — 스크립트 정책 데이터 수집")
    print(f"  목표       : {args.num_demos} 에피소드")
    print(f"  병렬 환경  : {num_envs}")
    print(f"  모드 비율  : Directed {args.directed_ratio*100:.0f}% / Search {(1-args.directed_ratio)*100:.0f}%")
    print(f"  카메라     : {'ON' if use_camera else 'OFF'}")
    print(f"  출력       : {output_path}")
    print("=" * 60 + "\n")

    # ── 환경 생성 ──
    env_cfg = Skill2EnvCfg()
    env_cfg.scene.num_envs = num_envs
    # Navigate는 접근만 하고 grasp 안 함 — 긴 에피소드 불필요
    env_cfg.episode_length_s = 15.0

    if args.dynamics_json:
        env_cfg.dynamics_json = os.path.expanduser(args.dynamics_json)
    if args.calibration_json:
        env_cfg.calibration_json = os.path.expanduser(args.calibration_json)
    if args.arm_limit_json:
        env_cfg.arm_limit_json = os.path.expanduser(args.arm_limit_json)
    if args.multi_object_json:
        env_cfg.multi_object_json = os.path.expanduser(args.multi_object_json)
    env_cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
    # 물체 거리 범위를 넓게 (Navigate는 먼 거리도 포함)
    env_cfg.object_dist_min = 1.0
    env_cfg.object_dist_max = 3.0
    env_cfg.curriculum_current_max_dist = 3.0

    if use_camera:
        env = Skill2EnvWithCam(
            cfg=env_cfg,
            base_cam_w=args.base_cam_width,
            base_cam_h=args.base_cam_height,
            wrist_cam_w=args.wrist_cam_width,
            wrist_cam_h=args.wrist_cam_height,
        )
    else:
        env = Skill2Env(cfg=env_cfg)

    device = env.device

    # ── Arm rest action 계산 ──
    obs, _ = env.reset()
    arm_rest_action = compute_rest_arm_action(env)  # (N, 6)

    # ── Per-env 상태 ──
    ep_mode: list[str] = ["directed"] * num_envs
    ep_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    ep_search_max_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    ep_search_wz = torch.zeros(num_envs, device=device)
    ep_search_vx = torch.zeros(num_envs, device=device)
    prev_action: torch.Tensor | None = None

    # Buffers
    ep_obs = [[] for _ in range(num_envs)]
    ep_act = [[] for _ in range(num_envs)]
    ep_base_img = [[] for _ in range(num_envs)]
    ep_wrist_img = [[] for _ in range(num_envs)]
    ep_robot_state = [[] for _ in range(num_envs)]
    ep_instruction = [""] * num_envs

    def _assign_mode(idx: int):
        """에피소드 시작 시 모드 할당."""
        if torch.rand(1).item() < args.directed_ratio:
            ep_mode[idx] = "directed"
        else:
            ep_mode[idx] = "search"
            # Random rotation direction and speed
            wz_dir = 1.0 if torch.rand(1).item() > 0.5 else -1.0
            ep_search_wz[idx] = wz_dir * (0.3 + torch.rand(1).item() * 0.4)  # 0.3~0.7
            ep_search_vx[idx] = torch.rand(1).item() * 0.3 - 0.1  # -0.1~0.2
            duration = args.search_duration_min + int(
                torch.rand(1).item() * (args.search_duration_max - args.search_duration_min)
            )
            ep_search_max_steps[idx] = duration
        obj_name = _get_object_name(env, idx)
        ep_instruction[idx] = _gen_instruction(ep_mode[idx], obj_name, 0.0)

    def _clear_buffers(idx: int):
        ep_obs[idx].clear()
        ep_act[idx].clear()
        ep_base_img[idx].clear()
        ep_wrist_img[idx].clear()
        ep_robot_state[idx].clear()

    # Assign initial modes
    for i in range(num_envs):
        _assign_mode(i)

    # ── HDF5 초기화 ──
    hdf5_file = h5py.File(output_path, "w")
    hdf5_file.attrs["skill"] = "navigate"
    hdf5_file.attrs["policy_type"] = "script_proportional"
    hdf5_file.attrs["has_camera"] = use_camera
    hdf5_file.attrs["has_robot_state"] = True
    hdf5_file.attrs["action_dim"] = 9
    hdf5_file.attrs["directed_ratio"] = args.directed_ratio
    if use_camera:
        hdf5_file.attrs["base_rgb_shape"] = [args.base_cam_height, args.base_cam_width, 3]
        hdf5_file.attrs["wrist_rgb_shape"] = [args.wrist_cam_height, args.wrist_cam_width, 3]

    saved = 0

    try:
        while saved < args.num_demos:
            # Robot state
            step_robot_state = extract_robot_state_9d(env)

            # Images
            base_rgb, wrist_rgb = None, None
            if use_camera:
                base_rgb = env.get_base_rgb()
                wrist_rgb = env.get_wrist_rgb()

            # Compute script action
            action = compute_navigate_action(
                env=env,
                ep_mode=ep_mode,
                ep_search_wz=ep_search_wz,
                ep_search_vx=ep_search_vx,
                arm_rest_action=arm_rest_action,
                prev_action=prev_action,
                noise_std=args.noise_std,
                repeat_prob=args.action_repeat_prob,
                speed_levels=args.speed_quantize_levels,
            )

            # Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_step_count += 1

            # Record
            for i in range(num_envs):
                ep_obs[i].append(obs["policy"][i].cpu().numpy())
                action_to_save = action[i].clone()
                # Gripper binary: always open = 1.0
                action_to_save[5] = 1.0
                ep_act[i].append(action_to_save.cpu().numpy())
                ep_robot_state[i].append(step_robot_state[i].cpu().numpy())
                if use_camera:
                    if base_rgb is not None:
                        ep_base_img[i].append(base_rgb[i].cpu().numpy())
                    if wrist_rgb is not None:
                        ep_wrist_img[i].append(wrist_rgb[i].cpu().numpy())

            # Check episode completion
            # env dones (timeout, out-of-bounds)
            env_done = terminated | truncated

            # Navigate-specific completion
            robot_xy = env.robot.data.root_pos_w[:, :2]
            target_xy = env.object_pos_w[:, :2]
            dist_to_obj = (target_xy - robot_xy).norm(dim=-1)

            directed_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
            search_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
            for i in range(num_envs):
                if ep_mode[i] == "directed":
                    directed_done[i] = dist_to_obj[i] < args.nav_arrive_dist
                else:
                    search_done[i] = ep_step_count[i] >= ep_search_max_steps[i]

            done = env_done | directed_done | search_done
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)

            for idx in done_ids.tolist():
                steps = len(ep_obs[idx])

                if steps >= args.min_steps and saved < args.num_demos:
                    grp = hdf5_file.create_group(f"episode_{saved}")
                    grp.create_dataset("obs", data=np.array(ep_obs[idx]))
                    grp.create_dataset("actions", data=np.array(ep_act[idx]))
                    grp.create_dataset("robot_state",
                                       data=np.array(ep_robot_state[idx], dtype=np.float32))

                    if use_camera:
                        img_grp = grp.create_group("images")
                        if ep_base_img[idx]:
                            img_grp.create_dataset(
                                "base_rgb",
                                data=np.array(ep_base_img[idx], dtype=np.uint8),
                                compression="gzip", compression_opts=4,
                                chunks=(1, args.base_cam_height, args.base_cam_width, 3),
                            )
                        if ep_wrist_img[idx]:
                            img_grp.create_dataset(
                                "wrist_rgb",
                                data=np.array(ep_wrist_img[idx], dtype=np.uint8),
                                compression="gzip", compression_opts=4,
                                chunks=(1, args.wrist_cam_height, args.wrist_cam_width, 3),
                            )

                    grp.attrs["num_steps"] = steps
                    grp.attrs["mode"] = ep_mode[idx]
                    grp.attrs["instruction"] = ep_instruction[idx]
                    grp.attrs["success"] = True
                    grp.attrs["final_dist_to_object"] = float(dist_to_obj[idx].item())
                    grp.attrs["has_images"] = use_camera
                    obj_name = _get_object_name(env, idx)
                    grp.attrs["object_name"] = obj_name
                    if hasattr(env, "object_bbox"):
                        grp.attrs["object_bbox_xyz"] = (
                            env.object_bbox[idx].detach().cpu().numpy()
                            .astype(np.float32).tolist()
                        )
                    if hasattr(env, "active_object_idx"):
                        grp.attrs["active_object_type_idx"] = int(
                            env.active_object_idx[idx].item()
                        )

                    hdf5_file.flush()
                    saved += 1

                    if saved % 50 == 0 or saved <= 5:
                        print(
                            f"  Demo {saved:>4}/{args.num_demos} | "
                            f"mode={ep_mode[idx]:>8} | steps={steps:>4} | "
                            f"dist={dist_to_obj[idx]:.3f}m"
                        )

                # Reset buffers and reassign mode
                _clear_buffers(idx)
                ep_step_count[idx] = 0
                _assign_mode(idx)

            prev_action = action.clone()
            obs = next_obs

    except KeyboardInterrupt:
        print("\n  중단됨")
    finally:
        hdf5_file.close()

    print("\n" + "=" * 60)
    print(f"  Navigate 데이터 수집 완료: {saved}/{args.num_demos}")
    print(f"  파일: {output_path}")
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  파일 크기: {size_mb:.1f} MB")
    print(f"\n  다음 단계: convert_hdf5_to_lerobot_v3.py로 변환")
    print("=" * 60)

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
