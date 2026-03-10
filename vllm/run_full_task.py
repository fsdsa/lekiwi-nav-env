#!/usr/bin/env python3
"""
VLM + VLA 전체 태스크 오케스트레이터

배포 아키텍처: VLM(instruction) + VLA(단일 Pi0-FAST, 9D action)
- VLM: base_cam RGB → instruction 생성 (2-4Hz)
- VLA: base_cam + wrist_cam + 9D state + instruction → 9D action chunk (5-10Hz)
- depth safety: base_cam depth → 전방 장애물 시 base 정지

3090 Desktop (Isaac Sim) ↔ A100 서버 (VLM + VLA) HTTP 통신

Usage:
    python run_full_task.py \
        --vlm_server http://<서버IP>:8000 \
        --vla_server http://<서버IP>:8002 \
        --target_object "medicine bottle" \
        --dest_object "red cup" \
        --object_usd <path> \
        --headless
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Full Task: VLM + VLA")

# Servers
parser.add_argument("--vlm_server", type=str, default="http://218.148.55.186:8000")
parser.add_argument("--vla_server", type=str, default="http://218.148.55.186:8002")
parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")

# Task
parser.add_argument("--target_object", type=str, default="medicine bottle")
parser.add_argument("--dest_object", type=str, default="red cup")
parser.add_argument("--num_trials", type=int, default=1)

# Camera
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=400)
parser.add_argument("--jpeg_quality", type=int, default=80)

# Safety
parser.add_argument("--safety_dist", type=float, default=0.3)
parser.add_argument("--enable_safety", action="store_true", default=True)

# Timing
parser.add_argument("--vlm_interval", type=int, default=30,
                    help="VLM query interval in sim steps (~3s at 10Hz)")
parser.add_argument("--max_total_steps", type=int, default=6000)

# Scene
parser.add_argument("--scene_usd", type=str, default="")
parser.add_argument("--scene_idx", type=int, default=0)

# Env
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--arm_limit_json", type=str, default="calibration/arm_limits_measured.json")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import base64
import io
import threading

import numpy as np
import requests
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg

from vlm_prompts import (
    NAVIGATE_SYSTEM_PROMPT,
    NAVIGATE_USER_TEMPLATE,
    SKILL_TRANSITION_SYSTEM_PROMPT,
    SKILL_TRANSITION_USER_TEMPLATE,
    COMMAND_TO_DIRECTION,
    PHASE_TO_SKILL,
)


# ═══════════════════════════════════════════════════════════════════════
#  VLM Client (비동기 HTTP → vLLM OpenAI API)
# ═══════════════════════════════════════════════════════════════════════

class VLMClient:
    """vLLM 서버와 HTTP 통신하는 클라이언트."""

    def __init__(self, server_url: str, model_name: str, jpeg_quality: int = 80):
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name
        self.jpeg_quality = jpeg_quality
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        self._lock = threading.Lock()
        self._latest_command = "FORWARD"
        self._latest_phase = "NAVIGATE_SEARCH"
        self._latest_instruction = "explore the room to find the target object"
        self._pending = False
        self._last_latency = 0.0

    def encode_image(self, rgb_array: np.ndarray) -> str:
        img = Image.fromarray(rgb_array.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def query_navigate_command(self, rgb_array: np.ndarray, target_object: str) -> str:
        t0 = time.time()
        b64_img = self.encode_image(rgb_array)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": NAVIGATE_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_img}",
                    }},
                    {"type": "text", "text": NAVIGATE_USER_TEMPLATE.format(
                        target_object=target_object
                    )},
                ]},
            ],
            "max_tokens": 10,
            "temperature": 0.0,
        }
        try:
            resp = self._session.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload, timeout=5.0,
            )
            resp.raise_for_status()
            result = resp.json()
            raw = result["choices"][0]["message"]["content"].strip().upper()
            cmd = raw.split()[0] if raw else "FORWARD"
            if cmd not in COMMAND_TO_DIRECTION:
                cmd = "FORWARD"
            self._last_latency = time.time() - t0
            return cmd
        except Exception as e:
            print(f"  [VLM] error: {e}")
            return self._latest_command

    def query_skill_phase(self, rgb_array: np.ndarray,
                          target_object: str, dest_object: str) -> str:
        t0 = time.time()
        b64_img = self.encode_image(rgb_array)
        system_prompt = SKILL_TRANSITION_SYSTEM_PROMPT.format(
            target_object=target_object, dest_object=dest_object,
        )
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_img}",
                    }},
                    {"type": "text", "text": SKILL_TRANSITION_USER_TEMPLATE},
                ]},
            ],
            "max_tokens": 20,
            "temperature": 0.0,
        }
        valid_phases = set(PHASE_TO_SKILL.keys())
        try:
            resp = self._session.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload, timeout=5.0,
            )
            resp.raise_for_status()
            result = resp.json()
            raw = result["choices"][0]["message"]["content"].strip().upper()
            phase = raw.split()[0] if raw else self._latest_phase
            if phase not in valid_phases:
                phase = self._latest_phase
            self._last_latency = time.time() - t0
            return phase
        except Exception as e:
            print(f"  [VLM] phase error: {e}")
            return self._latest_phase

    def query_async(self, rgb_array: np.ndarray,
                    target_object: str, dest_object: str, current_phase: str):
        """비동기 VLM 쿼리 (별도 스레드)."""
        if self._pending:
            return
        self._pending = True

        def _worker():
            try:
                # Navigate phase면 방향 command, 아니면 skill phase 판단
                if "NAVIGATE" in current_phase:
                    cmd = self.query_navigate_command(rgb_array, target_object)
                    with self._lock:
                        self._latest_command = cmd
                phase = self.query_skill_phase(rgb_array, target_object, dest_object)
                with self._lock:
                    self._latest_phase = phase
                    self._latest_instruction = self._phase_to_instruction(
                        phase, target_object, dest_object
                    )
            finally:
                self._pending = False

        threading.Thread(target=_worker, daemon=True).start()

    def get_latest(self) -> tuple[str, str, str]:
        """최신 (command, phase, instruction) 반환."""
        with self._lock:
            return self._latest_command, self._latest_phase, self._latest_instruction

    @staticmethod
    def _phase_to_instruction(phase: str, target: str, dest: str) -> str:
        """Phase → VLA instruction 문자열."""
        mapping = {
            "NAVIGATE_SEARCH": f"explore the room to find the {target}",
            "NAVIGATE_TO_TARGET": f"move toward the {target}",
            "APPROACH_AND_LIFT": f"approach and pick up the {target}",
            "NAVIGATE_TO_DEST": f"carry the {target} and find the {dest}",
            "NAVIGATE_TO_DEST_CLOSE": f"move toward the {dest} while holding the {target}",
            "CARRY_AND_PLACE": f"place the {target} next to the {dest}",
        }
        return mapping.get(phase, f"find the {target}")


# ═══════════════════════════════════════════════════════════════════════
#  VLA Client (HTTP → Pi0-FAST FastAPI)
# ═══════════════════════════════════════════════════════════════════════

class VLAClient:
    """Pi0-FAST VLA 서버와 HTTP 통신하는 클라이언트."""

    def __init__(self, server_url: str, jpeg_quality: int = 80):
        self.server_url = server_url.rstrip("/")
        self.jpeg_quality = jpeg_quality
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._action_buffer: list[list[float]] = []
        self._buffer_idx = 0

    def encode_image(self, rgb_array: np.ndarray) -> str:
        img = Image.fromarray(rgb_array.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def query_action(self, base_rgb: np.ndarray, wrist_rgb: np.ndarray,
                     state_9d: list[float], instruction: str) -> list[list[float]]:
        """동기 호출: images + state + instruction → action chunk."""
        payload = {
            "base_image": self.encode_image(base_rgb),
            "wrist_image": self.encode_image(wrist_rgb),
            "state": state_9d,
            "instruction": instruction,
        }
        try:
            resp = self._session.post(
                f"{self.server_url}/infer",
                json=payload, timeout=5.0,
            )
            resp.raise_for_status()
            result = resp.json()
            return result["actions"]
        except Exception as e:
            print(f"  [VLA] error: {e}")
            return []

    def get_action(self, base_rgb: np.ndarray, wrist_rgb: np.ndarray,
                   state_9d: list[float], instruction: str) -> np.ndarray | None:
        """Action chunk 버퍼에서 1개씩 반환. 버퍼 비면 새 쿼리."""
        if self._buffer_idx >= len(self._action_buffer):
            self._action_buffer = self.query_action(
                base_rgb, wrist_rgb, state_9d, instruction
            )
            self._buffer_idx = 0
            if not self._action_buffer:
                return None

        action = np.array(self._action_buffer[self._buffer_idx], dtype=np.float32)
        self._buffer_idx += 1
        return action

    def reset_buffer(self):
        self._action_buffer = []
        self._buffer_idx = 0

    def health_check(self) -> bool:
        try:
            resp = self._session.get(f"{self.server_url}/health", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════
#  Depth Safety Layer
# ═══════════════════════════════════════════════════════════════════════

def depth_safety_check(depth_image: np.ndarray, action: np.ndarray,
                       safety_dist: float = 0.3) -> tuple[np.ndarray, bool]:
    """
    D455 depth 이미지로 긴급 정지 판단.
    전방 중앙 1/3 영역의 min depth < safety_dist → base velocity 0.
    """
    H, W = depth_image.shape[:2]
    y1, y2 = H // 3, 2 * H // 3
    x1, x2 = W // 3, 2 * W // 3
    center_depth = depth_image[y1:y2, x1:x2]

    valid = (center_depth > 0.01) & (center_depth < 10.0)
    if valid.sum() < 10:
        return action, False

    min_depth = center_depth[valid].min()
    if min_depth < safety_dist:
        action = action.copy()
        action[6:9] = 0.0  # base velocity 정지
        return action, True

    return action, False


# ═══════════════════════════════════════════════════════════════════════
#  Isaac Sim 환경 + 카메라 세팅
# ═══════════════════════════════════════════════════════════════════════

def setup_env_with_cameras(args):
    """Skill2 환경 + base_cam(RGB+depth) + wrist_cam 생성."""
    from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg

    cfg = Skill2EnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.enable_domain_randomization = False
    cfg.arm_limit_write_to_sim = False
    cfg.episode_length_s = 600.0

    if args.object_usd:
        cfg.object_usd = os.path.expanduser(args.object_usd)
    if args.multi_object_json:
        cfg.multi_object_json = os.path.expanduser(args.multi_object_json)
    if args.dest_object_usd:
        cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
    if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
        cfg.arm_limit_json = args.arm_limit_json
    cfg.gripper_contact_prim_path = args.gripper_contact_prim_path

    env = Skill2Env(cfg=cfg)

    # ── base_cam: USD 내장 RealSense D455 RGB+Depth ──
    base_cam = Camera(CameraCfg(
        prim_path="/World/envs/env_0/Robot/LeKiwi/base_plate_layer1_v5"
                  "/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
        update_period=0.0,
        width=args.camera_width,
        height=args.camera_height,
        data_types=["rgb", "distance_to_image_plane"],
    ))

    # ── wrist_cam: USD 내장 wrist camera ──
    wrist_cam = Camera(CameraCfg(
        prim_path="/World/envs/env_0/Robot/LeKiwi"
                  "/Wrist_Roll_08c_v1/visuals/mesh_002_3/wrist_camera",
        update_period=0.0,
        width=args.camera_width,
        height=args.camera_height,
        data_types=["rgb"],
    ))

    return env, base_cam, wrist_cam


def get_robot_state_9d(env) -> list[float]:
    """현재 로봇 상태 9D: arm_pos(5) + gripper(1) + base_vel(3)."""
    joint_pos = env.robot.data.joint_pos[0]
    arm_pos = joint_pos[env.arm_idx[:5]].tolist()
    grip = joint_pos[env.gripper_idx].item()
    base_vel = env.robot.data.root_lin_vel_b[0].tolist()  # (vx, vy, vz)
    return arm_pos + [grip] + base_vel[:2] + [env.robot.data.root_ang_vel_b[0, 2].item()]


def capture_cameras(base_cam: Camera, wrist_cam: Camera, dt: float):
    """카메라 업데이트 후 RGB, depth numpy array 반환."""
    base_cam.update(dt=dt)
    wrist_cam.update(dt=dt)

    base_rgb_data = base_cam.data.output.get("rgb")
    depth_data = base_cam.data.output.get("distance_to_image_plane")
    wrist_rgb_data = wrist_cam.data.output.get("rgb")

    # (N, H, W, C) → (H, W, C) numpy, env 0
    base_rgb = base_rgb_data[0, ..., :3].cpu().numpy() if base_rgb_data is not None else None
    depth = depth_data[0, ..., 0].cpu().numpy() if depth_data is not None else None
    wrist_rgb = wrist_rgb_data[0, ..., :3].cpu().numpy() if wrist_rgb_data is not None else None

    return base_rgb, depth, wrist_rgb


# ═══════════════════════════════════════════════════════════════════════
#  메인 루프
# ═══════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda:0")

    # ── 클라이언트 ──
    vlm = VLMClient(args.vlm_server, args.vlm_model, args.jpeg_quality)
    vla = VLAClient(args.vla_server, args.jpeg_quality)

    print(f"\n{'='*60}")
    print(f"  Full Task: VLM + VLA (Pi0-FAST)")
    print(f"  VLM: {args.vlm_server}")
    print(f"  VLA: {args.vla_server}")
    print(f"  Target: {args.target_object} → Dest: {args.dest_object}")
    print(f"  Camera: {args.camera_width}x{args.camera_height}")
    print(f"  Safety: dist={args.safety_dist}m, enabled={args.enable_safety}")
    print(f"{'='*60}\n")

    # ── 환경 + 카메라 ──
    env, base_cam, wrist_cam = setup_env_with_cameras(args)
    dt = env.sim.cfg.dt

    print("  Resetting environment...")
    obs, info = env.reset()

    # ── 서버 헬스 체크 ──
    vla_ok = vla.health_check()
    print(f"  [VLA] health: {'OK' if vla_ok else 'FAIL'}")
    if not vla_ok:
        print("  [WARN] VLA server not reachable. Actions will be zero.")

    # ── 상태 ──
    current_phase = "NAVIGATE_SEARCH"
    current_command = "FORWARD"
    current_instruction = f"explore the room to find the {args.target_object}"
    total_steps = 0
    phase_steps = 0
    safety_stops = 0
    phase_history: list[tuple[str, int]] = []

    for trial in range(args.num_trials):
        print(f"\n  === Trial {trial + 1}/{args.num_trials} ===")
        obs, info = env.reset()
        current_phase = "NAVIGATE_SEARCH"
        total_steps = 0
        phase_steps = 0
        vla.reset_buffer()

        while current_phase != "DONE" and total_steps < args.max_total_steps:
            if not simulation_app.is_running():
                break

            # 1. 카메라 캡처
            base_rgb, depth, wrist_rgb = capture_cameras(base_cam, wrist_cam, dt)

            if base_rgb is None or wrist_rgb is None:
                # 카메라 데이터 아직 준비 안 됨
                env.sim.step()
                total_steps += 1
                continue

            # 2. VLM 비동기 쿼리 (vlm_interval 마다)
            if total_steps % args.vlm_interval == 0:
                vlm.query_async(
                    base_rgb, args.target_object, args.dest_object, current_phase
                )

            # 3. VLM 최신 결과 가져오기
            cmd, phase, instruction = vlm.get_latest()
            if phase != current_phase:
                print(f"  Phase: {current_phase} → {phase} (step {total_steps})")
                phase_history.append((current_phase, phase_steps))
                current_phase = phase
                phase_steps = 0
                vla.reset_buffer()
            current_command = cmd
            current_instruction = instruction

            # 4. VLA action 생성
            state_9d = get_robot_state_9d(env)
            action = vla.get_action(base_rgb, wrist_rgb, state_9d, current_instruction)

            if action is None:
                action = np.zeros(9, dtype=np.float32)

            # 5. Depth safety (Navigate phase에서만)
            stopped = False
            if args.enable_safety and "NAVIGATE" in current_phase and depth is not None:
                action, stopped = depth_safety_check(depth, action, args.safety_dist)
                if stopped:
                    safety_stops += 1

            # 6. env step
            action_tensor = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            obs, reward, terminated, truncated, info = env.step(action_tensor)
            total_steps += 1
            phase_steps += 1

            # 7. 로그
            if total_steps % 50 == 0:
                stop_str = " [SAFETY STOP]" if stopped else ""
                print(f"    [t={total_steps:4d}] phase={current_phase} cmd={current_command} "
                      f"inst=\"{current_instruction[:40]}...\"{stop_str}")

            # 8. 종료 체크
            done = terminated.any() or truncated.any()
            if done:
                success = info.get("task_success", torch.zeros(1)).any().item()
                result = "SUCCESS" if success else "TIMEOUT"
                print(f"  Trial {trial + 1} → {result} | steps={total_steps} "
                      f"safety_stops={safety_stops}")
                phase_history.append((current_phase, phase_steps))
                break

        # Trial 요약
        print(f"  Phase history: {phase_history}")
        phase_history.clear()

    simulation_app.close()


if __name__ == "__main__":
    main()
