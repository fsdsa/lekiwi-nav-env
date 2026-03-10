#!/usr/bin/env python3
"""
VLM + BC Navigate — Isaac Sim ProcTHOR 가정집 환경

VLM (Qwen2.5-VL-7B, A100 서버)이 2-4Hz로 방향 판단,
BC policy (3090 로컬)가 10Hz로 base velocity 실행.

구조:
    [3090 로컬]                              [A100 서버]
    Isaac Sim + D455 카메라                    vLLM serve
    BC policy (10Hz)                          Qwen2.5-VL-7B
         │                                        │
         │── RGB 640×400 JPEG ──→ HTTP POST ──→   │
         │←── "FORWARD" / "TURN_LEFT" ←─────────  │
         │                                        │
    depth safety layer
         │
    Robot action

Usage:
    # 1) A100 서버에서 VLM 서버 실행
    bash launch_vlm_server.sh

    # 2) 3090 로컬에서 실행
    python run_vlm_navigate.py \
        --vlm_server http://<서버IP>:8000 \
        --bc_checkpoint checkpoints/dp_bc_nav/dp_bc.pt \
        --target_object "medicine bottle" \
        --scene_idx 0
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import threading
import time

# ── AppLauncher 먼저 ──
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VLM + BC Navigate in ProcTHOR")

# VLM 서버
parser.add_argument("--vlm_server", type=str, required=True,
                    help="vLLM 서버 URL (예: http://123.456.78.90:8000)")
parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="vLLM에 로드된 모델 이름")

# BC Policy
parser.add_argument("--bc_checkpoint", type=str, required=True,
                    help="Navigate BC checkpoint 경로")
parser.add_argument("--resip_checkpoint", type=str, default="",
                    help="Navigate ResiP checkpoint (선택, 있으면 BC+RL)")
parser.add_argument("--inference_steps", type=int, default=4,
                    help="Diffusion inference steps")

# Task
parser.add_argument("--target_object", type=str, default="medicine bottle",
                    help="찾을 물체 이름")
parser.add_argument("--dest_object", type=str, default="red cup",
                    help="목적지 물체 이름 (Carry&Place용)")

# Scene
parser.add_argument("--scene_idx", type=int, default=0,
                    help="ProcTHOR 장면 인덱스")
parser.add_argument("--scene_usd", type=str, default="",
                    help="직접 지정할 USD 경로 (scene_idx 대신)")

# Camera
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=400)
parser.add_argument("--jpeg_quality", type=int, default=80,
                    help="JPEG 압축 품질 (70-90, 낮을수록 빠름)")

# Safety
parser.add_argument("--safety_dist", type=float, default=0.3,
                    help="depth safety layer 긴급 정지 거리 (m)")
parser.add_argument("--enable_safety", action="store_true", default=True)

# Control
parser.add_argument("--max_steps", type=int, default=3000,
                    help="최대 스텝 수 (10Hz 기준 300초)")
parser.add_argument("--vlm_interval", type=float, default=0.0,
                    help="VLM 호출 최소 간격 (0=최대한 빠르게)")

# Calibration (기존 환경과 동일)
parser.add_argument("--arm_limit_json", type=str, default="calibration/arm_limits_measured.json")
parser.add_argument("--dynamics_json", type=str, default="")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # 카메라 필수
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── 나머지 import ──
import json
import math
import numpy as np
import requests
import torch
from PIL import Image

import isaaclab.sim as sim_utils

from diffusion_policy import DiffusionPolicyAgent, ResidualPolicy
from vlm_prompts import (
    NAVIGATE_SYSTEM_PROMPT,
    NAVIGATE_USER_TEMPLATE,
    SKILL_TRANSITION_SYSTEM_PROMPT,
    SKILL_TRANSITION_USER_TEMPLATE,
    COMMAND_TO_DIRECTION,
    PHASE_TO_SKILL,
)


# ═══════════════════════════════════════════════════════════════════════
#  VLM Client (비동기 HTTP)
# ═══════════════════════════════════════════════════════════════════════

class VLMClient:
    """vLLM 서버와 HTTP 통신하는 클라이언트."""

    def __init__(self, server_url: str, model_name: str, jpeg_quality: int = 80):
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name
        self.jpeg_quality = jpeg_quality
        self._session = requests.Session()
        # Connection pool 재사용 (매번 TCP handshake 안 함)
        self._session.headers.update({"Content-Type": "application/json"})

        # 비동기 결과 버퍼
        self._lock = threading.Lock()
        self._latest_command = "FORWARD"
        self._latest_phase = "NAVIGATE_SEARCH"
        self._pending = False
        self._last_latency = 0.0

    def encode_image(self, rgb_array: np.ndarray) -> str:
        """numpy RGB array → base64 JPEG string."""
        img = Image.fromarray(rgb_array.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def query_navigate_command(self, rgb_array: np.ndarray, target_object: str) -> str:
        """동기 호출: RGB 이미지 → direction command 문자열."""
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
                json=payload,
                timeout=5.0,
            )
            resp.raise_for_status()
            result = resp.json()
            text = result["choices"][0]["message"]["content"].strip().upper()

            # 유효한 command만 파싱
            for cmd in COMMAND_TO_DIRECTION:
                if cmd in text:
                    self._last_latency = time.time() - t0
                    return cmd

            # 파싱 실패 → 이전 command 유지
            self._last_latency = time.time() - t0
            return self._latest_command

        except Exception as e:
            print(f"  [VLM] Error: {e}")
            self._last_latency = time.time() - t0
            return self._latest_command

    def query_skill_phase(self, rgb_array: np.ndarray,
                          target_object: str, dest_object: str) -> str:
        """동기 호출: RGB → skill phase 문자열."""
        t0 = time.time()
        b64_img = self.encode_image(rgb_array)

        sys_prompt = SKILL_TRANSITION_SYSTEM_PROMPT.format(
            target_object=target_object,
            dest_object=dest_object,
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_img}",
                    }},
                    {"type": "text", "text": SKILL_TRANSITION_USER_TEMPLATE},
                ]},
            ],
            "max_tokens": 15,
            "temperature": 0.0,
        }

        try:
            resp = self._session.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                timeout=5.0,
            )
            resp.raise_for_status()
            result = resp.json()
            text = result["choices"][0]["message"]["content"].strip().upper()

            for phase in PHASE_TO_SKILL:
                if phase in text:
                    self._last_latency = time.time() - t0
                    return phase

            self._last_latency = time.time() - t0
            return self._latest_phase

        except Exception as e:
            print(f"  [VLM] Phase error: {e}")
            self._last_latency = time.time() - t0
            return self._latest_phase

    def query_async(self, rgb_array: np.ndarray, target_object: str):
        """비동기 VLM 호출 (별도 스레드)."""
        if self._pending:
            return  # 이전 요청이 아직 진행 중

        self._pending = True

        def _worker():
            cmd = self.query_navigate_command(rgb_array, target_object)
            with self._lock:
                self._latest_command = cmd
                self._pending = False

        threading.Thread(target=_worker, daemon=True).start()

    def get_latest_command(self) -> str:
        with self._lock:
            return self._latest_command

    @property
    def latency(self) -> float:
        return self._last_latency

    @property
    def is_pending(self) -> bool:
        return self._pending


# ═══════════════════════════════════════════════════════════════════════
#  Depth Safety Layer
# ═══════════════════════════════════════════════════════════════════════

def depth_safety_check(depth_image: np.ndarray, action: np.ndarray,
                       safety_dist: float = 0.3) -> tuple[np.ndarray, bool]:
    """
    D455 depth 이미지로 긴급 정지 판단.

    depth_image: (H, W) float, 단위 m
    action: (9,) — v6 format [arm5, grip1, base3]
    safety_dist: 정지 거리 (m)

    Returns: (modified_action, stopped)
    """
    H, W = depth_image.shape[:2]

    # 전방 중앙 영역 (FOV 중심 1/3)
    y1, y2 = H // 3, 2 * H // 3
    x1, x2 = W // 3, 2 * W // 3
    center_depth = depth_image[y1:y2, x1:x2]

    # 유효 depth만 (0보다 크고 10m 이하)
    valid = (center_depth > 0.01) & (center_depth < 10.0)
    if valid.sum() < 10:
        return action, False  # depth 데이터 부족 → 통과

    min_depth = float(center_depth[valid].min())

    if min_depth < safety_dist:
        action = action.copy()
        # base velocity 차단 (v6: action[6:9])
        action[6] = 0.0  # vx
        action[7] = 0.0  # vy
        # wz는 유지 (제자리 회전은 허용)
        return action, True

    return action, False


# ═══════════════════════════════════════════════════════════════════════
#  BC Policy 로더
# ═══════════════════════════════════════════════════════════════════════

def load_bc_policy(checkpoint_path: str, device: torch.device,
                   inference_steps: int = 4) -> DiffusionPolicyAgent:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    agent = DiffusionPolicyAgent(
        obs_dim=cfg["obs_dim"],
        act_dim=cfg["act_dim"],
        pred_horizon=cfg["pred_horizon"],
        action_horizon=cfg["action_horizon"],
        num_diffusion_iters=cfg["num_diffusion_iters"],
        inference_steps=inference_steps,
        down_dims=cfg.get("down_dims", [256, 512, 1024]),
    ).to(device)

    sd = ckpt["model_state_dict"]
    agent.model.load_state_dict(
        {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")})
    agent.normalizer.load_state_dict(
        {k[len("normalizer."):]: v for k, v in sd.items() if k.startswith("normalizer.")},
        device=device)

    for p in agent.parameters():
        p.requires_grad = False
    agent.eval()

    print(f"  [BC] Loaded: obs={cfg['obs_dim']}, act={cfg['act_dim']}, "
          f"pred_h={cfg['pred_horizon']}, act_h={cfg['action_horizon']}")
    return agent


def load_residual_policy(checkpoint_path: str, obs_dim: int, act_dim: int,
                         device: torch.device) -> ResidualPolicy | None:
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return None

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    rpol = ResidualPolicy(
        obs_dim=obs_dim, action_dim=act_dim,
        actor_hidden_size=saved_args.get("actor_hidden_size", 256),
        actor_num_layers=saved_args.get("actor_num_layers", 2),
        critic_hidden_size=saved_args.get("critic_hidden_size", 256),
        critic_num_layers=saved_args.get("critic_num_layers", 2),
        action_scale=saved_args.get("action_scale", 0.1),
        init_logstd=saved_args.get("init_logstd", -1.0),
        action_head_std=saved_args.get("action_head_std", 0.0),
        learn_std=False,
    ).to(device)

    rpol.load_state_dict(ckpt["residual_policy_state_dict"])
    rpol.eval()
    print(f"  [RL] Loaded residual policy: {checkpoint_path}")
    return rpol


# ═══════════════════════════════════════════════════════════════════════
#  Isaac Sim 환경 + D455 카메라 세팅
# ═══════════════════════════════════════════════════════════════════════

def setup_env_with_camera(args):
    """Navigate 환경 + D455 카메라 생성."""
    from lekiwi_skill1_env import Skill1Env, Skill1EnvCfg

    cfg = Skill1EnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.enable_domain_randomization = False
    cfg.arm_limit_write_to_sim = False
    cfg.force_tucked_pose = True
    cfg.episode_length_s = 600.0  # 10분
    cfg.obstacle_none_prob = 1.0  # tensor 장애물 비활성 (집 벽이 장애물)

    if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
        cfg.arm_limit_json = args.arm_limit_json
    if args.dynamics_json and os.path.isfile(args.dynamics_json):
        cfg.dynamics_json = args.dynamics_json

    env = Skill1Env(cfg=cfg)

    # ── ProcTHOR 장면 로드 ──
    if args.scene_usd:
        scene_usda = args.scene_usd
    else:
        scene_dir = f"assets/usd/scenes/procthor-10k-train/train_{args.scene_idx}"
        scene_usda = os.path.join(scene_dir, "scene.usda")

    if os.path.isfile(scene_usda):
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        scene_prim = stage.DefinePrim("/World/ProcTHORScene", "Xform")
        scene_prim.GetReferences().AddReference(str(os.path.abspath(scene_usda)))
        stage.Load()
        print(f"  [Scene] ProcTHOR loaded: {scene_usda}")
    else:
        print(f"  [Scene] WARNING: {scene_usda} not found, running without house")

    # ── D455 카메라 (Isaac Sim TiledCamera) ──
    # 카메라는 env에 직접 안 넣고 별도로 생성
    # 로봇 body에 attach하는 방식
    from isaaclab.sensors import Camera

    camera = Camera(CameraCfg(
        prim_path="/World/envs/env_0/Robot/D455Camera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,         # D455 FOV 87° 매칭
            horizontal_aperture=3.68,
        ),
        width=args.camera_width,
        height=args.camera_height,
        update_period=1 / 30,  # 30Hz 렌더링 (VLM이 2-4Hz로 가져감)
        data_types=["rgb", "distance_to_image_plane"],
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.05, 0.22),  # D455 mount position on LeKiwi
            rot=(0.5, -0.5, 0.5, -0.5),  # ROS convention
            convention="ros",
        ),
    ))

    return env, camera


# ═══════════════════════════════════════════════════════════════════════
#  메인 루프
# ═══════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda:0")

    # ── 환경 + 카메라 ──
    env, camera = setup_env_with_camera(args)
    print(f"\n  [Env] Navigate environment ready")
    print(f"  [Camera] D455 sim: {args.camera_width}×{args.camera_height}")

    # ── VLM 클라이언트 ──
    vlm = VLMClient(args.vlm_server, args.vlm_model, args.jpeg_quality)
    print(f"  [VLM] Server: {args.vlm_server}")
    print(f"  [VLM] Model: {args.vlm_model}")

    # VLM 연결 테스트
    print(f"  [VLM] Testing connection...")
    try:
        test_img = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
        test_cmd = vlm.query_navigate_command(test_img, args.target_object)
        print(f"  [VLM] Connection OK (test={test_cmd}, latency={vlm.latency:.0f}ms)")
    except Exception as e:
        print(f"  [VLM] Connection FAILED: {e}")
        print(f"  [VLM] A100 서버에서 launch_vlm_server.sh 실행했는지 확인!")
        env.close()
        simulation_app.close()
        return

    # ── BC Policy ──
    bc_policy = load_bc_policy(args.bc_checkpoint, device, args.inference_steps)
    obs_dim = bc_policy.normalizer.params_dict["obs"]["input_stats"]["min"].shape[-1]
    act_dim = bc_policy.normalizer.params_dict["action"]["input_stats"]["min"].shape[-1]

    # ── Residual Policy (선택) ──
    resip = load_residual_policy(args.resip_checkpoint, obs_dim, act_dim, device)
    resip_scale = torch.zeros(act_dim, device=device)
    resip_scale[6:9] = 0.25  # base only
    use_resip = resip is not None

    # ── 상태 초기화 ──
    obs, _ = env.reset()
    bc_policy.reset()
    current_command = "FORWARD"
    direction_cmd = torch.tensor(COMMAND_TO_DIRECTION["FORWARD"],
                                 dtype=torch.float32, device=device)

    # ── 타이밍 ──
    last_vlm_time = 0.0
    vlm_call_count = 0
    vlm_total_latency = 0.0
    safety_stop_count = 0
    step_count = 0

    print(f"\n{'='*60}")
    print(f"  VLM + BC Navigate")
    print(f"  Target: {args.target_object}")
    print(f"  Safety: {'ON' if args.enable_safety else 'OFF'} ({args.safety_dist}m)")
    print(f"  Residual RL: {'ON' if use_resip else 'OFF'}")
    print(f"  Max steps: {args.max_steps}")
    print(f"{'='*60}\n")

    # ═══════════════════════════════════════════════════════════════════
    #  메인 제어 루프 (10Hz)
    # ═══════════════════════════════════════════════════════════════════

    try:
        while step_count < args.max_steps and simulation_app.is_running():
            loop_start = time.time()

            # ── 1. 카메라 이미지 캡처 ──
            camera.update(dt=env.sim.cfg.dt)
            rgb_data = camera.data.output.get("rgb")
            depth_data = camera.data.output.get("distance_to_image_plane")

            has_image = rgb_data is not None and rgb_data.numel() > 0

            # ── 2. VLM 비동기 호출 (이미지 있을 때만) ──
            now = time.time()
            vlm_ready = (now - last_vlm_time) >= args.vlm_interval
            if has_image and vlm_ready and not vlm.is_pending:
                # (H, W, 4) RGBA → (H, W, 3) RGB numpy
                rgb_np = rgb_data[0, :, :, :3].cpu().numpy().astype(np.uint8)
                vlm.query_async(rgb_np, args.target_object)
                last_vlm_time = now

            # ── 3. VLM 결과 반영 ──
            new_command = vlm.get_latest_command()
            if new_command != current_command:
                print(f"  [VLM] {current_command} → {new_command} "
                      f"(latency={vlm.latency*1000:.0f}ms)")
                current_command = new_command
                vlm_call_count += 1
                vlm_total_latency += vlm.latency

            # direction_cmd 업데이트
            cmd_vec = COMMAND_TO_DIRECTION.get(current_command, [0, 0, 0])
            direction_cmd = torch.tensor(cmd_vec, dtype=torch.float32, device=device)

            # ── 4. obs에 direction_cmd 주입 ──
            obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
            obs_tensor = obs_tensor.to(device)

            # obs[9:12] = direction_cmd (기존 환경이 넣어주지만, 여기서 VLM 결과로 덮어씀)
            obs_tensor[0, 9:12] = direction_cmd

            # obs[12:20] = lidar → 전부 1.0 (장애물 없음 — VLM이 장애물 판단)
            obs_tensor[0, 12:20] = 1.0

            # ── 5. BC Policy 실행 ──
            with torch.no_grad():
                base_naction = bc_policy.base_action_normalized(obs_tensor)

                if use_resip:
                    nobs = bc_policy.normalizer(obs_tensor, "obs", forward=True)
                    nobs = torch.clamp(nobs, -3, 3)
                    residual_input = torch.cat([nobs, base_naction], dim=-1)
                    residual_naction = resip.actor_mean(residual_input)
                    residual_naction = torch.clamp(residual_naction, -1.0, 1.0)
                    naction = base_naction + residual_naction * resip_scale
                else:
                    naction = base_naction

                action = bc_policy.normalizer(naction, "action", forward=False)

            action_np = action[0].cpu().numpy()

            # ── 6. STOP command 처리 ──
            if current_command == "STOP":
                action_np[6:9] = 0.0  # base 정지

            # ── 7. Depth Safety Layer ──
            stopped = False
            if args.enable_safety and depth_data is not None and depth_data.numel() > 0:
                depth_np = depth_data[0, :, :, 0].cpu().numpy()
                action_np, stopped = depth_safety_check(
                    depth_np, action_np, args.safety_dist
                )
                if stopped:
                    safety_stop_count += 1

            # ── 8. 환경 step ──
            action_tensor = torch.tensor(action_np, dtype=torch.float32,
                                         device=device).unsqueeze(0)
            obs, reward, terminated, truncated, info = env.step(action_tensor)
            step_count += 1

            # ── 9. 로깅 ──
            if step_count % 50 == 0:
                vx = env.robot.data.root_lin_vel_b[0, 0].item()
                vy = env.robot.data.root_lin_vel_b[0, 1].item()
                wz = env.robot.data.root_ang_vel_b[0, 2].item()
                pos = env.robot.data.root_pos_w[0, :2].cpu().numpy()
                avg_vlm_ms = (vlm_total_latency / max(vlm_call_count, 1)) * 1000

                safety_str = f" SAFETY_STOP" if stopped else ""
                print(
                    f"  [t={step_count:4d}] cmd={current_command:12s} | "
                    f"vel=(vx={vx:+.2f},vy={vy:+.2f},wz={wz:+.2f}) | "
                    f"pos=({pos[0]:+.2f},{pos[1]:+.2f}) | "
                    f"vlm={avg_vlm_ms:.0f}ms ({vlm_call_count} calls) | "
                    f"safety_stops={safety_stop_count}{safety_str}"
                )

            # ── 10. 에피소드 종료 체크 ──
            if terminated.any() or truncated.any():
                print(f"\n  [DONE] Episode ended at step {step_count}")
                obs, _ = env.reset()
                bc_policy.reset()
                current_command = "FORWARD"
                step_count = 0

    except KeyboardInterrupt:
        print("\n\n  중단됨 (Ctrl+C)")

    # ── 요약 ──
    avg_vlm_ms = (vlm_total_latency / max(vlm_call_count, 1)) * 1000
    print(f"\n{'='*60}")
    print(f"  실행 완료")
    print(f"  총 스텝: {step_count}")
    print(f"  VLM 호출: {vlm_call_count}회, 평균 {avg_vlm_ms:.0f}ms")
    print(f"  Safety 정지: {safety_stop_count}회")
    print(f"{'='*60}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
