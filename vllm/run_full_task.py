#!/usr/bin/env python3
"""
VLM + VLA 전체 태스크 실행기

아키텍처:
  [A100 서버 — 최초 1회]
    유저 지시어 → VLM /classify → source_object + dest_object 추출

  [실행 루프]
    3090 로컬 (Isaac Sim + 카메라)
      ├─ base_cam + wrist_cam + depth (omni.replicator)
      └─ depth < 0.3m → 긴급정지

    A100 서버
      ├─ VLM (0.3Hz, 비동기): base_cam → 자연어 instruction 생성
      └─ VLA (5-10Hz, 동기): base_cam + wrist_cam + instruction + 9D state → 9D action

    3090 로컬
      └─ env.step(action)

Usage:
    # SSH 터널 먼저
    ssh -f -N -L 8000:localhost:8000 -L 8002:localhost:8002 jovyan@218.148.55.186 -p 30179

    python run_full_task.py \
        --user_command "약병을 찾아서 빨간 컵 옆에 놓아" \
        --object_usd <path> \
        --headless
"""
from __future__ import annotations

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Full Task: VLM + VLA")

# Servers
parser.add_argument("--vlm_server", type=str, default="http://localhost:8000")
parser.add_argument("--vla_server", type=str, default="http://localhost:8002")
parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")

# Task — 유저가 직접 입력하는 지시어
parser.add_argument("--user_command", type=str,
                    default="find the medicine bottle and place it next to the red cup",
                    help="유저 지시어 (VLM이 source/dest 자동 추출)")
# 수동 override (VLM classify 건너뛸 때)
parser.add_argument("--target_object", type=str, default="",
                    help="수동 지정 시 VLM classify 건너뜀")
parser.add_argument("--dest_object", type=str, default="",
                    help="수동 지정 시 VLM classify 건너뜀")

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
                    help="VLM 호출 간격 (sim steps, 30 = ~3s at 10Hz)")
parser.add_argument("--max_total_steps", type=int, default=6000,
                    help="최대 스텝 (6000 = 10분 at 10Hz)")

# Env
parser.add_argument("--object_usd", type=str, default="")
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

import numpy as np
import requests
import torch
from PIL import Image

import omni.replicator.core as rep

from vlm_orchestrator import classify_user_request, RelativePlacementOrchestrator


# ═══════════════════════════════════════════════════════════════════════
#  VLA Client
# ═══════════════════════════════════════════════════════════════════════

class VLAClient:
    """Pi0-FAST VLA 서버 클라이언트. Action chunk 버퍼링 지원."""

    def __init__(self, server_url: str, jpeg_quality: int = 80):
        self.server_url = server_url.rstrip("/")
        self.jpeg_quality = jpeg_quality
        self._session = requests.Session()
        self._action_buffer: list[list[float]] = []
        self._buffer_idx = 0
        self._last_latency = 0.0

    def encode_image(self, rgb_array: np.ndarray) -> str:
        img = Image.fromarray(rgb_array.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def query_action(self, base_rgb: np.ndarray, wrist_rgb: np.ndarray,
                     state_9d: list[float], instruction: str) -> list[list[float]]:
        """동기 호출: images + state + instruction → action chunk."""
        t0 = time.perf_counter()
        payload = {
            "base_image_b64": self.encode_image(base_rgb),
            "wrist_image_b64": self.encode_image(wrist_rgb),
            "state": state_9d,
            "instruction": instruction,
        }
        try:
            resp = self._session.post(
                f"{self.server_url}/act", json=payload, timeout=10.0,
            )
            resp.raise_for_status()
            result = resp.json()
            self._last_latency = time.perf_counter() - t0
            return result["actions"]
        except Exception as e:
            print(f"  [VLA] error: {e}")
            self._last_latency = time.perf_counter() - t0
            return []

    def get_action_9d(self, base_rgb: np.ndarray, wrist_rgb: np.ndarray,
                      state_9d: list[float], instruction: str) -> np.ndarray:
        """Action chunk에서 1개 반환 (9D). 버퍼 소진 시 새 쿼리."""
        if self._buffer_idx >= len(self._action_buffer):
            self._action_buffer = self.query_action(
                base_rgb, wrist_rgb, state_9d, instruction
            )
            self._buffer_idx = 0
            if not self._action_buffer:
                return np.zeros(9, dtype=np.float32)

        raw = np.array(self._action_buffer[self._buffer_idx], dtype=np.float32)
        self._buffer_idx += 1
        # 32D → 9D truncate (fine-tuned 모델은 9D 직접 반환)
        if len(raw) >= 9:
            return raw[:9]
        return np.pad(raw, (0, 9 - len(raw)))

    def reset_buffer(self):
        self._action_buffer = []
        self._buffer_idx = 0

    def health_check(self) -> dict | None:
        try:
            resp = self._session.get(f"{self.server_url}/health", timeout=3.0)
            return resp.json() if resp.status_code == 200 else None
        except Exception:
            return None

    @property
    def latency(self) -> float:
        return self._last_latency


# ═══════════════════════════════════════════════════════════════════════
#  Depth Safety Layer
# ═══════════════════════════════════════════════════════════════════════

def depth_safety_check(depth_image: np.ndarray, action: np.ndarray,
                       safety_dist: float = 0.3) -> tuple[np.ndarray, bool]:
    """전방 중앙 1/3 min depth < safety_dist → base 정지 (회전은 허용)."""
    H, W = depth_image.shape[:2]
    center = depth_image[H//3:2*H//3, W//3:2*W//3]
    valid = (center > 0.01) & (center < 10.0)
    if valid.sum() < 10:
        return action, False
    if center[valid].min() < safety_dist:
        action = action.copy()
        action[6:8] = 0.0  # vx, vy 정지, wz(회전)는 유지
        return action, True
    return action, False


# ═══════════════════════════════════════════════════════════════════════
#  Isaac Sim 환경 + 카메라
# ═══════════════════════════════════════════════════════════════════════

def setup_env(args):
    """Skill2 환경 + omni.replicator 카메라."""
    from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg

    cfg = Skill2EnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.enable_domain_randomization = False
    cfg.arm_limit_write_to_sim = False
    cfg.episode_length_s = 600.0

    if args.object_usd:
        cfg.object_usd = os.path.expanduser(args.object_usd)
    if args.dest_object_usd:
        cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
    if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
        cfg.arm_limit_json = args.arm_limit_json
    cfg.gripper_contact_prim_path = args.gripper_contact_prim_path

    env = Skill2Env(cfg=cfg)

    # omni.replicator 카메라
    base_rp = rep.create.render_product(
        "/World/envs/env_0/Robot/LeKiwi/base_plate_layer1_v5"
        "/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
        (args.camera_width, args.camera_height),
    )
    wrist_rp = rep.create.render_product(
        "/World/envs/env_0/Robot/LeKiwi"
        "/Wrist_Roll_08c_v1/visuals/mesh_002_3/wrist_camera",
        (args.camera_width, args.camera_height),
    )

    base_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    base_rgb_annot.attach([base_rp])
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    depth_annot.attach([base_rp])
    wrist_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    wrist_rgb_annot.attach([wrist_rp])

    cams = {"base_rgb": base_rgb_annot, "depth": depth_annot, "wrist_rgb": wrist_rgb_annot}
    return env, cams


def capture(env, cams: dict):
    """render + annotator read → numpy arrays."""
    env.sim.render()
    b = cams["base_rgb"].get_data()
    d = cams["depth"].get_data()
    w = cams["wrist_rgb"].get_data()
    base_rgb = np.array(b)[..., :3] if b is not None else None
    depth = np.array(d) if d is not None else None
    wrist_rgb = np.array(w)[..., :3] if w is not None else None
    return base_rgb, depth, wrist_rgb


def get_state_9d(env) -> list[float]:
    jp = env.robot.data.joint_pos[0]
    arm = jp[env.arm_idx[:5]].tolist()
    grip = jp[env.gripper_idx].item()
    bv = env.robot.data.root_lin_vel_b[0].tolist()
    wz = env.robot.data.root_ang_vel_b[0, 2].item()
    return arm + [grip] + bv[:2] + [wz]


# ═══════════════════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda:0")

    # ── 0. 서버 헬스 체크 ──
    print(f"\n{'='*60}")
    print(f"  Full Task: VLM + VLA")
    print(f"  VLM: {args.vlm_server} ({args.vlm_model})")
    print(f"  VLA: {args.vla_server}")
    print(f"{'='*60}")

    print(f"\n  [Health Check]")
    try:
        r = requests.get(f"{args.vlm_server}/v1/models", timeout=3)
        print(f"  VLM: OK ({r.json()['data'][0]['id']})")
    except Exception as e:
        print(f"  VLM: FAIL — {e}")
        print(f"  → SSH 터널: ssh -f -N -L 8000:localhost:8000 -L 8002:localhost:8002 ...")
        simulation_app.close()
        return

    vla = VLAClient(args.vla_server, args.jpeg_quality)
    vla_health = vla.health_check()
    if vla_health:
        print(f"  VLA: OK ({vla_health['model']}, {vla_health['gpu_memory_mb']:.0f}MB)")
    else:
        print(f"  VLA: FAIL")
        simulation_app.close()
        return

    # ── 1. VLM /classify: 유저 지시어 → source/dest 추출 (1회) ──
    if args.target_object and args.dest_object:
        source = args.target_object
        dest = args.dest_object
        print(f"\n  [Classify] Manual override: {source} → {dest}")
    else:
        print(f"\n  [Classify] User command: \"{args.user_command}\"")
        task_info = classify_user_request(
            args.vlm_server, args.vlm_model, args.user_command
        )
        source = task_info["source_object"]
        dest = task_info.get("dest_object", "")
        print(f"  [Classify] Result: mode={task_info['mode']}, "
              f"source=\"{source}\", dest=\"{dest}\"")

    user_request = args.user_command if args.user_command else f"find the {source} and place it next to the {dest}"

    # ── 2. 환경 + 카메라 ──
    print(f"\n  Setting up environment...")
    env, cams = setup_env(args)
    obs, _ = env.reset()

    # warm up
    for _ in range(5):
        env.sim.step()
        env.sim.render()

    # ── 3. Orchestrator 생성 ──
    orch = RelativePlacementOrchestrator(
        vlm_server=args.vlm_server,
        vlm_model=args.vlm_model,
        source_object=source,
        dest_object=dest,
        user_request=user_request,
        jpeg_quality=args.jpeg_quality,
    )

    print(f"\n{'='*60}")
    print(f"  Task: \"{user_request}\"")
    print(f"  Source: {source}, Dest: {dest}")
    print(f"  Camera: {args.camera_width}x{args.camera_height}")
    print(f"  Safety: {args.safety_dist}m, VLM interval: {args.vlm_interval} steps")
    print(f"  Max steps: {args.max_total_steps}")
    print(f"{'='*60}\n")

    # ── 4. 메인 루프 ──
    for trial in range(args.num_trials):
        print(f"  === Trial {trial + 1}/{args.num_trials} ===")
        obs, _ = env.reset()
        vla.reset_buffer()
        total_steps = 0
        safety_stops = 0
        t_start = time.time()

        try:
            while total_steps < args.max_total_steps and simulation_app.is_running():
                # 4a. 카메라 캡처
                base_rgb, depth, wrist_rgb = capture(env, cams)
                if base_rgb is None or wrist_rgb is None:
                    env.sim.step()
                    total_steps += 1
                    continue

                # 4b. VLM 비동기 호출 (vlm_interval 마다)
                if total_steps % args.vlm_interval == 0:
                    orch.query_async(base_rgb)

                # 4c. VLM "done" 체크
                if orch.is_done:
                    print(f"\n  [DONE] VLM said 'done' at step {total_steps}")
                    break

                # 4d. VLA action (최신 VLM instruction 사용)
                instruction = orch.instruction
                state = get_state_9d(env)
                action = vla.get_action_9d(base_rgb, wrist_rgb, state, instruction)

                # 4e. Depth safety
                stopped = False
                if args.enable_safety and depth is not None:
                    action, stopped = depth_safety_check(depth, action, args.safety_dist)
                    if stopped:
                        safety_stops += 1

                # 4f. env step
                action_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                obs, rew, term, trunc, info = env.step(action_t)
                total_steps += 1

                # 4g. 로그
                if total_steps % 50 == 0:
                    elapsed = time.time() - t_start
                    hz = total_steps / elapsed if elapsed > 0 else 0
                    stop_str = " [SAFETY]" if stopped else ""
                    print(f"    [t={total_steps:4d} {elapsed:.0f}s {hz:.1f}Hz] "
                          f"inst=\"{instruction[:50]}\" "
                          f"vlm={orch.avg_latency*1000:.0f}ms({orch.call_count}) "
                          f"vla={vla.latency*1000:.0f}ms{stop_str}")

                # 4h. 에피소드 종료
                if term.any() or trunc.any():
                    success = info.get("task_success", torch.zeros(1)).any().item()
                    print(f"\n  Trial {trial+1} → {'SUCCESS' if success else 'TIMEOUT'} "
                          f"| steps={total_steps} safety={safety_stops}")
                    break

        except KeyboardInterrupt:
            print("\n  중단 (Ctrl+C)")
            break

        # Trial 요약
        elapsed = time.time() - t_start
        print(f"  Trial {trial+1} summary: {total_steps} steps, {elapsed:.0f}s, "
              f"VLM {orch.call_count} calls (avg {orch.avg_latency*1000:.0f}ms), "
              f"safety {safety_stops}")

    simulation_app.close()


if __name__ == "__main__":
    main()
