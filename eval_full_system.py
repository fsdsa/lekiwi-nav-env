#!/usr/bin/env python3
"""
Sim Full-System Evaluation — VLM + VLA closed-loop (Phase 4.5).

Isaac Sim 안에서 VLM→VLA 전체 파이프라인을 closed-loop으로 검증한다.
3090 Desktop에서 sim 렌더링을, A100 Server에서 VLM+VLA 추론을 수행.

사용법:
    # Relative placement — source 물체를 destination 옆에 놓기
    python eval_full_system.py \
        --vlm_server http://<A100_IP>:8001 \
        --vla_server http://<A100_IP>:8002 \
        --num_trials 30 \
        --task "find the medicine bottle and place it next to the red cup" \
        --multi_object_json object_catalog.json

    # Skill별 단독 평가 (VLM 없이, 고정 instruction)
    python eval_full_system.py \
        --vla_server http://<A100_IP>:8002 \
        --eval_mode skill_only --skill navigate \
        --instruction "move forward" --num_trials 50

    # Headless (서버에서)
    python eval_full_system.py --headless --vlm_server ... --vla_server ...
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import time

os.environ["CARB_LOG_LEVEL"] = "fatal"
os.environ["OMNI_LOG_LEVEL"] = "fatal"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sim Full-System Evaluation (Phase 4.5)")
# Server endpoints
parser.add_argument("--vlm_server", type=str, default=None,
                    help="VLM 서버 URL (e.g. http://218.148.55.186:8001). None이면 VLM 없이 고정 instruction.")
parser.add_argument("--vla_server", type=str, required=True,
                    help="VLA 서버 URL (e.g. http://218.148.55.186:8002)")
# Task
parser.add_argument("--task", type=str, default="find the medicine bottle and place it next to the red cup",
                    help="사용자 명령 (VLM에 전달)")
parser.add_argument("--instruction", type=str, default="move forward",
                    help="고정 instruction (skill_only 모드에서 사용)")
# Eval mode
parser.add_argument("--eval_mode", type=str, default="full",
                    choices=["full", "skill_only"],
                    help="full: VLM+VLA 통합, skill_only: VLA만 (고정 instruction)")
parser.add_argument("--skill", type=str, default="navigate",
                    choices=["navigate", "approach_and_grasp", "carry_and_place"],
                    help="skill_only 모드에서 평가할 skill")
# Env
parser.add_argument("--num_trials", type=int, default=30, help="평가 횟수")
parser.add_argument("--max_steps_per_trial", type=int, default=1200,
                    help="trial당 최대 step (10Hz × 120s = 1200)")
parser.add_argument("--multi_object_json", type=str, default="",
                    help="Multi-object catalog JSON")
parser.add_argument("--dynamics_json", type=str, default=None)
parser.add_argument("--arm_limit_json", type=str, default=None)
# VLM timing
parser.add_argument("--vlm_interval_steps", type=int, default=33,
                    help="VLM 호출 간격 (steps). 10Hz × 3.3s ≈ 0.3Hz")
# Output
parser.add_argument("--output_log", type=str, default="eval_full_system_log.json",
                    help="평가 결과 JSON 로그")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
sim_app = launcher.app

import numpy as np
import requests
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_full_system")

# Suppress Isaac Sim verbose logs
try:
    import omni.log
    log_iface = omni.log.get_log()
    for ch in ["omni.physx.tensors.plugin", "omni.physx.plugin",
               "usdrt.hydra.fabric_scene_delegate.plugin", "omni.fabric.plugin"]:
        log_iface.set_channel_enabled(ch, False)
except Exception:
    pass


# ─── Environment setup ───────────────────────────────────────────

from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg
from isaaclab.sensors import CameraCfg, Camera

ENV_PRIM = "/World/envs/env_.*/Robot"
BASE_RGB_CAM_PRIM = (
    f"{ENV_PRIM}/base_plate_layer1_v5/Realsense/RSD455"
    f"/Camera_OmniVision_OV9782_Color"
)
WRIST_CAM_PRIM = (
    f"{ENV_PRIM}/Wrist_Roll_08c_v1/visuals/mesh_002_3"
    f"/wrist_camera"
)


class FullSystemEnv(Skill2Env):
    """Skill2Env + 카메라 (Phase 4.5 Full-System Evaluation용)."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        super()._setup_scene()
        base_cam_cfg = CameraCfg(
            prim_path=BASE_RGB_CAM_PRIM, spawn=None, update_period=0.0,
            height=720, width=1280, data_types=["rgb"],
        )
        self.base_cam = Camera(base_cam_cfg)
        self.scene.sensors["base_cam"] = self.base_cam

        wrist_cam_cfg = CameraCfg(
            prim_path=WRIST_CAM_PRIM, spawn=None, update_period=0.0,
            height=480, width=640, data_types=["rgb"],
        )
        self.wrist_cam = Camera(wrist_cam_cfg)
        self.scene.sensors["wrist_cam"] = self.wrist_cam
        log.info("[Camera] base=1280x720, wrist=640x480")

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


# ─── Remote inference clients ────────────────────────────────────

def _tensor_to_jpeg_b64(tensor: torch.Tensor) -> str:
    """(H, W, 3) uint8 tensor → base64 JPEG string."""
    arr = tensor.cpu().numpy()
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def query_vlm(server_url: str, base_rgb: torch.Tensor,
              task: str, prev_phase: str,
              system_prompt: str | None = None) -> dict:
    """VLM 서버에 이미지를 보내고 instruction + phase를 받는다."""
    payload = {
        "image": _tensor_to_jpeg_b64(base_rgb),
        "task": task,
        "prev_phase": prev_phase,
    }
    if system_prompt:
        payload["system_prompt"] = system_prompt
    try:
        resp = requests.post(f"{server_url}/infer", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning(f"VLM query failed: {e}")
        return {"instruction": "stop", "phase": prev_phase, "done": False, "reasoning": str(e)}


def query_vla(server_url: str, base_rgb: torch.Tensor,
              wrist_rgb: torch.Tensor, state_9d: list[float],
              instruction: str) -> list[list[float]]:
    """VLA 서버에 이미지+state+instruction을 보내고 action chunk를 받는다."""
    payload = {
        "base_image": _tensor_to_jpeg_b64(base_rgb),
        "wrist_image": _tensor_to_jpeg_b64(wrist_rgb),
        "state": state_9d,
        "instruction": instruction,
    }
    try:
        resp = requests.post(f"{server_url}/infer", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data["actions"]
    except Exception as e:
        log.warning(f"VLA query failed: {e}")
        return [[0.0] * 9]  # zero action fallback


def extract_robot_state_9d(env: Skill2Env) -> list[float]:
    """Extract 9D robot state: arm_pos(5) + gripper(1) + vx + vy + wz."""
    arm_pos = env.robot.data.joint_pos[0, env.arm_idx].cpu().tolist()  # 6D (5 arm + 1 gripper)
    vx = float(env.robot.data.root_lin_vel_b[0, 0])
    vy = float(env.robot.data.root_lin_vel_b[0, 1])
    wz = float(env.robot.data.root_ang_vel_b[0, 2])
    return arm_pos[:5] + [arm_pos[5]] + [vx, vy, wz]  # 9D


# ─── Success detection ────────────────────────────────────────────

def check_success(env: Skill2Env, phase: str) -> bool:
    """현재 환경 상태로 task 성공 여부 판단."""
    if phase == "done":
        return True
    # Skill-3 env: dest object 기반 place 성공 판정
    if hasattr(env, "_check_place_success"):
        return bool(env._check_place_success()[0].item())
    # Fallback: home 기반 (Skill-2 env)
    if not hasattr(env, "object_pos_w") or not hasattr(env, "home_pos_w"):
        return False
    obj_pos = env.object_pos_w[0, :2]
    home_pos = env.home_pos_w[0, :2]
    dist = torch.norm(obj_pos - home_pos).item()
    gripper_pos = env.robot.data.joint_pos[0, env.arm_idx[5]].item()
    return dist < 0.05 and gripper_pos > 0.5


def classify_failure(env: Skill2Env, phase: str, step: int, max_steps: int) -> str:
    """실패 유형 분류."""
    if step >= max_steps:
        return f"timeout_in_{phase}"
    return f"failure_in_{phase}"


# ─── Main evaluation loop ────────────────────────────────────────

from vlm_orchestrator import (
    classify_user_request,
    RelativePlacementOrchestrator,
)


def run_evaluation():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Environment config
    env_cfg = Skill2EnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.arm_limit_write_to_sim = False
    if args.multi_object_json:
        env_cfg.multi_object_json = args.multi_object_json
    if args.dynamics_json:
        env_cfg.dynamics_json = args.dynamics_json
    if args.arm_limit_json:
        env_cfg.arm_limit_json = args.arm_limit_json

    log.info("Creating FullSystemEnv ...")
    env = FullSystemEnv(cfg=env_cfg)
    log.info(f"Env ready. device={env.device}")

    # ── 모드 분류 + 오케스트레이터 선택 ──
    use_orchestrator = (args.eval_mode == "full" and args.vlm_server)
    task_info = None
    if use_orchestrator:
        task_info = classify_user_request(args.task, vlm_url=args.vlm_server)
        log.info(f"Task classification: {task_info}")

    # Stats
    results = []
    success_count = 0

    for trial in range(args.num_trials):
        log.info(f"\n{'='*60}")
        log.info(f"Trial {trial + 1}/{args.num_trials}")
        log.info(f"{'='*60}")

        obs, info = env.reset()

        # 오케스트레이터 초기화 (매 trial마다)
        if use_orchestrator:
            orchestrator = RelativePlacementOrchestrator(
                source_object=task_info.get("source_object", "medicine bottle"),
                destination_object=task_info.get("destination_object", "red cup"),
                user_request=args.task,
            )
            phase = orchestrator.phase
        else:
            orchestrator = None
            phase = args.skill

        instruction = args.instruction
        done = False
        trial_log = {
            "trial": trial + 1,
            "task": args.task,
            "mode": task_info["mode"] if task_info else "skill_only",
            "steps": 0,
            "success": False,
            "failure_type": None,
            "phases": [],
            "vlm_calls": 0,
            "objects_completed": [],
        }
        action_chunk = []
        chunk_idx = 0

        for step in range(args.max_steps_per_trial):
            # Get camera images
            base_rgb = env.get_base_rgb()
            wrist_rgb = env.get_wrist_rgb()
            if base_rgb is None or wrist_rgb is None:
                obs, _, _, _, _ = env.step(torch.zeros(1, 9, device=env.device))
                continue

            base_img = base_rgb[0]   # (H, W, 3)
            wrist_img = wrist_rgb[0]

            # ── VLM query (at interval) ──
            if orchestrator and step % args.vlm_interval_steps == 0:
                system_prompt = orchestrator.get_system_prompt()
                vlm_raw = query_vlm(
                    args.vlm_server, base_img, args.task, phase,
                    system_prompt=system_prompt,
                )
                trial_log["vlm_calls"] += 1

                result = orchestrator.process_vlm_response(vlm_raw)
                new_phase = result["phase"]
                new_instruction = result["instruction"]

                if new_phase != phase:
                    log.info(f"  [VLM] Phase: {phase} -> {new_phase}")
                    trial_log["phases"].append({
                        "step": step, "from": phase, "to": new_phase,
                        "instruction": new_instruction,
                    })
                    phase = new_phase
                instruction = new_instruction

                if result["done"]:
                    done = True
                    trial_log["success"] = True
                    break

            # ── VLA query (when action chunk exhausted) ──
            if chunk_idx >= len(action_chunk):
                state_9d = extract_robot_state_9d(env)
                action_chunk = query_vla(
                    args.vla_server, base_img, wrist_img, state_9d, instruction,
                )
                chunk_idx = 0

            # Execute action
            action = torch.tensor(
                [action_chunk[chunk_idx]], dtype=torch.float32, device=env.device,
            )
            chunk_idx += 1

            obs, reward, terminated, truncated, info = env.step(action)

            # Check success (physics-based)
            if check_success(env, phase):
                done = True
                trial_log["success"] = True
                break

            if terminated.any() or truncated.any():
                break

            # Periodic log
            if step > 0 and step % 100 == 0:
                state = extract_robot_state_9d(env)
                log.info(
                    f"  step={step:4d}  phase={phase:20s}  "
                    f"inst=\"{instruction[:30]}\"  "
                    f"vel=[{state[6]:+.3f},{state[7]:+.3f},{state[8]:+.3f}]"
                )

        # Trial end
        trial_log["steps"] = step + 1
        if not trial_log["success"]:
            trial_log["failure_type"] = classify_failure(env, phase, step + 1, args.max_steps_per_trial)

        results.append(trial_log)
        if trial_log["success"]:
            success_count += 1

        status = "SUCCESS" if trial_log["success"] else f"FAIL ({trial_log['failure_type']})"
        log.info(f"  Trial {trial+1}: {status} in {trial_log['steps']} steps, "
                 f"VLM calls={trial_log['vlm_calls']}")

    # ── Summary ──
    total = len(results)
    log.info(f"\n{'='*60}")
    log.info(f"FINAL: {success_count}/{total} success ({100*success_count/max(total,1):.1f}%)")
    log.info(f"{'='*60}")

    # Failure breakdown
    failure_types: dict[str, int] = {}
    for r in results:
        ft = r.get("failure_type")
        if ft:
            failure_types[ft] = failure_types.get(ft, 0) + 1
    if failure_types:
        log.info("Failure breakdown:")
        for ft, count in sorted(failure_types.items(), key=lambda x: -x[1]):
            log.info(f"  {ft}: {count}")

    # Save log
    summary = {
        "eval_mode": args.eval_mode,
        "task": args.task,
        "mode": task_info["mode"] if task_info else "skill_only",
        "total_trials": total,
        "successes": success_count,
        "success_rate": round(success_count / max(total, 1), 4),
        "failure_breakdown": failure_types,
        "trials": results,
    }
    with open(args.output_log, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"Log saved: {args.output_log}")

    env.close()


# ─── Entry ────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_evaluation()
    sim_app.close()
