#!/usr/bin/env python3
"""
전체 태스크 오케스트레이터 — VLM + 3 Skill Policies

태스크: "약통을 찾아서 빨간컵 옆에 놓기"

Phase 1: NAVIGATE_SEARCH     → VLM이 방향 지시, BC navigate 실행
Phase 2: NAVIGATE_TO_TARGET  → VLM이 물체 방향 지시, BC navigate 실행
Phase 3: APPROACH_AND_LIFT   → BC+RL approach&lift policy 실행
Phase 4: NAVIGATE_TO_DEST    → 물체 들고 VLM 방향 지시, BC navigate 실행
Phase 5: CARRY_AND_PLACE     → BC+RL carry&place policy 실행
Phase 6: DONE

Usage:
    python run_full_task.py \
        --vlm_server http://<서버IP>:8000 \
        --nav_bc checkpoints/dp_bc_nav/dp_bc.pt \
        --approach_bc checkpoints/dp_bc_skill2/dp_bc.pt \
        --approach_rl checkpoints/resip_skill2/resip_best.pt \
        --place_bc checkpoints/dp_bc_skill3/dp_bc.pt \
        --place_rl checkpoints/resip_skill3/resip_best.pt \
        --target_object "medicine bottle" \
        --dest_object "red cup" \
        --scene_idx 0
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Full Task: VLM + 3 Skills")

# VLM
parser.add_argument("--vlm_server", type=str, required=True)
parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")

# Policy checkpoints
parser.add_argument("--nav_bc", type=str, required=True, help="Navigate BC checkpoint")
parser.add_argument("--nav_rl", type=str, default="", help="Navigate ResiP (선택)")
parser.add_argument("--approach_bc", type=str, required=True, help="Approach&Lift BC")
parser.add_argument("--approach_rl", type=str, default="", help="Approach&Lift ResiP")
parser.add_argument("--place_bc", type=str, required=True, help="Carry&Place BC")
parser.add_argument("--place_rl", type=str, default="", help="Carry&Place ResiP")

# Task
parser.add_argument("--target_object", type=str, default="medicine bottle")
parser.add_argument("--dest_object", type=str, default="red cup")

# Scene / Camera
parser.add_argument("--scene_idx", type=int, default=0)
parser.add_argument("--scene_usd", type=str, default="")
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=400)
parser.add_argument("--jpeg_quality", type=int, default=80)

# Safety
parser.add_argument("--safety_dist", type=float, default=0.3)
parser.add_argument("--enable_safety", action="store_true", default=True)

# Limits
parser.add_argument("--max_steps_per_phase", type=int, default=1500)
parser.add_argument("--max_total_steps", type=int, default=6000)

# Calibration
parser.add_argument("--arm_limit_json", type=str, default="calibration/arm_limits_measured.json")
parser.add_argument("--dynamics_json", type=str, default="")
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════════
#  Import (run_vlm_navigate.py와 공유)
# ═══════════════════════════════════════════════════════════════════════

from run_vlm_navigate import (
    VLMClient,
    depth_safety_check,
    load_bc_policy,
    load_residual_policy,
)
from vlm_prompts import COMMAND_TO_DIRECTION, PHASE_TO_SKILL


# ═══════════════════════════════════════════════════════════════════════
#  Phase 정의
# ═══════════════════════════════════════════════════════════════════════

PHASES = [
    "NAVIGATE_SEARCH",       # 1: 물체 탐색
    "NAVIGATE_TO_TARGET",    # 2: 물체 방향으로 이동
    "APPROACH_AND_LIFT",     # 3: 접근 + 파지 + 들기
    "NAVIGATE_TO_DEST",      # 4: 목적지 탐색 (물체 들고)
    "NAVIGATE_TO_DEST_CLOSE",# 5: 목적지 방향으로 이동
    "CARRY_AND_PLACE",       # 6: 놓기
    "DONE",                  # 7: 완료
]


def main():
    device = torch.device("cuda:0")

    # ── VLM 클라이언트 ──
    vlm = VLMClient(args.vlm_server, args.vlm_model, args.jpeg_quality)
    print(f"  [VLM] Server: {args.vlm_server}")

    # ── Policy 로드 ──
    print("\n  Loading policies...")
    nav_bc = load_bc_policy(args.nav_bc, device)
    nav_rl = load_residual_policy(args.nav_rl, 20, 9, device) if args.nav_rl else None

    approach_bc = load_bc_policy(args.approach_bc, device)
    approach_rl = load_residual_policy(args.approach_rl, 30, 9, device) if args.approach_rl else None

    place_bc = load_bc_policy(args.place_bc, device)
    place_rl = load_residual_policy(args.place_rl, 29, 9, device) if args.place_rl else None

    # Residual scales (skill별로 다름)
    nav_scale = torch.zeros(9, device=device)
    nav_scale[6:9] = 0.25

    approach_scale = torch.zeros(9, device=device)
    approach_scale[0:5] = 0.15
    approach_scale[5] = 0.20
    approach_scale[6:9] = 0.25

    place_scale = torch.zeros(9, device=device)
    place_scale[0:5] = 0.15
    place_scale[5] = 0.20
    place_scale[6:9] = 0.25

    # ── Skill 매핑 ──
    skill_policies = {
        "navigate": (nav_bc, nav_rl, nav_scale),
        "approach_and_lift": (approach_bc, approach_rl, approach_scale),
        "carry_and_place": (place_bc, place_rl, place_scale),
    }

    # ── 상태 ──
    current_phase = "NAVIGATE_SEARCH"
    current_command = "FORWARD"
    total_steps = 0
    phase_steps = 0
    phase_history = []

    print(f"\n{'='*60}")
    print(f"  Full Task: {args.target_object} → {args.dest_object}")
    print(f"  Policies: nav={'BC+RL' if nav_rl else 'BC'}, "
          f"approach={'BC+RL' if approach_rl else 'BC'}, "
          f"place={'BC+RL' if place_rl else 'BC'}")
    print(f"  Phase: {current_phase}")
    print(f"{'='*60}\n")

    # ── TODO: 환경 세팅 ──
    # 여기에 ProcTHOR 장면 + 물체 배치 + 카메라 세팅
    # 현재는 run_vlm_navigate.py의 setup_env_with_camera() 참고
    #
    # 전체 태스크를 하려면 통합 환경이 필요:
    #   - Navigate: Skill1Env (arm tucked)
    #   - Approach&Lift: Skill2Env (arm active)
    #   - Carry&Place: Skill3Env (arm active, object grasped)
    #
    # 방법 1: 세 환경을 skill 전환 시 교체 (간단하지만 비효율)
    # 방법 2: 통합 환경 하나에서 모드 전환 (깔끔하지만 작업 필요)
    #
    # 우선은 Navigate만 실행하고, skill 전환 시점을 확인하는 게 첫 단계.

    print("  [NOTE] 현재는 Navigate skill만 실행 가능")
    print("  [NOTE] run_vlm_navigate.py로 Navigate를 먼저 검증하세요")
    print("  [NOTE] Approach/Place 통합은 환경 통합 후 진행")

    # ── 메인 루프 (개념 코드) ──
    """
    while current_phase != "DONE" and total_steps < args.max_total_steps:

        # 1. 현재 phase에 맞는 skill 결정
        skill_name = PHASE_TO_SKILL[current_phase]
        bc, rl, scale = skill_policies[skill_name]

        # 2. VLM으로 현재 이미지 판단
        if skill_name == "navigate":
            # Navigate: VLM이 방향 command 결정
            rgb = capture_camera()
            command = vlm.query_navigate_command(rgb, args.target_object)
            direction_cmd = COMMAND_TO_DIRECTION[command]

            # obs 구성 + direction_cmd 주입
            obs[0, 9:12] = direction_cmd
            obs[0, 12:20] = 1.0  # lidar dummy

        # 3. BC (+RL) action 계산
        with torch.no_grad():
            base_naction = bc.base_action_normalized(obs)
            if rl is not None:
                nobs = bc.normalizer(obs, "obs", forward=True)
                residual = rl.actor_mean(torch.cat([nobs, base_naction], -1))
                naction = base_naction + residual * scale
            else:
                naction = base_naction
            action = bc.normalizer(naction, "action", forward=False)

        # 4. Safety check (Navigate 시)
        if skill_name == "navigate" and args.enable_safety:
            depth = capture_depth()
            action, stopped = depth_safety_check(depth, action, args.safety_dist)

        # 5. 환경 step
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        phase_steps += 1

        # 6. Phase 전환 판단 (VLM, 주기적)
        if total_steps % 30 == 0:  # ~3초마다
            rgb = capture_camera()
            new_phase = vlm.query_skill_phase(rgb, args.target_object, args.dest_object)
            if new_phase != current_phase:
                print(f"  Phase: {current_phase} → {new_phase} (step {total_steps})")
                phase_history.append((current_phase, phase_steps))
                current_phase = new_phase
                phase_steps = 0
                bc.reset()  # diffusion policy state reset

        # 7. STOP → phase 전환 트리거
        if command == "STOP" and skill_name == "navigate":
            if "TARGET" in current_phase:
                current_phase = "APPROACH_AND_LIFT"
            elif "DEST" in current_phase:
                current_phase = "CARRY_AND_PLACE"
    """

    simulation_app.close()


if __name__ == "__main__":
    main()
