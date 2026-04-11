#!/usr/bin/env python3
"""
Heading Offset Verification Test.
로봇이 초기 상태(yaw=0)에서 forward(action[6]=+0.5)로 전진할 때:
  - _get_robot_heading() 값 확인
  - pseudo-lidar ray 0 방향 확인
  - body_vel 축 확인

사용법 (Desktop 3090, GUI):
    cd ~/IsaacLab/scripts/lekiwi_nav_env
    python test_heading_offset.py

사용법 (headless):
    python test_heading_offset.py --headless
"""
from __future__ import annotations
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
sim_app = launcher.app

import math
import torch
from lekiwi_skill1_env import Skill1Env, Skill1EnvCfg


def main():
    env_cfg = Skill1EnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.enable_domain_randomization = False  # 노이즈 제거
    env = Skill1Env(cfg=env_cfg)

    device = env.device
    obs, _ = env.reset()

    # 초기 yaw를 0으로 강제 (world +X 방향)
    root_state = env.robot.data.default_root_state[0:1].clone()
    root_state[:, 3] = 1.0  # quat w=1 (identity, yaw=0)
    root_state[:, 4:7] = 0.0
    env.robot.write_root_state_to_sim(root_state, env_ids=torch.tensor([0], device=device))

    # 장애물 하나를 로봇 정면(+X)에 배치
    env._obstacle_pos[0, 0, 0] = root_state[0, 0].item() + 1.0  # x: 정면 1m
    env._obstacle_pos[0, 0, 1] = root_state[0, 1].item()         # y: 같은 라인
    env._obstacle_radius[0, 0] = 0.2
    env._obstacle_valid[0, :] = False
    env._obstacle_valid[0, 0] = True

    # forward action
    action = torch.zeros(1, env.cfg.action_space, device=device)
    action[:, 6] = 0.5  # x.vel forward

    print("\n" + "="*70)
    print("  HEADING OFFSET TEST")
    print("  Robot at yaw=0 (facing world +X)")
    print("  Obstacle at +X direction (robot front)")
    print("  Action: forward (action[6]=+0.5)")
    print("="*70)

    for step in range(60):
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 10 == 0:
            # Body velocity
            bv = env._read_base_body_vel()
            vx_b, vy_b, wz_b = bv[0, 0].item(), bv[0, 1].item(), bv[0, 2].item()

            # Heading
            quat = env.robot.data.root_quat_w[0]
            yaw = 2.0 * math.atan2(quat[3].item(), quat[0].item())
            heading_current = yaw + math.pi / 2  # 현재 코드
            heading_no_offset = yaw               # π/2 제거

            # Lidar scan
            scan = env._compute_lidar_scan()  # (1, 8)
            scan_vals = scan[0].tolist()

            # Ray 0 방향 (world frame)
            ray0_with_offset = env._lidar_ray_angles[0].item() + heading_current
            ray0_no_offset = env._lidar_ray_angles[0].item() + heading_no_offset

            print(f"\n  step {step:3d}:")
            print(f"    body_vel = (vx={vx_b:+.3f}, vy={vy_b:+.3f}, wz={wz_b:+.3f})")
            print(f"    yaw = {math.degrees(yaw):+.1f}°")
            print(f"    heading (with π/2) = {math.degrees(heading_current):+.1f}°")
            print(f"    heading (no offset) = {math.degrees(heading_no_offset):+.1f}°")
            print(f"    ray0 direction (with π/2) = {math.degrees(ray0_with_offset):+.1f}°")
            print(f"    ray0 direction (no offset) = {math.degrees(ray0_no_offset):+.1f}°")
            print(f"    lidar scan (8 rays) = [{', '.join(f'{v:.2f}' for v in scan_vals)}]")
            
            # 장애물은 +X(0°) 방향에 있음
            # ray가 장애물을 감지하면 해당 값이 < 1.0
            min_ray = min(scan_vals)
            min_ray_idx = scan_vals.index(min_ray)
            if min_ray < 0.95:
                print(f"    → 장애물 감지: ray[{min_ray_idx}] = {min_ray:.2f}")
            else:
                print(f"    → 장애물 미감지 (모든 ray > 0.95)")

    print("\n" + "="*70)
    print("  판단 기준:")
    print("  - 장애물은 world +X (0°) 방향에 있음")
    print("  - ray[0]이 장애물을 감지해야 heading이 정확한 것")
    print()
    print("  IF 'heading (with π/2)'에서 ray[0]이 감지 → π/2 유지")
    print("  IF 'heading (no offset)'에서 ray[0]이 감지 → π/2 제거 필요")
    print()
    print("  실제로 어떤 ray가 장애물을 감지했는지 보고:")
    print("  - ray[0]이면 heading 정확")
    print("  - ray[2] (90° 오프셋)이면 π/2가 문제")
    print("="*70 + "\n")

    sim_app.close()


if __name__ == "__main__":
    main()
