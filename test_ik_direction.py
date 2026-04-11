#!/usr/bin/env python3
"""
⚠️ 레거시: direction-based navigate 검증. 목적지 기반 전환 후 폐기.

Direction Command + Compliance 검증 테스트.

각 direction_cmd에 맞는 action을 보내고:
1. 로봇이 올바른 방향으로 움직이는지 (눈으로 확인)
2. compliance = dot(direction_cmd, body_vel_normalized) > 0 인지 (숫자 확인)

사용법:
    python test_ik_direction.py          # GUI
    python test_ik_direction.py --headless  # headless
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Direction Compliance Verification")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
sim_app = launcher.app

import torch
from lekiwi_skill1_env import Skill1Env, Skill1EnvCfg

# direction_cmd (body frame) ↔ action 매핑
# body frame: vx=right, vy=forward, wz>0=CW
# After body→IK conversion in _apply_action: action ∝ direction_cmd (index+sign match)
TESTS = [
    {
        "label": "FORWARD",
        "direction_cmd": [0.0, 1.0, 0.0],   # vy_body > 0
        "action": {7: 0.5},                  # action[7] = body vy+ = forward
        "expect": "로봇이 앞으로 직진",
    },
    {
        "label": "BACKWARD",
        "direction_cmd": [0.0, -1.0, 0.0],  # vy_body < 0
        "action": {7: -0.5},                 # action[7] = body vy- = backward
        "expect": "로봇이 뒤로 후진",
    },
    {
        "label": "STRAFE LEFT",
        "direction_cmd": [-1.0, 0.0, 0.0],  # vx_body < 0 (leftward)
        "action": {6: -0.5},                 # action[6] = body vx- = left
        "expect": "로봇이 왼쪽으로 횡이동",
    },
    {
        "label": "STRAFE RIGHT",
        "direction_cmd": [1.0, 0.0, 0.0],   # vx_body > 0 (rightward)
        "action": {6: 0.5},                  # action[6] = body vx+ = right
        "expect": "로봇이 오른쪽으로 횡이동",
    },
    {
        "label": "TURN LEFT (CCW)",
        "direction_cmd": [0.0, 0.0, 1.0],   # wz_body > 0 = CCW (Isaac Sim right-hand rule)
        "action": {8: 0.5},                  # action[8] = body wz+ = CCW
        "expect": "로봇이 반시계 방향 회전",
    },
    {
        "label": "TURN RIGHT (CW)",
        "direction_cmd": [0.0, 0.0, -1.0],  # wz_body < 0 = CW
        "action": {8: -0.5},                 # action[8] = body wz- = CW
        "expect": "로봇이 시계 방향 회전",
    },
]


def run_test(env: Skill1Env, test: dict, steps: int = 150):
    """Run one direction test and print compliance."""
    label = test["label"]
    print(f"\n{'='*65}")
    print(f"  TEST: {label}")
    print(f"  direction_cmd = {test['direction_cmd']}")
    print(f"  기대 동작: {test['expect']}")
    print(f"{'='*65}")

    obs, _ = env.reset()
    device = env.device

    action = torch.zeros(env.num_envs, env.cfg.action_space, device=device)
    for idx, val in test["action"].items():
        action[:, idx] = val

    cmd = torch.tensor([test["direction_cmd"]], dtype=torch.float32, device=device)
    max_lin = env.cfg.max_lin_vel
    max_ang = env.cfg.max_ang_vel

    compliance_sum = 0.0
    n = 0

    for step in range(steps):
        obs, reward, terminated, truncated, info = env.step(action)
        body_vel = env._read_base_body_vel()  # (1, 3): vx_body, vy_body, wz_body

        # Normalize like _compute_metrics does
        vel_norm = body_vel.clone()
        vel_norm[:, :2] = vel_norm[:, :2] / (max_lin + 1e-6)
        vel_norm[:, 2] = vel_norm[:, 2] / (max_ang + 1e-6)
        compliance = (cmd * vel_norm).sum(dim=-1).item()

        if step >= 20:  # skip acceleration
            compliance_sum += compliance
            n += 1

        if step % 50 == 0:
            vx_b = body_vel[0, 0].item()
            vy_b = body_vel[0, 1].item()
            wz_b = body_vel[0, 2].item()
            print(
                f"  step {step:3d} | "
                f"body_vel=({vx_b:+.3f}, {vy_b:+.3f}, {wz_b:+.3f}) | "
                f"compliance={compliance:+.3f}"
            )

    avg_compliance = compliance_sum / max(n, 1)
    result = "PASS" if avg_compliance > 0.1 else "FAIL"
    print(f"\n  >> avg compliance = {avg_compliance:+.4f}  [{result}]")
    return avg_compliance


def main():
    env_cfg = Skill1EnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.arm_limit_write_to_sim = False  # avoid PhysX limit errors in test
    env = Skill1Env(cfg=env_cfg)

    results = []
    for test in TESTS:
        c = run_test(env, test)
        results.append((test["label"], c))

    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    all_pass = True
    for label, c in results:
        status = "PASS" if c > 0.1 else "FAIL"
        if c <= 0.1:
            all_pass = False
        print(f"  {label:20s} | compliance={c:+.4f} | {status}")
    print(f"{'='*65}")
    print(f"  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'='*65}\n")

    sim_app.close()


if __name__ == "__main__":
    main()
