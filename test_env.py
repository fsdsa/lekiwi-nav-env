"""
LeKiwi Nav 환경 — 빠른 검증.

환경이 정상 동작하는지 random action으로 확인.
학습 전에 반드시 실행하여 obs/action shape, reward, reset 검증.

Usage:
    cd /home/yubin11/IsaacLab/scripts/lekiwi_nav_env
    python test_env.py --num_envs 4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="LeKiwi Nav — env sanity check")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--gripper_contact_prim_path", type=str, default="")
parser.add_argument("--dynamics_json", type=str, default=None,
                    help="tune_sim_dynamics.py 출력 JSON (best_params) 경로")
parser.add_argument("--arm_limit_json", type=str, default=None,
                    help="optional arm joint limit JSON (real2sim calibration)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
sim_app = launcher.app

import torch
from lekiwi_nav_env import LeKiwiNavEnv, LeKiwiNavEnvCfg


def main():
    cfg = LeKiwiNavEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.object_usd:
        cfg.object_usd = os.path.expanduser(args.object_usd)
    if args.multi_object_json:
        cfg.multi_object_json = os.path.expanduser(args.multi_object_json)
    if args.gripper_contact_prim_path:
        cfg.gripper_contact_prim_path = str(args.gripper_contact_prim_path)
    if args.dynamics_json:
        cfg.dynamics_json = os.path.expanduser(args.dynamics_json)
    if args.arm_limit_json:
        cfg.arm_limit_json = os.path.expanduser(args.arm_limit_json)
    env = LeKiwiNavEnv(cfg=cfg)

    print("\n" + "=" * 60)
    print(f"  LeKiwi Nav — 환경 검증")
    print(f"  num_envs: {args.num_envs}")
    print(f"  obs_space: {env.observation_space}")
    print(f"  act_space: {env.action_space}")
    print(f"  device: {env.device}")
    if args.multi_object_json:
        print(f"  multi_object_json: {os.path.expanduser(args.multi_object_json)}")
    if args.object_usd:
        print(f"  object_usd: {os.path.expanduser(args.object_usd)}")
    print("=" * 60)

    obs, info = env.reset()
    print(f"\n  [reset] obs shape: {obs['policy'].shape}")
    print(f"  [reset] obs range: [{obs['policy'].min():.3f}, {obs['policy'].max():.3f}]")

    total_rewards = torch.zeros(args.num_envs, device=env.device)
    num_steps = 500  # 10초 (50Hz / decimation=2 = 25Hz → 25 × 10 = 250 실제 step, 하지만 step 500번)

    for step in range(num_steps):
        # Random actions in [-1, 1]
        actions = torch.randn(args.num_envs, env.cfg.action_space, device=env.device) * 0.5
        obs, reward, terminated, truncated, info = env.step(actions)
        total_rewards += reward

        if step % 100 == 0:
            root_pos = env.robot.data.root_pos_w[0].cpu().numpy()
            goal_pos = env.goal_pos_w[0].cpu().numpy()
            dist = ((root_pos[0]-goal_pos[0])**2 + (root_pos[1]-goal_pos[1])**2) ** 0.5
            phase = int(env.phase[0].item()) if hasattr(env, "phase") else -1
            visible = bool(env.object_visible[0].item()) if hasattr(env, "object_visible") else False
            grasped = bool(env.object_grasped[0].item()) if hasattr(env, "object_grasped") else False
            success = bool(env.task_success[0].item()) if hasattr(env, "task_success") else False
            bbox_msg = ""
            if hasattr(env, "object_bbox"):
                bbox = env.object_bbox[0].detach().cpu().numpy()
                cat = int(env.object_category_id[0].item()) if hasattr(env, "object_category_id") else -1
                bbox_msg = f" | bbox=({bbox[0]:.3f},{bbox[1]:.3f},{bbox[2]:.3f}) cat={cat}"
            print(
                f"  step {step:4d} | "
                f"pos=({root_pos[0]:+.2f},{root_pos[1]:+.2f}) | "
                f"goal=({goal_pos[0]:+.2f},{goal_pos[1]:+.2f}) | "
                f"dist={dist:.2f}m | "
                f"phase={phase} vis={int(visible)} grasp={int(grasped)} succ={int(success)}"
                f"{bbox_msg} | "
                f"rew={reward[0]:+.3f} | "
                f"term={terminated[0].item()} trunc={truncated[0].item()}"
            )

        # Reset 발생 확인
        reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0 and step < 100:
            print(f"    ↳ reset envs: {reset_ids.tolist()} at step {step}")

    mean_return = total_rewards.mean().item()
    print(f"\n  ── 결과 ──")
    print(f"  {num_steps} steps 완료")
    print(f"  평균 return: {mean_return:.2f}")
    print(f"  goal reached: {env.reached_count}회")
    print(f"  ✅ 환경 정상 동작!" if not torch.isnan(total_rewards).any() else "  ⚠ NaN detected!")

    if not args.headless:
        print(f"\n  🖥️  GUI 유지 중... Ctrl+C")
        try:
            while sim_app.is_running():
                env.step(torch.zeros(args.num_envs, env.cfg.action_space, device=env.device))
        except KeyboardInterrupt:
            pass

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
