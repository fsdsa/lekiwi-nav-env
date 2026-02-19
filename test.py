#!/usr/bin/env python3
"""
Isaac Lab API 검증 — standalone 실행
=====================================
Script Editor가 아닌 터미널에서 실행:

    cd ~/IsaacLab/scripts/lekiwi_nav_env
    conda activate env_isaaclab && source ~/isaacsim/setup_conda_env.sh
    python verify_isaaclab_api.py --headless

이 스크립트는 codefix.md에서 사용하는 Isaac Lab API가
실제로 존재하고 올바른 shape을 반환하는지 검증한다.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
sim_app = launcher.app

import torch
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg

# v8 env import (기존 환경을 빌려서 API 확인)
try:
    from lekiwi_nav_env import LeKiwiNavEnv, LeKiwiNavEnvCfg
    USE_V8 = True
    print("[OK] v8 env imported")
except ImportError:
    USE_V8 = False
    print("[SKIP] v8 env not found, using minimal setup")


def check_api():
    if not USE_V8:
        print("\n[ERROR] v8 env import failed. Cannot verify APIs without env.")
        print("Make sure lekiwi_nav_env.py is in the current directory.")
        return

    # Minimal env
    cfg = LeKiwiNavEnvCfg()
    cfg.scene.num_envs = 2
    env = LeKiwiNavEnv(cfg=cfg)
    obs, _ = env.reset()

    robot = env.robot
    print("\n" + "=" * 60)
    print("ISAAC LAB API VERIFICATION")
    print("=" * 60)

    # -------------------------------------------------------
    # 1. root_lin_vel_b / root_ang_vel_b
    # -------------------------------------------------------
    print("\n--- 1. Body-frame velocity API ---")
    has_lin_vel_b = hasattr(robot.data, "root_lin_vel_b")
    has_ang_vel_b = hasattr(robot.data, "root_ang_vel_b")
    print(f"  root_lin_vel_b exists: {has_lin_vel_b}")
    print(f"  root_ang_vel_b exists: {has_ang_vel_b}")

    if has_lin_vel_b:
        val = robot.data.root_lin_vel_b
        print(f"  root_lin_vel_b shape: {val.shape}  (expect: ({cfg.scene.num_envs}, 3))")
        print(f"  root_lin_vel_b sample: {val[0].tolist()}")
    else:
        print("  [FALLBACK NEEDED] root_lin_vel_b not available.")
        print("  Use: quat_apply_inverse(root_quat_w, root_lin_vel_w)")
        # Check world-frame exists
        has_w = hasattr(robot.data, "root_lin_vel_w")
        print(f"  root_lin_vel_w exists: {has_w}")
        if has_w:
            print(f"  root_lin_vel_w shape: {robot.data.root_lin_vel_w.shape}")

    if has_ang_vel_b:
        val = robot.data.root_ang_vel_b
        print(f"  root_ang_vel_b shape: {val.shape}")
        print(f"  root_ang_vel_b sample: {val[0].tolist()}")
    else:
        print("  [FALLBACK NEEDED] root_ang_vel_b not available.")
        has_w = hasattr(robot.data, "root_ang_vel_w")
        print(f"  root_ang_vel_w exists: {has_w}")

    # -------------------------------------------------------
    # 2. body_pos_w shape and gripper body
    # -------------------------------------------------------
    print("\n--- 2. body_pos_w and gripper body ---")
    has_body_pos = hasattr(robot.data, "body_pos_w")
    print(f"  body_pos_w exists: {has_body_pos}")
    if has_body_pos:
        val = robot.data.body_pos_w
        print(f"  body_pos_w shape: {val.shape}  (expect: (num_envs, num_bodies, 3))")
        print(f"  body_pos_w ndim: {val.ndim}")

    # find_bodies
    print("\n  Searching for gripper body 'Moving_Jaw_08d_v1'...")
    try:
        body_ids, body_names = robot.find_bodies(["Moving_Jaw_08d_v1"])
        print(f"  [OK] Found: ids={body_ids}, names={body_names}")
        if has_body_pos and len(body_ids) > 0:
            idx = body_ids[0]
            if val.ndim == 3:
                grip_pos = val[:, idx, :]
            else:
                grip_pos = val[:, idx]
            print(f"  Gripper body pos (env 0): {grip_pos[0].tolist()}")
    except Exception as e:
        print(f"  [ERROR] find_bodies failed: {e}")
        print("  Listing all body names...")
        if hasattr(robot, "body_names"):
            for i, name in enumerate(robot.body_names):
                if "jaw" in name.lower() or "grip" in name.lower() or "finger" in name.lower():
                    print(f"    [{i}] {name}  << candidate")
                else:
                    print(f"    [{i}] {name}")

    # -------------------------------------------------------
    # 3. Joint info
    # -------------------------------------------------------
    print("\n--- 3. Joint info ---")
    print(f"  num_joints: {robot.num_joints}")
    if hasattr(robot, "joint_names"):
        for i, name in enumerate(robot.joint_names):
            print(f"    [{i}] {name}")

    print(f"\n  soft_joint_pos_limits exists: {hasattr(robot.data, 'soft_joint_pos_limits')}")
    if hasattr(robot.data, "soft_joint_pos_limits"):
        limits = robot.data.soft_joint_pos_limits[0]
        print(f"  soft_joint_pos_limits shape: {limits.shape}")
        inf_count = 0
        for i in range(limits.shape[0]):
            lo, hi = limits[i, 0].item(), limits[i, 1].item()
            is_inf = (lo == float("-inf") or hi == float("inf") or
                      abs(lo) > 1e6 or abs(hi) > 1e6)
            if is_inf:
                inf_count += 1
            name = robot.joint_names[i] if hasattr(robot, "joint_names") else f"joint_{i}"
            flag = " *** INF ***" if is_inf else ""
            print(f"    [{i}] {name}: [{lo:.4f}, {hi:.4f}]{flag}")
        print(f"\n  Joints with inf limits: {inf_count}")

    # -------------------------------------------------------
    # 4. Verify step produces velocity data
    # -------------------------------------------------------
    print("\n--- 4. Velocity after step ---")
    action = torch.zeros(cfg.scene.num_envs, env.num_actions, device=env.device)
    # Give some base velocity
    if env.num_actions >= 3:
        action[:, 0] = 0.5  # first action channel
    obs, reward, term, trunc, info = env.step(action)
    for _ in range(10):
        obs, reward, term, trunc, info = env.step(action)

    if has_lin_vel_b:
        vel = robot.data.root_lin_vel_b[0]
        print(f"  After 10 steps with action, root_lin_vel_b[0]: {vel.tolist()}")
        print(f"  Non-zero: {vel.abs().sum().item() > 1e-6}")
    elif hasattr(robot.data, "root_lin_vel_w"):
        vel = robot.data.root_lin_vel_w[0]
        print(f"  After 10 steps, root_lin_vel_w[0]: {vel.tolist()}")

    # -------------------------------------------------------
    # 5. scene.env_origins
    # -------------------------------------------------------
    print("\n--- 5. env_origins ---")
    has_origins = hasattr(env.scene, "env_origins")
    print(f"  scene.env_origins exists: {has_origins}")
    if has_origins:
        print(f"  env_origins shape: {env.scene.env_origins.shape}")
        print(f"  env_origins[0]: {env.scene.env_origins[0].tolist()}")

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    critical = []
    if not has_lin_vel_b:
        critical.append("root_lin_vel_b NOT found - need fallback in _read_base_body_vel()")
    if not has_ang_vel_b:
        critical.append("root_ang_vel_b NOT found - need fallback in _read_base_body_vel()")
    if not has_body_pos:
        critical.append("body_pos_w NOT found - grasp break detection broken")

    if critical:
        print("[CRITICAL ISSUES]")
        for c in critical:
            print(f"  - {c}")
    else:
        print("[ALL CRITICAL APIs AVAILABLE]")
        print("  root_lin_vel_b: OK")
        print("  root_ang_vel_b: OK")
        print("  body_pos_w: OK")
        print("  find_bodies('Moving_Jaw_08d_v1'): check above")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    check_api()
