#!/usr/bin/env python3
"""
ProcTHOR Scene spawn 검증 + 충돌 테스트.
record_teleop_scene.py와 동일한 환경 설정 사용.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

parser = argparse.ArgumentParser(description="Validate MolmoSpaces spawn layout")
parser.add_argument("--scene_idx", type=int, default=1302)
parser.add_argument("--scene_usd", type=str, default="")
parser.add_argument("--scene_install_dir", type=str, default="~/molmospaces/assets/usd")
parser.add_argument("--scene_floor_z", type=float, default=None)
parser.add_argument("--scene_object_rest_z", type=float, default=0.033)
parser.add_argument("--scene_robot_x", type=float, default=None)
parser.add_argument("--scene_robot_y", type=float, default=None)
parser.add_argument("--scene_robot_yaw_deg", type=float, default=None)
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--dest_object_usd", type=str, default="")
parser.add_argument("--arm_limit_json", type=str, default="calibration/arm_limits_measured.json")
parser.add_argument("--gripper_contact_prim_path", type=str,
                    default="/World/envs/env_.*/Robot/LeKiwi/Moving_Jaw_08d_v1")
parser.add_argument("--scene_scale", type=float, default=0.6,
                    help="Scene 전체 스케일 (record_teleop_scene과 동일, 기본 0.6)")
parser.add_argument("--num_trials", type=int, default=5)
parser.add_argument("--settle_steps", type=int, default=120)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np

import isaaclab.sim as sim_utils
from lekiwi_skill2_env import Skill2Env, Skill2EnvCfg
from procthor_scene import (
    SceneSpawnCfg,
    apply_scene_task_layout,
    estimate_spawn_clearance,
    load_scene_reference,
    resolve_scene_usd,
    sample_scene_task_layout,
)


def make_env(scene_usd_path: str = "", floor_z: float = 0.0, scene_scale: float = 0.6) -> Skill2Env:
    """record_teleop_scene.py와 동일한 환경 설정."""
    cfg = Skill2EnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 1.0
    cfg.sim.device = "cpu" if scene_usd_path else "cuda:0"
    cfg.enable_domain_randomization = False
    cfg.arm_limit_write_to_sim = False
    cfg.episode_length_s = 60.0

    cfg.object_scale = 0.7
    cfg.dest_object_scale = 0.56
    cfg.dest_object_fixed = False
    cfg.dest_object_mass = 50.0
    cfg.grasp_contact_threshold = 0.55
    cfg.grasp_gripper_threshold = 0.65
    cfg.grasp_success_height = 0.05
    cfg.lift_hold_steps = 500
    cfg.max_dist_from_origin = 50.0

    # Scene
    cfg.use_builtin_ground = True
    cfg.builtin_ground_z = floor_z
    if scene_usd_path:
        cfg.scene_reference_usd = scene_usd_path
        cfg.scene_scale = scene_scale

    if args.object_usd:
        cfg.object_usd = os.path.expanduser(args.object_usd)
    if args.dest_object_usd:
        cfg.dest_object_usd = os.path.expanduser(args.dest_object_usd)
    cfg.gripper_contact_prim_path = args.gripper_contact_prim_path
    if args.arm_limit_json and os.path.isfile(args.arm_limit_json):
        cfg.arm_limit_json = args.arm_limit_json

    return Skill2Env(cfg=cfg)


def main():
    scene_path = resolve_scene_usd(args.scene_idx, args.scene_usd, args.scene_install_dir)
    if scene_path is None:
        raise FileNotFoundError(f"Scene not found: idx={args.scene_idx}, usd={args.scene_usd}")

    from procthor_scene import _load_support_floor_z, SCENE_PRESETS
    preset = SCENE_PRESETS.get(args.scene_idx)
    if preset and args.scene_floor_z is None:
        floor_z = _load_support_floor_z(str(scene_path.resolve()), preset.support_floor_prim_path)
    else:
        floor_z = args.scene_floor_z or 0.0

    scaled_floor_z = floor_z * args.scene_scale
    print(f"[Scene] {scene_path}, floor_z={floor_z:.4f}, scaled_floor_z={scaled_floor_z:.4f}, "
          f"scene_scale={args.scene_scale}")

    env = make_env(scene_usd_path=str(scene_path), floor_z=scaled_floor_z,
                   scene_scale=args.scene_scale)

    robot_xy = None
    if args.scene_robot_x is not None and args.scene_robot_y is not None:
        robot_xy = (float(args.scene_robot_x), float(args.scene_robot_y))
    robot_yaw_rad = None if args.scene_robot_yaw_deg is None else math.radians(float(args.scene_robot_yaw_deg))

    ss = float(args.scene_scale) if args.scene_scale > 0 else 1.0
    source_override = SceneSpawnCfg(
        min_robot_dist=float(getattr(env.cfg, "object_dist_min", 0.8)) / ss,
        max_robot_dist=float(getattr(env.cfg, "object_dist_max", 1.2)) / ss,
        clearance_radius=0.14,
    )

    passes = 0
    for trial in range(args.num_trials):
        env.reset()

        for _retry in range(20):
            try:
                layout = sample_scene_task_layout(
                    args.scene_idx,
                    scene_usd=scene_path,
                    robot_xy=robot_xy,
                    robot_yaw_rad=robot_yaw_rad,
                    source_rest_z=args.scene_object_rest_z,
                    floor_z=args.scene_floor_z,
                    scene_scale=args.scene_scale,
                    source_spawn_override=source_override,
                    robot_faces_source=True,
                    randomize_robot_xy=True,
                )
                break
            except RuntimeError:
                if _retry < 19:
                    continue
                raise

        apply_scene_task_layout(env, layout)

        for _ in range(args.settle_steps):
            env.sim.step()
            env.sim.render()

        # 측정
        obj_dist = math.dist(layout.robot_xy, layout.source_xy)
        robot_z = float(env.robot.data.root_pos_w[0, 2].item() - layout.floor_z)
        robot_z_ref = float(env.robot.data.default_root_state[0, 2].item())

        src_z = 0.0
        src_drift = 0.0
        if getattr(env, "object_rigid", None) is not None:
            src_xy0 = torch.tensor(layout.source_xy, device=env.device)
            src_xy = env.object_rigid.data.root_pos_w[0, :2]
            src_z = float(env.object_rigid.data.root_pos_w[0, 2].item() - layout.floor_z)
            src_drift = float(torch.norm(src_xy - src_xy0).item())

        dst_z = 0.0
        dst_drift = 0.0
        if getattr(env, "_dest_object_rigid", None) is not None:
            dst_xy0 = torch.tensor(layout.dest_xy, device=env.device)
            dst_xy = env._dest_object_rigid.data.root_pos_w[0, :2]
            dst_z = float(env._dest_object_rigid.data.root_pos_w[0, 2].item() - layout.floor_z)
            dst_drift = float(torch.norm(dst_xy - dst_xy0).item())

        ok = (
            src_drift < 0.25
            and dst_drift < 0.25
            and abs(robot_z - robot_z_ref) < 0.15
            and obj_dist > 0.3
        )
        passes += int(ok)
        print(
            f"[Trial {trial + 1}] {'PASS' if ok else 'FAIL'} "
            f"robot=({layout.robot_xy[0]:.2f}, {layout.robot_xy[1]:.2f}) "
            f"src=({layout.source_xy[0]:.2f}, {layout.source_xy[1]:.2f}) "
            f"obj_dist={obj_dist:.2f}m "
            f"robot_z={robot_z:.3f} src_z={src_z:.3f} dst_z={dst_z:.3f} "
            f"src_drift={src_drift:.3f} dst_drift={dst_drift:.3f}"
        )

    print(f"[Summary] pass={passes}/{args.num_trials}")

    # Interactive: GUI 유지 (전진 없이 스폰 결과만 확인)
    if not args.headless:
        print("[Interactive] 스폰 결과 확인 중. 창 닫으면 종료.")
        while simulation_app.is_running():
            env.sim.step()
            env.sim.render()

    env.close()
    simulation_app.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
