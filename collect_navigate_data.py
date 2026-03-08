#!/usr/bin/env python3
"""
LeKiwi Navigate (Skill-1) — Scripted proportional controller data collection.

Generates VLA training data using a P-controller that drives toward a target
object, with noise injection for trajectory diversity.  No RL checkpoint is
required; the policy is fully scripted.

Controller:
  - P_linear  = 2.0,  P_angular = 3.0
  - Gaussian steering noise  sigma = --noise_std (default 0.05)
  - Forward speed quantised to [0, 0.3, 0.6, 1.0]
  - Action repetition: same action held for 2-4 steps (random)
  - Success (handoff to Skill-2): distance to object < --approach_threshold

HDF5 layout (matches collect_demos.py v8):
    episode_N/
        obs              (T, obs_dim)       float32
        actions          (T, 9)             float32
        robot_state      (T, 9)             float32
        images/
          base_rgb       (T, H, W, 3)       uint8  gzip-4
          wrist_rgb      (T, H, W, 3)       uint8  gzip-4
        attrs:
          num_steps, final_object_dist, final_dist, success, instruction
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="LeKiwi Navigate (Skill-1) — scripted P-controller demo collection"
)
parser.add_argument("--num_demos", type=int, default=50)
parser.add_argument(
    "--num_envs",
    type=int,
    default=4,
    help="Parallel envs (camera on → 1-8 recommended for VRAM)",
)
parser.add_argument("--max_attempts", type=int, default=200)
parser.add_argument("--output", type=str, default=None)

# Environment config JSONs
parser.add_argument("--dynamics_json", type=str, default=None)
parser.add_argument("--calibration_json", type=str, default=None)
parser.add_argument("--arm_limit_json", type=str, default=None)
parser.add_argument("--arm_limit_margin_rad", type=float, default=0.0)
parser.add_argument("--multi_object_json", type=str, default="")
parser.add_argument("--object_usd", type=str, default="")
parser.add_argument("--object_mass", type=float, default=0.3)
parser.add_argument("--object_scale_phys", type=float, default=1.0)
parser.add_argument(
    "--gripper_contact_prim_path", type=str, default="",
    help="Contact sensor prim path for gripper body",
)
parser.add_argument("--grasp_gripper_threshold", type=float, default=-0.3)
parser.add_argument("--grasp_contact_threshold", type=float, default=0.5)
parser.add_argument("--grasp_max_object_dist", type=float, default=0.25)
parser.add_argument("--grasp_attach_height", type=float, default=0.15)

# Camera
parser.add_argument("--no_camera", action="store_true", help="State-only (no images)")
parser.add_argument("--base_cam_width", type=int, default=1280)
parser.add_argument("--base_cam_height", type=int, default=720)
parser.add_argument("--wrist_cam_width", type=int, default=640)
parser.add_argument("--wrist_cam_height", type=int, default=480)

# Navigate controller
parser.add_argument(
    "--noise_std", type=float, default=0.05,
    help="Gaussian noise std for steering diversity",
)
parser.add_argument(
    "--approach_threshold", type=float, default=0.5,
    help="Distance (m) to object for navigation success (Skill-1 handoff)",
)

# SpawnManager
parser.add_argument("--objects_index", type=str, default=None)
parser.add_argument("--object_scale", type=float, default=0.7)
parser.add_argument("--object_cap", type=int, default=0)
parser.add_argument("--min_steps", type=int, default=10)

dr_group = parser.add_mutually_exclusive_group()
dr_group.add_argument("--dr_lighting", dest="dr_lighting", action="store_true")
dr_group.add_argument("--no_dr_lighting", dest="dr_lighting", action="store_false")
parser.set_defaults(dr_lighting=True)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
sim_app = launcher.app

# ---------------------------------------------------------------------------
# Imports after AppLauncher (Isaac Sim requirement)
# ---------------------------------------------------------------------------
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

from lekiwi_nav_env import LeKiwiNavEnv, LeKiwiNavEnvCfg  # noqa: E402

try:
    from spawn_manager import SpawnManager
except ImportError:
    SpawnManager = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENV_PRIM = "/World/envs/env_.*/Robot"

BASE_RGB_CAM_PRIM = (
    f"{ENV_PRIM}/base_plate_layer1_v5/Realsense/RSD455"
    f"/Camera_OmniVision_OV9782_Color"
)
WRIST_CAM_PRIM = (
    f"{ENV_PRIM}/Wrist_Roll_08c_v1/visuals/mesh_002_3/wrist_camera"
)

# Proportional gains
P_LINEAR = 2.0
P_ANGULAR = 3.0

# Speed quantisation levels
SPEED_LEVELS = [0.0, 0.3, 0.6, 1.0]

NAVIGATE_INSTRUCTION = "navigate toward the target object"


# ---------------------------------------------------------------------------
# Camera-augmented env (reused from collect_demos.py)
# ---------------------------------------------------------------------------
class LeKiwiNavEnvWithCam(LeKiwiNavEnv):
    """LeKiwiNavEnv + base_rgb / wrist RGB cameras."""

    def __init__(
        self,
        cfg,
        base_cam_w: int = 1280,
        base_cam_h: int = 720,
        wrist_cam_w: int = 640,
        wrist_cam_h: int = 480,
        render_mode=None,
        **kwargs,
    ):
        self._base_cam_w = base_cam_w
        self._base_cam_h = base_cam_h
        self._wrist_cam_w = wrist_cam_w
        self._wrist_cam_h = wrist_cam_h
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        super()._setup_scene()
        base_cam_cfg = CameraCfg(
            prim_path=BASE_RGB_CAM_PRIM,
            spawn=None,
            update_period=0.0,
            height=self._base_cam_h,
            width=self._base_cam_w,
            data_types=["rgb"],
        )
        self.base_cam = Camera(base_cam_cfg)
        self.scene.sensors["base_cam"] = self.base_cam

        wrist_cam_cfg = CameraCfg(
            prim_path=WRIST_CAM_PRIM,
            spawn=None,
            update_period=0.0,
            height=self._wrist_cam_h,
            width=self._wrist_cam_w,
            data_types=["rgb"],
        )
        self.wrist_cam = Camera(wrist_cam_cfg)
        self.scene.sensors["wrist_cam"] = self.wrist_cam

        print(f"  [Camera] base_rgb : {self._base_cam_w}x{self._base_cam_h}")
        print(f"  [Camera] wrist_rgb: {self._wrist_cam_w}x{self._wrist_cam_h}")

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


# ---------------------------------------------------------------------------
# Helper: extract robot state 9D (arm_pos(6) + wheel_vel(3))
# ---------------------------------------------------------------------------
def extract_robot_state_9d(env: LeKiwiNavEnv) -> torch.Tensor:
    arm_pos = env.robot.data.joint_pos[:, env.arm_idx]
    wheel_vel = env.robot.data.joint_vel[:, env.wheel_idx]
    return torch.cat([arm_pos, wheel_vel], dim=-1)


# ---------------------------------------------------------------------------
# Scripted Navigate Policy
# ---------------------------------------------------------------------------
def _quantise_speed(v: float) -> float:
    """Snap forward speed magnitude to the nearest quantisation level."""
    best = SPEED_LEVELS[0]
    best_d = abs(v - best)
    for lvl in SPEED_LEVELS[1:]:
        d = abs(v - lvl)
        if d < best_d:
            best, best_d = lvl, d
    return math.copysign(best, v)


def navigate_policy_batched(
    object_xy_body: torch.Tensor,
    noise_std: float = 0.05,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Proportional controller with noise for a batch of envs.

    Args:
        object_xy_body: (N, 2) — object relative position in body frame.
        noise_std: Gaussian noise std for action diversity.
        device: torch device.

    Returns:
        actions: (N, 9) — [vx, vy, wz, 0, 0, 0, 0, 0, 0]
    """
    N = object_xy_body.shape[0]
    dx = object_xy_body[:, 0]
    dy = object_xy_body[:, 1]
    angle = torch.atan2(dy, dx)

    # Proportional control
    vx = P_LINEAR * dx
    vy = P_LINEAR * dy
    wz = P_ANGULAR * angle

    # Clamp to [-1, 1]
    vx = vx.clamp(-1.0, 1.0)
    vy = vy.clamp(-1.0, 1.0)
    wz = wz.clamp(-1.0, 1.0)

    # Add Gaussian noise
    if noise_std > 0:
        vx = vx + torch.randn(N, device=device) * noise_std
        vy = vy + torch.randn(N, device=device) * noise_std
        wz = wz + torch.randn(N, device=device) * noise_std

    # Quantise forward speed
    vx_np = vx.cpu().numpy()
    vx_q = np.array([_quantise_speed(float(v)) for v in vx_np], dtype=np.float32)
    vx = torch.from_numpy(vx_q).to(device)

    # Final clamp after noise
    vx = vx.clamp(-1.0, 1.0)
    vy = vy.clamp(-1.0, 1.0)
    wz = wz.clamp(-1.0, 1.0)

    # Build 9D action: [vx, vy, wz, 0,0,0,0,0,0]  (arm stays at rest)
    actions = torch.zeros(N, 9, device=device)
    actions[:, 0] = vx
    actions[:, 1] = vy
    actions[:, 2] = wz

    return actions


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------
def _clear_buffers(idx, ep_obs, ep_act, ep_base_img, ep_wrist_img, ep_robot_state):
    ep_obs[idx].clear()
    ep_act[idx].clear()
    ep_base_img[idx].clear()
    ep_wrist_img[idx].clear()
    ep_robot_state[idx].clear()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    use_camera = not args.no_camera
    physics_grasp_mode = bool(str(args.object_usd).strip()) or bool(
        str(args.multi_object_json).strip()
    )
    multi_object_mode = bool(str(args.multi_object_json).strip())
    use_spawn = (
        args.objects_index is not None
        and use_camera
        and (not physics_grasp_mode)
        and SpawnManager is not None
    )

    # Output path
    if args.output:
        output_path = args.output
    else:
        os.makedirs("navigate_demos", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = "_cam" if use_camera else ""
        output_path = f"navigate_demos/navigate_skill1{suffix}_{timestamp}.hdf5"

    # SpawnManager (optional)
    spawn_mgr = None
    if use_spawn:
        index_path = os.path.expanduser(args.objects_index)
        spawn_mgr = SpawnManager(
            index_path=index_path,
            num_envs=args.num_envs,
            object_scale=args.object_scale,
            object_cap=args.object_cap,
        )

    use_dr_lighting = bool(args.dr_lighting and use_spawn and spawn_mgr is not None)

    print("\n" + "=" * 60)
    print("  LeKiwi Navigate (Skill-1) — Scripted P-Controller Collection")
    print(f"  Target demos      : {args.num_demos}")
    print(f"  Parallel envs     : {args.num_envs}")
    print(f"  P_linear={P_LINEAR}, P_angular={P_ANGULAR}")
    print(f"  Noise std          : {args.noise_std}")
    print(f"  Approach threshold : {args.approach_threshold} m")
    if use_camera:
        print(
            f"  Camera             : base ({args.base_cam_width}x{args.base_cam_height})"
            f" + wrist ({args.wrist_cam_width}x{args.wrist_cam_height})"
        )
    else:
        print("  Camera             : disabled")
    if use_spawn and spawn_mgr is not None:
        print(f"  SpawnMgr           : {len(spawn_mgr.objects_list)} objects")
        print(f"  Lighting DR        : {'ON' if use_dr_lighting else 'OFF'}")
    if args.dynamics_json:
        print(f"  Dynamics           : {os.path.expanduser(args.dynamics_json)}")
    if args.calibration_json is not None:
        cal = str(args.calibration_json).strip()
        if cal:
            print(f"  Calibration        : {os.path.expanduser(cal)}")
        else:
            print("  Calibration        : (disabled)")
    if args.arm_limit_json:
        print(f"  Arm limits         : {os.path.expanduser(args.arm_limit_json)}")
    print(f"  Output             : {output_path}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Build environment
    # ------------------------------------------------------------------
    env_cfg = LeKiwiNavEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.calibration_json is not None:
        raw = str(args.calibration_json).strip()
        env_cfg.calibration_json = os.path.expanduser(raw) if raw else ""
    if args.dynamics_json:
        env_cfg.dynamics_json = os.path.expanduser(args.dynamics_json)
    if args.arm_limit_json:
        env_cfg.arm_limit_json = os.path.expanduser(args.arm_limit_json)
        env_cfg.arm_limit_margin_rad = float(args.arm_limit_margin_rad)
    if multi_object_mode:
        env_cfg.multi_object_json = os.path.expanduser(args.multi_object_json)
    if physics_grasp_mode:
        env_cfg.object_usd = os.path.expanduser(args.object_usd)
        env_cfg.object_mass = float(args.object_mass)
        env_cfg.object_scale = float(args.object_scale_phys)
        env_cfg.gripper_contact_prim_path = str(args.gripper_contact_prim_path)
        env_cfg.grasp_gripper_threshold = float(args.grasp_gripper_threshold)
        env_cfg.grasp_contact_threshold = float(args.grasp_contact_threshold)
        env_cfg.grasp_max_object_dist = float(args.grasp_max_object_dist)
        env_cfg.grasp_attach_height = float(args.grasp_attach_height)

    if use_camera:
        env = LeKiwiNavEnvWithCam(
            cfg=env_cfg,
            base_cam_w=args.base_cam_width,
            base_cam_h=args.base_cam_height,
            wrist_cam_w=args.wrist_cam_width,
            wrist_cam_h=args.wrist_cam_height,
        )
    else:
        env = LeKiwiNavEnv(cfg=env_cfg)

    device = env.device

    print(
        f"  Geometry: wheel={env.wheel_radius:.6f}, base={env.base_radius:.6f}"
    )

    obs, info = env.reset()

    if use_spawn and spawn_mgr is not None:
        if use_dr_lighting:
            spawn_mgr.randomize_lighting()
        spawn_mgr.spawn_all(env.object_pos_w)
        print(f"  Initial object spawn done ({args.num_envs} envs)\n")

    # ------------------------------------------------------------------
    # Episode buffers
    # ------------------------------------------------------------------
    ep_obs = [[] for _ in range(args.num_envs)]
    ep_act = [[] for _ in range(args.num_envs)]
    ep_base_img = [[] for _ in range(args.num_envs)]
    ep_wrist_img = [[] for _ in range(args.num_envs)]
    ep_robot_state = [[] for _ in range(args.num_envs)]

    ep_object_name = ["" for _ in range(args.num_envs)]
    ep_spawn_meta = [{} for _ in range(args.num_envs)]
    if use_spawn and spawn_mgr is not None:
        for i in range(args.num_envs):
            ep_object_name[i] = spawn_mgr.get_object_name(i)
            ep_spawn_meta[i] = spawn_mgr.get_spawn_metadata(i)

    # Action repetition state: per-env counters
    repeat_action = torch.zeros(args.num_envs, 9, device=device)
    repeat_remaining = torch.zeros(args.num_envs, dtype=torch.long, device=device)

    saved = 0
    attempts = 0
    dists: list[float] = []

    # ------------------------------------------------------------------
    # HDF5 file
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    hdf5_file = h5py.File(output_path, "w")
    hdf5_file.attrs["has_camera"] = use_camera
    hdf5_file.attrs["has_robot_state"] = True
    hdf5_file.attrs["obs_dim"] = int(env.observation_space.shape[0])
    hdf5_file.attrs["action_dim"] = int(env.action_space.shape[0])
    hdf5_file.attrs["has_spawn_metadata"] = bool(use_spawn)
    hdf5_file.attrs["collection_policy"] = "proportional_controller"
    hdf5_file.attrs["p_linear"] = P_LINEAR
    hdf5_file.attrs["p_angular"] = P_ANGULAR
    hdf5_file.attrs["noise_std"] = float(args.noise_std)
    hdf5_file.attrs["approach_threshold"] = float(args.approach_threshold)
    hdf5_file.attrs["speed_levels"] = json.dumps(SPEED_LEVELS)
    hdf5_file.attrs["skill"] = "navigate"
    hdf5_file.attrs["skill_id"] = 1
    if use_camera:
        hdf5_file.attrs["base_rgb_shape"] = [args.base_cam_height, args.base_cam_width, 3]
        hdf5_file.attrs["wrist_rgb_shape"] = [args.wrist_cam_height, args.wrist_cam_width, 3]
    if use_spawn and spawn_mgr is not None:
        hdf5_file.attrs["objects_index_path"] = str(
            os.path.expanduser(args.objects_index)
        )
        hdf5_file.attrs["object_scale"] = float(args.object_scale)
        hdf5_file.attrs["num_objects_in_library"] = int(len(spawn_mgr.objects_list))
    if args.dynamics_json:
        hdf5_file.attrs["dynamics_json"] = str(
            os.path.expanduser(args.dynamics_json)
        )
        hdf5_file.attrs["dynamics_scaled_max_lin_vel"] = float(env.cfg.max_lin_vel)
        hdf5_file.attrs["dynamics_scaled_max_ang_vel"] = float(env.cfg.max_ang_vel)

    # ------------------------------------------------------------------
    # Collection loop
    # ------------------------------------------------------------------
    try:
        while saved < args.num_demos and attempts < args.max_attempts:
            # --- Compute object_xy_body from env internals ---
            root_pos_w = env.robot.data.root_pos_w
            root_quat_w = env.robot.data.root_quat_w

            from isaaclab.utils.math import quat_apply_inverse

            object_delta_w = env.object_pos_w - root_pos_w
            object_pos_b = quat_apply_inverse(root_quat_w, object_delta_w)
            object_xy_b = object_pos_b[:, :2]  # (N, 2)
            object_dist = torch.norm(object_xy_b, dim=-1)  # (N,)

            # --- Action repetition: only recompute when counter exhausted ---
            needs_new = repeat_remaining <= 0  # (N,)
            if needs_new.any():
                new_actions = navigate_policy_batched(
                    object_xy_body=object_xy_b,
                    noise_std=args.noise_std,
                    device=device,
                )
                # Draw new repetition counts (2-4) for envs that need them
                new_counts = torch.randint(2, 5, (args.num_envs,), device=device)
                repeat_action[needs_new] = new_actions[needs_new]
                repeat_remaining[needs_new] = new_counts[needs_new]

            action = repeat_action.clone()
            repeat_remaining -= 1

            # --- Capture robot state and images at time t ---
            step_robot_state = extract_robot_state_9d(env)

            base_rgb = None
            wrist_rgb = None
            if use_camera:
                base_rgb = env.get_base_rgb()
                wrist_rgb = env.get_wrist_rgb()

            # --- Record into episode buffers ---
            for i in range(args.num_envs):
                ep_obs[i].append(obs["policy"][i].cpu().numpy())
                ep_act[i].append(action[i].cpu().numpy())
                ep_robot_state[i].append(step_robot_state[i].cpu().numpy())
                if use_camera:
                    if base_rgb is not None:
                        ep_base_img[i].append(base_rgb[i].cpu().numpy())
                    if wrist_rgb is not None:
                        ep_wrist_img[i].append(wrist_rgb[i].cpu().numpy())

            # --- Step environment ---
            next_obs, reward, terminated, truncated, info = env.step(action)

            if use_spawn and spawn_mgr is not None:
                spawn_mgr.update_all_positions(env, env.object_pos_w)

            # --- Handle episode termination ---
            done = terminated | truncated
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)

            for idx in done_ids.tolist():
                attempts += 1
                if len(ep_obs[idx]) < 5:
                    _clear_buffers(idx, ep_obs, ep_act, ep_base_img, ep_wrist_img, ep_robot_state)
                    repeat_remaining[idx] = 0
                    if use_spawn and spawn_mgr is not None:
                        spawn_mgr.respawn_for_env(idx, env.object_pos_w[idx])
                        ep_object_name[idx] = spawn_mgr.get_object_name(idx)
                        ep_spawn_meta[idx] = spawn_mgr.get_spawn_metadata(idx)
                    continue

                # Compute final distance to object
                root_xy = env.robot.data.root_pos_w[idx, :2].cpu().numpy()
                obj_xy = env.object_pos_w[idx, :2].cpu().numpy()
                dist = float(np.linalg.norm(root_xy - obj_xy))

                # Navigate success: reached within approach_threshold
                success = dist < args.approach_threshold

                if success and saved < args.num_demos:
                    grp = hdf5_file.create_group(f"episode_{saved}")
                    grp.create_dataset("obs", data=np.array(ep_obs[idx]))
                    grp.create_dataset(
                        "actions", data=np.array(ep_act[idx], dtype=np.float32)
                    )
                    grp.create_dataset(
                        "robot_state",
                        data=np.array(ep_robot_state[idx], dtype=np.float32),
                    )

                    if use_camera:
                        img_grp = grp.create_group("images")
                        if len(ep_base_img[idx]) > 0:
                            img_grp.create_dataset(
                                "base_rgb",
                                data=np.array(ep_base_img[idx], dtype=np.uint8),
                                compression="gzip",
                                compression_opts=4,
                                chunks=(
                                    1,
                                    args.base_cam_height,
                                    args.base_cam_width,
                                    3,
                                ),
                            )
                        if len(ep_wrist_img[idx]) > 0:
                            img_grp.create_dataset(
                                "wrist_rgb",
                                data=np.array(ep_wrist_img[idx], dtype=np.uint8),
                                compression="gzip",
                                compression_opts=4,
                                chunks=(
                                    1,
                                    args.wrist_cam_height,
                                    args.wrist_cam_width,
                                    3,
                                ),
                            )

                    grp.attrs["num_steps"] = len(ep_obs[idx])
                    grp.attrs["final_object_dist"] = dist
                    grp.attrs["final_dist"] = dist  # legacy alias
                    grp.attrs["success"] = True
                    grp.attrs["has_images"] = bool(use_camera)
                    grp.attrs["num_base_imgs"] = len(ep_base_img[idx])
                    grp.attrs["num_wrist_imgs"] = len(ep_wrist_img[idx])
                    grp.attrs["approach_threshold"] = float(args.approach_threshold)
                    grp.attrs["skill"] = "navigate"

                    if use_spawn and spawn_mgr is not None:
                        obj_name = ep_object_name[idx]
                        grp.attrs["object_name"] = obj_name
                        grp.attrs["instruction"] = (
                            f"navigate toward the {obj_name}"
                            if obj_name and obj_name != "target object"
                            else NAVIGATE_INSTRUCTION
                        )
                        grp.attrs["spawn_meta_json"] = json.dumps(
                            ep_spawn_meta[idx], ensure_ascii=False
                        )
                        grp.attrs["object_usd"] = str(
                            ep_spawn_meta[idx].get("object_usd", "")
                        )
                        grp.attrs["object_scale"] = float(
                            ep_spawn_meta[idx].get("object_scale", args.object_scale)
                        )
                        spawn_mgr.record_saved(idx)
                    else:
                        grp.attrs["instruction"] = NAVIGATE_INSTRUCTION

                    hdf5_file.flush()

                    saved += 1
                    dists.append(dist)

                    img_info = f" | imgs={len(ep_base_img[idx])}" if use_camera else ""
                    obj_info = ""
                    if use_spawn and spawn_mgr is not None:
                        obj_info = f" | obj={ep_object_name[idx]}"
                    print(
                        f"  Demo {saved:>3}/{args.num_demos} | "
                        f"steps={len(ep_obs[idx]):>4} | dist={dist:.3f}m | "
                        f"att={attempts}{img_info}{obj_info}"
                    )

                # Clear buffers regardless of success
                _clear_buffers(idx, ep_obs, ep_act, ep_base_img, ep_wrist_img, ep_robot_state)
                repeat_remaining[idx] = 0

                if use_spawn and spawn_mgr is not None:
                    if use_dr_lighting:
                        spawn_mgr.randomize_lighting()
                    spawn_mgr.respawn_for_env(idx, env.object_pos_w[idx])
                    ep_object_name[idx] = spawn_mgr.get_object_name(idx)
                    ep_spawn_meta[idx] = spawn_mgr.get_spawn_metadata(idx)

            obs = next_obs

    except KeyboardInterrupt:
        print("\n  Interrupted by user")
    finally:
        hdf5_file.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    rate = saved / max(attempts, 1) * 100
    print("\n" + "=" * 60)
    print(f"  Collection done: {saved}/{args.num_demos}")
    print(f"  Attempts: {attempts} (success rate: {rate:.1f}%)")
    if dists:
        print(f"  Mean final object dist: {np.mean(dists):.3f} m")
    if use_camera:
        print(
            f"  Camera: base ({args.base_cam_width}x{args.base_cam_height})"
            f" + wrist ({args.wrist_cam_width}x{args.wrist_cam_height})"
        )
    if use_spawn and spawn_mgr is not None:
        stats = spawn_mgr.get_object_stats()
        print(
            f"  Object diversity: {stats['total_objects_used']}/{stats['total_library']} types"
        )
        if stats["top_5"]:
            print(f"  Top 5: {stats['top_5']}")
    print(f"  Output: {output_path}")
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")
    print("=" * 60)

    if use_spawn and spawn_mgr is not None:
        spawn_mgr.despawn_all()
    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
