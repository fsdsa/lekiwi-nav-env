#!/usr/bin/env python3
"""Navigate demo collection on Skill2Env (cuboid ground, friction=0.5).
Lookup table action, 6 directions × N reps, headless.
Saves 20D obs + 9D robot_state + 9D actions.

Tucked pose: j3=-0.4 (카메라 시야 확보).
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_reps", type=int, default=20)
parser.add_argument("--steps_per_ep", type=int, default=200)
parser.add_argument("--output", type=str, default="demos/navigate_skill2_24ep.hdf5")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
launcher = AppLauncher(args)
sim_app = launcher.app

import h5py, math, torch, numpy as np
from lekiwi_skill2_eval import Skill2Env, Skill2EnvCfg

cfg = Skill2EnvCfg()
cfg.scene.num_envs = 1
cfg.sim.device = "cuda:0"
cfg.enable_domain_randomization = False
cfg.arm_limit_write_to_sim = False
cfg.episode_length_s = 3600.0
cfg.max_dist_from_origin = 50.0
cfg.dr_action_delay_steps = 0

env = Skill2Env(cfg=cfg)

# Tucked pose in normalized [-1,1] action space (arm_action_to_limits 매핑 역변환)
# target: [-0.02966, -0.213839, 0.09066, -0.4, 0.058418, -0.201554]
# = (target - center) / half, clamped to [-1, 1]
_TUCKED_ARM_NORM = torch.tensor([-0.000791, -1.0, 1.0, 0.658716, -0.537318], device=env.device)
_TUCKED_GRIP_NORM = torch.tensor(-0.999472, device=env.device)

# init_arm_pose: 에피소드 시작 시점의 실제 joint_pos (매 에피소드마다 갱신)

_orig_dones = env._get_dones
def _no_dones():
    t, tr = _orig_dones()
    t[:] = False; tr[:] = False
    return t, tr
env._get_dones = _no_dones

DIRECTIONS = {
    "navigate forward":      [0.0, 0.5, 0.0],
    "navigate backward":     [0.0, -0.5, 0.0],
    "navigate strafe left":  [-0.5, 0.0, 0.0],
    "navigate strafe right": [0.5, 0.0, 0.0],
    "navigate turn left":    [0.0, 0.0, -0.33],
    "navigate turn right":   [0.0, 0.0, 0.33],
}

# direction_cmd 매핑 (20D obs용)
DIR_CMD = {
    "navigate forward":      [0.0, 1.0, 0.0],
    "navigate backward":     [0.0, -1.0, 0.0],
    "navigate strafe left":  [-1.0, 0.0, 0.0],
    "navigate strafe right": [1.0, 0.0, 0.0],
    "navigate turn left":    [0.0, 0.0, 1.0],
    "navigate turn right":   [0.0, 0.0, -1.0],
}

os.makedirs(os.path.dirname(args.output) or "demos", exist_ok=True)
hf = h5py.File(args.output, "w")
ep_idx = 0
total = len(DIRECTIONS) * args.num_reps
t0 = time.time()

for inst, base_cmd in DIRECTIONS.items():
    dir_cmd = DIR_CMD[inst]
    for rep in range(args.num_reps):
        obs, _ = env.reset()
        # 에피소드 시작 시점의 arm pose 캡처 (init_arm_pose)
        jp0 = env.robot.data.joint_pos[0]
        init_arm = jp0[env.arm_idx[:5]].cpu().numpy().astype(np.float32)
        init_grip = np.array([jp0[env.arm_idx[5]].item()], dtype=np.float32)
        init_arm_pose = np.concatenate([init_arm, init_grip])  # (6,)

        obs_list, states, actions_list = [], [], []
        for step in range(args.steps_per_ep):
            action = torch.zeros(1, 9, device=env.device)
            # arm action in [-1,1] normalized space + 미세 노이즈 (normalizer range 확보)
            action[0, 0:5] = _TUCKED_ARM_NORM + torch.randn(5, device=env.device) * 0.02
            action[0, 5] = _TUCKED_GRIP_NORM + torch.randn(1, device=env.device).item() * 0.02
            action[0, 6] = base_cmd[0]
            action[0, 7] = base_cmd[1]
            action[0, 8] = base_cmd[2]

            # 9D robot_state (실제 joint_pos)
            jp = env.robot.data.joint_pos[0]
            arm = jp[env.arm_idx[:5]].cpu().numpy()
            grip = jp[env.arm_idx[5]].item()
            bv = env.robot.data.root_lin_vel_b[0].cpu().numpy()
            wz = env.robot.data.root_ang_vel_b[0, 2].item()
            state = np.concatenate([arm, [grip, bv[0], bv[1], wz]]).astype(np.float32)
            states.append(state)

            # 26D obs: arm(5)+grip(1)+base_vel(3)+dir_cmd(3)+lidar(8)+init_arm(5)+init_grip(1)
            obs_26d = np.concatenate([
                arm.astype(np.float32),
                np.array([grip], dtype=np.float32),
                np.array([bv[0], bv[1], wz], dtype=np.float32),
                np.array(dir_cmd, dtype=np.float32),
                np.ones(8, dtype=np.float32),
                init_arm_pose,  # 에피소드 시작 시점 값 (매 에피소드마다 다름)
            ])
            obs_list.append(obs_26d)
            actions_list.append(action[0].cpu().numpy().astype(np.float32))

            obs, _, _, _, _ = env.step(action)

        grp = hf.create_group(f"episode_{ep_idx}")
        grp.create_dataset("obs", data=np.array(obs_list))
        grp.create_dataset("robot_state", data=np.array(states))
        grp.create_dataset("actions", data=np.array(actions_list))
        grp.attrs["instruction"] = inst
        grp.attrs["num_steps"] = args.steps_per_ep
        ep_idx += 1
        elapsed = time.time() - t0
        print(f"  [{ep_idx}/{total}] {inst} rep={rep} | {elapsed:.0f}s")

hf.close()
print(f"\nDone: {ep_idx} episodes -> {args.output}")
sim_app.close()
