"""
Expert data collection using trained ResiP policy (Frozen DP + Residual).

This script:
  1. Loads the frozen DP BC checkpoint + trained residual policy
  2. Rolls out the combined policy in the environment
  3. Saves successful trajectories as HDF5 for downstream VLA fine-tuning

The combined policy operates as:
    base_action = frozen_dp(obs)                    # macro planning (every action_horizon steps)
    residual = residual_policy(obs, base_action)    # micro correction (every step)
    final_action = base_action + 0.1 * residual     # combined

Usage:
    python collect_resip_demos.py \\
        --bc_checkpoint checkpoints/dp_bc.pt \\
        --resip_checkpoint checkpoints/resip_best.pt \\
        --env_cfg skill2 \\
        --num_envs 64 \\
        --num_episodes 1000 \\
        --output_path demos_expert/expert_skill2.hdf5
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from diffusion_policy import (
    DiffusionPolicyAgent,
    ResidualPolicy,
)


def load_resip_policy(bc_ckpt_path, resip_ckpt_path, device):
    """
    Load the full ResiP policy: frozen DP + trained residual.
    """
    # 1. Load frozen DP
    bc_ckpt = torch.load(bc_ckpt_path, map_location=device, weights_only=False)
    dp_cfg = bc_ckpt["config"]

    dp_agent = DiffusionPolicyAgent(
        obs_dim=dp_cfg["obs_dim"],
        act_dim=dp_cfg["act_dim"],
        pred_horizon=dp_cfg["pred_horizon"],
        action_horizon=dp_cfg["action_horizon"],
        num_diffusion_iters=dp_cfg["num_diffusion_iters"],
        inference_steps=dp_cfg.get("inference_steps", 16),
        down_dims=dp_cfg.get("down_dims", [256, 512, 1024]),
    ).to(device)

    state_dict = bc_ckpt["model_state_dict"]
    model_state = {
        k[len("model."):]: v for k, v in state_dict.items()
        if k.startswith("model.")
    }
    norm_state = {
        k[len("normalizer."):]: v for k, v in state_dict.items()
        if k.startswith("normalizer.")
    }
    dp_agent.model.load_state_dict(model_state)
    dp_agent.normalizer.load_state_dict(norm_state)
    dp_agent.eval()
    for p in dp_agent.parameters():
        p.requires_grad = False

    # Faster inference for data collection
    dp_agent.inference_steps = 4

    # 2. Load residual policy
    resip_ckpt = torch.load(resip_ckpt_path, map_location=device, weights_only=False)
    resip_args = resip_ckpt.get("args", {})

    residual_policy = ResidualPolicy(
        obs_dim=dp_cfg["obs_dim"],
        action_dim=dp_cfg["act_dim"],
        actor_hidden_size=resip_args.get("actor_hidden_size", 256),
        actor_num_layers=resip_args.get("actor_num_layers", 2),
        critic_hidden_size=resip_args.get("critic_hidden_size", 256),
        critic_num_layers=resip_args.get("critic_num_layers", 2),
        action_scale=resip_args.get("action_scale", 0.1),
        action_head_std=resip_args.get("action_head_std", 0.0),
        init_logstd=resip_args.get("init_logstd", -1.0),
    ).to(device)

    residual_policy.load_state_dict(resip_ckpt["residual_policy_state_dict"])
    residual_policy.eval()

    print(f"Loaded ResiP policy:")
    print(f"  DP: {bc_ckpt_path}")
    print(f"  Residual: {resip_ckpt_path}")
    print(f"  Best SR: {resip_ckpt.get('success_rate', 'N/A')}")

    return dp_agent, residual_policy, dp_cfg


@torch.no_grad()
def collect_episode(dp_agent, residual_policy, env, max_steps, device):
    """
    Collect one episode using the ResiP policy.

    Returns dict with obs, actions, rewards, success flag.
    Works with vectorized env but collects per-env trajectories.
    """
    num_envs = env.num_envs
    obs_dim = dp_agent.obs_dim
    act_dim = dp_agent.act_dim

    # Storage per env
    all_obs = [[] for _ in range(num_envs)]
    all_actions = [[] for _ in range(num_envs)]
    all_rewards = [[] for _ in range(num_envs)]
    env_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    obs = env.reset()
    dp_agent.reset()

    for step in range(max_steps):
        # Get base action from frozen DP
        base_naction = dp_agent.base_action_normalized(obs)

        # Normalize obs
        nobs = dp_agent.normalizer(obs, "obs", forward=True)
        nobs = torch.clamp(nobs, -3, 3)

        # Get residual (deterministic = actor mean)
        residual_nobs = torch.cat([nobs, base_naction], dim=-1)
        residual_action = residual_policy.get_action(residual_nobs)

        # Combine
        naction = base_naction + residual_action
        action = dp_agent.normalizer(naction, "action", forward=False)

        # Record (only for envs not yet done)
        for i in range(num_envs):
            if not env_done[i]:
                all_obs[i].append(obs[i].cpu().numpy())
                all_actions[i].append(action[i].cpu().numpy())

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        for i in range(num_envs):
            if not env_done[i]:
                all_rewards[i].append(reward[i].cpu().item())
                if terminated[i]:
                    env_success[i] = True

        env_done = env_done | done.view(-1).bool()

        if env_done.all():
            break

    # Build episodes
    episodes = []
    for i in range(num_envs):
        if len(all_obs[i]) > 0:
            episodes.append({
                "obs": np.array(all_obs[i], dtype=np.float32),
                "actions": np.array(all_actions[i], dtype=np.float32),
                "rewards": np.array(all_rewards[i], dtype=np.float32),
                "success": env_success[i].item(),
                "num_steps": len(all_obs[i]),
            })

    return episodes


@torch.no_grad()
def collect_carry_episode(
    s2_dp, s2_rpol, s2_scale,
    carry_dp, carry_rpol, carry_scale,
    env, device,
    s2_max_steps=800, carry_steps=600,
    s2_lift_hold=400,
):
    """
    Collect carry expert episodes: S2 expert → lift → carry expert (arm=interp, base=RL).

    Key: arm action은 보간값으로 기록 (RL output 무시).
    """
    N = env.num_envs

    # S3 arm interpolation constants
    S3_ARM_END = torch.tensor([+0.002, -0.193, +0.295, -1.306, +0.006], dtype=torch.float32, device=device)
    S3_GRIP_END = 0.15
    INTERP_STEPS = float(carry_steps)

    # 6 directions
    ALL_DIRS = torch.tensor([[0,1,0],[0,-1,0],[-1,0,0],[1,0,0],[0,0,1],[0,0,-1]], dtype=torch.float32, device=device)

    # Per-env state
    phase = torch.zeros(N, dtype=torch.long, device=device)  # 0=S2, 1=carry
    lift_counter = torch.zeros(N, dtype=torch.long, device=device)
    carry_step = torch.zeros(N, device=device)
    direction_cmd = ALL_DIRS[torch.randint(0, 6, (N,), device=device)]
    carry_arm_start = torch.zeros(N, 5, device=device)
    carry_grip_start = torch.zeros(N, device=device)

    # Storage (carry phase only)
    all_obs = [[] for _ in range(N)]
    all_actions = [[] for _ in range(N)]
    env_done = torch.zeros(N, dtype=torch.bool, device=device)

    # Arm limit cache for normalization
    override_lim = getattr(env.env, "_arm_action_limits_override", None)
    if override_lim is not None:
        lim = override_lim
    else:
        lim = env.env.robot.data.soft_joint_pos_limits[:, env.env.arm_idx]
    lo, hi = lim[..., 0], lim[..., 1]
    arm_ctr = 0.5 * (lo + hi)
    arm_hlf = 0.5 * (hi - lo)
    fin = torch.isfinite(arm_ctr) & torch.isfinite(arm_hlf) & (arm_hlf.abs() > 1e-6)
    arm_hlf = torch.where(fin, arm_hlf, torch.ones_like(arm_hlf))
    arm_ctr = torch.where(fin, arm_ctr, torch.zeros_like(arm_ctr))

    obs_dict, _ = env.reset()
    s2_dp.reset()
    carry_dp.reset()

    env_origins = env.env.scene.env_origins

    for step in range(s2_max_steps + carry_steps + 100):
        obs_t = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        is_s2 = (phase == 0)
        is_carry = (phase == 1)

        # ── S2 expert action ──
        action = torch.zeros(N, 9, device=device)
        if is_s2.any():
            s2_ba = s2_dp.base_action_normalized(obs_t)
            if s2_rpol is not None:
                s2_no = s2_dp.normalizer(obs_t, "obs", forward=True)
                s2_no = torch.nan_to_num(torch.clamp(s2_no, -3, 3), nan=0.0)
                s2_ro = torch.cat([s2_no, s2_ba], dim=-1)
                s2_ram = s2_rpol.actor_mean(s2_ro)
                s2_ram = torch.clamp(s2_ram, -1, 1)
                s2_na = s2_ba + s2_ram * s2_scale
            else:
                s2_na = s2_ba
            s2_act = s2_dp.normalizer(s2_na, "action", forward=False)
            action[is_s2] = s2_act[is_s2]

        # ── Carry expert action ──
        if is_carry.any():
            carry_obs = torch.cat([obs_t, direction_cmd], dim=-1)  # 33D
            c_ba = carry_dp.base_action_normalized(carry_obs)
            if carry_rpol is not None:
                c_no = carry_dp.normalizer(carry_obs, "obs", forward=True)
                c_no = torch.nan_to_num(torch.clamp(c_no, -3, 3), nan=0.0)
                c_ro = torch.cat([c_no, c_ba], dim=-1)
                c_ram = carry_rpol.actor_mean(c_ro)
                c_ram = torch.clamp(c_ram, -1, 1)
                c_na = c_ba + c_ram * carry_scale
            else:
                c_na = c_ba
            c_act = carry_dp.normalizer(c_na, "action", forward=False)
            action[is_carry] = c_act[is_carry]

            # Arm interpolation override (보간값으로 기록)
            carry_step[is_carry] += 1
            t_interp = (carry_step / INTERP_STEPS).clamp(0, 1).unsqueeze(-1)
            arm_t = carry_arm_start * (1 - t_interp) + S3_ARM_END.unsqueeze(0) * t_interp
            grip_t = carry_grip_start * (1 - t_interp.squeeze(-1)) + S3_GRIP_END * t_interp.squeeze(-1)
            arm6 = torch.cat([arm_t, grip_t.unsqueeze(-1)], dim=-1)
            arm_norm = ((arm6 - arm_ctr) / arm_hlf).clamp(-1, 1)
            action[is_carry, :6] = arm_norm[is_carry]

        action = action.clamp(-1, 1)

        # Record carry phase
        for i in range(N):
            if is_carry[i] and not env_done[i]:
                all_obs[i].append(obs_t[i].cpu().numpy())
                all_actions[i].append(action[i].cpu().numpy())

        # Step
        obs_dict, reward, terminated, truncated, info = env.step(action)

        # S2 lift detection
        objZ = env.env.object_pos_w[:, 2] - env_origins[:, 2]
        grasped = env.env.object_grasped if hasattr(env.env, 'object_grasped') else torch.zeros(N, dtype=torch.bool, device=device)
        lifted = is_s2 & grasped & (objZ > 0.05)
        lift_counter[lifted] += 1
        lift_counter[is_s2 & ~lifted] = 0

        # S2→carry transition
        transition = is_s2 & (lift_counter >= s2_lift_hold)
        if transition.any():
            t_ids = transition.nonzero(as_tuple=False).squeeze(-1)
            carry_arm_start[t_ids] = env.env.robot.data.joint_pos[t_ids][:, env.env.arm_idx[:5]]
            carry_grip_start[t_ids] = env.env.robot.data.joint_pos[t_ids, env.env.gripper_idx]
            carry_step[t_ids] = 0
            phase[t_ids] = 1
            lift_counter[t_ids] = 0

        # S2 topple → reset
        s2_topple = is_s2 & (objZ < 0.026) & (step > 20)
        if s2_topple.any():
            pass  # These envs will auto-reset via env.step terminated

        # Carry done (600 steps) or drop
        carry_done = is_carry & (carry_step >= carry_steps)
        carry_drop = is_carry & (objZ < 0.03)
        env_done = env_done | carry_done | carry_drop

        if env_done.all():
            break

    # Build episodes
    episodes = []
    for i in range(N):
        if len(all_obs[i]) > 10:
            episodes.append({
                "obs": np.array(all_obs[i], dtype=np.float32),
                "actions": np.array(all_actions[i], dtype=np.float32),
                "success": len(all_obs[i]) >= carry_steps - 10,
                "num_steps": len(all_obs[i]),
                "direction_cmd": direction_cmd[i].cpu().numpy(),
            })
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Collect expert demos with ResiP")

    parser.add_argument("--bc_checkpoint", type=str, required=True,
                        help="Frozen DP BC checkpoint (S2 for skill2, carry BC for carry)")
    parser.add_argument("--resip_checkpoint", type=str, required=True,
                        help="Residual policy checkpoint")
    parser.add_argument("--skill", type=str, default="approach_and_grasp",
                        choices=["approach_and_grasp", "carry"],
                        help="Skill to collect demos for")
    # Carry mode: S2 expert checkpoints
    parser.add_argument("--s2_bc_checkpoint", type=str, default="",
                        help="carry: S2 frozen DP checkpoint")
    parser.add_argument("--s2_resip_checkpoint", type=str, default="",
                        help="carry: S2 residual policy checkpoint")
    parser.add_argument("--object_usd", type=str, default="")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Total episodes to collect")
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--carry_steps", type=int, default=600)
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output HDF5 path")
    parser.add_argument("--only_success", action="store_true", default=True,
                        help="Only save successful episodes")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.skill == "carry":
        # ── Carry mode: S2 expert + carry expert ──
        if not args.s2_bc_checkpoint:
            print("ERROR: --skill carry requires --s2_bc_checkpoint")
            return
        # Load S2 expert
        s2_dp, s2_rpol, s2_cfg = load_resip_policy(
            args.s2_bc_checkpoint, args.s2_resip_checkpoint, device
        ) if args.s2_resip_checkpoint else (None, None, None)
        if s2_dp is None:
            s2_dp, s2_cfg = load_resip_policy(args.s2_bc_checkpoint, "", device)[:2]
        s2_scale = torch.zeros(s2_cfg["act_dim"], device=device)
        s2_scale[0:5] = 0.20; s2_scale[5] = 0.25; s2_scale[6:9] = 0.35

        # Load carry expert
        carry_dp, carry_rpol, carry_cfg = load_resip_policy(
            args.bc_checkpoint, args.resip_checkpoint, device
        )
        carry_scale = torch.zeros(carry_cfg["act_dim"], device=device)
        carry_scale[0:5] = 0.0; carry_scale[5] = 0.0
        carry_scale[6:9] = 0.35  # base only

        # Create env
        from train_resip import make_env
        # Fake args for make_env
        class _Args:
            s3_dest_spawn_dist_min = 0.6
            s3_dest_spawn_dist_max = 0.9
            s3_dest_heading_max_rad = 0.5
            enable_domain_randomization = False
            handoff_buffer = ""
        env = make_env("carry", args.num_envs, _Args())

        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_collected = 0
        total_success = 0
        total_attempted = 0
        start_time = time.time()

        with h5py.File(str(output_path), "w") as f:
            f.attrs["skill"] = "carry"
            f.attrs["obs_dim"] = carry_cfg["obs_dim"]
            f.attrs["act_dim"] = carry_cfg["act_dim"]
            while total_collected < args.num_episodes:
                episodes = collect_carry_episode(
                    s2_dp, s2_rpol, s2_scale,
                    carry_dp, carry_rpol, carry_scale,
                    env, device,
                    carry_steps=args.carry_steps,
                )
                total_attempted += len(episodes)
                for ep in episodes:
                    if args.only_success and not ep["success"]:
                        continue
                    if total_collected >= args.num_episodes:
                        break
                    grp = f.create_group(f"episode_{total_collected}")
                    grp.create_dataset("obs", data=ep["obs"])
                    grp.create_dataset("actions", data=ep["actions"])
                    grp.attrs["success"] = ep["success"]
                    grp.attrs["num_steps"] = ep["num_steps"]
                    grp.attrs["direction_cmd"] = ep["direction_cmd"]
                    total_collected += 1
                    if ep["success"]:
                        total_success += 1

                sr = total_success / total_attempted if total_attempted > 0 else 0
                elapsed = time.time() - start_time
                print(f"Collected: {total_collected}/{args.num_episodes} | "
                      f"Attempted: {total_attempted} | SR: {sr:.2%} | Time: {elapsed:.0f}s")
    else:
        # ── Original approach_and_grasp mode ──
        dp_agent, residual_policy, dp_cfg = load_resip_policy(
            args.bc_checkpoint, args.resip_checkpoint, device
        )
        from train_resip import make_env
        class _Args:
            enable_domain_randomization = False
            handoff_buffer = ""
        env = make_env(args.skill, args.num_envs, _Args())

        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_collected = 0
        total_success = 0
        total_attempted = 0
        start_time = time.time()

        with h5py.File(str(output_path), "w") as f:
            while total_collected < args.num_episodes:
                episodes = collect_episode(
                    dp_agent, residual_policy, env, args.max_steps, device
                )
                total_attempted += len(episodes)
                for ep in episodes:
                    if args.only_success and not ep["success"]:
                        continue
                    if total_collected >= args.num_episodes:
                        break
                    ep_group = f.create_group(f"episode_{total_collected}")
                    ep_group.create_dataset("obs", data=ep["obs"])
                    ep_group.create_dataset("actions", data=ep["actions"])
                    ep_group.create_dataset("rewards", data=ep["rewards"])
                    ep_group.attrs["success"] = ep["success"]
                    ep_group.attrs["num_steps"] = ep["num_steps"]
                    total_collected += 1
                    if ep["success"]:
                        total_success += 1

                sr = total_success / total_attempted if total_attempted > 0 else 0
                elapsed = time.time() - start_time
                print(f"Collected: {total_collected}/{args.num_episodes} | "
                      f"Attempted: {total_attempted} | SR: {sr:.2%} | Time: {elapsed:.0f}s")

    total_time = time.time() - start_time
    print(f"\nDone! Collected {total_collected} episodes in {total_time:.0f}s")
    print(f"Success rate: {total_success}/{total_attempted} = "
          f"{total_success/total_attempted:.2%}" if total_attempted > 0 else "N/A")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
