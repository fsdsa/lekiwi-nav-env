"""
Generate expert demonstration data using trained ACT-RL policy.

Rolls out the fine-tuned ACT-RL policy in Isaac Lab and saves
(obs, action) trajectories as HDF5 files for VLA fine-tuning.

Pipeline position:
    Teleop demos → ACT BC → ACT-RL (PPO) → THIS SCRIPT → VLA fine-tune data

Usage:
    python generate_expert_data.py \
        --act_rl_checkpoint checkpoints/act_rl/act_rl_approach_and_grasp_best.pt \
        --act_checkpoint checkpoints/act/act_approach_and_grasp_best.pt \
        --skill approach_and_grasp \
        --num_episodes 1000 \
        --num_envs 64 \
        --output_dir expert_demos/ \
        --headless
"""

import argparse
import os
import time
from datetime import datetime

import h5py
import numpy as np
import torch

from act_model import ACTPolicy
from train_act_rl import ACTRLPolicy, create_isaac_env


def load_act_rl_policy(
    act_checkpoint: str,
    act_rl_checkpoint: str,
    device: torch.device,
) -> ACTRLPolicy:
    """Load trained ACT-RL policy (frozen ACT + fine-tuned exploration head + value head)."""
    # Load ACT config and base model
    act_ckpt = torch.load(act_checkpoint, map_location=device, weights_only=False)
    cfg = act_ckpt["config"]

    act_policy = ACTPolicy(
        obs_dim=cfg["obs_dim"],
        action_dim=cfg["action_dim"],
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        chunk_size=cfg["chunk_size"],
        n_heads=cfg["n_heads"],
        n_enc_layers=cfg["n_enc_layers"],
        n_dec_layers=cfg["n_dec_layers"],
        ff_dim=cfg["ff_dim"],
    )
    act_policy.load_state_dict(act_ckpt["model_state_dict"])
    act_policy.to(device)

    # Create ACT-RL wrapper and load fine-tuned weights
    rl_policy = ACTRLPolicy(
        act_policy=act_policy,
        obs_dim=cfg["obs_dim"],
        action_dim=cfg["action_dim"],
        chunk_size=cfg["chunk_size"],
    ).to(device)

    rl_ckpt = torch.load(act_rl_checkpoint, map_location=device, weights_only=False)
    rl_policy.load_state_dict(rl_ckpt["policy_state_dict"])
    rl_policy.eval()

    print(f"[Expert Gen] Loaded ACT-RL policy")
    print(f"  ACT checkpoint: {act_checkpoint}")
    print(f"  ACT-RL checkpoint: {act_rl_checkpoint}")
    print(f"  obs_dim={cfg['obs_dim']}, action_dim={cfg['action_dim']}, chunk_size={cfg['chunk_size']}")

    return rl_policy, cfg


def rollout_expert(
    policy: ACTRLPolicy,
    env,
    num_episodes: int,
    max_steps: int,
    device: torch.device,
    use_mean: bool = True,
    temporal_agg: bool = True,
) -> list[dict]:
    """
    Roll out expert policy and collect trajectory data.

    Args:
        policy:        Trained ACT-RL policy
        env:           Isaac Lab vectorized environment
        num_episodes:  Total episodes to collect
        max_steps:     Max steps per episode
        device:        torch device
        use_mean:      If True, use ACT mean (deterministic). If False, sample from Gaussian.
        temporal_agg:  If True, use temporal ensembling over chunks

    Returns:
        List of episode dicts, each containing:
            obs:          (T, obs_dim)
            actions:      (T, action_dim)
            rewards:      (T,)
            robot_pos_w:  (T, 3)    — if available
            robot_quat_w: (T, 4)    — if available
            object_pos_w: (T, 3)    — if available
            object_quat_w:(T, 4)    — if available
            success:      bool
    """
    policy.eval()
    chunk_size = policy.chunk_size

    episodes = []
    num_envs = env.num_envs if hasattr(env, 'num_envs') else 1

    # Setup temporal ensembling on the underlying ACT
    if temporal_agg:
        policy.act.enable_temporal_agg(True)

    episode_data = {i: {
        "obs": [], "actions": [], "rewards": [],
        "robot_pos_w": [], "robot_quat_w": [],
        "object_pos_w": [], "object_quat_w": [],
    } for i in range(num_envs)}
    episode_steps = {i: 0 for i in range(num_envs)}

    # Track which envs are actively collecting
    active_envs = set(range(num_envs))

    # Reset
    obs_dict = env.reset()
    if isinstance(obs_dict, dict):
        obs = obs_dict.get("policy", obs_dict.get("obs"))
    elif isinstance(obs_dict, tuple):
        obs = obs_dict[0]
    else:
        obs = obs_dict
    obs = obs.to(device)

    print(f"[Expert Gen] Starting rollout: {num_episodes} episodes, max {max_steps} steps")

    collected = 0
    total_steps = 0
    t0 = time.time()

    # For chunk-level execution
    chunk_buffer = {}  # env_idx → (remaining actions, current position)

    while collected < num_episodes:
        # Get actions — either from chunk buffer or generate new chunk
        actions = torch.zeros(num_envs, policy.action_dim, device=device)

        for i in range(num_envs):
            if i not in active_envs:
                continue

            # Check if we need a new chunk
            if i not in chunk_buffer or chunk_buffer[i][1] >= chunk_size:
                with torch.no_grad():
                    if use_mean:
                        # Deterministic: use ACT mean directly
                        act_out = policy.act(obs[i:i+1])
                        chunk = act_out["pred_actions"][0]  # (K, 9)
                    else:
                        # Stochastic: sample from Gaussian
                        sample = policy.sample_chunk(obs[i:i+1])
                        chunk = sample["actions"][0]  # (K, 9)

                chunk_buffer[i] = (chunk, 0)

            # Get current action from chunk
            chunk, pos = chunk_buffer[i]
            actions[i] = chunk[pos]
            chunk_buffer[i] = (chunk, pos + 1)

        # Step environment
        obs_dict = env.step(actions)

        if isinstance(obs_dict, dict):
            next_obs = obs_dict.get("policy", obs_dict.get("obs", obs))
            rewards = obs_dict.get("reward", torch.zeros(num_envs, device=device))
            terminated = obs_dict.get("terminated", torch.zeros(num_envs, dtype=torch.bool, device=device))
            truncated = obs_dict.get("truncated", torch.zeros_like(terminated))
            infos = obs_dict.get("info", {})
        elif isinstance(obs_dict, tuple):
            next_obs, rewards, terminated, truncated, infos = obs_dict
        else:
            next_obs = obs_dict
            rewards = torch.zeros(num_envs, device=device)
            terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
            truncated = torch.zeros_like(terminated)
            infos = {}

        dones = terminated | truncated

        # Record data for each active env
        for i in list(active_envs):
            episode_data[i]["obs"].append(obs[i].cpu().numpy())
            episode_data[i]["actions"].append(actions[i].cpu().numpy())
            episode_data[i]["rewards"].append(rewards[i].item())

            # Extract world state if available
            if isinstance(infos, dict):
                if "robot_pos_w" in infos:
                    episode_data[i]["robot_pos_w"].append(
                        infos["robot_pos_w"][i].cpu().numpy()
                    )
                if "robot_quat_w" in infos:
                    episode_data[i]["robot_quat_w"].append(
                        infos["robot_quat_w"][i].cpu().numpy()
                    )
                if "object_pos_w" in infos:
                    episode_data[i]["object_pos_w"].append(
                        infos["object_pos_w"][i].cpu().numpy()
                    )
                if "object_quat_w" in infos:
                    episode_data[i]["object_quat_w"].append(
                        infos["object_quat_w"][i].cpu().numpy()
                    )

            episode_steps[i] += 1

            # Check if episode is done
            if dones[i] or episode_steps[i] >= max_steps:
                success = False
                if isinstance(infos, dict) and "success" in infos:
                    success = infos["success"][i].item()

                ep = {
                    "obs": np.array(episode_data[i]["obs"]),
                    "actions": np.array(episode_data[i]["actions"]),
                    "rewards": np.array(episode_data[i]["rewards"]),
                    "success": bool(success),
                    "num_steps": episode_steps[i],
                }

                # Add world state if collected
                for key in ["robot_pos_w", "robot_quat_w", "object_pos_w", "object_quat_w"]:
                    if episode_data[i][key]:
                        ep[key] = np.array(episode_data[i][key])

                episodes.append(ep)
                collected += 1

                if collected % 50 == 0 or collected == num_episodes:
                    elapsed = time.time() - t0
                    success_rate = np.mean([e["success"] for e in episodes])
                    print(
                        f"  Collected {collected}/{num_episodes} episodes | "
                        f"Success: {success_rate:.1%} | "
                        f"{elapsed:.0f}s"
                    )

                if collected >= num_episodes:
                    break

                # Reset this env's data
                episode_data[i] = {
                    "obs": [], "actions": [], "rewards": [],
                    "robot_pos_w": [], "robot_quat_w": [],
                    "object_pos_w": [], "object_quat_w": [],
                }
                episode_steps[i] = 0
                chunk_buffer.pop(i, None)  # Force new chunk on next step

        obs = next_obs.to(device)
        total_steps += num_envs

    return episodes


def save_episodes_hdf5(
    episodes: list[dict],
    output_path: str,
    only_success: bool = True,
):
    """
    Save collected episodes to HDF5 in the same format as record_teleop.py.

    This ensures compatibility with existing VLA training pipelines.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if only_success:
        success_episodes = [ep for ep in episodes if ep["success"]]
        print(f"[Save] Filtering: {len(success_episodes)}/{len(episodes)} successful episodes")
        episodes = success_episodes

    if not episodes:
        print("[Save] WARNING: No episodes to save!")
        return

    with h5py.File(output_path, "w") as f:
        for idx, ep in enumerate(episodes):
            grp = f.create_group(f"episode_{idx}")

            # Core data (same format as record_teleop.py)
            grp.create_dataset("obs", data=ep["obs"].astype(np.float32))
            grp.create_dataset("actions", data=ep["actions"].astype(np.float64))
            grp.create_dataset("rewards", data=ep["rewards"].astype(np.float32))

            # Teleop active flag (all 1s for expert policy)
            T = ep["obs"].shape[0]
            grp.create_dataset("teleop_active", data=np.ones(T, dtype=np.int8))

            # Robot state (same as obs[:, :9] for compatibility — arm6 + base3)
            # Note: robot_state in original format is [arm6, base_vel3]
            # We approximate from actions which are [arm5, grip1, base3]
            robot_state = np.zeros((T, 9), dtype=np.float32)
            robot_state[:, :6] = ep["obs"][:, :6]  # arm5 + grip1 from obs
            robot_state[:, 6:9] = ep["actions"][:, 6:9]  # base vel from actions
            grp.create_dataset("robot_state", data=robot_state)

            # World state (if available)
            for key in ["robot_pos_w", "robot_quat_w", "object_pos_w", "object_quat_w"]:
                if key in ep and ep[key] is not None:
                    grp.create_dataset(key, data=ep[key].astype(np.float32))

            # Attributes
            grp.attrs["num_steps"] = T
            grp.attrs["num_active_steps"] = T
            grp.attrs["success"] = ep["success"]
            grp.attrs["source"] = "act_rl_expert"

            if "robot_pos_w" in ep and len(ep["robot_pos_w"]) > 0:
                grp.attrs["robot_init_pos"] = ep["robot_pos_w"][0]
            if "robot_quat_w" in ep and len(ep["robot_quat_w"]) > 0:
                grp.attrs["robot_init_quat"] = ep["robot_quat_w"][0]
            if "object_pos_w" in ep and len(ep["object_pos_w"]) > 0:
                grp.attrs["object_init_pos"] = ep["object_pos_w"][0]
            if "object_quat_w" in ep and len(ep["object_quat_w"]) > 0:
                grp.attrs["object_init_quat"] = ep["object_quat_w"][0]

    total_steps = sum(ep["obs"].shape[0] for ep in episodes)
    success_rate = np.mean([ep["success"] for ep in episodes])
    print(f"[Save] Saved {len(episodes)} episodes ({total_steps:,} steps) to {output_path}")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Avg episode length: {total_steps / len(episodes):.0f} steps")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate expert data with ACT-RL policy")

    # Checkpoints
    parser.add_argument("--act_checkpoint", type=str, required=True,
                        help="Path to ACT BC checkpoint (for architecture config)")
    parser.add_argument("--act_rl_checkpoint", type=str, required=True,
                        help="Path to ACT-RL fine-tuned checkpoint")

    # Environment
    parser.add_argument("--skill", type=str, default="approach_and_grasp")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--multi_object_json", type=str, default=None)
    parser.add_argument("--handoff_buffer", type=str, default=None)

    # Rollout settings
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Number of expert episodes to generate")
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="Max steps per episode")
    parser.add_argument("--use_mean", action="store_true", default=True,
                        help="Use deterministic ACT mean (recommended for expert data)")
    parser.add_argument("--stochastic", dest="use_mean", action="store_false",
                        help="Use stochastic sampling (adds diversity)")
    parser.add_argument("--temporal_agg", action="store_true", default=True,
                        help="Use temporal ensembling")
    parser.add_argument("--only_success", action="store_true", default=True,
                        help="Only save successful episodes")

    # Output
    parser.add_argument("--output_dir", type=str, default="expert_demos/")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device)

    # Load policy
    rl_policy, cfg = load_act_rl_policy(
        args.act_checkpoint, args.act_rl_checkpoint, device
    )

    # Create environment
    env = create_isaac_env(args)

    # Rollout
    episodes = rollout_expert(
        policy=rl_policy,
        env=env,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        device=device,
        use_mean=args.use_mean,
        temporal_agg=args.temporal_agg,
    )

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir,
        f"expert_{args.skill}_{timestamp}.hdf5",
    )
    save_episodes_hdf5(episodes, output_path, only_success=args.only_success)

    # Summary statistics
    total = len(episodes)
    successes = sum(1 for ep in episodes if ep["success"])
    avg_reward = np.mean([ep["rewards"].sum() for ep in episodes])
    avg_len = np.mean([ep["num_steps"] for ep in episodes])

    print(f"\n{'='*60}")
    print(f"Expert Data Generation Summary")
    print(f"{'='*60}")
    print(f"  Total episodes:  {total}")
    print(f"  Successful:      {successes} ({successes/total:.1%})")
    print(f"  Avg reward:      {avg_reward:.2f}")
    print(f"  Avg length:      {avg_len:.0f} steps")
    print(f"  Output:          {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
