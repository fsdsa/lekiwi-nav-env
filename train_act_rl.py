"""
ACT-RL: Fine-tune frozen ACT policy with chunk-level PPO.

Following "ACT-RL: Efficient Fine-Tuning through RL for Robotic Manipulation"
(Sikand, Stone, Yeung — Stanford CS224R 2025):
  1. Freeze the entire ACT encoder-decoder backbone
  2. ACT deterministic output → mean of diagonal Gaussian policy
  3. Learnable log_std exploration head (per action dim)
  4. Small MLP value network
  5. Chunk-level PPO: one optimization step per K-action macro-chunk
  6. KL anchor to frozen ACT prior to prevent drift

Integration with Isaac Lab environment for LeKiwi robot.

Usage:
    python train_act_rl.py \
        --act_checkpoint checkpoints/act/act_approach_and_grasp_best.pt \
        --skill approach_and_grasp \
        --num_envs 2048 \
        --total_timesteps 2000000 \
        --headless
"""

import argparse
import math
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from act_model import ACTPolicy


# ===========================================================================
# ACT-RL Policy: Frozen ACT + Exploration Head + Value Head
# ===========================================================================
class ACTRLPolicy(nn.Module):
    """
    Wraps a frozen ACT policy for RL fine-tuning.

    Architecture:
        ACT backbone (FROZEN) → action_chunk_mean (K, 9)
        Learnable log_std     → action_chunk_std  (K, 9)
        Value MLP             → V(s) scalar

    Supports two modes:
        - chunk-level: predict entire chunk, execute all K actions, one PPO update
        - action-level: predict chunk, but PPO update per individual action (fallback)
    """

    def __init__(
        self,
        act_policy: ACTPolicy,
        obs_dim: int,
        action_dim: int = 9,
        chunk_size: int = 20,
        init_log_std: float = -1.0,
        value_hidden: int = 256,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        # ---- Frozen ACT backbone ----
        self.act = act_policy
        self.act.eval()
        for p in self.act.parameters():
            p.requires_grad = False

        # ---- Learnable exploration head ----
        # Per-action-in-chunk log_std: (chunk_size, action_dim)
        self.log_std = nn.Parameter(
            torch.full((chunk_size, action_dim), init_log_std)
        )

        # ---- Value network ----
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, value_hidden),
            nn.GELU(),
            nn.Linear(value_hidden, value_hidden),
            nn.GELU(),
            nn.Linear(value_hidden, 1),
        )

    def get_chunk_distribution(self, obs: torch.Tensor) -> tuple:
        """
        Get action chunk distribution.

        Args:
            obs: (B, obs_dim)
        Returns:
            mean:   (B, K, action_dim)  — from frozen ACT
            std:    (B, K, action_dim)  — from learnable log_std
            value:  (B,)
        """
        # Frozen ACT forward
        with torch.no_grad():
            act_out = self.act(obs)
            mean = act_out["pred_actions"]  # (B, K, action_dim)

        # Exploration std
        std = self.log_std.exp().unsqueeze(0).expand_as(mean)  # (B, K, action_dim)

        # Value
        value = self.value_net(obs).squeeze(-1)  # (B,)

        return mean, std, value

    def sample_chunk(self, obs: torch.Tensor) -> dict:
        """
        Sample an action chunk for environment interaction.

        Returns:
            dict with: actions (B, K, 9), log_probs (B,), values (B,), mean (B, K, 9)
        """
        mean, std, value = self.get_chunk_distribution(obs)
        dist = Normal(mean, std)
        actions = dist.sample()

        # Chunk-level log prob: sum over (K, action_dim)
        log_prob = dist.log_prob(actions).sum(dim=(-2, -1))  # (B,)

        return {
            "actions": actions,
            "log_probs": log_prob,
            "values": value,
            "mean": mean,
        }

    def evaluate_chunk(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple:
        """
        Evaluate actions for PPO update.

        Args:
            obs:     (B, obs_dim)
            actions: (B, K, action_dim)
        Returns:
            log_probs: (B,)
            values:    (B,)
            entropy:   (B,)
        """
        mean, std, value = self.get_chunk_distribution(obs)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(dim=(-2, -1))  # (B,)
        entropy = dist.entropy().sum(dim=(-2, -1))  # (B,)

        return log_prob, value, entropy

    def get_kl_from_prior(self, obs: torch.Tensor) -> torch.Tensor:
        """
        KL divergence from current policy to frozen ACT prior.
        KL(pi_current || pi_prior) where prior has std → 0 (deterministic).
        Approximated as: 0.5 * (log_std²) + 0.5 * log_std.exp()²
        (since prior mean = current mean, only std matters)
        """
        std = self.log_std.exp()  # (K, action_dim)
        # KL of N(mu, sigma) from N(mu, 0+) ≈ -log(sigma) + 0.5*sigma^2
        # Simplified: deviation of std from 0
        kl = -self.log_std + 0.5 * std.pow(2) - 0.5
        return kl.sum()  # scalar


# ===========================================================================
# Rollout Buffer for Chunk-Level PPO
# ===========================================================================
@dataclass
class ChunkRollout:
    """Single chunk-level rollout entry."""
    obs: torch.Tensor         # (obs_dim,)
    actions: torch.Tensor     # (chunk_size, action_dim)
    log_prob: float
    value: float
    chunk_return: float       # discounted return over the chunk
    advantage: float


class ChunkRolloutBuffer:
    """Buffer for chunk-level PPO rollouts."""

    def __init__(self, buffer_size: int = 2048):
        self.buffer_size = buffer_size
        self.obs_list = []
        self.actions_list = []
        self.log_probs_list = []
        self.values_list = []
        self.returns_list = []
        self.advantages_list = []

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        chunk_return: torch.Tensor,
    ):
        self.obs_list.append(obs)
        self.actions_list.append(actions)
        self.log_probs_list.append(log_prob)
        self.values_list.append(value)
        self.returns_list.append(chunk_return)

    def finalize(self):
        """Compute advantages with normalization."""
        values = torch.stack(self.values_list)
        returns = torch.stack(self.returns_list)
        advantages = returns - values.detach()

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        self.advantages_list = ((advantages - adv_mean) / adv_std).tolist()

    def get_batches(self, batch_size: int, device: torch.device):
        """Yield random mini-batches."""
        n = len(self.obs_list)
        indices = torch.randperm(n)

        obs_t = torch.stack(self.obs_list).to(device)
        actions_t = torch.stack(self.actions_list).to(device)
        log_probs_t = torch.stack(self.log_probs_list).to(device)
        values_t = torch.stack(self.values_list).to(device)
        returns_t = torch.stack(self.returns_list).to(device)
        advantages_t = torch.tensor(self.advantages_list, device=device)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield {
                "obs": obs_t[idx],
                "actions": actions_t[idx],
                "old_log_probs": log_probs_t[idx],
                "old_values": values_t[idx],
                "returns": returns_t[idx],
                "advantages": advantages_t[idx],
            }

    def clear(self):
        self.obs_list.clear()
        self.actions_list.clear()
        self.log_probs_list.clear()
        self.values_list.clear()
        self.returns_list.clear()
        self.advantages_list.clear()

    def __len__(self):
        return len(self.obs_list)


# ===========================================================================
# Chunk-Level PPO Trainer
# ===========================================================================
class ChunkPPOTrainer:
    """
    Chunk-level PPO following ACT-RL paper.

    Key differences from standard PPO:
      - One optimization step per K-action chunk (not per single action)
      - Discount accumulates over chunk: δ = Σγ^i r_{t+i} + γ^K V(s_{t+K}) - V(s_t)
      - KL anchor to frozen ACT prior
    """

    def __init__(
        self,
        policy: ACTRLPolicy,
        lr: float = 3e-5,
        clip_range: float = 0.1,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        kl_coef: float = 0.05,
        kl_decay: float = 0.999,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        value_clip: float = 0.2,
    ):
        self.policy = policy
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.kl_coef = kl_coef
        self.kl_decay = kl_decay
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.value_clip = value_clip

        # Only optimize exploration head + value network
        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr, eps=1e-5)

        self.buffer = ChunkRolloutBuffer()

    def collect_chunk_rollout(
        self,
        env,
        obs: torch.Tensor,
        device: torch.device,
    ) -> tuple:
        """
        Execute one chunk in the environment.

        Args:
            env:  Isaac Lab environment (vectorized)
            obs:  (num_envs, obs_dim) current observations
        Returns:
            next_obs:      (num_envs, obs_dim)
            chunk_rewards: (num_envs,) — discounted reward sum over chunk
            dones:         (num_envs,) — whether episode ended during chunk
            infos:         list of info dicts
        """
        self.policy.eval()
        K = self.policy.chunk_size

        with torch.no_grad():
            sample = self.policy.sample_chunk(obs)

        action_chunk = sample["actions"]  # (num_envs, K, 9)
        log_probs = sample["log_probs"]   # (num_envs,)
        values = sample["values"]         # (num_envs,)

        # Execute all K actions in sequence
        chunk_rewards = torch.zeros(obs.shape[0], device=device)
        dones = torch.zeros(obs.shape[0], dtype=torch.bool, device=device)
        next_obs = obs.clone()
        infos = {}

        for k in range(K):
            actions_k = action_chunk[:, k]  # (num_envs, 9)

            # Step environment
            obs_dict = env.step(actions_k)

            # Handle Isaac Lab obs dict format
            if isinstance(obs_dict, dict):
                next_obs = obs_dict.get("policy", obs_dict.get("obs", next_obs))
                step_rewards = obs_dict.get("reward", torch.zeros(obs.shape[0], device=device))
                step_dones = obs_dict.get("terminated", torch.zeros(obs.shape[0], dtype=torch.bool, device=device))
                step_dones = step_dones | obs_dict.get("truncated", torch.zeros_like(step_dones))
                infos = obs_dict.get("info", {})
            elif isinstance(obs_dict, tuple):
                # Gym-style: obs, reward, terminated, truncated, info
                next_obs, step_rewards, terminated, truncated, infos = obs_dict
                step_dones = terminated | truncated
            else:
                next_obs = obs_dict
                step_rewards = torch.zeros(obs.shape[0], device=device)
                step_dones = torch.zeros(obs.shape[0], dtype=torch.bool, device=device)

            # Accumulate discounted reward within chunk
            chunk_rewards += (self.gamma ** k) * step_rewards * (~dones).float()
            dones = dones | step_dones

        # Compute chunk return: chunk_rewards + γ^K * V(s_{t+K}) for non-done envs
        with torch.no_grad():
            _, _, next_values = self.policy.get_chunk_distribution(next_obs)

        chunk_returns = chunk_rewards + (self.gamma ** K) * next_values * (~dones).float()

        # Store in buffer (per-env)
        for i in range(obs.shape[0]):
            self.buffer.add(
                obs=obs[i].cpu(),
                actions=action_chunk[i].cpu(),
                log_prob=log_probs[i].cpu(),
                value=values[i].cpu(),
                chunk_return=chunk_returns[i].cpu(),
            )

        return next_obs, chunk_rewards, dones, infos

    def update(self, device: torch.device) -> dict:
        """
        Run PPO update on collected chunk rollouts.

        Returns:
            dict with loss metrics
        """
        self.buffer.finalize()
        self.policy.train()

        total_metrics = {
            "policy_loss": 0, "value_loss": 0, "entropy": 0,
            "kl_loss": 0, "total_loss": 0, "clip_frac": 0,
        }
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size, device):
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                old_values = batch["old_values"]
                returns = batch["returns"]
                advantages = batch["advantages"]

                # Evaluate current policy
                new_log_probs, new_values, entropy = self.policy.evaluate_chunk(
                    obs, actions
                )

                # ---- Policy loss (clipped PPO) ----
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- Value loss (clipped) ----
                if self.value_clip > 0:
                    value_clipped = old_values + torch.clamp(
                        new_values - old_values, -self.value_clip, self.value_clip
                    )
                    v_loss1 = (new_values - returns).pow(2)
                    v_loss2 = (value_clipped - returns).pow(2)
                    value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    value_loss = 0.5 * (new_values - returns).pow(2).mean()

                # ---- Entropy bonus ----
                entropy_loss = -entropy.mean()

                # ---- KL anchor to frozen ACT prior ----
                kl_loss = self.policy.get_kl_from_prior(obs)

                # ---- Total loss ----
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                    + self.kl_coef * kl_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in self.policy.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    clip_frac = ((ratio - 1).abs() > self.clip_range).float().mean()

                total_metrics["policy_loss"] += policy_loss.item()
                total_metrics["value_loss"] += value_loss.item()
                total_metrics["entropy"] += -entropy_loss.item()
                total_metrics["kl_loss"] += kl_loss.item()
                total_metrics["total_loss"] += total_loss.item()
                total_metrics["clip_frac"] += clip_frac.item()
                n_updates += 1

        # Decay KL coefficient
        self.kl_coef *= self.kl_decay

        # Average metrics
        for k in total_metrics:
            total_metrics[k] /= max(1, n_updates)

        self.buffer.clear()
        return total_metrics


# ===========================================================================
# Main training loop
# ===========================================================================
def load_act_checkpoint(checkpoint_path: str, device: torch.device) -> ACTPolicy:
    """Load a trained ACT policy from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = ACTPolicy(
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"[ACT-RL] Loaded ACT checkpoint from {checkpoint_path}")
    print(f"  obs_dim={cfg['obs_dim']}, action_dim={cfg['action_dim']}, "
          f"chunk_size={cfg['chunk_size']}")
    return model


def train_act_rl(args):
    device = torch.device(args.device)

    # ---- Load frozen ACT ----
    act_policy = load_act_checkpoint(args.act_checkpoint, device)
    cfg = torch.load(args.act_checkpoint, map_location="cpu", weights_only=False)["config"]

    # ---- Create ACT-RL policy ----
    rl_policy = ACTRLPolicy(
        act_policy=act_policy,
        obs_dim=cfg["obs_dim"],
        action_dim=cfg["action_dim"],
        chunk_size=cfg["chunk_size"],
        init_log_std=args.init_log_std,
        value_hidden=args.value_hidden,
    ).to(device)

    trainable = sum(p.numel() for p in rl_policy.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in rl_policy.parameters() if not p.requires_grad)
    print(f"[ACT-RL] Trainable params: {trainable:,} | Frozen params: {frozen:,}")

    # ---- Create environment ----
    # Import Isaac Lab environment (lazy import — only needed at runtime)
    env = create_isaac_env(args)
    obs_dim_env = cfg["obs_dim"]

    # ---- PPO Trainer ----
    trainer = ChunkPPOTrainer(
        policy=rl_policy,
        lr=args.lr,
        clip_range=args.clip_range,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        kl_coef=args.kl_coef,
        kl_decay=args.kl_decay,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.ppo_epochs,
        batch_size=args.mini_batch_size,
        value_clip=args.value_clip,
    )

    # ---- Training loop ----
    os.makedirs(args.save_dir, exist_ok=True)

    chunk_size = cfg["chunk_size"]
    total_chunks = args.total_timesteps // (chunk_size * args.num_envs)
    chunks_per_update = args.rollout_chunks  # how many chunks to collect before PPO update

    print(f"\n[ACT-RL] Starting training:")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Num envs: {args.num_envs}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Chunks per update: {chunks_per_update}")

    # Reset environment
    obs_dict = env.reset()
    if isinstance(obs_dict, dict):
        obs = obs_dict.get("policy", obs_dict.get("obs"))
    elif isinstance(obs_dict, tuple):
        obs = obs_dict[0]
    else:
        obs = obs_dict
    obs = obs.to(device)

    best_reward = -float("inf")
    reward_history = deque(maxlen=100)
    success_history = deque(maxlen=100)
    global_step = 0
    t0 = time.time()

    for update_idx in range(total_chunks // chunks_per_update):
        # ---- Collect rollouts ----
        update_rewards = []
        update_successes = []

        for _ in range(chunks_per_update):
            obs, chunk_reward, dones, infos = trainer.collect_chunk_rollout(
                env, obs, device
            )
            update_rewards.append(chunk_reward.mean().item())

            # Track success from info
            if isinstance(infos, dict) and "success" in infos:
                update_successes.extend(infos["success"].cpu().tolist())

            global_step += chunk_size * args.num_envs

            # Handle resets for done envs
            if dones.any():
                # Isaac Lab auto-resets; just update obs
                pass

        # ---- PPO Update ----
        metrics = trainer.update(device)

        # ---- Logging ----
        avg_reward = np.mean(update_rewards)
        reward_history.append(avg_reward)
        if update_successes:
            success_history.extend(update_successes)

        if (update_idx + 1) % args.log_freq == 0:
            elapsed = time.time() - t0
            fps = global_step / elapsed
            avg_success = np.mean(list(success_history)) if success_history else 0.0

            print(
                f"Update {update_idx+1:5d} | "
                f"Steps: {global_step:>10,} | "
                f"Reward: {avg_reward:>8.3f} (avg100: {np.mean(list(reward_history)):>8.3f}) | "
                f"Success: {avg_success:.1%} | "
                f"PL: {metrics['policy_loss']:.4f} VL: {metrics['value_loss']:.4f} "
                f"Ent: {metrics['entropy']:.4f} KL: {metrics['kl_loss']:.4f} "
                f"Clip: {metrics['clip_frac']:.2f} | "
                f"FPS: {fps:.0f} | {elapsed:.0f}s"
            )

            # Log std statistics
            with torch.no_grad():
                std_mean = rl_policy.log_std.exp().mean().item()
                std_min = rl_policy.log_std.exp().min().item()
                std_max = rl_policy.log_std.exp().max().item()
            print(f"  log_std → mean={std_mean:.4f}, min={std_min:.4f}, max={std_max:.4f}")

        # ---- Save checkpoints ----
        if avg_reward > best_reward:
            best_reward = avg_reward
            save_path = os.path.join(args.save_dir, f"act_rl_{args.skill}_best.pt")
            torch.save({
                "policy_state_dict": rl_policy.state_dict(),
                "act_config": cfg,
                "global_step": global_step,
                "best_reward": best_reward,
            }, save_path)

        if (update_idx + 1) % args.save_freq == 0:
            save_path = os.path.join(
                args.save_dir, f"act_rl_{args.skill}_step{global_step}.pt"
            )
            torch.save({
                "policy_state_dict": rl_policy.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "act_config": cfg,
                "global_step": global_step,
            }, save_path)

        # ---- Early stopping ----
        if len(success_history) >= 100 and np.mean(list(success_history)) >= args.early_stop_success:
            print(f"\n[ACT-RL] Early stopping: success rate {np.mean(list(success_history)):.1%} >= {args.early_stop_success:.1%}")
            break

    # ---- Final save ----
    save_path = os.path.join(args.save_dir, f"act_rl_{args.skill}_final.pt")
    torch.save({
        "policy_state_dict": rl_policy.state_dict(),
        "act_config": cfg,
        "global_step": global_step,
    }, save_path)
    print(f"\n[ACT-RL] Training complete. Best reward: {best_reward:.4f}")
    print(f"  Models saved to {args.save_dir}")


# ===========================================================================
# Environment creation (Isaac Lab integration)
# ===========================================================================
def create_isaac_env(args):
    """
    Create Isaac Lab environment for the specified skill.
    This is a template — adapt import paths to your setup.
    """
    try:
        # Try Isaac Lab import
        from omni.isaac.lab.app import AppLauncher

        launcher = AppLauncher(headless=args.headless)
        simulation_app = launcher.app

        if args.skill == "approach_and_grasp":
            from lekiwi_skill2_env import LeKiwiSkill2Env, LeKiwiSkill2EnvCfg

            cfg = LeKiwiSkill2EnvCfg()
            cfg.scene.num_envs = args.num_envs
            if args.multi_object_json:
                cfg.multi_object_json = args.multi_object_json
            env = LeKiwiSkill2Env(cfg=cfg)

        elif args.skill == "carry_and_place":
            from lekiwi_skill3_env import LeKiwiSkill3Env, LeKiwiSkill3EnvCfg

            cfg = LeKiwiSkill3EnvCfg()
            cfg.scene.num_envs = args.num_envs
            if args.handoff_buffer:
                cfg.handoff_buffer_path = args.handoff_buffer
            env = LeKiwiSkill3Env(cfg=cfg)
        else:
            raise ValueError(f"Unknown skill: {args.skill}")

        return env

    except ImportError:
        print("[ACT-RL] Isaac Lab not available, creating dummy environment for testing")
        return DummyVecEnv(
            num_envs=args.num_envs,
            obs_dim=30 if args.skill == "approach_and_grasp" else 29,
            action_dim=9,
        )


class DummyVecEnv:
    """Dummy vectorized env for testing without Isaac Lab."""

    def __init__(self, num_envs=4, obs_dim=30, action_dim=9):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._step = 0

    def reset(self):
        self._step = 0
        return {"policy": torch.randn(self.num_envs, self.obs_dim)}

    def step(self, actions):
        self._step += 1
        obs = torch.randn(self.num_envs, self.obs_dim)
        reward = torch.randn(self.num_envs) * 0.1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = (torch.rand(self.num_envs) < 0.001)
        return {
            "policy": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": {"success": torch.zeros(self.num_envs, dtype=torch.bool)},
        }


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="ACT-RL: Chunk-level PPO Fine-tuning")

    # ACT checkpoint
    parser.add_argument("--act_checkpoint", type=str, required=True,
                        help="Path to trained ACT BC checkpoint")

    # Environment
    parser.add_argument("--skill", type=str, default="approach_and_grasp",
                        choices=["approach_and_grasp", "carry_and_place"])
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--multi_object_json", type=str, default=None)
    parser.add_argument("--handoff_buffer", type=str, default=None)

    # PPO hyperparameters (conservative, following ACT-RL paper)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--clip_range", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--kl_coef", type=float, default=0.05,
                        help="KL anchor coefficient to frozen ACT prior")
    parser.add_argument("--kl_decay", type=float, default=0.999)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=64)
    parser.add_argument("--value_clip", type=float, default=0.2)

    # Exploration head
    parser.add_argument("--init_log_std", type=float, default=-1.0)
    parser.add_argument("--value_hidden", type=int, default=256)

    # Training
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--rollout_chunks", type=int, default=24,
                        help="Number of chunks to collect before each PPO update")

    # Save/log
    parser.add_argument("--save_dir", type=str, default="checkpoints/act_rl")
    parser.add_argument("--log_freq", type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--early_stop_success", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    train_act_rl(args)


if __name__ == "__main__":
    main()
