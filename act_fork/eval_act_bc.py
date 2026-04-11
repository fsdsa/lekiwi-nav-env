"""
Evaluate trained state-only ACT policy in Isaac Sim.

Usage:
    python eval_act_bc.py \
        --ckpt_dir checkpoints/act_skill2 \
        --ckpt_name policy_best.ckpt \
        --temporal_agg
"""
import torch
import numpy as np
import os
import pickle
import argparse

from policy import ACTPolicy


def load_policy(ckpt_dir, ckpt_name="policy_best.ckpt"):
    """Load trained ACT policy and normalization stats."""

    # Load config
    config_path = os.path.join(ckpt_dir, "config.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    # Load stats
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    # Build policy config
    policy_config = {
        'lr': config.get('lr', 1e-4),
        'num_queries': config['chunk_size'],
        'kl_weight': config['kl_weight'],
        'hidden_dim': config['hidden_dim'],
        'dim_feedforward': config['dim_feedforward'],
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': config.get('enc_layers', 4),
        'dec_layers': config.get('dec_layers', 7),
        'nheads': config.get('nheads', 8),
        'camera_names': [],
        'state_only': True,
        'state_dim': config['state_dim'],
        'action_dim': config['action_dim'],
    }

    # Build and load policy
    policy = ACTPolicy(policy_config)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print(f"Loaded: {ckpt_path} ({loading_status})")
    policy.cuda()
    policy.eval()

    # Create pre/post processing functions
    obs_mean = torch.from_numpy(stats['obs_mean']).float().cuda()
    obs_std = torch.from_numpy(stats['obs_std']).float().cuda()
    action_mean = torch.from_numpy(stats['action_mean']).float().cuda()
    action_std = torch.from_numpy(stats['action_std']).float().cuda()

    def pre_process(obs):
        """Normalize observation. obs: (obs_dim,) numpy or tensor."""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().cuda()
        return (obs - obs_mean) / obs_std

    def post_process(action):
        """Denormalize action. action: (action_dim,) or (K, action_dim) tensor."""
        return action * action_std + action_mean

    return policy, pre_process, post_process, config


class ACTInference:
    """
    ACT inference wrapper with temporal ensembling.

    Usage:
        act = ACTInference(ckpt_dir, temporal_agg=True)
        act.reset()
        for step in episode:
            action = act.get_action(obs)
    """

    def __init__(self, ckpt_dir, ckpt_name="policy_best.ckpt", temporal_agg=False):
        self.policy, self.pre_process, self.post_process, self.config = \
            load_policy(ckpt_dir, ckpt_name)
        self.temporal_agg = temporal_agg
        self.chunk_size = self.config['chunk_size']
        self.action_dim = self.config['action_dim']

        # Temporal ensembling state
        self.all_time_actions = None
        self.step_counter = 0

    def reset(self):
        """Reset temporal ensembling state. Call at episode start."""
        self.all_time_actions = None
        self.step_counter = 0

    @torch.no_grad()
    def get_action(self, obs):
        """
        Get single action from ACT policy.

        Args:
            obs: numpy array (obs_dim,) — raw (unnormalized) observation

        Returns:
            action: numpy array (action_dim,) — raw (denormalized) action
        """
        # Normalize obs
        qpos = self.pre_process(obs).unsqueeze(0)  # (1, obs_dim)

        # Forward pass (inference: z=0)
        all_actions = self.policy(qpos)  # (1, K, action_dim) normalized

        if not self.temporal_agg:
            # No ensembling: return first action of chunk
            raw_action = all_actions[:, 0]  # (1, action_dim)
            action = self.post_process(raw_action).squeeze(0).cpu().numpy()
            return action

        # Temporal ensembling
        K = self.chunk_size
        t = self.step_counter

        if self.all_time_actions is None:
            max_T = 10000
            self.all_time_actions = torch.zeros(
                max_T, max_T + K, self.action_dim, device='cuda'
            )

        self.all_time_actions[t, t:t + K] = all_actions[0]
        self.step_counter += 1

        # Weighted average over overlapping chunks
        actions_for_t = self.all_time_actions[:t + 1, t]  # (t+1, action_dim)
        actions_populated = torch.all(actions_for_t != 0, dim=1)
        actions_for_t = actions_for_t[actions_populated]

        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_t)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1).float()
        raw_action = (actions_for_t * exp_weights).sum(dim=0, keepdim=True)  # (1, action_dim)

        action = self.post_process(raw_action).squeeze(0).cpu().numpy()
        return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate ACT policy")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="policy_best.ckpt")
    parser.add_argument("--temporal_agg", action="store_true")
    args = parser.parse_args()

    act = ACTInference(args.ckpt_dir, args.ckpt_name, args.temporal_agg)
    print(f"\nPolicy loaded successfully!")
    print(f"  state_dim: {act.config['state_dim']}")
    print(f"  action_dim: {act.config['action_dim']}")
    print(f"  chunk_size: {act.chunk_size}")
    print(f"  temporal_agg: {act.temporal_agg}")

    # Quick test with random obs
    obs = np.random.randn(act.config['state_dim']).astype(np.float32)
    act.reset()
    action = act.get_action(obs)
    print(f"\n  Test obs shape: {obs.shape}")
    print(f"  Test action shape: {action.shape}")
    print(f"  Test action values: {action}")
    print("\nReady for Isaac Sim evaluation.")
