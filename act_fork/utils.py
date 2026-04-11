"""
Data loading utilities for state-only ACT training.
Reads LeKiwi HDF5 format: episode_*/obs, episode_*/actions, episode_*/teleop_active
Preserves official ACT normalization and data pipeline.
"""
import numpy as np
import torch
import os
import glob
import h5py
from torch.utils.data import DataLoader


class EpisodicDataset(torch.utils.data.Dataset):
    """
    Episodic dataset for state-only ACT.

    Each __getitem__ returns one (obs_t, action_chunk_{t:t+K}, is_pad) sample
    by randomly sampling a start timestep within an episode.

    This matches the official ACT sampling strategy: each epoch samples
    one random timestep per episode.
    """
    def __init__(self, episode_list, chunk_size, norm_stats):
        """
        Args:
            episode_list: list of (obs_array, action_array) tuples
            chunk_size: action chunk length (num_queries)
            norm_stats: dict with obs_mean, obs_std, action_mean, action_std
        """
        super().__init__()
        self.episode_list = episode_list
        self.chunk_size = chunk_size
        self.norm_stats = norm_stats

    def __len__(self):
        return len(self.episode_list)

    def __getitem__(self, index):
        obs_all, actions_all = self.episode_list[index]
        episode_len = obs_all.shape[0]
        K = self.chunk_size
        action_dim = actions_all.shape[1]

        # Random start timestep (official ACT strategy)
        start_ts = np.random.choice(episode_len)

        # Get observation at start_ts
        qpos = obs_all[start_ts]

        # Get action chunk: actions from start_ts onwards, padded to chunk_size
        action = actions_all[start_ts:]
        action_len = min(len(action), K)

        # Pad to fixed chunk_size (so all samples have same shape for batching)
        padded_action = np.zeros((K, action_dim), dtype=np.float32)
        padded_action[:action_len] = action[:action_len]
        is_pad = np.zeros(K)
        is_pad[action_len:] = 1

        # Convert to tensors
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # Normalize (critical for ACT training stability)
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["obs_mean"]) / self.norm_stats["obs_std"]

        return qpos_data, action_data, is_pad


def load_episodes_from_hdf5(demo_dir, skill="approach_and_grasp", filter_active=True):
    """
    Load episodes from LeKiwi HDF5 files.

    Args:
        demo_dir: directory containing HDF5 files
        skill: skill name for file pattern matching
        filter_active: if True, filter by teleop_active flag

    Returns:
        episode_list: list of (obs, actions) numpy arrays
    """
    # File patterns for each skill
    patterns = {
        "approach_and_grasp": ["combined_skill2_*.hdf5", "teleop_skill2_*.hdf5"],
        "carry_and_place": ["combined_skill3_*.hdf5", "teleop_skill3_*.hdf5"],
        "navigate": ["teleop_nav_*.hdf5", "teleop_*.hdf5"],
    }
    file_patterns = patterns.get(skill, ["*.hdf5"])

    hdf5_files = []
    for pat in file_patterns:
        hdf5_files.extend(glob.glob(os.path.join(demo_dir, pat)))
    hdf5_files = sorted(set(hdf5_files))

    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {demo_dir} for skill={skill}")

    print(f"[Data] Found {len(hdf5_files)} files for skill={skill}")

    episode_list = []
    total_steps = 0

    for fpath in hdf5_files:
        with h5py.File(fpath, "r") as f:
            for ep_key in sorted(f.keys()):
                if not ep_key.startswith("episode_"):
                    continue

                ep = f[ep_key]
                obs = ep["obs"][:]          # (T, obs_dim)
                actions = ep["actions"][:]  # (T, action_dim)

                # Filter by teleop_active if requested
                if filter_active and "teleop_active" in ep:
                    active = ep["teleop_active"][:].astype(bool)
                    obs = obs[active]
                    actions = actions[active]

                T = obs.shape[0]
                if T < 2:
                    continue

                episode_list.append((obs.astype(np.float32), actions.astype(np.float32)))
                total_steps += T

    print(f"[Data] Loaded {len(episode_list)} episodes, {total_steps} total steps")
    return episode_list


def get_norm_stats(episode_list):
    """
    Compute normalization statistics across all episodes.
    Matches official ACT: mean/std over [episode, timestep] dims, clipped std.
    """
    all_obs = []
    all_actions = []
    for obs, actions in episode_list:
        all_obs.append(torch.from_numpy(obs))
        all_actions.append(torch.from_numpy(actions))

    # Concat all timesteps
    all_obs = torch.cat(all_obs, dim=0)        # (total_steps, obs_dim)
    all_actions = torch.cat(all_actions, dim=0)  # (total_steps, action_dim)

    # Compute stats
    obs_mean = all_obs.mean(dim=0)
    obs_std = all_obs.std(dim=0)
    obs_std = torch.clip(obs_std, 1e-2, np.inf)

    action_mean = all_actions.mean(dim=0)
    action_std = all_actions.std(dim=0)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    stats = {
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }

    print(f"[Norm] obs_mean range: [{obs_mean.min():.3f}, {obs_mean.max():.3f}]")
    print(f"[Norm] obs_std range:  [{obs_std.min():.3f}, {obs_std.max():.3f}]")
    print(f"[Norm] act_mean range: [{action_mean.min():.3f}, {action_mean.max():.3f}]")
    print(f"[Norm] act_std range:  [{action_std.min():.3f}, {action_std.max():.3f}]")

    return stats


def load_data(demo_dir, skill, chunk_size, batch_size_train, batch_size_val,
              filter_active=True):
    """
    Full data loading pipeline: load episodes → compute stats → create dataloaders.

    Returns:
        train_dataloader, val_dataloader, norm_stats
    """
    print(f'\nData from: {demo_dir}\n')

    # Load all episodes
    episode_list = load_episodes_from_hdf5(demo_dir, skill, filter_active)
    num_episodes = len(episode_list)

    # Train/val split (80/20)
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # Ensure at least 1 val episode
    if len(val_indices) == 0:
        val_indices = train_indices[-1:]
        train_indices = train_indices[:-1]

    train_episodes = [episode_list[i] for i in train_indices]
    val_episodes = [episode_list[i] for i in val_indices]

    # Compute normalization stats from training set only
    norm_stats = get_norm_stats(train_episodes)

    # Create datasets
    train_dataset = EpisodicDataset(train_episodes, chunk_size, norm_stats)
    val_dataset = EpisodicDataset(val_episodes, chunk_size, norm_stats)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        prefetch_factor=2,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    print(f"[Data] Train: {len(train_episodes)} episodes, Val: {len(val_episodes)} episodes")

    return train_dataloader, val_dataloader, norm_stats


# ---------------------------------------------------------------------------
# Helper functions (from official ACT)
# ---------------------------------------------------------------------------
def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
