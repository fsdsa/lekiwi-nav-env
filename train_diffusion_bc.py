"""
Diffusion Policy BC training for LeKiwi — adapted from official ResiP codebase.

Source: https://github.com/ankile/robust-rearrangement
  - Training logic from src/train/bc.py
  - Data handling adapted from src/dataset/dataset.py (zarr → HDF5)
  - Hyperparameters from official configs

This script:
  1. Loads teleop demos from HDF5 (same format as record_teleop.py output)
  2. Creates (obs, action_sequence) pairs with pred_horizon padding
  3. Fits LinearNormalizer on the data (min-max to [-1, 1])
  4. Trains ConditionalUnet1D with DDPM epsilon-prediction loss
  5. Uses EMA for stable inference weights
  6. Saves checkpoint compatible with residual RL stage

Usage:
    python train_diffusion_bc.py \\
        --demo_path demos_skill2/combined_skill2_20260227_091123.hdf5 \\
        --obs_dim 30 \\
        --epochs 500 \\
        --eval
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from diffusion_policy import (
    DiffusionPolicyAgent,
    LinearNormalizer,
    SwitchEMA,
)


# =============================================================================
# Dataset — adapted from src/dataset/dataset.py for our HDF5 format
# =============================================================================
class DiffusionPolicyDataset(Dataset):
    """
    Creates (obs, action_sequence) pairs from episodic HDF5 data.

    For each timestep t in each episode:
        obs: obs[t]  (obs_dim,)
        action_seq: actions[t:t+pred_horizon]  (pred_horizon, act_dim)
        If t + pred_horizon > episode_length, pad with last action.

    The official code uses zarr with obs_horizon > 1 (stacking past obs).
    For our low-dim state (30D with velocities already included),
    obs_horizon=1 is sufficient — no stacking needed.
    """

    def __init__(self, obs_list, act_list, pred_horizon):
        """
        Args:
            obs_list: list of (T_i, obs_dim) arrays per episode
            act_list: list of (T_i, act_dim) arrays per episode
            pred_horizon: number of future actions to predict
        """
        self.pred_horizon = pred_horizon
        self.samples = []

        for obs_ep, act_ep in zip(obs_list, act_list):
            T = len(obs_ep)
            for t in range(T):
                obs = obs_ep[t]

                # Build action sequence of length pred_horizon
                if t + pred_horizon <= T:
                    action_seq = act_ep[t:t + pred_horizon]
                else:
                    # Pad with last action (same as official padding strategy)
                    available = act_ep[t:]
                    pad_len = pred_horizon - len(available)
                    padding = np.tile(act_ep[-1:], (pad_len, 1))
                    action_seq = np.concatenate([available, padding], axis=0)

                self.samples.append((
                    obs.astype(np.float32),
                    action_seq.astype(np.float32),
                ))

    def __len__(self):
        return len(self.samples)

    def __init_augmentation__(self, vel_dropout_prob=0.0, grip_noise_std=0.0,
                               armvel_dropout_prob=0.0, arm_noise_std=0.0):
        """Set augmentation parameters after construction."""
        self.vel_dropout_prob = vel_dropout_prob
        self.grip_noise_std = grip_noise_std
        self.armvel_dropout_prob = armvel_dropout_prob
        self.arm_noise_std = arm_noise_std

    def __getitem__(self, idx):
        obs, action_seq = self.samples[idx]
        obs_t = torch.from_numpy(obs)
        need_clone = False
        # arm3 (wrist_flex) noise: obs[3] only
        if self.arm_noise_std > 0:
            if not need_clone:
                obs_t = obs_t.clone(); need_clone = True
            obs_t[3] = obs_t[3] + torch.randn(1).item() * self.arm_noise_std
        # base_vel random dropout: obs[6:15] (bvx,bvy,bwz,lvx,lvy,lvz,avx,avy,avz)
        if self.vel_dropout_prob > 0 and torch.rand(1).item() < self.vel_dropout_prob:
            if not need_clone:
                obs_t = obs_t.clone(); need_clone = True
            obs_t[6:15] = 0.0
        # gripper position noise: obs[5]
        if self.grip_noise_std > 0:
            if not need_clone:
                obs_t = obs_t.clone(); need_clone = True
            obs_t[5] = obs_t[5] + torch.randn(1).item() * self.grip_noise_std
        # arm+grip velocity dropout: obs[15:21]
        if self.armvel_dropout_prob > 0 and torch.rand(1).item() < self.armvel_dropout_prob:
            if not need_clone:
                obs_t = obs_t.clone(); need_clone = True
            obs_t[15:21] = 0.0
        return obs_t, torch.from_numpy(action_seq)


def load_demos_from_hdf5(path: str, obs_key="obs", act_key="actions"):
    """
    Load episodic demos from a single HDF5 file.
    Compatible with record_teleop.py output format:
        /episode_N/obs: (T, obs_dim)
        /episode_N/actions: (T, act_dim)
        /episode_N/teleop_active: (T,) optional

    Returns: obs_list, act_list (lists of arrays per episode)
    """
    obs_list = []
    act_list = []

    with h5py.File(path, "r") as f:
        # Get all episode keys and sort numerically
        ep_keys = sorted(
            [k for k in f.keys() if k.startswith("episode_")],
            key=lambda x: int(x.split("_")[1]),
        )
        print(f"Found {len(ep_keys)} episodes in {path}")

        for ep_key in ep_keys:
            ep = f[ep_key]

            # Check for success attribute (skip failed episodes)
            if "success" in ep.attrs and not ep.attrs["success"]:
                print(f"  Skipping {ep_key} (not successful)")
                continue

            obs = ep[obs_key][:]
            actions = ep[act_key][:].astype(np.float32)  # float64 → float32

            # Optional: filter by teleop_active
            if "teleop_active" in ep:
                active = ep["teleop_active"][:].astype(bool)
                if active.sum() > 0:
                    # Find contiguous active region
                    first_active = np.argmax(active)
                    last_active = len(active) - 1 - np.argmax(active[::-1])
                    obs = obs[first_active:last_active + 1]
                    actions = actions[first_active:last_active + 1]

            obs_list.append(obs.astype(np.float32))
            act_list.append(actions)
            print(f"  {ep_key}: {len(obs)} steps")

    print(f"Loaded {len(obs_list)} episodes, "
          f"total {sum(len(o) for o in obs_list)} steps")
    return obs_list, act_list


# =============================================================================
# Training loop
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy BC")

    # Data
    parser.add_argument("--demo_path", type=str, required=True,
                        help="Path to HDF5 demo file")
    parser.add_argument("--obs_dim", type=int, required=True,
                        help="Observation dimension (30 for Skill-2, 29 for Skill-3)")
    parser.add_argument("--act_dim", type=int, default=9,
                        help="Action dimension")

    # Architecture (official defaults from config)
    parser.add_argument("--pred_horizon", type=int, default=16,
                        help="Prediction horizon (official default: 16)")
    parser.add_argument("--action_horizon", type=int, default=8,
                        help="Execution horizon (official default: 8)")
    parser.add_argument("--down_dims", type=int, nargs="+", default=[256, 512, 1024],
                        help="UNet channel dims (official default: 256 512 1024)")

    # Diffusion (official defaults)
    parser.add_argument("--num_diffusion_iters", type=int, default=100,
                        help="Training diffusion steps (official: 100)")
    parser.add_argument("--inference_steps", type=int, default=16,
                        help="DDIM inference steps (official: 16)")

    # Training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (official uses 1e-4 for BC)")
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--ema_decay", type=float, default=0.995,
                        help="EMA decay (0 to disable)")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Save checkpoint every N epochs")

    # Output
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_name", type=str, default="dp_bc.pt")
    parser.add_argument("--vel_dropout", type=float, default=0.0,
                        help="Probability of zeroing base_vel obs[6:15] per sample (0.0=off)")
    parser.add_argument("--grip_noise", type=float, default=0.0,
                        help="Std of Gaussian noise added to gripper obs[5] (0.0=off)")
    parser.add_argument("--armvel_dropout", type=float, default=0.0,
                        help="Probability of zeroing arm_vel obs[15:21] per sample (0.0=off)")
    parser.add_argument("--arm_noise", type=float, default=0.0,
                        help="Std of Gaussian noise added to arm+grip obs[0:6] (0.0=off)")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation after training")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # =========================================================================
    # 1. Load data
    # =========================================================================
    obs_list, act_list = load_demos_from_hdf5(args.demo_path)

    assert obs_list[0].shape[1] == args.obs_dim, \
        f"obs_dim mismatch: data has {obs_list[0].shape[1]}, expected {args.obs_dim}"
    assert act_list[0].shape[1] == args.act_dim, \
        f"act_dim mismatch: data has {act_list[0].shape[1]}, expected {args.act_dim}"

    # =========================================================================
    # 2. Fit normalizer (official: min-max to [-1, 1])
    # =========================================================================
    all_obs = np.concatenate(obs_list, axis=0)
    all_actions = np.concatenate(act_list, axis=0)

    normalizer = LinearNormalizer()
    normalizer.fit({
        "obs": torch.from_numpy(all_obs),
        "action": torch.from_numpy(all_actions),
    })
    print(f"Normalizer fitted: obs range [{all_obs.min():.3f}, {all_obs.max():.3f}], "
          f"action range [{all_actions.min():.3f}, {all_actions.max():.3f}]")

    # =========================================================================
    # 3. Create dataset
    # =========================================================================
    dataset = DiffusionPolicyDataset(obs_list, act_list, args.pred_horizon)
    dataset.__init_augmentation__(
        vel_dropout_prob=args.vel_dropout,
        grip_noise_std=args.grip_noise,
        armvel_dropout_prob=args.armvel_dropout,
        arm_noise_std=args.arm_noise,
    )
    if args.vel_dropout > 0:
        print(f"Velocity dropout enabled: p={args.vel_dropout} (obs[6:15] zeroed)")
    if args.grip_noise > 0:
        print(f"Gripper noise enabled: std={args.grip_noise} (obs[5])")
    if args.arm_noise > 0:
        print(f"Arm noise enabled: std={args.arm_noise} (obs[0:6])")
    if args.armvel_dropout > 0:
        print(f"Arm velocity dropout enabled: p={args.armvel_dropout} (obs[15:21] zeroed)")
    print(f"Dataset: {len(dataset)} samples (all data used for training)")

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    print(f"Train: {len(dataset)} samples")

    # =========================================================================
    # 4. Create model
    # =========================================================================
    agent = DiffusionPolicyAgent(
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        pred_horizon=args.pred_horizon,
        action_horizon=args.action_horizon,
        num_diffusion_iters=args.num_diffusion_iters,
        inference_steps=args.inference_steps,
        down_dims=args.down_dims,
    ).to(device)

    # Share normalizer
    agent.normalizer.load_state_dict(normalizer.state_dict(), device=device)

    num_params = agent.get_num_params()
    print(f"Model parameters: {num_params:,}")

    # =========================================================================
    # 5. Optimizer + scheduler (official uses cosine annealing)
    # =========================================================================
    optimizer = torch.optim.AdamW(
        agent.model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.95, 0.999),  # official uses (0.95, 0.999) for BC
        eps=1e-8,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # EMA
    ema = None
    if args.ema_decay > 0:
        ema = SwitchEMA(agent.model, decay=args.ema_decay)
        ema.register()
        print(f"EMA enabled with decay={args.ema_decay}")

    # =========================================================================
    # 6. Training loop
    # =========================================================================
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        agent.model.train()
        train_losses = []

        for obs_batch, action_batch in train_loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)

            # Normalize
            nobs = agent.normalizer(obs_batch, "obs", forward=True)
            naction = agent.normalizer(
                action_batch.reshape(-1, args.act_dim), "action", forward=True
            ).reshape(action_batch.shape)

            loss = agent.compute_loss(nobs, naction)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
            optimizer.step()

            if ema is not None:
                ema.update()

            train_losses.append(loss.item())

        lr_scheduler.step()
        avg_train_loss = np.mean(train_losses)

        # --- Log ---
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train_loss={avg_train_loss:.6f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"time={elapsed:.0f}s"
            )

        # --- Save periodically ---
        if epoch % args.save_every == 0 or epoch == args.epochs:
            if ema is not None:
                ema.apply_shadow()

            # Save epoch-specific checkpoint
            save_path = save_dir / f"dp_bc_epoch{epoch}.pt"
            ckpt_data = {
                "model_state_dict": {
                    **{f"model.{k}": v for k, v in agent.model.state_dict().items()},
                    **{f"normalizer.{k}": v for k, v in agent.normalizer.state_dict().items()},
                },
                "config": {
                    "obs_dim": args.obs_dim,
                    "act_dim": args.act_dim,
                    "pred_horizon": args.pred_horizon,
                    "action_horizon": args.action_horizon,
                    "num_diffusion_iters": args.num_diffusion_iters,
                    "inference_steps": args.inference_steps,
                    "down_dims": args.down_dims,
                },
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "normalizer_state_dict": agent.normalizer.state_dict(),
            }
            torch.save(ckpt_data, save_path)

            # Also overwrite "latest" for convenience
            torch.save(ckpt_data, save_dir / args.save_name)

            if ema is not None:
                ema.restore()

            print(f"  ★ Saved checkpoint (epoch={epoch}, loss={avg_train_loss:.6f}) → {save_path.name}")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s")
    print(f"Final train loss: {avg_train_loss:.6f}")
    print(f"Checkpoint: {save_dir / args.save_name}")

    # =========================================================================
    # 7. Evaluation
    # =========================================================================
    if args.eval:
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)

        # Load last checkpoint
        ckpt = torch.load(save_dir / args.save_name, map_location=device, weights_only=False)

        # Reconstruct model
        cfg = ckpt["config"]
        eval_agent = DiffusionPolicyAgent(
            obs_dim=cfg["obs_dim"],
            act_dim=cfg["act_dim"],
            pred_horizon=cfg["pred_horizon"],
            action_horizon=cfg["action_horizon"],
            num_diffusion_iters=cfg["num_diffusion_iters"],
            inference_steps=cfg["inference_steps"],
            down_dims=cfg["down_dims"],
        ).to(device)

        # Load weights (same format as official load_base_state_dict)
        state_dict = ckpt["model_state_dict"]
        model_state = {
            k[len("model."):]: v for k, v in state_dict.items()
            if k.startswith("model.")
        }
        norm_state = {
            k[len("normalizer."):]: v for k, v in state_dict.items()
            if k.startswith("normalizer.")
        }
        eval_agent.model.load_state_dict(model_state)
        eval_agent.normalizer.load_state_dict(norm_state)
        eval_agent.eval()

        # Evaluate on some episodes
        total_first_mae = []
        total_chunk_mae = []

        for ep_idx, (obs_ep, act_ep) in enumerate(zip(obs_list, act_list)):
            if ep_idx >= 5:
                break

            ep_first_maes = []
            eval_agent.reset()

            for t in range(0, len(obs_ep), args.action_horizon):
                obs_t = torch.from_numpy(obs_ep[t:t+1]).float().to(device)
                pred_actions = eval_agent.predict_action(obs_t)  # (1, pred_horizon, act_dim)

                # Compare first action
                gt_first = torch.from_numpy(act_ep[t:t+1]).float().to(device)
                pred_first = pred_actions[0, 0:1, :]
                mae_first = (pred_first - gt_first).abs().mean().item()
                ep_first_maes.append(mae_first)

                # Compare full chunk
                end_t = min(t + args.action_horizon, len(act_ep))
                chunk_len = end_t - t
                if chunk_len > 0:
                    gt_chunk = torch.from_numpy(act_ep[t:end_t]).float().to(device)
                    pred_chunk = pred_actions[0, :chunk_len, :]
                    mae_chunk = (pred_chunk - gt_chunk).abs().mean().item()
                    total_chunk_mae.append(mae_chunk)

            avg_ep_mae = np.mean(ep_first_maes)
            total_first_mae.append(avg_ep_mae)
            print(f"  Episode {ep_idx}: first_action_mae={avg_ep_mae:.4f}")

        print(f"\n  Overall first_action_mae: {np.mean(total_first_mae):.4f}")
        print(f"  Overall chunk_mae: {np.mean(total_chunk_mae):.4f}")


if __name__ == "__main__":
    main()
