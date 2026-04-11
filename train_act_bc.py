"""
Train State-Only ACT via Behavioral Cloning on LeKiwi HDF5 demos.

Reads existing HDF5 files (from record_teleop.py) and trains ACT with
L1 reconstruction + KL divergence loss.

Features:
  - AMP (bf16) for ~2-3x speedup on A100
  - torch.compile for additional kernel fusion speedup
  - Cosine annealing with warmup

Usage:
    python train_act_bc.py \
        --demo_dir demos/ \
        --skill approach_and_grasp \
        --chunk_size 20 \
        --epochs 3000 \
        --batch_size 256 \
        --lr 4e-4 \
        --kl_weight 10.0 \
        --num_workers 8 \
        --save_dir checkpoints/act
"""

import argparse
import glob
import math
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from act_model import ACTPolicy, act_loss, count_params


# ---------------------------------------------------------------------------
# Dataset: load HDF5 demos → (obs, action_chunk) pairs
# ---------------------------------------------------------------------------
class ACTDemoDataset(Dataset):
    """
    Loads episodes from HDF5 files and creates (obs_t, action_chunk_{t:t+k}) pairs.

    Each sample:
        obs:          (obs_dim,)         — observation at timestep t
        action_chunk: (chunk_size, 9)    — future actions [t, t+1, ..., t+k-1]

    Handles padding for episodes shorter than chunk_size at the end
    by repeating the last action.
    """

    def __init__(
        self,
        demo_dir: str,
        chunk_size: int = 20,
        skill: str = "approach_and_grasp",
        filter_active: bool = True,
        stride: int = 1,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []

        # Find HDF5 files
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
            raise FileNotFoundError(
                f"No HDF5 files found in {demo_dir} for skill={skill}"
            )

        print(f"[ACTDemoDataset] Found {len(hdf5_files)} files for skill={skill}")

        total_episodes = 0
        total_steps = 0

        for fpath in hdf5_files:
            with h5py.File(fpath, "r") as f:
                for ep_key in sorted(f.keys()):
                    if not ep_key.startswith("episode_"):
                        continue

                    ep = f[ep_key]
                    obs = ep["obs"][:]          # (T, obs_dim)
                    actions = ep["actions"][:]  # (T, 9)

                    # Filter by teleop_active if requested
                    if filter_active and "teleop_active" in ep:
                        active = ep["teleop_active"][:].astype(bool)
                        obs = obs[active]
                        actions = actions[active]

                    T = obs.shape[0]
                    if T < 2:
                        continue

                    # Create (obs, action_chunk) pairs with stride
                    for t in range(0, T, stride):
                        # Action chunk: actions[t:t+k], pad if necessary
                        end = min(t + chunk_size, T)
                        chunk = actions[t:end].copy()

                        if chunk.shape[0] < chunk_size:
                            # Pad by repeating last action
                            pad_len = chunk_size - chunk.shape[0]
                            pad = np.tile(chunk[-1:], (pad_len, 1))
                            chunk = np.concatenate([chunk, pad], axis=0)

                        self.samples.append(
                            (obs[t].astype(np.float32), chunk.astype(np.float32))
                        )

                    total_episodes += 1
                    total_steps += T

        print(
            f"[ACTDemoDataset] Loaded {total_episodes} episodes, "
            f"{total_steps} total steps, {len(self.samples)} training samples "
            f"(chunk_size={chunk_size}, stride={stride})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, chunk = self.samples[idx]
        return torch.from_numpy(obs), torch.from_numpy(chunk)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device(args.device)

    # ---- Dataset ----
    dataset = ACTDemoDataset(
        demo_dir=args.demo_dir,
        chunk_size=args.chunk_size,
        skill=args.skill,
        filter_active=args.filter_active,
        stride=args.stride,
    )

    # Determine obs_dim from data
    obs_dim = dataset.samples[0][0].shape[0]
    action_dim = dataset.samples[0][1].shape[1]
    print(f"[Train] obs_dim={obs_dim}, action_dim={action_dim}")

    # Train/val split
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Model ----
    model = ACTPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        chunk_size=args.chunk_size,
        n_heads=args.n_heads,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    params = count_params(model)
    print(f"[Train] Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # ---- torch.compile (PyTorch 2.x) ----
    compiled_model = model
    if args.compile:
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
            print("[Train] torch.compile enabled (reduce-overhead mode)")
        except Exception as e:
            print(f"[Train] torch.compile failed ({e}), falling back to eager mode")
            compiled_model = model

    # ---- AMP setup ----
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    # Note: bf16 on A100 doesn't need GradScaler, but we keep it for fp16 fallback
    if use_amp:
        print(f"[Train] AMP enabled with {amp_dtype}")
    else:
        print("[Train] AMP disabled (fp32)")

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Cosine annealing with warmup
    total_steps = args.epochs * len(train_loader)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(
            1, total_steps - args.warmup_steps
        )
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Training ----
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    global_step = 0

    print(f"\n[Train] Starting training for {args.epochs} epochs...")
    print(f"  Train samples: {train_size}, Val samples: {val_size}")
    print(f"  Batch size: {args.batch_size}, Steps/epoch: {len(train_loader)}")
    print(f"  KL weight: {args.kl_weight}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {"total": 0, "l1": 0, "kl": 0}
        t0 = time.time()

        for obs, action_chunks in train_loader:
            obs = obs.to(device, non_blocking=True)
            action_chunks = action_chunks.to(device, non_blocking=True)

            # Forward pass with AMP
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = compiled_model(obs, action_chunks)
                loss_dict = act_loss(
                    out["pred_actions"],
                    action_chunks,
                    out["mu"],
                    out["logvar"],
                    kl_weight=args.kl_weight,
                )

            optimizer.zero_grad(set_to_none=True)

            if use_amp and amp_dtype == torch.float16:
                # fp16 needs GradScaler
                scaler.scale(loss_dict["total"]).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # bf16 or fp32: no scaler needed
                loss_dict["total"].backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()

            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k].item()

            global_step += 1

        n_batches = len(train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        # ---- Validation ----
        model.eval()
        val_losses = {"total": 0, "l1": 0, "kl": 0}
        with torch.no_grad():
            for obs, action_chunks in val_loader:
                obs = obs.to(device, non_blocking=True)
                action_chunks = action_chunks.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    out = model(obs, action_chunks)
                    loss_dict = act_loss(
                        out["pred_actions"],
                        action_chunks,
                        out["mu"],
                        out["logvar"],
                        kl_weight=args.kl_weight,
                    )

                for k in val_losses:
                    val_losses[k] += loss_dict[k].item()

        n_val = len(val_loader)
        for k in val_losses:
            val_losses[k] /= max(1, n_val)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        # ---- Logging ----
        if epoch % args.log_freq == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"Train L1={epoch_losses['l1']:.5f} KL={epoch_losses['kl']:.5f} "
                f"Total={epoch_losses['total']:.5f} | "
                f"Val L1={val_losses['l1']:.5f} Total={val_losses['total']:.5f} | "
                f"LR={lr_now:.2e} | {elapsed:.1f}s"
            )

        # ---- Save best ----
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            save_path = os.path.join(args.save_dir, f"act_{args.skill}_best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "config": {
                        "obs_dim": obs_dim,
                        "action_dim": action_dim,
                        "hidden_dim": args.hidden_dim,
                        "latent_dim": args.latent_dim,
                        "chunk_size": args.chunk_size,
                        "n_heads": args.n_heads,
                        "n_enc_layers": args.n_enc_layers,
                        "n_dec_layers": args.n_dec_layers,
                        "ff_dim": args.ff_dim,
                    },
                },
                save_path,
            )
            if epoch % args.log_freq == 0:
                print(f"  → Saved best model (val_loss={best_val_loss:.5f})")

        # ---- Periodic checkpoint ----
        if epoch % args.save_freq == 0:
            save_path = os.path.join(
                args.save_dir, f"act_{args.skill}_epoch{epoch}.pt"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": {
                        "obs_dim": obs_dim,
                        "action_dim": action_dim,
                        "hidden_dim": args.hidden_dim,
                        "latent_dim": args.latent_dim,
                        "chunk_size": args.chunk_size,
                        "n_heads": args.n_heads,
                        "n_enc_layers": args.n_enc_layers,
                        "n_dec_layers": args.n_dec_layers,
                        "ff_dim": args.ff_dim,
                    },
                },
                save_path,
            )

    # ---- Final save ----
    save_path = os.path.join(args.save_dir, f"act_{args.skill}_final.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": args.epochs,
            "config": {
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "hidden_dim": args.hidden_dim,
                "latent_dim": args.latent_dim,
                "chunk_size": args.chunk_size,
                "n_heads": args.n_heads,
                "n_enc_layers": args.n_enc_layers,
                "n_dec_layers": args.n_dec_layers,
                "ff_dim": args.ff_dim,
            },
        },
        save_path,
    )
    print(f"\n[Train] Done. Best val_loss={best_val_loss:.5f}")
    print(f"  Best model: {args.save_dir}/act_{args.skill}_best.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train State-Only ACT via BC")

    # Data
    parser.add_argument("--demo_dir", type=str, required=True)
    parser.add_argument("--skill", type=str, default="approach_and_grasp",
                        choices=["approach_and_grasp", "carry_and_place", "navigate"])
    parser.add_argument("--filter_active", action="store_true", default=True)
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride for sampling (obs, chunk) pairs from episodes")

    # Model architecture
    parser.add_argument("--chunk_size", type=int, default=20,
                        help="Number of future actions to predict (20 @ 60Hz = 0.33s)")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_enc_layers", type=int, default=4)
    parser.add_argument("--n_dec_layers", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=10.0)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=8)

    # AMP & compile
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Enable AMP bf16 (default: True)")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable AMP (force fp32)")
    parser.add_argument("--compile", action="store_true", default=True,
                        help="Enable torch.compile (default: True)")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile")

    # Save/log
    parser.add_argument("--save_dir", type=str, default="checkpoints/act")
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Handle negation flags
    if args.no_amp:
        args.amp = False
    if args.no_compile:
        args.compile = False

    train(args)


if __name__ == "__main__":
    main()
