#!/usr/bin/env python3
"""
LeKiwi Navigation — Behavioral Cloning 학습.

record_teleop.py가 수집한 HDF5 데모에서 (obs, action) 쌍을 추출하여 supervised learning.
네트워크 구조가 train_lekiwi.py의 PolicyNet.net + mean_layer과 **동일**해야
BC 가중치를 PPO Actor에 그대로 로드할 수 있음.

Usage:
    cd ~/IsaacLab/scripts/lekiwi_nav_env

    # 37D 텔레옵 데모(BC->RL 37D 메인 실험용) 학습
    python train_bc.py --demo_dir demos/ --epochs 200 --expected_obs_dim 37

    # 검증 포함
    python train_bc.py --demo_dir demos/ --epochs 200 --expected_obs_dim 37 --eval

결과:
    checkpoints/bc_nav.pt          — BC 가중치 (net + mean_layer)
    checkpoints/bc_nav_norm.npz    — obs 정규화 파라미터 (mean, std, --normalize 사용 시)

다음 단계:
    python train_lekiwi.py --num_envs 2048 --bc_checkpoint checkpoints/bc_nav.pt --headless
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, random_split


# ═══════════════════════════════════════════════════════════════════════
#  BC Policy — train_lekiwi.py PolicyNet과 동일한 구조
# ═══════════════════════════════════════════════════════════════════════

class BCPolicy(nn.Module):
    """
    train_lekiwi.py PolicyNet의 net + mean_layer과 완전히 동일한 구조.

    State dict keys:
        net.0.weight, net.0.bias    (Linear obs_dim→256)
        net.2.weight, net.2.bias    (Linear 256→128)
        net.4.weight, net.4.bias    (Linear 128→64)
        mean_layer.weight, mean_layer.bias  (Linear 64→9)

    PolicyNet에 있는 log_std_parameter는 여기 없음 (BC는 deterministic).
    → load_bc_into_policy()가 net + mean_layer만 로드, log_std는 유지.
    """
    def __init__(self, obs_dim: int = 33, act_dim: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.mean_layer = nn.Linear(64, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_layer(self.net(obs))


# ═══════════════════════════════════════════════════════════════════════
#  데모 로드
# ═══════════════════════════════════════════════════════════════════════

def load_demos(demo_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    record_teleop.py가 생성한 HDF5에서 (obs, action) 추출.

    Expected HDF5 구조:
        /episode_0/obs      → (T, obs_dim)
        /episode_0/actions  → (T, 9)
        /episode_1/...
    """
    demo_dir = Path(demo_dir)
    hdf5_files = sorted(demo_dir.glob("*.hdf5")) + sorted(demo_dir.glob("*.h5"))

    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {demo_dir}")

    obs_list, act_list = [], []

    for fpath in hdf5_files:
        file_episode_count = 0
        with h5py.File(fpath, "r") as f:
            episode_keys = sorted([k for k in f.keys() if k.startswith("episode")])

            if episode_keys:
                for key in episode_keys:
                    grp = f[key]
                    obs_list.append(grp["obs"][:])
                    act_list.append(grp["actions"][:])
                    file_episode_count += 1
            elif "obs" in f and "actions" in f:
                obs_list.append(f["obs"][:])
                act_list.append(f["actions"][:])
                file_episode_count = 1

        print(f"  Loaded: {fpath.name} ({file_episode_count} episodes)")

    if not obs_list:
        raise ValueError(f"No valid episodes found in {demo_dir}")

    obs_all = np.concatenate(obs_list, axis=0)
    act_all = np.concatenate(act_list, axis=0)

    print(f"\n  총 {len(obs_list)} 에피소드, {len(obs_all)} 스텝")
    print(f"  obs shape: {obs_all.shape}, action shape: {act_all.shape}")

    return obs_all, act_all


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LeKiwi Nav — BC Training")
    parser.add_argument("--demo_dir", type=str, default="demos/")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--save_dir", type=str, default="checkpoints/")
    parser.add_argument("--expected_obs_dim", type=int, required=True,
                        help="관측 차원 강제 검증 (예: 30=Skill-2, 29=Skill-3)")
    parser.add_argument("--expected_act_dim", type=int, default=9,
                        help="액션 차원 검증")
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument("--normalize", action="store_true",
                            help="obs 정규화 적용 (기본: 비활성)")
    norm_group.add_argument("--no_normalize", action="store_true",
                            help="deprecated: obs 정규화 비활성화 (기본값과 동일)")
    parser.add_argument("--eval", action="store_true",
                        help="학습 후 간단한 평가")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  LeKiwi Nav — Behavioral Cloning")
    print("=" * 60)

    # —— 데모 로드 ————————————————————————————————————————————
    obs_data, act_data = load_demos(args.demo_dir)
    obs_dim = obs_data.shape[-1]
    act_dim = act_data.shape[-1]

    assert obs_dim == args.expected_obs_dim, (
        f"Expected obs_dim={args.expected_obs_dim}, got {obs_dim}"
    )
    assert act_dim == args.expected_act_dim, (
        f"Expected act_dim={args.expected_act_dim}, got {act_dim}"
    )

    # —— 정규화 ——————————————————————————————————————————————
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    use_normalize = bool(args.normalize) and not bool(args.no_normalize)

    if use_normalize:
        obs_mean = obs_data.mean(axis=0)
        obs_std = obs_data.std(axis=0) + 1e-8
        obs_data = (obs_data - obs_mean) / obs_std

        norm_path = save_dir / "bc_nav_norm.npz"
        np.savez(norm_path, mean=obs_mean, std=obs_std)
        print(f"\n  Obs 정규화 적용, 저장: {norm_path}")
    else:
        norm_path = save_dir / "bc_nav_norm.npz"
        if norm_path.exists():
            # stale 파일로 인해 BC->PPO 정규화 mismatch가 생기지 않도록 제거
            norm_path.unlink()
            print(f"\n  Obs 정규화 미적용 (기본). stale norm 파일 제거: {norm_path}")
        else:
            print("\n  Obs 정규화 미적용 (기본, PPO RunningStandardScaler와 정합)")

    # —— 데이터셋 ————————————————————————————————————————————
    dataset = TensorDataset(
        torch.FloatTensor(obs_data),
        torch.FloatTensor(act_data),
    )
    train_size = max(int(len(dataset) * args.train_split), 1)
    train_size = min(train_size, len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if val_size > 0 else None

    # —— 모델 ————————————————————————————————————————————————
    policy = BCPolicy(obs_dim, act_dim).cuda()
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.MSELoss()

    print(f"\n  Network: {obs_dim} → [256, 128, 64] → {act_dim}")
    print(f"  Train: {train_size} steps, Val: {val_size} steps")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print()

    # —— 학습 루프 ————————————————————————————————————————————
    best_val_loss = float("inf")
    save_path = save_dir / "bc_nav.pt"

    for epoch in range(args.epochs):
        policy.train()
        train_loss = 0.0
        for obs_b, act_b in train_loader:
            obs_b, act_b = obs_b.cuda(), act_b.cuda()
            pred = policy(obs_b)
            loss = loss_fn(pred, act_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        policy.eval()
        if val_loader is not None and len(val_loader) > 0:
            val_loss = 0.0
            with torch.no_grad():
                for obs_b, act_b in val_loader:
                    obs_b, act_b = obs_b.cuda(), act_b.cuda()
                    val_loss += loss_fn(policy(obs_b), act_b).item()
            val_loss /= len(val_loader)
        else:
            val_loss = train_loss

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(policy.state_dict(), save_path)

        if (epoch + 1) % 20 == 0 or epoch == 0 or is_best:
            mark = " ← best" if is_best else ""
            print(f"  Epoch {epoch+1:>3}/{args.epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}{mark}")

    print(f"\n  ✅ BC 학습 완료")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  저장: {save_path}")

    # —— 평가 ————————————————————————————————————————————————
    if args.eval:
        print(f"\n  ── 평가 ──")
        policy.load_state_dict(torch.load(save_path, weights_only=True))
        policy.eval()

        with torch.no_grad():
            sample_obs = torch.FloatTensor(obs_data[:1000]).cuda()
            sample_act = torch.FloatTensor(act_data[:1000]).cuda()
            pred = policy(sample_obs)

            mae = (pred - sample_act).abs().mean(dim=0).cpu().numpy()
            names = ["arm0", "arm1", "arm2", "arm3", "arm4", "gripper", "vx", "vy", "wz"]

            print(f"  Action별 MAE:")
            for name, val in zip(names, mae):
                print(f"    {name:>5}: {val:.4f}")

    print(f"\n  다음 단계:")
    print(f"    python train_lekiwi.py --num_envs 2048 --bc_checkpoint {save_path} --headless")


if __name__ == "__main__":
    main()
