#!/usr/bin/env python3
"""
LeKiwi Navigation — Behavioral Cloning 학습.

record_teleop.py가 수집한 HDF5 데모에서 (obs, action) 쌍을 추출하여 supervised learning.
네트워크 구조가 train_lekiwi.py의 PolicyNet.net + mean_layer과 **동일**해야
BC 가중치를 PPO Actor에 그대로 로드할 수 있음.

Usage:
    cd ~/IsaacLab/scripts/lekiwi_nav_env

    # MSE (기존)
    python train_bc.py --demo_dir demos_skill2/ --epochs 200 --expected_obs_dim 30

    # GMM (권장 — mean regression 방지)
    python train_bc.py --demo_dir demos_skill2/ --epochs 200 --expected_obs_dim 30 --loss gmm

결과:
    checkpoints/bc_nav.pt          — BC 가중치 (net + mean_layer, RL warm-start 호환)
    checkpoints/bc_nav_gmm.pt      — GMM 전체 가중치 (--loss gmm 사용 시)
    checkpoints/bc_nav_norm.npz    — obs 정규화 파라미터 (mean, std, --normalize 사용 시)

다음 단계:
    python train_lekiwi.py --num_envs 2048 --bc_checkpoint checkpoints/bc_nav.pt --headless
"""
from __future__ import annotations

import argparse
import math
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, random_split


# ═══════════════════════════════════════════════════════════════════════
#  BC Policy — MSE (train_lekiwi.py PolicyNet과 동일한 구조)
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
#  BC Policy — GMM (Gaussian Mixture Model)
# ═══════════════════════════════════════════════════════════════════════

class BCPolicyGMM(nn.Module):
    """
    Mixture Density Network: K개의 Gaussian 컴포넌트로 다봉 분포 학습.

    MSE의 mean regression 문제 해결:
    - MSE: 극단값(-0.997) → 평균 쪽으로 축소(-0.75)
    - GMM: 각 컴포넌트가 서로 다른 행동 모드를 학습 → 극단값도 보존

    backbone (net)은 BCPolicy와 동일 → RL warm-start 호환.
    """
    def __init__(self, obs_dim: int = 33, act_dim: int = 9, n_components: int = 5):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_components = n_components

        # backbone: BCPolicy/PolicyNet과 동일
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        # GMM heads
        self.pi_layer = nn.Linear(64, n_components)
        self.mu_layer = nn.Linear(64, n_components * act_dim)
        self.log_std_layer = nn.Linear(64, n_components * act_dim)

        # mu, log_std 초기화: 작은 값으로 시작
        nn.init.zeros_(self.log_std_layer.bias)
        nn.init.xavier_uniform_(self.mu_layer.weight, gain=0.1)

    def forward(self, obs: torch.Tensor):
        """Returns (pi_logits, mu, log_std)."""
        h = self.net(obs)
        pi_logits = self.pi_layer(h)                                          # (B, K)
        mu = self.mu_layer(h).view(-1, self.n_components, self.act_dim)       # (B, K, D)
        log_std = self.log_std_layer(h).view(-1, self.n_components, self.act_dim)
        log_std = torch.clamp(log_std, -5.0, 2.0)                            # (B, K, D)
        return pi_logits, mu, log_std

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        """Inference: 가장 가중치 높은 컴포넌트의 mean 반환."""
        pi_logits, mu, _ = self.forward(obs)
        best_k = pi_logits.argmax(dim=-1)  # (B,)
        idx = best_k.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.act_dim)
        return mu.gather(1, idx).squeeze(1)  # (B, D)

    def extract_bc_state_dict(self, obs_sample: torch.Tensor | None = None) -> dict:
        """
        BCPolicy 호환 state dict 추출 (RL warm-start용).

        backbone (net) 그대로 복사 + 가장 많이 선택되는 컴포넌트의
        mu weights를 mean_layer로 변환.
        """
        sd = {}
        # backbone 복사
        for k, v in self.net.state_dict().items():
            sd[f"net.{k}"] = v.clone()

        # dominant component 찾기
        if obs_sample is not None and len(obs_sample) > 0:
            with torch.no_grad():
                pi_logits, _, _ = self.forward(obs_sample)
                best_k = pi_logits.argmax(dim=-1)
                dominant_k = int(best_k.mode().values.item())
        else:
            dominant_k = 0

        # mu_layer에서 dominant component의 weights 추출
        K, D = self.n_components, self.act_dim
        w = self.mu_layer.weight.data.view(K, D, -1)  # (K, D, 64)
        b = self.mu_layer.bias.data.view(K, D)         # (K, D)
        sd["mean_layer.weight"] = w[dominant_k].clone()  # (D, 64)
        sd["mean_layer.bias"] = b[dominant_k].clone()    # (D,)

        return sd


def gmm_nll_loss(pi_logits: torch.Tensor, mu: torch.Tensor,
                 log_std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    GMM Negative Log-Likelihood.

    Args:
        pi_logits: (B, K) mixture weight logits
        mu: (B, K, D) component means
        log_std: (B, K, D) component log standard deviations
        target: (B, D) ground truth actions

    Returns:
        scalar loss (mean NLL)
    """
    target = target.unsqueeze(1)  # (B, 1, D)
    var = torch.exp(2 * log_std) + 1e-6
    # per-component log prob
    log_normal = -0.5 * (((target - mu) ** 2) / var + 2 * log_std + math.log(2 * math.pi))
    log_normal = log_normal.sum(dim=-1)     # (B, K) sum over action dims
    # mixture log prob
    log_pi = F.log_softmax(pi_logits, dim=-1)  # (B, K)
    log_mixture = torch.logsumexp(log_pi + log_normal, dim=-1)  # (B,)
    return -log_mixture.mean()


# ═══════════════════════════════════════════════════════════════════════
#  데모 로드
# ═══════════════════════════════════════════════════════════════════════

def load_demos(demo_dir: str, filter_active: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    record_teleop.py가 생성한 HDF5에서 (obs, action) 추출.

    Expected HDF5 구조:
        /episode_0/obs            → (T, obs_dim)
        /episode_0/actions        → (T, 9)
        /episode_0/teleop_active  → (T,) int8, optional
        /episode_1/...

    Args:
        demo_dir: HDF5 파일이 있는 디렉토리 경로
        filter_active: True면 teleop_active=1인 스텝만 사용 (idle 프레임 제거)
    """
    demo_dir = Path(demo_dir)
    hdf5_files = sorted(demo_dir.glob("*.hdf5")) + sorted(demo_dir.glob("*.h5"))

    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {demo_dir}")

    obs_list, act_list = [], []
    total_filtered = 0

    for fpath in hdf5_files:
        file_episode_count = 0
        with h5py.File(fpath, "r") as f:
            episode_keys = sorted([k for k in f.keys() if k.startswith("episode")])

            if episode_keys:
                for key in episode_keys:
                    grp = f[key]
                    obs_data = grp["obs"][:]
                    act_data = grp["actions"][:]
                    if filter_active and "teleop_active" in grp:
                        mask = grp["teleop_active"][:].astype(bool)
                        n_before = len(obs_data)
                        obs_data = obs_data[mask]
                        act_data = act_data[mask]
                        total_filtered += n_before - len(obs_data)
                    obs_list.append(obs_data)
                    act_list.append(act_data)
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
    if filter_active and total_filtered > 0:
        print(f"  (idle 프레임 {total_filtered}개 제거됨)")
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
    parser.add_argument("--loss", type=str, default="gmm", choices=["mse", "gmm"],
                        help="손실 함수: mse(기존) 또는 gmm(권장, mean regression 방지)")
    parser.add_argument("--n_components", type=int, default=5,
                        help="GMM 컴포넌트 수 (--loss gmm 시)")
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument("--normalize", action="store_true",
                            help="obs 정규화 적용 (기본: 비활성)")
    norm_group.add_argument("--no_normalize", action="store_true",
                            help="deprecated: obs 정규화 비활성화 (기본값과 동일)")
    parser.add_argument("--eval", action="store_true",
                        help="학습 후 간단한 평가")
    parser.add_argument("--filter_active", action="store_true",
                        help="teleop_active=True 프레임만 사용 (idle 제거, 텔레옵 데이터 권장)")
    args = parser.parse_args()

    use_gmm = (args.loss == "gmm")

    print("\n" + "=" * 60)
    print(f"  LeKiwi Nav — Behavioral Cloning ({'GMM' if use_gmm else 'MSE'})")
    print("=" * 60)

    # —— 데모 로드 ————————————————————————————————————————————
    obs_data, act_data = load_demos(args.demo_dir, filter_active=args.filter_active)
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
            norm_path.unlink()
            print(f"\n  Obs 정규화 미적용 (기본). stale norm 파일 제거: {norm_path}")
        else:
            print("\n  Obs 정규화 미적용 (기본, PPO RunningStandardScaler와 정합)")

    # —— 데이터셋 ————————————————————————————————————————————
    obs_tensor = torch.FloatTensor(obs_data)
    act_tensor = torch.FloatTensor(act_data)
    dataset = TensorDataset(obs_tensor, act_tensor)
    train_size = max(int(len(dataset) * args.train_split), 1)
    train_size = min(train_size, len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if val_size > 0 else None

    # —— 모델 ————————————————————————————————————————————————
    if use_gmm:
        model = BCPolicyGMM(obs_dim, act_dim, n_components=args.n_components).cuda()
        loss_str = f"GMM NLL (K={args.n_components})"
    else:
        model = BCPolicy(obs_dim, act_dim).cuda()
        loss_str = "MSE"

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(f"\n  Network: {obs_dim} → [256, 128, 64] → {act_dim}")
    print(f"  Loss: {loss_str}")
    print(f"  Train: {train_size} steps, Val: {val_size} steps")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print()

    # —— 학습 루프 ————————————————————————————————————————————
    best_val_loss = float("inf")
    save_path_rl = save_dir / "bc_nav.pt"        # RL warm-start용 (BCPolicy 호환)
    save_path_gmm = save_dir / "bc_nav_gmm.pt"   # GMM 전체 (eval_bc용)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for obs_b, act_b in train_loader:
            obs_b, act_b = obs_b.cuda(), act_b.cuda()

            if use_gmm:
                pi, mu, log_std = model(obs_b)
                loss = gmm_nll_loss(pi, mu, log_std, act_b)
            else:
                pred = model(obs_b)
                loss = F.mse_loss(pred, act_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        if val_loader is not None and len(val_loader) > 0:
            val_loss = 0.0
            with torch.no_grad():
                for obs_b, act_b in val_loader:
                    obs_b, act_b = obs_b.cuda(), act_b.cuda()
                    if use_gmm:
                        pi, mu, log_std = model(obs_b)
                        val_loss += gmm_nll_loss(pi, mu, log_std, act_b).item()
                    else:
                        val_loss += F.mse_loss(model(obs_b), act_b).item()
            val_loss /= len(val_loader)
        else:
            val_loss = train_loss

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            if use_gmm:
                # GMM 전체 저장
                torch.save(model.state_dict(), save_path_gmm)
                # RL warm-start 호환 state dict 추출
                obs_sample = obs_tensor[:2000].cuda()
                rl_sd = model.extract_bc_state_dict(obs_sample)
                torch.save(rl_sd, save_path_rl)
            else:
                torch.save(model.state_dict(), save_path_rl)

        if (epoch + 1) % 20 == 0 or epoch == 0 or is_best:
            mark = " *" if is_best else ""
            print(f"  Epoch {epoch+1:>3}/{args.epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}{mark}")

    print(f"\n  BC 학습 완료")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  RL warm-start: {save_path_rl}")
    if use_gmm:
        print(f"  GMM full:      {save_path_gmm}")

    # —— 평가 ————————————————————————————————————————————————
    if args.eval:
        print(f"\n  ── 평가 ──")
        if use_gmm:
            model.load_state_dict(torch.load(save_path_gmm, weights_only=True))
        else:
            model.load_state_dict(torch.load(save_path_rl, weights_only=True))
        model.eval()

        with torch.no_grad():
            sample_obs = torch.FloatTensor(obs_data[:1000]).cuda()
            sample_act = torch.FloatTensor(act_data[:1000]).cuda()

            if use_gmm:
                pred = model.predict(sample_obs)
            else:
                pred = model(sample_obs)

            mae = (pred - sample_act).abs().mean(dim=0).cpu().numpy()
            names = ["arm0", "arm1", "arm2", "arm3", "arm4", "gripper", "vx", "vy", "wz"]

            print(f"  Action별 MAE:")
            for name, val in zip(names, mae):
                print(f"    {name:>8}: {val:.4f}")

    print(f"\n  다음 단계:")
    print(f"    python train_lekiwi.py --num_envs 2048 --bc_checkpoint {save_path_rl} --headless")


if __name__ == "__main__":
    main()
