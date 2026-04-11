#!/usr/bin/env python3
"""
LeKiwi — State-only ACT (Action Chunking Transformer) BC 학습.

=== 구조 ===

  [Training]
  obs (B, obs_dim) + action_chunk (B, C, act_dim)
       ↓
  ┌─────────────────────────────────────────┐
  │  CVAE Encoder                            │
  │  [CLS | obs_token | a0..a_{C-1}]         │
  │  → TransformerEncoder → CLS output       │
  │  → μ (latent_dim), log_var (latent_dim)  │
  └──────────────────┬──────────────────────┘
                     │  z ~ N(μ, σ²)  (reparameterization)
                     ↓
  ┌─────────────────────────────────────────┐
  │  Decoder                                 │
  │  Memory: [z_token | obs_token]           │
  │  Queries: step_embed[0..C-1]             │
  │  → TransformerDecoder                    │
  │  → Linear → chunk (B, C, act_dim)       │
  └─────────────────────────────────────────┘
  Loss = L1(pred_chunk, gt_chunk) + β * KL(z || N(0,I))

  [Inference]
  z = zeros (prior mean)
  obs → obs_token → Decoder → chunk (C, act_dim)

=== RL Warm-start ===
  checkpoints/bc_act.pt  — ACT 전체 가중치
  policy_act_rl.py의 StateOnlyACT.load_state_dict()로 로드 후
  ACT freeze + log_std + value_head만 PPO로 학습

Usage:
    # Skill-2 (30D)
    python train_bc_act.py --demo_dir demos_skill2/ --epochs 300 --expected_obs_dim 30

    # Skill-3 (29D)
    python train_bc_act.py --demo_dir demos_skill3/ --epochs 300 --expected_obs_dim 29

    # 대규모 데이터 (50+ 에피소드)
    python train_bc_act.py --demo_dir demos_skill2/ --epochs 500 \\
        --expected_obs_dim 30 --d_model 256 --n_layers 4 --batch_size 512
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ═══════════════════════════════════════════════════════════════════════
#  1. 데이터셋 (record_teleop.py HDF5 호환)
# ═══════════════════════════════════════════════════════════════════════

def load_episodes(demo_dir: str) -> list[dict]:
    demo_dir = Path(demo_dir)
    hdf5_files = sorted(demo_dir.glob("*.hdf5")) + sorted(demo_dir.glob("*.h5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files in {demo_dir}")

    episodes: list[dict] = []
    for fpath in hdf5_files:
        n = 0
        with h5py.File(fpath, "r") as f:
            for key in sorted(k for k in f.keys() if k.startswith("episode")):
                grp = f[key]
                ep: dict = {"obs": grp["obs"][:], "actions": grp["actions"][:]}
                if "teleop_active" in grp:
                    ep["teleop_active"] = grp["teleop_active"][:]
                episodes.append(ep)
                n += 1
            if n == 0 and "obs" in f and "actions" in f:
                episodes.append({"obs": f["obs"][:], "actions": f["actions"][:]})
                n = 1
        print(f"  Loaded: {fpath.name}  ({n} episodes)")

    total = sum(len(ep["obs"]) for ep in episodes)
    print(f"  → 총 {len(episodes)} 에피소드,  {total} 스텝\n")
    return episodes


class ACTDataset(Dataset):
    """
    (obs_current, action_chunk) 쌍 생성.
    - obs_current: 현재 관측 (obs_dim,) — Decoder context
    - action_chunk: 현재부터 C스텝 (C, act_dim) — CVAE target
    """

    def __init__(
        self,
        episodes: list[dict],
        chunk_size: int = 20,
        filter_active: bool = False,
    ):
        self.C = chunk_size
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []

        for ep in episodes:
            obs, acts = ep["obs"], ep["actions"]
            if filter_active and "teleop_active" in ep:
                mask = ep["teleop_active"].astype(bool)
                # 연속 구간만 사용
                segs = self._split_contiguous(obs, acts, mask)
            else:
                segs = [(obs, acts)]
            for seg_obs, seg_acts in segs:
                self._build_samples(seg_obs, seg_acts)

        print(f"  [ACTDataset] {len(self.samples)} samples  (C={chunk_size})")

    @staticmethod
    def _split_contiguous(obs, acts, active):
        segs, start = [], None
        for t in range(len(active)):
            if active[t]:
                if start is None:
                    start = t
            else:
                if start is not None and (t - start) >= 2:
                    segs.append((obs[start:t], acts[start:t]))
                start = None
        if start is not None and (len(active) - start) >= 2:
            segs.append((obs[start:], acts[start:]))
        return segs

    def _build_samples(self, obs: np.ndarray, acts: np.ndarray):
        T = len(obs)
        if T < 2:
            return
        for t in range(T):
            chunk = np.stack([acts[min(t + c, T - 1)] for c in range(self.C)])
            self.samples.append((obs[t].copy(), chunk))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, chunk = self.samples[idx]
        return torch.from_numpy(obs).float(), torch.from_numpy(chunk).float()


# ═══════════════════════════════════════════════════════════════════════
#  2. State-only ACT 모델
# ═══════════════════════════════════════════════════════════════════════

class StateOnlyACT(nn.Module):
    """
    이미지 없는 State-only ACT.
    obs (obs_dim,) → Transformer → chunk (C, act_dim)

    Training: CVAE encoder가 action chunk에서 style z 추출
    Inference: z = 0 (prior mean)

    RL warm-start 호환:
      predict_chunk(obs) → (B, C, act_dim)  ← ACT-RL에서 policy mean으로 사용
    """

    def __init__(
        self,
        obs_dim: int = 30,
        act_dim: int = 9,
        chunk_size: int = 20,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.C = chunk_size
        self.d_model = d_model
        self.latent_dim = latent_dim

        # ── CVAE Encoder (training only) ──
        # 입력 토큰: [CLS(1), obs(1), a_0..a_{C-1}(C)]
        self.enc_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.enc_obs_proj = nn.Linear(obs_dim, d_model)
        self.enc_act_proj = nn.Linear(act_dim, d_model)
        self.enc_pos = nn.Parameter(torch.randn(1, 1 + 1 + chunk_size, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.mu_layer = nn.Linear(d_model, latent_dim)
        self.log_var_layer = nn.Linear(d_model, latent_dim)

        # ── Decoder ──
        # Memory: [z_token(1), obs_token(1)]
        # Queries: step_embed[0..C-1] (C tokens)
        self.dec_latent_proj = nn.Linear(latent_dim, d_model)
        self.dec_obs_proj = nn.Linear(obs_dim, d_model)
        self.dec_step_embed = nn.Embedding(chunk_size, d_model)
        self.dec_pos_mem = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        self.dec_pos_q = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.action_head = nn.Linear(d_model, act_dim)

        # 초기화
        nn.init.xavier_uniform_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)
        nn.init.zeros_(self.mu_layer.weight)
        nn.init.zeros_(self.mu_layer.bias)
        nn.init.zeros_(self.log_var_layer.weight)
        nn.init.zeros_(self.log_var_layer.bias)

    # ── CVAE Encoder ──

    def encode(self, obs: torch.Tensor, action_chunk: torch.Tensor):
        """
        CVAE Encoder: (obs, action_chunk) → (μ, log_var)

        Args:
            obs:          (B, obs_dim)
            action_chunk: (B, C, act_dim)
        Returns:
            mu, log_var: each (B, latent_dim)
        """
        B = obs.size(0)
        cls_tok = self.enc_cls.expand(B, -1, -1)            # (B, 1, d_model)
        obs_tok = self.enc_obs_proj(obs).unsqueeze(1)        # (B, 1, d_model)
        act_tok = self.enc_act_proj(action_chunk)            # (B, C, d_model)
        tokens = torch.cat([cls_tok, obs_tok, act_tok], dim=1)  # (B, 1+1+C, d_model)
        tokens = tokens + self.enc_pos[:, :tokens.size(1)]

        encoded = self.encoder(tokens)                       # (B, 1+1+C, d_model)
        cls_out = encoded[:, 0]                              # (B, d_model)
        mu = self.mu_layer(cls_out)
        log_var = self.log_var_layer(cls_out)
        return mu, log_var

    # ── Decoder ──

    def decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Decoder: (obs, z) → chunk

        Args:
            obs: (B, obs_dim)
            z:   (B, latent_dim)
        Returns:
            chunk: (B, C, act_dim)
        """
        B = obs.size(0)
        z_tok = self.dec_latent_proj(z).unsqueeze(1)         # (B, 1, d_model)
        obs_tok = self.dec_obs_proj(obs).unsqueeze(1)        # (B, 1, d_model)
        memory = torch.cat([z_tok, obs_tok], dim=1)          # (B, 2, d_model)
        memory = memory + self.dec_pos_mem

        step_ids = torch.arange(self.C, device=obs.device)
        queries = self.dec_step_embed(step_ids).unsqueeze(0).expand(B, -1, -1)  # (B, C, d_model)
        queries = queries + self.dec_pos_q

        decoded = self.decoder(tgt=queries, memory=memory)   # (B, C, d_model)
        return self.action_head(decoded)                      # (B, C, act_dim)

    # ── Forward (Training) ──

    def forward(self, obs: torch.Tensor, action_chunk: torch.Tensor):
        """
        Training forward.

        Returns:
            pred_chunk: (B, C, act_dim)
            mu:         (B, latent_dim)
            log_var:    (B, latent_dim)
        """
        mu, log_var = self.encode(obs, action_chunk)
        # Reparameterization
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        pred_chunk = self.decode(obs, z)
        return pred_chunk, mu, log_var

    # ── Inference ──

    @torch.no_grad()
    def predict_chunk(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Inference: z=0 (prior mean) → chunk.

        Args:  obs (B, obs_dim)  or  (obs_dim,)
        Returns: chunk (B, C, act_dim)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.size(0)
        z = torch.zeros(B, self.latent_dim, device=obs.device, dtype=obs.dtype)
        return self.decode(obs, z)

    @torch.no_grad()
    def predict_first_action(self, obs: torch.Tensor) -> torch.Tensor:
        """첫 번째 action만 반환 (B, act_dim)."""
        return self.predict_chunk(obs)[:, 0]


# ═══════════════════════════════════════════════════════════════════════
#  3. Loss
# ═══════════════════════════════════════════════════════════════════════

def act_loss(
    pred_chunk: torch.Tensor,
    gt_chunk: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
    step_decay: float = 0.1,
    chunk_size: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ACT Loss = L1(reconstruction) + β * KL

    Step-wise weight: 가까운 미래 스텝에 높은 가중치.
    """
    device = pred_chunk.device
    step_w = torch.exp(-step_decay * torch.arange(chunk_size, dtype=torch.float32, device=device))
    step_w = step_w / step_w.sum()

    # L1 per step
    per_step_l1 = (pred_chunk - gt_chunk).abs().mean(dim=(0, 2))  # (C,)
    l1 = (per_step_l1 * step_w).sum()

    # KL divergence: -0.5 * Σ(1 + log_var - μ² - σ²)
    kl = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()

    loss = l1 + beta * kl
    return loss, l1, kl


# ═══════════════════════════════════════════════════════════════════════
#  4. Temporal Ensemble (eval_bc 호환)
# ═══════════════════════════════════════════════════════════════════════

class TemporalEnsemble:
    """여러 시점의 chunk 예측을 가중 평균 → 부드러운 action."""

    def __init__(self, act_dim: int, chunk_size: int, decay: float = 0.1):
        self.act_dim = act_dim
        self.C = chunk_size
        self.weights = torch.exp(-decay * torch.arange(chunk_size, dtype=torch.float32))
        self._buffer: list[torch.Tensor] = []

    def reset(self):
        self._buffer.clear()

    def update(self, chunk: torch.Tensor) -> torch.Tensor:
        """chunk: (C, act_dim) → (act_dim,)"""
        self._buffer.append(chunk.detach().cpu())
        weighted_sum = torch.zeros(self.act_dim)
        weight_sum = 0.0
        alive = []
        for i, buf in enumerate(self._buffer):
            offset = len(self._buffer) - 1 - i
            if offset >= self.C:
                continue
            alive.append(buf)
            w = self.weights[offset].item()
            weighted_sum += w * buf[offset]
            weight_sum += w
        self._buffer = alive
        return weighted_sum / max(weight_sum, 1e-8)


# ═══════════════════════════════════════════════════════════════════════
#  5. Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser(description="LeKiwi — State-only ACT BC")
    pa.add_argument("--demo_dir", type=str, default="demos/")
    pa.add_argument("--epochs", type=int, default=300)
    pa.add_argument("--batch_size", type=int, default=256)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--weight_decay", type=float, default=1e-4)
    pa.add_argument("--train_split", type=float, default=0.9)
    pa.add_argument("--save_dir", type=str, default="checkpoints/")
    pa.add_argument("--expected_obs_dim", type=int, required=True)
    pa.add_argument("--expected_act_dim", type=int, default=9)
    pa.add_argument("--chunk_size", type=int, default=20)
    pa.add_argument("--d_model", type=int, default=128,
                     help="Transformer hidden dim (128=소량, 256=50+에피소드)")
    pa.add_argument("--n_layers", type=int, default=3)
    pa.add_argument("--n_heads", type=int, default=4)
    pa.add_argument("--latent_dim", type=int, default=32)
    pa.add_argument("--beta", type=float, default=1.0,
                     help="KL weight (β-VAE). 낮을수록 reconstruction 중심)")
    pa.add_argument("--step_decay", type=float, default=0.1,
                     help="chunk step-wise weight decay (가까운 미래 우선)")
    pa.add_argument("--dropout", type=float, default=0.1)
    pa.add_argument("--filter_active", action="store_true")
    pa.add_argument("--eval", action="store_true")
    args = pa.parse_args()

    C = args.chunk_size

    print("\n" + "=" * 60)
    print("  LeKiwi — State-only ACT BC")
    print("=" * 60)
    print(f"  d_model:    {args.d_model}")
    print(f"  n_layers:   {args.n_layers}  (encoder + decoder 각각)")
    print(f"  n_heads:    {args.n_heads}")
    print(f"  latent_dim: {args.latent_dim}")
    print(f"  chunk_size: {C}")
    print(f"  beta (KL):  {args.beta}")

    # ── 에피소드 로드 ──
    episodes = load_episodes(args.demo_dir)
    obs_dim = episodes[0]["obs"].shape[-1]
    act_dim = episodes[0]["actions"].shape[-1]
    assert obs_dim == args.expected_obs_dim, f"obs_dim mismatch: {obs_dim} vs {args.expected_obs_dim}"
    assert act_dim == args.expected_act_dim, f"act_dim mismatch: {act_dim} vs {args.expected_act_dim}"

    # ── 데이터셋 ──
    ds = ACTDataset(episodes, chunk_size=C, filter_active=args.filter_active)
    n_train = max(int(len(ds) * args.train_split), 1)
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size) if n_val > 0 else None

    # ── 모델 ──
    model = StateOnlyACT(
        obs_dim=obs_dim,
        act_dim=act_dim,
        chunk_size=C,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    ).cuda()

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = (sum(p.numel() for p in model.encoder.parameters()) +
             sum(p.numel() for p in model.mu_layer.parameters()) +
             sum(p.numel() for p in model.log_var_layer.parameters()))
    n_dec = n_params - n_enc
    print(f"\n  Parameters: {n_params:,} total")
    print(f"    Encoder: {n_enc:,}  Decoder: {n_dec:,}")
    print(f"  Train: {n_train},  Val: {n_val}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    p_full = save_dir / "bc_act.pt"
    p_cfg = save_dir / "bc_act_config.pt"

    best_val = float("inf")

    # ── 학습 루프 ──
    for epoch in range(args.epochs):
        model.train()
        s_loss, s_l1, s_kl, n_batch = 0.0, 0.0, 0.0, 0

        for obs_b, chunk_b in train_dl:
            obs_b, chunk_b = obs_b.cuda(), chunk_b.cuda()
            pred, mu, log_var = model(obs_b, chunk_b)
            loss, l1, kl = act_loss(pred, chunk_b, mu, log_var,
                                     beta=args.beta,
                                     step_decay=args.step_decay,
                                     chunk_size=C)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            s_loss += loss.item(); s_l1 += l1.item(); s_kl += kl.item(); n_batch += 1

        scheduler.step()
        avg_loss = s_loss / max(n_batch, 1)
        avg_l1 = s_l1 / max(n_batch, 1)
        avg_kl = s_kl / max(n_batch, 1)

        # Validation
        model.eval()
        v_loss = avg_loss
        if val_dl is not None:
            v_sum, v_n = 0.0, 0
            with torch.no_grad():
                for obs_b, chunk_b in val_dl:
                    obs_b, chunk_b = obs_b.cuda(), chunk_b.cuda()
                    pred, mu, lv = model(obs_b, chunk_b)
                    vl, _, _ = act_loss(pred, chunk_b, mu, lv,
                                        beta=args.beta, step_decay=args.step_decay, chunk_size=C)
                    v_sum += vl.item(); v_n += 1
            v_loss = v_sum / max(v_n, 1)

        is_best = v_loss < best_val
        if is_best:
            best_val = v_loss
            torch.save(model.state_dict(), p_full)
            torch.save({
                "obs_dim": obs_dim, "act_dim": act_dim, "chunk_size": C,
                "d_model": args.d_model, "n_heads": args.n_heads,
                "n_layers": args.n_layers, "latent_dim": args.latent_dim,
                "dropout": args.dropout,
            }, p_cfg)

        if (epoch + 1) % 20 == 0 or epoch == 0 or is_best:
            lr = optimizer.param_groups[0]["lr"]
            mark = " ★" if is_best else ""
            print(f"  Epoch {epoch+1:>3}/{args.epochs}  "
                  f"loss={avg_loss:.4f}  L1={avg_l1:.4f}  KL={avg_kl:.4f}  "
                  f"val={v_loss:.4f}  lr={lr:.1e}{mark}")

    print(f"\n  ── 학습 완료 ──")
    print(f"  Best val: {best_val:.4f}")
    print(f"  ACT checkpoint: {p_full}")
    print(f"  Config:         {p_cfg}")

    # ── 평가 ──
    if args.eval:
        print(f"\n  ── 평가 ──")
        model.load_state_dict(torch.load(p_full, weights_only=True))
        model.eval()

        n_eval = min(1000, len(ds))
        e_obs = torch.stack([ds[i][0] for i in range(n_eval)]).cuda()
        e_chunk = torch.stack([ds[i][1] for i in range(n_eval)]).cuda()
        names = ["arm0", "arm1", "arm2", "arm3", "arm4", "gripper", "vx", "vy", "wz"]

        with torch.no_grad():
            pred_chunk = model.predict_chunk(e_obs)  # z=0 (inference)
            first_mae = (pred_chunk[:, 0] - e_chunk[:, 0]).abs().mean(0).cpu().numpy()
            print(f"\n  [z=0 inference] First-action MAE:")
            for name, val in zip(names, first_mae):
                print(f"    {name:>8}: {val:.4f}")
            print(f"  총 MAE: {first_mae.sum():.4f}")

    print(f"\n  ── 다음 단계 (RL fine-tuning) ──")
    print(f"    python train_lekiwi_act.py \\")
    print(f"        --act_checkpoint {p_full} \\")
    print(f"        --act_config {p_cfg} \\")
    print(f"        --num_envs 2048 --headless")


if __name__ == "__main__":
    main()
