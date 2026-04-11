"""
LeKiwi — ACT-RL Policy 래퍼.

Stanford CS224R 2025 논문 "ACT-RL" 구현:
  - ACT encoder-decoder 전체 freeze
  - ACT decoder 출력 = Gaussian policy의 mean
  - 학습 파라미터: log_std (act_dim,) + value_head MLP
  - Chunk-level PPO: C스텝 실행 후 reward 합산, PPO 업데이트

사용법:
    # 1. ACT checkpoint 로드
    from train_bc_act import StateOnlyACT
    from policy_act_rl import ACTActorCritic, ChunkRolloutWrapper

    cfg = torch.load("checkpoints/bc_act_config.pt")
    act = StateOnlyACT(**{k: cfg[k] for k in
                          ["obs_dim","act_dim","chunk_size","d_model",
                           "n_heads","n_layers","latent_dim","dropout"]})
    act.load_state_dict(torch.load("checkpoints/bc_act.pt"))

    policy = ACTActorCritic(act, value_hidden=256)

    # 2. skrl PPO에 연결 (train_lekiwi.py 수정)
    #    - policy.parameters() → log_std + value_head만 최적화
    #    - ChunkRolloutWrapper로 환경 wrapping

=== 핵심 설계 ===

  obs (B, obs_dim)
       ↓ frozen ACT
  chunk_mean (B, C, act_dim)
       ↓
  action = chunk_mean[:, step_in_chunk] + noise(log_std)
       ↓ C스텝 실행
  reward_sum = Σ r_t (chunk 단위 합산)
       ↓ PPO 업데이트 (frozen ACT, only log_std + value_head)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ═══════════════════════════════════════════════════════════════════════
#  ACT Actor-Critic (skrl 호환 인터페이스)
# ═══════════════════════════════════════════════════════════════════════

class ACTActorCritic(nn.Module):
    """
    Frozen ACT + 학습 가능한 log_std + value_head.

    skrl GaussianMixin / DeterministicMixin 대신 직접 구현.
    PPO optimizer에는 log_std + value_head 파라미터만 전달.

    Args:
        act_model:    학습된 StateOnlyACT (weights freeze됨)
        value_hidden: value head MLP hidden dim
        log_std_init: 초기 exploration magnitude
    """

    def __init__(
        self,
        act_model: nn.Module,
        value_hidden: int = 256,
        log_std_init: float = -2.0,
    ):
        super().__init__()

        # ACT 전체 freeze
        self.act = act_model
        for p in self.act.parameters():
            p.requires_grad = False

        obs_dim = act_model.obs_dim
        act_dim = act_model.act_dim
        self.act_dim = act_dim
        self.chunk_size = act_model.C

        # ── 학습 파라미터 1: exploration (per-dim log std) ──
        self.log_std = nn.Parameter(
            torch.full((act_dim,), log_std_init)
        )

        # ── 학습 파라미터 2: value head ──
        self.value_head = nn.Sequential(
            nn.Linear(obs_dim, value_hidden),
            nn.ELU(),
            nn.Linear(value_hidden, value_hidden // 2),
            nn.ELU(),
            nn.Linear(value_hidden // 2, 1),
        )

        n_trainable = (
            self.log_std.numel()
            + sum(p.numel() for p in self.value_head.parameters())
        )
        n_frozen = sum(p.numel() for p in self.act.parameters())
        print(f"  [ACTActorCritic] Frozen ACT: {n_frozen:,} params")
        print(f"  [ACTActorCritic] Trainable: {n_trainable:,} params "
              f"(log_std={self.log_std.numel()}, value_head={n_trainable - self.log_std.numel()})")

    def trainable_parameters(self):
        """PPO optimizer에 넘길 파라미터 (ACT 제외)."""
        return list(self.value_head.parameters()) + [self.log_std]

    # ── Actor ──

    def get_chunk_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Frozen ACT → chunk mean.

        Args:  obs (B, obs_dim)
        Returns: chunk_mean (B, C, act_dim)
        """
        with torch.no_grad():
            return self.act.predict_chunk(obs)

    def get_action_dist(self, obs: torch.Tensor, step_in_chunk: int = 0):
        """
        특정 chunk step의 Gaussian distribution.

        Args:
            obs:           (B, obs_dim)
            step_in_chunk: 0 ~ C-1
        Returns:
            dist: Normal distribution
            mean: (B, act_dim)
        """
        chunk_mean = self.get_chunk_mean(obs)       # (B, C, act_dim)
        mean = chunk_mean[:, step_in_chunk]          # (B, act_dim)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std), mean

    def get_full_chunk_dist(self, obs: torch.Tensor):
        """
        전체 chunk의 Gaussian distribution (chunk-level PPO용).

        Returns:
            dist: Normal (B, C, act_dim)
            chunk_mean: (B, C, act_dim)
        """
        chunk_mean = self.get_chunk_mean(obs)        # (B, C, act_dim)
        std = self.log_std.exp().unsqueeze(0).unsqueeze(0).expand_as(chunk_mean)
        return Normal(chunk_mean, std), chunk_mean

    def act_and_log_prob(self, obs: torch.Tensor, step_in_chunk: int = 0):
        """
        Action 샘플 + log_prob.

        Returns:
            action:   (B, act_dim) — clipped to [-1, 1]
            log_prob: (B,)
        """
        dist, _ = self.get_action_dist(obs, step_in_chunk)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        action = torch.clamp(action, -1.0, 1.0)
        return action, log_prob

    # ── Critic ──

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Value 추정.

        Args:  obs (B, obs_dim)
        Returns: value (B, 1)
        """
        return self.value_head(obs)

    # ── skrl 호환 인터페이스 ──

    def act(self, inputs: dict, role: str = "") -> tuple:
        """skrl Actor.act() 호환."""
        obs = inputs.get("states", inputs.get("obs"))
        action, log_prob = self.act_and_log_prob(obs)
        return action, log_prob, {}

    def compute(self, inputs: dict, role: str = "") -> tuple:
        """skrl 호환: (action, value, log_prob) 반환."""
        obs = inputs.get("states", inputs.get("obs"))
        action, log_prob = self.act_and_log_prob(obs)
        value = self.get_value(obs)
        return action, value, log_prob


# ═══════════════════════════════════════════════════════════════════════
#  Chunk Rollout Wrapper
# ═══════════════════════════════════════════════════════════════════════

class ChunkRolloutCollector:
    """
    Chunk-level PPO rollout 수집.

    ACT가 C-step chunk를 예측하면:
    1. 같은 obs에서 chunk 전체를 한 번에 예측
    2. C step 실행 → reward 합산
    3. (obs, chunk_mean, reward_sum, done, value)를 PPO buffer에 저장

    사용법:
        collector = ChunkRolloutCollector(policy, env, chunk_size=20)
        transitions = collector.collect(n_chunks=100)
        # transitions: list of (obs, action_chunk, reward_sum, done, value, log_prob)
    """

    def __init__(self, policy: ACTActorCritic, env, chunk_size: int = 20):
        self.policy = policy
        self.env = env
        self.C = chunk_size

    @torch.no_grad()
    def collect_step(self, obs: torch.Tensor) -> tuple:
        """
        1 chunk (C steps) 수행.

        Returns:
            obs:         (B, obs_dim) — chunk 시작 시점 관측
            action_chunk:(B, C, act_dim) — 예측된 chunk
            reward_sum:  (B,) — C step 누적 reward
            done:        (B,) bool
            value:       (B, 1)
            log_prob:    (B,)
        """
        # 1. Chunk 전체 예측 (frozen ACT)
        chunk_mean = self.policy.get_chunk_mean(obs)  # (B, C, act_dim)
        std = self.policy.log_std.exp()
        eps = torch.randn_like(chunk_mean)
        chunk_sampled = torch.clamp(chunk_mean + eps * std, -1.0, 1.0)

        # log_prob: chunk 전체의 합 (chunk-level credit assignment)
        dist = Normal(chunk_mean, std.unsqueeze(0).unsqueeze(0).expand_as(chunk_mean))
        log_prob = dist.log_prob(chunk_sampled).sum(dim=(-1, -2))  # (B,)

        # 2. C step 실행, reward 합산
        reward_sum = torch.zeros(obs.shape[0], device=obs.device)
        done = torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
        next_obs = obs

        for step_i in range(self.C):
            action = chunk_sampled[:, step_i]
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            reward_sum += reward.squeeze(-1) if reward.dim() > 1 else reward
            done = done | terminated.squeeze(-1) | truncated.squeeze(-1)

        # 3. Value 추정 (시작 obs 기준)
        value = self.policy.get_value(obs)

        return obs, chunk_sampled, reward_sum, done, value, log_prob, next_obs


# ═══════════════════════════════════════════════════════════════════════
#  Chunk-level PPO Loss
# ═══════════════════════════════════════════════════════════════════════

def chunk_ppo_loss(
    policy: ACTActorCritic,
    obs: torch.Tensor,
    old_action_chunks: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.1,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> tuple[torch.Tensor, dict]:
    """
    Chunk-level PPO Loss.

    Args:
        obs:               (B, obs_dim)
        old_action_chunks: (B, C, act_dim)
        old_log_probs:     (B,)
        advantages:        (B,)
        returns:           (B,)

    Returns:
        loss: scalar
        info: dict (policy_loss, value_loss, entropy)
    """
    # 새 chunk mean (frozen ACT, no_grad)
    with torch.no_grad():
        chunk_mean = policy.get_chunk_mean(obs)

    std = policy.log_std.exp()
    dist = Normal(chunk_mean, std.unsqueeze(0).unsqueeze(0).expand_as(chunk_mean))
    new_log_probs = dist.log_prob(old_action_chunks).sum(dim=(-1, -2))  # (B,)
    entropy = dist.entropy().sum(dim=(-1, -2)).mean()

    # PPO clipped ratio
    ratio = (new_log_probs - old_log_probs).exp()
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_loss = -torch.min(
        ratio * adv,
        torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv,
    ).mean()

    # Value loss
    values = policy.get_value(obs).squeeze(-1)
    value_loss = F.mse_loss(values, returns)

    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    return loss, {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "approx_kl": ((ratio - 1) - ratio.log()).mean().item(),
    }


# ═══════════════════════════════════════════════════════════════════════
#  ACTActorCritic 로드 헬퍼
# ═══════════════════════════════════════════════════════════════════════

def load_act_actor_critic(
    act_checkpoint: str,
    act_config: str,
    value_hidden: int = 256,
    log_std_init: float = -2.0,
    device: str = "cuda",
) -> ACTActorCritic:
    """
    ACT checkpoint에서 ACTActorCritic 생성.

    Usage:
        policy = load_act_actor_critic(
            "checkpoints/bc_act.pt",
            "checkpoints/bc_act_config.pt",
        )
        # PPO optimizer: only policy.trainable_parameters()
        optimizer = torch.optim.Adam(policy.trainable_parameters(), lr=3e-5)
    """
    import os
    from train_bc_act import StateOnlyACT

    cfg = torch.load(act_config, weights_only=True, map_location=device)
    act = StateOnlyACT(
        obs_dim=cfg["obs_dim"],
        act_dim=cfg["act_dim"],
        chunk_size=cfg["chunk_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        latent_dim=cfg["latent_dim"],
        dropout=cfg.get("dropout", 0.1),
    )
    state_dict = torch.load(act_checkpoint, weights_only=True, map_location=device)
    act.load_state_dict(state_dict)
    act = act.to(device)

    policy = ACTActorCritic(
        act_model=act,
        value_hidden=value_hidden,
        log_std_init=log_std_init,
    ).to(device)

    print(f"  [load_act_actor_critic] Loaded from {os.path.basename(act_checkpoint)}")
    print(f"  [load_act_actor_critic] obs={cfg['obs_dim']} act={cfg['act_dim']} "
          f"chunk={cfg['chunk_size']} d_model={cfg['d_model']}")
    return policy
