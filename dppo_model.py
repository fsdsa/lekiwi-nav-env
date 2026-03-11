"""
DPPO Model classes for LeKiwi.

Standalone module — no Isaac Lab or AppLauncher dependencies.
Imported by both train_dppo.py and eval_dppo.py.

Contains:
  - cosine_beta_schedule: noise schedule computation
  - CriticMLP: value function network
  - DPPODiffusion: 2-layer MDP (environment × denoising) model
"""
from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from diffusion_policy import LinearNormalizer


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule — matches DPPO official & diffusers squaredcos_cap_v2."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)


class RunningRewardScaler:
    """Per-env running reward normalization (from DPPO official)."""
    def __init__(self, num_envs, gamma=0.99, epsilon=1e-8):
        import numpy as np
        self.num_envs = num_envs
        self.gamma = gamma
        self.epsilon = epsilon
        self.running_sum = np.zeros(num_envs)
        self.running_sumsq = np.zeros(num_envs)
        self.count = 0

    def __call__(self, reward, first):
        import numpy as np
        self.running_sum = self.running_sum * self.gamma * (1 - first) + reward
        self.running_sumsq = self.running_sumsq * self.gamma**2 * (1 - first) + reward**2
        self.count += 1
        mean = np.mean(self.running_sum)
        var = max(np.mean(self.running_sumsq) - mean**2, 0)
        std = max(np.sqrt(var), self.epsilon)
        return reward / std


# ═══════════════════════════════════════════════════════════════════════════════
#  Critic
# ═══════════════════════════════════════════════════════════════════════════════

class CriticMLP(nn.Module):
    """State-only value function (matches DPPO CriticObs)."""

    def __init__(self, input_dim, hidden_dims=(256, 256, 256)):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Mish())
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  DPPODiffusion — Core DPPO model
# ═══════════════════════════════════════════════════════════════════════════════

class DPPODiffusion(nn.Module):
    """
    DPPO: 2-layer MDP (environment × denoising).

    Wraps a pre-trained ConditionalUnet1D with:
      - actor (frozen original) + actor_ft (fine-tuned copy)
      - DDIM denoising with explicit chain tracking
      - Per-denoising-step Gaussian logprob computation
      - Separate critic MLP for value estimation
    """

    def __init__(
        self,
        unet_pretrained: nn.Module,
        normalizer: LinearNormalizer,
        obs_dim: int,
        act_dim: int,
        pred_horizon: int = 16,
        act_steps: int = 8,
        denoising_steps: int = 100,
        ddim_steps: int = 4,
        ft_denoising_steps: int = 4,
        min_sampling_denoising_std: float = 0.08,
        min_logprob_denoising_std: float = 0.08,
        gamma_denoising: float = 0.9,
        clip_ploss_coef: float = 0.01,
        clip_ploss_coef_base: float = 0.001,
        clip_ploss_coef_rate: float = 3.0,
        denoised_clip_value: float = 1.0,
        randn_clip_value: float = 3.0,
        final_action_clip_value: float = 1.0,
        eta: float = 1.0,
        norm_adv: bool = True,
        critic_dims: tuple = (256, 256, 256),
        device: str = "cuda:0",
    ):
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pred_horizon = pred_horizon
        self.act_steps = act_steps
        self.denoising_steps = denoising_steps
        self.ddim_steps = ddim_steps
        self.ft_denoising_steps = ft_denoising_steps
        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.min_logprob_denoising_std = min_logprob_denoising_std
        self.gamma_denoising = gamma_denoising
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate
        self.denoised_clip_value = denoised_clip_value
        self.randn_clip_value = randn_clip_value
        self.final_action_clip_value = final_action_clip_value
        self.eta_val = eta
        self.norm_adv = norm_adv

        assert ft_denoising_steps <= ddim_steps

        # Frozen actor (original BC UNet) — no gradients
        self.actor = copy.deepcopy(unet_pretrained)
        for p in self.actor.parameters():
            p.requires_grad = False

        # Fine-tuned actor — receives PPO gradients
        self.actor_ft = copy.deepcopy(unet_pretrained)

        # Normalizer (frozen)
        self.normalizer = normalizer
        for p in self.normalizer.parameters():
            p.requires_grad = False

        # Critic
        self.critic = CriticMLP(obs_dim, critic_dims)

        # DDIM parameters
        self._setup_ddim(denoising_steps, ddim_steps)

        n_actor = sum(p.numel() for p in self.actor_ft.parameters() if p.requires_grad)
        n_critic = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        print(f"  [DPPO] actor_ft params: {n_actor:,}")
        print(f"  [DPPO] critic params: {n_critic:,}")
        print(f"  [DPPO] DDIM: {ddim_steps} steps, ft={ft_denoising_steps}, eta={eta}")

    def _setup_ddim(self, denoising_steps, ddim_steps):
        """Compute DDIM parameters from cosine beta schedule."""
        betas = cosine_beta_schedule(denoising_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # DDIM timesteps (uniform spacing, then flip for high→low noise)
        step_ratio = denoising_steps // ddim_steps
        ddim_t = torch.arange(0, ddim_steps) * step_ratio
        ddim_alphas = alphas_cumprod[ddim_t].float()
        ddim_alphas_prev = torch.cat([
            torch.tensor([1.0]),
            alphas_cumprod[ddim_t[:-1]],
        ]).float()
        ddim_sqrt_one_minus_alphas = (1.0 - ddim_alphas).sqrt()

        # Sigma: η * sqrt((1-α_{t-1})/(1-α_t) * (1 - α_t/α_{t-1}))
        sigma_sq = (
            (1 - ddim_alphas_prev) / (1 - ddim_alphas)
            * (1 - ddim_alphas / ddim_alphas_prev)
        )
        ddim_sigmas = self.eta_val * torch.sqrt(torch.clamp(sigma_sq, min=1e-20))

        # Flip all: denoising goes from highest noise to lowest
        self.register_buffer("ddim_t", torch.flip(ddim_t, [0]).long())
        self.register_buffer("ddim_alphas", torch.flip(ddim_alphas, [0]))
        self.register_buffer("ddim_alphas_prev", torch.flip(ddim_alphas_prev, [0]))
        self.register_buffer("ddim_sqrt_one_minus_alphas",
                             torch.flip(ddim_sqrt_one_minus_alphas, [0]))
        self.register_buffer("ddim_sigmas", torch.flip(ddim_sigmas, [0]))

    # ── Denoising step ──

    def _denoise_step_mean_sigma(self, x, i, nobs, use_ft=True):
        """Single DDIM step: compute mean and sigma of transition distribution."""
        B = x.shape[0]
        t = self.ddim_t[i]
        t_batch = torch.full((B,), t, device=x.device, dtype=torch.long)

        actor = self.actor_ft if use_ft else self.actor
        noise_pred = actor(x, t_batch, global_cond=nobs)

        alpha = self.ddim_alphas[i]
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alphas[i]
        x_recon = (x - sqrt_one_minus_alpha * noise_pred) / (alpha ** 0.5)
        x_recon = x_recon.clamp(-self.denoised_clip_value, self.denoised_clip_value)

        # Re-derive noise from clamped x0
        noise_pred = (x - (alpha ** 0.5) * x_recon) / sqrt_one_minus_alpha

        alpha_prev = self.ddim_alphas_prev[i]
        sigma_base = self.ddim_sigmas[i]

        dir_xt = torch.clamp(1.0 - alpha_prev - sigma_base ** 2, min=0).sqrt() * noise_pred
        mu = (alpha_prev ** 0.5) * x_recon + dir_xt

        return mu, sigma_base

    # ── Rollout: sample actions with chain tracking ──

    @torch.no_grad()
    def sample_actions(self, obs, deterministic=False):
        """
        Run DDIM denoising, return action chunk + chain.

        Args:
            obs: (B, obs_dim) raw observations
            deterministic: if True, sigma=0 (eval mode)

        Returns:
            actions_norm: (B, pred_horizon, act_dim) in normalized [-1,1] space
            chain: (B, ft_denoising_steps+1, pred_horizon, act_dim)
        """
        B = obs.shape[0]
        nobs = self.normalizer(obs, "obs", forward=True)

        x = torch.randn(B, self.pred_horizon, self.act_dim, device=obs.device)
        chain = []

        for i in range(self.ddim_steps):
            is_ft_step = (i >= self.ddim_steps - self.ft_denoising_steps)

            if is_ft_step and len(chain) == 0:
                chain.append(x.clone())

            mu, sigma_base = self._denoise_step_mean_sigma(
                x, i, nobs, use_ft=is_ft_step)

            if deterministic:
                sigma = torch.zeros_like(mu)
            else:
                sigma = max(sigma_base.item(), self.min_sampling_denoising_std)
                sigma = torch.full_like(mu, sigma)

            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value)
            x = mu + sigma * noise

            if i == self.ddim_steps - 1 and self.final_action_clip_value is not None:
                x = x.clamp(-self.final_action_clip_value, self.final_action_clip_value)

            if is_ft_step:
                chain.append(x.clone())

        chain = torch.stack(chain, dim=1)  # (B, K+1, T, D)
        return x, chain

    # ── PPO update: logprob computation ──

    def get_logprobs_subsample(self, obs, chains_prev, chains_next, denoising_inds):
        """
        Compute log-probabilities for sampled denoising transitions.

        Args:
            obs: (B, obs_dim) raw observations
            chains_prev: (B, pred_horizon, act_dim) x_t
            chains_next: (B, pred_horizon, act_dim) x_{t-1}
            denoising_inds: (B,) indices [0..K-1]

        Returns:
            logprobs: (B, pred_horizon, act_dim)
        """
        nobs = self.normalizer(obs, "obs", forward=True)
        B = obs.shape[0]

        ddim_offset = self.ddim_steps - self.ft_denoising_steps
        ddim_indices = ddim_offset + denoising_inds

        t_all = self.ddim_t[ddim_indices]
        alpha = self.ddim_alphas[ddim_indices].view(B, 1, 1)
        alpha_prev = self.ddim_alphas_prev[ddim_indices].view(B, 1, 1)
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alphas[ddim_indices].view(B, 1, 1)
        sigma_base = self.ddim_sigmas[ddim_indices].view(B, 1, 1)

        noise_pred = self.actor_ft(chains_prev, t_all, global_cond=nobs)

        x_recon = (chains_prev - sqrt_one_minus_alpha * noise_pred) / (alpha ** 0.5)
        x_recon = x_recon.clamp(-self.denoised_clip_value, self.denoised_clip_value)
        noise_pred = (chains_prev - (alpha ** 0.5) * x_recon) / sqrt_one_minus_alpha

        dir_xt = torch.clamp(1.0 - alpha_prev - sigma_base ** 2, min=0).sqrt() * noise_pred
        mu = (alpha_prev ** 0.5) * x_recon + dir_xt

        std = torch.clamp(sigma_base, min=self.min_logprob_denoising_std)
        dist = Normal(mu, std)
        return dist.log_prob(chains_next).clamp(-5, 2)

    def get_logprobs_all(self, obs, chains):
        """
        Compute logprobs for all ft_denoising_steps in a chain.

        Args:
            obs: (B, obs_dim)
            chains: (B, K+1, pred_horizon, act_dim)

        Returns:
            logprobs: (B, K, pred_horizon, act_dim)
        """
        B = obs.shape[0]
        K = self.ft_denoising_steps

        obs_rep = obs.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        chains_prev = chains[:, :-1].reshape(B * K, self.pred_horizon, self.act_dim)
        chains_next = chains[:, 1:].reshape(B * K, self.pred_horizon, self.act_dim)
        denoising_inds = torch.arange(K, device=obs.device).unsqueeze(0).expand(B, -1).reshape(-1)

        logprobs_flat = self.get_logprobs_subsample(
            obs_rep, chains_prev, chains_next, denoising_inds)
        return logprobs_flat.reshape(B, K, self.pred_horizon, self.act_dim)

    def get_value(self, obs):
        nobs = self.normalizer(obs, "obs", forward=True)
        return self.critic(nobs)

    def loss_ppo(
        self, obs, chains_prev, chains_next, denoising_inds,
        returns, oldvalues, advantages, oldlogprobs,
        reward_horizon=8,
    ):
        """
        PPO loss with DPPO modifications.

        Returns: (pg_loss, v_loss, approx_kl, clipfrac)
        """
        newlogprobs = self.get_logprobs_subsample(
            obs, chains_prev, chains_next, denoising_inds)

        newlogprobs = newlogprobs[:, :reward_horizon, :]
        oldlogprobs = oldlogprobs[:, :reward_horizon, :]

        newlp = newlogprobs.mean(dim=(-1, -2))
        oldlp = oldlogprobs.mean(dim=(-1, -2))

        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Denoising discount
        discount = torch.tensor([
            self.gamma_denoising ** (self.ft_denoising_steps - i - 1)
            for i in denoising_inds.cpu().tolist()
        ], device=obs.device)
        advantages = advantages * discount

        # Exponential clip coefficient interpolation
        if self.ft_denoising_steps > 1:
            t_norm = denoising_inds.float() / (self.ft_denoising_steps - 1)
            t_norm = t_norm.to(obs.device)
            clip_coef = self.clip_ploss_coef_base + (
                self.clip_ploss_coef - self.clip_ploss_coef_base
            ) * (torch.exp(self.clip_ploss_coef_rate * t_norm) - 1) / (
                math.exp(self.clip_ploss_coef_rate) - 1)
        else:
            clip_coef = torch.full_like(advantages, self.clip_ploss_coef)

        logratio = newlp - oldlp
        ratio = logratio.exp()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        newvalues = self.get_value(obs)
        v_loss = 0.5 * ((newvalues - returns) ** 2).mean()

        return pg_loss, v_loss, approx_kl.item(), clipfrac
