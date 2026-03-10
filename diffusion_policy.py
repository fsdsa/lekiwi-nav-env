"""
Diffusion Policy for LeKiwi BC — ported from official ResiP codebase.

Source: https://github.com/ankile/robust-rearrangement
Paper: "From Imitation to Refinement — Residual RL for Precise Assembly" (Ankile et al. 2024)

Components taken from official code:
  - ConditionalUnet1D (src/models/unet.py) — exact copy
  - SinusoidalPosEmb (src/models/pos_embed.py) — exact copy
  - LinearNormalizer (src/dataset/normalizer.py) — adapted for flat obs/actions
  - SwitchEMA (src/models/ema.py) — exact copy
  - DiffusionPolicy (src/behavior/diffusion.py) — adapted for LeKiwi env

Adaptations from official code:
  - Official uses dict obs (robot_state, parts_poses) → we use flat obs (30D/29D)
  - Official uses zarr data → we use HDF5
  - Official uses FurnitureBench env → we use IsaacLab LeKiwi env
  - Official uses Franka 7DoF → we use LeKiwi 3DoF base + 6DoF arm

Everything else (architecture, schedulers, hyperparameters) follows official code.
"""

import math
from collections import deque
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


# =============================================================================
# SinusoidalPosEmb — exact copy from src/models/pos_embed.py
# =============================================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# =============================================================================
# ConditionalUnet1D — exact copy from src/models/unet.py
# =============================================================================
class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        # FiLM modulation https://arxiv.org/abs/1709.07871
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1))
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """
    Exact copy from src/models/unet.py.

    input_dim: Dim of actions.
    global_cond_dim: Dim of global conditioning (obs_dim, flat).
    diffusion_step_embed_dim: Size of positional encoding for diffusion step k.
    down_dims: Channel size for each UNet level.
    kernel_size: Conv kernel size.
    n_groups: Number of groups for GroupNorm.
    """

    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in, dim_out, cond_dim=cond_dim,
                            kernel_size=kernel_size, n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out, dim_out, cond_dim=cond_dim,
                            kernel_size=kernel_size, n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2, dim_in, cond_dim=cond_dim,
                            kernel_size=kernel_size, n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in, dim_in, cond_dim=cond_dim,
                            kernel_size=kernel_size, n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond=None,
    ):
        """
        sample: (B, T, input_dim)
        timestep: (B,) or int
        global_cond: (B, global_cond_dim)
        output: (B, T, input_dim)
        """
        sample = sample.moveaxis(-1, -2)  # (B,T,C) -> (B,C,T)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.moveaxis(-1, -2)  # (B,C,T) -> (B,T,C)
        return x


# =============================================================================
# LinearNormalizer — adapted from src/dataset/normalizer.py
# Official uses min-max normalization to [-1, 1]
# =============================================================================
class LinearNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stats = nn.ParameterDict()

    def fit(self, data_dict: dict):
        """
        Fit normalizer on data.
        data_dict: {"obs": tensor(N, obs_dim), "action": tensor(N, act_dim)}
        """
        for key, tensor in data_dict.items():
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor, dtype=torch.float32)
            min_value = tensor.min(dim=0)[0]
            max_value = tensor.max(dim=0)[0]

            diff = max_value - min_value
            constant_columns = diff == 0
            min_value[constant_columns] -= 1
            max_value[constant_columns] += 1

            self.stats[key] = nn.ParameterDict(
                {
                    "min": nn.Parameter(min_value, requires_grad=False),
                    "max": nn.Parameter(max_value, requires_grad=False),
                }
            )
        self._turn_off_gradients()

    def _normalize(self, x, key):
        stats = self.stats[key]
        x = (x - stats["min"]) / (stats["max"] - stats["min"])
        x = 2 * x - 1
        return x

    def _denormalize(self, x, key):
        stats = self.stats[key]
        x = (x + 1) / 2
        x = x * (stats["max"] - stats["min"]) + stats["min"]
        return x

    def forward(self, x, key, forward=True):
        if forward:
            return self._normalize(x, key)
        else:
            return self._denormalize(x, key)

    def _turn_off_gradients(self):
        for key in self.stats.keys():
            for stat in self.stats[key].keys():
                self.stats[key][stat].requires_grad = False

    def load_state_dict(self, state_dict, device=None):
        """
        Custom load from official src/dataset/normalizer.py.
        Rebuilds nested ParameterDict from flat state_dict keys.
        """
        stats = nn.ParameterDict()
        for key, value in state_dict.items():
            if key.startswith("stats."):
                param_key = key[6:]
                keys = param_key.split(".")
                current_dict = stats
                for k in keys[:-1]:
                    if k not in current_dict:
                        current_dict[k] = nn.ParameterDict()
                    current_dict = current_dict[k]
                if device is not None:
                    value = value.to(device)
                current_dict[keys[-1]] = nn.Parameter(value)

        self.stats = stats
        self._turn_off_gradients()


# =============================================================================
# SwitchEMA — exact copy from src/models/ema.py
# =============================================================================
class SwitchEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                )
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# ResidualPolicy — exact copy from src/models/residual.py
# Official config (src/config/actor/residual_diffusion.yaml):
#   action_scale: 0.1
#   action_head_std: 0.0
#   init_logstd: -1.0
#   learn_std: false
#   actor_hidden_size: 256, actor_num_layers: 2
#   critic_hidden_size: 256, critic_num_layers: 2
#   actor_activation: ReLU, critic_activation: ReLU
#   critic_last_layer_bias_const: 0.25
#   critic_last_layer_std: 0.25
# =============================================================================
def _layer_init(layer, nonlinearity="ReLU", std=math.sqrt(2), bias_const=0.0):
    """From official src/models/residual.py layer_init()."""
    if isinstance(layer, nn.Linear):
        if nonlinearity in ("ReLU", "SiLU"):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif nonlinearity == "Tanh":
            torch.nn.init.orthogonal_(layer.weight, std)
        else:
            nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def _build_mlp(
    input_dim, hidden_sizes, output_dim, activation,
    output_std=1.0, bias_on_last_layer=True, last_layer_bias_const=0.0,
):
    """From official src/models/residual.py build_mlp()."""
    act_func = getattr(nn, activation)
    layers = []
    layers.append(_layer_init(nn.Linear(input_dim, hidden_sizes[0]), nonlinearity=activation))
    layers.append(act_func())
    for i in range(1, len(hidden_sizes)):
        layers.append(
            _layer_init(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]), nonlinearity=activation)
        )
        layers.append(act_func())
    layers.append(
        _layer_init(
            nn.Linear(hidden_sizes[-1], output_dim, bias=bias_on_last_layer),
            std=output_std, nonlinearity="Tanh", bias_const=last_layer_bias_const,
        )
    )
    return nn.Sequential(*layers)


class ResidualPolicy(nn.Module):
    """
    Exact copy from src/models/residual.py ResidualPolicy.

    Input: [normalized_obs, base_action] concatenated
    Output: residual action correction (scaled by action_scale)

    In the RL loop:
        final_action = base_action + action_scale * residual_policy(obs, base_action)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_size: int = 256,
        actor_num_layers: int = 2,
        critic_hidden_size: int = 256,
        critic_num_layers: int = 2,
        actor_activation: str = "ReLU",
        critic_activation: str = "ReLU",
        init_logstd: float = -1.0,
        action_head_std: float = 0.0,
        action_scale: float = 0.1,
        learn_std: bool = False,
        critic_last_layer_bias_const: float = 0.25,
        critic_last_layer_std: float = 0.25,
    ):
        super().__init__()

        self.action_dim = action_dim
        # Input to residual = obs + base_action
        self.obs_dim = obs_dim + action_dim
        self.action_scale = action_scale

        self.actor_mean = _build_mlp(
            input_dim=self.obs_dim,
            hidden_sizes=[actor_hidden_size] * actor_num_layers,
            output_dim=action_dim,
            activation=actor_activation,
            output_std=action_head_std,
            bias_on_last_layer=False,
        )

        self.critic = _build_mlp(
            input_dim=self.obs_dim,
            hidden_sizes=[critic_hidden_size] * critic_num_layers,
            output_dim=1,
            activation=critic_activation,
            output_std=critic_last_layer_std,
            bias_on_last_layer=True,
            last_layer_bias_const=critic_last_layer_bias_const,
        )

        self.actor_logstd = nn.Parameter(
            torch.ones(1, action_dim) * init_logstd,
            requires_grad=learn_std,
        )

    def get_value(self, nobs: torch.Tensor) -> torch.Tensor:
        return self.critic(nobs)

    def get_action_and_value(self, nobs, action=None):
        from torch.distributions import Normal

        action_mean = self.actor_mean(nobs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        # NaN safety: prevent crash from physics/normalizer instability
        action_mean = torch.nan_to_num(action_mean, nan=0.0)
        action_std = torch.nan_to_num(action_std, nan=1.0).clamp(min=1e-6)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action).sum(dim=1),
            probs.entropy().sum(dim=1),
            self.critic(nobs),
            action_mean,
        )

    def get_action(self, nobs: torch.Tensor) -> torch.Tensor:
        """Deterministic action (for eval) scaled by action_scale."""
        return self.actor_mean(nobs) * self.action_scale


# =============================================================================
# DiffusionPolicy — adapted from src/behavior/diffusion.py
#
# Official config (src/config/actor/diffusion.yaml):
#   beta_schedule: squaredcos_cap_v2
#   prediction_type: epsilon
#   inference_steps: 16
#   num_diffusion_iters: 100
#   clip_sample: true
# =============================================================================
class DiffusionPolicyAgent(nn.Module):
    """
    Full Diffusion Policy agent with normalization, inference, and action queue.

    Follows the official ResiP implementation:
      - DDPM training with epsilon prediction
      - DDIM inference with warm-starting
      - squaredcos_cap_v2 noise schedule
      - ConditionalUnet1D denoiser
      - LinearNormalizer for obs/action normalization

    Args:
        obs_dim: Observation dim (30 for Skill-2, 29 for Skill-3)
        act_dim: Action dim (9 for LeKiwi)
        pred_horizon: Actions to predict (default 16, official default)
        action_horizon: Actions to execute before re-planning (default 8)
        num_diffusion_iters: Training diffusion steps (default 100)
        inference_steps: DDIM inference steps (default 16)
        down_dims: UNet channel dims (default [256, 512, 1024])
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        num_diffusion_iters: int = 100,
        inference_steps: int = 16,
        down_dims: list = None,
        diffusion_step_embed_dim: int = 256,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()

        if down_dims is None:
            down_dims = [256, 512, 1024]

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.inference_steps = inference_steps

        # UNet denoiser
        self.model = ConditionalUnet1D(
            input_dim=act_dim,
            global_cond_dim=obs_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

        # Noise schedulers (from diffusers, same as official)
        self.train_noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.inference_noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        # Normalizer
        self.normalizer = LinearNormalizer()

        # Warm-start inference (from official)
        self.warmstart_timestep = 50
        self.prev_naction = None
        self.eta = 0.0

        # Action buffer for receding horizon
        self.base_nactions = deque(maxlen=action_horizon)

    def compute_loss(self, nobs: torch.Tensor, naction: torch.Tensor) -> torch.Tensor:
        """
        DDPM training loss (epsilon prediction).

        Args:
            nobs: Normalized observations (B, obs_dim)
            naction: Normalized action sequences (B, pred_horizon, act_dim)
        Returns:
            loss: scalar
        """
        noise = torch.randn(naction.shape, device=naction.device)

        timesteps = torch.randint(
            0,
            self.train_noise_scheduler.config.num_train_timesteps,
            (nobs.shape[0],),
            device=nobs.device,
        ).long()

        noisy_action = self.train_noise_scheduler.add_noise(naction, noise, timesteps)

        noise_pred = self.model(
            sample=noisy_action,
            timestep=timesteps,
            global_cond=nobs.float(),
        )

        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def _predict_normalized_action(self, nobs: torch.Tensor) -> torch.Tensor:
        """
        DDIM inference with warm-start.
        From src/behavior/diffusion.py _normalized_action().
        """
        B = nobs.shape[0]

        if self.prev_naction is None or self.prev_naction.shape[0] != B:
            self.prev_naction = torch.zeros(
                (B, self.pred_horizon, self.act_dim), device=nobs.device
            )

        noise = torch.randn(
            (B, self.pred_horizon, self.act_dim), device=nobs.device
        )

        self.inference_noise_scheduler.set_timesteps(self.inference_steps)

        naction = self.inference_noise_scheduler.add_noise(
            self.prev_naction, noise,
            torch.full((B,), self.warmstart_timestep, device=nobs.device).long(),
        )

        for k in self.inference_noise_scheduler.timesteps:
            noise_pred = self.model(sample=naction, timestep=k, global_cond=nobs)
            naction = self.inference_noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction, eta=self.eta,
            ).prev_sample

        # Store for warm-start
        self.prev_naction[:, :self.pred_horizon - self.action_horizon, :] = \
            naction[:, self.action_horizon:, :]

        return naction

    @torch.no_grad()
    def base_action_normalized(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get normalized base action for the current step.
        Used in the residual RL loop — matches official residual_ppo.py line 353.

        If action queue is empty, run diffusion to fill it.
        Returns: (B, act_dim) normalized base action for this step.
        """
        if not self.base_nactions:
            nobs = self.normalizer(obs, "obs", forward=True)
            naction_pred = self._predict_normalized_action(nobs)
            for i in range(self.action_horizon):
                self.base_nactions.append(naction_pred[:, i, :])

        return self.base_nactions.popleft()

    @torch.no_grad()
    def predict_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Full pipeline: normalize → predict → denormalize.
        Returns: (B, pred_horizon, act_dim)
        """
        nobs = self.normalizer(obs, "obs", forward=True)
        naction = self._predict_normalized_action(nobs)

        B, T, A = naction.shape
        action = self.normalizer(
            naction.reshape(B * T, A), "action", forward=False
        ).reshape(B, T, A)
        return action

    def reset(self):
        self.prev_naction = None
        self.base_nactions.clear()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
