"""
LeKiwi Navigation — Shared RL model definitions.

PolicyNet/ValueNet은 train_lekiwi.py, collect_demos.py 등에서 공통 사용.
구조 변경 시 이 파일만 수정하면 모든 스크립트에 반영됨.

PolicyNet.net + mean_layer 구조는 train_bc.py의 BCPolicy와 동일해야
BC → PPO warm-start가 정상 동작함.

AsymmetricValueNet: Asymmetric Actor-Critic (AAC)용 Critic.
Actor obs_dim과 다른 critic_obs_dim을 받아 value를 출력.
Skill-2: actor 14D / critic 21D, Skill-3: actor 13D / critic 20D.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class PolicyNet(GaussianMixin, Model):
    """Gaussian Policy (Actor).

    구조가 train_bc.py의 BCPolicy와 동일 (net + mean_layer).
    """

    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=-5.0,
            max_log_std=2.0,
        )

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.mean_layer = nn.Linear(64, act_dim)
        self.log_std_parameter = nn.Parameter(torch.full((act_dim,), -1.0))

    def compute(self, inputs, role):
        x = self.net(inputs["states"])
        return self.mean_layer(x), self.log_std_parameter, {}


class ValueNet(DeterministicMixin, Model):
    """Value Function (Critic). Same obs space as actor."""

    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions=False)

        obs_dim = observation_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class AsymmetricValueNet(DeterministicMixin, Model):
    """Asymmetric Actor-Critic (AAC) Critic.

    Actor obs_dim과 다른 critic_obs_dim을 사용.
    skrl의 state_space를 통해 privileged critic obs를 받음.

    Skill-2: critic_obs_dim=21 (actor 14D + obj_bbox 6D + obj_mass 1D)
    Skill-3: critic_obs_dim=20 (actor 13D + obj_dims 3D + obj_mass 1D + grip_rel 3D)
    """

    def __init__(self, observation_space, action_space, device,
                 critic_obs_dim: int | None = None, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions=False)

        if critic_obs_dim is not None:
            obs_dim = critic_obs_dim
        elif hasattr(observation_space, "shape"):
            obs_dim = observation_space.shape[0]
        else:
            obs_dim = int(observation_space)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def make_asymmetric_models(
    actor_obs_dim: int,
    critic_obs_dim: int,
    act_dim: int,
    device: str | torch.device,
) -> dict[str, Model]:
    """Create PolicyNet + AsymmetricValueNet for AAC training.

    Returns dict with "policy" and "value" keys for skrl PPO agent.
    """
    actor_obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(actor_obs_dim,), dtype=np.float32
    )
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
    )
    critic_obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(critic_obs_dim,), dtype=np.float32
    )

    return {
        "policy": PolicyNet(actor_obs_space, action_space, device),
        "value": AsymmetricValueNet(
            critic_obs_space, action_space, device,
            critic_obs_dim=critic_obs_dim,
        ),
    }
