"""
LeKiwi Navigation — Shared RL model definitions.

PolicyNet/ValueNet은 train_lekiwi.py, collect_demos.py 등에서 공통 사용.
구조 변경 시 이 파일만 수정하면 모든 스크립트에 반영됨.

PolicyNet.net + mean_layer 구조는 train_bc.py의 BCPolicy와 동일해야
BC → PPO warm-start가 정상 동작함.
"""
from __future__ import annotations

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
    """Value Function (Critic)."""

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


class CriticNet(DeterministicMixin, Model):
    """Asymmetric Critic — Actor보다 넓은 observation을 받는다."""

    def __init__(self, observation_space, action_space, device,
                 critic_obs_dim=None, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions=False)

        obs_dim = critic_obs_dim if critic_obs_dim is not None else observation_space.shape[0]

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
