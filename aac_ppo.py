"""
Asymmetric Actor-Critic PPO for skrl 1.4.3.

skrl 1.4.3의 PPO는 actor와 critic에 동일한 observation을 전달한다.
이 서브클래스는 critic에 별도의 privileged observation (37D/36D)을
전달하여 Asymmetric Actor-Critic을 구현한다.

사용법:
    from aac_ppo import AAC_PPO
    agent = AAC_PPO(
        models={"policy": PolicyNet(...), "value": CriticNet(critic_obs_dim=37)},
        memory=memory, cfg=cfg_ppo,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        critic_observation_space=env._aac_state_space,  # gym.spaces.Box(37,)
    )
"""
from __future__ import annotations

import itertools
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl import config
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR


class AAC_PPO(PPO):
    """PPO with Asymmetric Actor-Critic support.

    Critic receives separate (wider) privileged observations stored in
    a dedicated memory tensor 'critic_states'.

    AAC trainer must set the following attributes before record_transition():
        self._current_critic_states: torch.Tensor
        self._current_next_critic_states: torch.Tensor
    """

    def __init__(
        self,
        models,
        memory=None,
        observation_space=None,
        action_space=None,
        device=None,
        cfg=None,
        critic_observation_space=None,
    ):
        self._critic_observation_space = critic_observation_space
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )

        # Critic state preprocessor (별도 RunningStandardScaler)
        if self._critic_observation_space is not None:
            critic_pp = self.cfg.get("critic_state_preprocessor", None)
            critic_pp_kwargs = self.cfg.get("critic_state_preprocessor_kwargs", {})
            if critic_pp is not None:
                self._critic_state_preprocessor = critic_pp(**critic_pp_kwargs)
                self.checkpoint_modules["critic_state_preprocessor"] = self._critic_state_preprocessor
            else:
                self._critic_state_preprocessor = self._empty_preprocessor
        else:
            # Fallback: critic uses same preprocessor as actor (symmetric)
            self._critic_state_preprocessor = self._state_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Memory에 critic_states 텐서를 추가 생성."""
        super().init(trainer_cfg=trainer_cfg)

        if self.memory is not None and self._critic_observation_space is not None:
            self.memory.create_tensor(
                name="critic_states",
                size=self._critic_observation_space,
                dtype=torch.float32,
            )
            # _tensors_names에 critic_states 추가
            # 순서: states, critic_states, actions, log_prob, values, returns, advantages
            self._tensors_names = [
                "states", "critic_states",
                "actions", "log_prob", "values", "returns", "advantages",
            ]

        # Temporary variables
        self._current_critic_states = None
        self._current_next_critic_states = None

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Critic obs를 사용하여 value를 계산하고 memory에 저장."""
        # Agent base class tracking (episode rewards/timesteps)
        from skrl.agents.torch import Agent
        Agent.record_transition(
            self, states, actions, rewards, next_states,
            terminated, truncated, infos, timestep, timesteps,
        )

        if self.memory is not None:
            self._current_next_states = next_states

            # Reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # Critic obs for value computation
            critic_states = self._current_critic_states
            if critic_states is None:
                # Fallback: symmetric (no critic obs available)
                critic_states = states

            # Compute values using CRITIC obs
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                values, _, _ = self.value.act(
                    {"states": self._critic_state_preprocessor(critic_states)},
                    role="value",
                )
                values = self._value_preprocessor(values, inverse=True)

            # Time-limit bootstrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # Store transition (includes critic_states)
            self.memory.add_samples(
                states=states,
                critic_states=critic_states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
            )
            for secondary_memory in self.secondary_memories:
                secondary_memory.add_samples(
                    states=states,
                    critic_states=critic_states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=self._current_log_prob,
                    values=values,
                )

    def _update(self, timestep: int, timesteps: int) -> None:
        """PPO update with asymmetric critic observations.

        skrl 1.4.3 PPO._update()를 기반으로 critic_states를 value network에
        전달하도록 수정. Policy는 actor obs(30D), Value는 critic obs(37D) 사용.
        """
        def compute_gae(rewards, dones, values, next_values,
                        discount_factor=0.99, lambda_coefficient=0.95):
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]
            for i in reversed(range(memory_size)):
                nv = values[i + 1] if i < memory_size - 1 else next_values
                advantage = (
                    rewards[i] - values[i]
                    + discount_factor * not_dones[i] * (nv + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        # ── GAE computation (critic obs 사용) ──────────────────────────
        next_critic_states = self._current_next_critic_states
        if next_critic_states is None:
            next_critic_states = self._current_next_states

        with torch.no_grad(), torch.autocast(
            device_type=self._device_type, enabled=self._mixed_precision,
        ):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._critic_state_preprocessor(next_critic_states.float())},
                role="value",
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=(
                self.memory.get_tensor_by_name("terminated")
                | self.memory.get_tensor_by_name("truncated")
            ),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # ── Mini-batch training ────────────────────────────────────────
        sampled_batches = self.memory.sample_all(
            names=self._tensors_names, mini_batches=self._mini_batches,
        )

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        for epoch in range(self._learning_epochs):
            kl_divergences = []

            for batch in sampled_batches:
                # Unpack: states, critic_states, actions, log_prob, values, returns, advantages
                (
                    sampled_states,
                    sampled_critic_states,
                    sampled_actions,
                    sampled_log_prob,
                    sampled_values,
                    sampled_returns,
                    sampled_advantages,
                ) = batch

                with torch.autocast(
                    device_type=self._device_type, enabled=self._mixed_precision,
                ):
                    # Actor: policy obs
                    pp_states = self._state_preprocessor(sampled_states, train=not epoch)
                    # Critic: privileged obs
                    pp_critic = self._critic_state_preprocessor(
                        sampled_critic_states, train=not epoch,
                    )

                    # Policy forward pass
                    _, next_log_prob, _ = self.policy.act(
                        {"states": pp_states, "taken_actions": sampled_actions},
                        role="policy",
                    )

                    # KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # Entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = (
                            -self._entropy_loss_scale
                            * self.policy.get_entropy(role="policy").mean()
                        )
                    else:
                        entropy_loss = 0

                    # Policy loss (clipped surrogate)
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip,
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # Value loss — CRITIC obs 사용
                    predicted_values, _, _ = self.value.act(
                        {"states": pp_critic}, role="value",
                    )
                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values,
                            min=-self._value_clip,
                            max=self._value_clip,
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(
                        sampled_returns, predicted_values,
                    )

                # Optimization
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(
                            self.policy.parameters(), self._grad_norm_clip,
                        )
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(
                                self.policy.parameters(), self.value.parameters(),
                            ),
                            self._grad_norm_clip,
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # LR scheduler
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # Logging
        n = self._learning_epochs * self._mini_batches
        self.track_data("Loss / Policy loss", cumulative_policy_loss / n)
        self.track_data("Loss / Value loss", cumulative_value_loss / n)
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / n)

        self.track_data(
            "Policy / Standard deviation",
            self.policy.distribution(role="policy").stddev.mean().item(),
        )
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
