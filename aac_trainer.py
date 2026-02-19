"""
Asymmetric Actor-Critic trainer for skrl 1.4.3.

skrl 1.4.3의 SequentialTrainer는 critic obs를 추적하지 않는다.
이 서브클래스는 env.state()를 통해 critic obs를 별도 추적하여
AAC_PPO agent에 전달한다.
"""
from __future__ import annotations

import sys

import torch
import tqdm
from skrl.trainers.torch import SequentialTrainer


class AACSequentialTrainer(SequentialTrainer):
    """SequentialTrainer with Asymmetric Actor-Critic support.

    env.state()로 critic observation을 읽어 agent의
    _current_critic_states / _current_next_critic_states에 설정.
    """

    def single_agent_train(self) -> None:
        assert self.num_simultaneous_agents == 1
        assert self.env.num_agents == 1

        # Reset
        states, infos = self.env.reset()
        critic_states = self._get_critic_states()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps),
            disable=self.disable_progressbar,
            file=sys.stdout,
        ):
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                actions = self.agents.act(
                    states, timestep=timestep, timesteps=self.timesteps,
                )[0]

                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                next_critic_states = self._get_critic_states()

                if not self.headless:
                    self.env.render()

                # AAC: critic states를 agent에 주입
                self.agents._current_critic_states = critic_states
                self.agents._current_next_critic_states = next_critic_states

                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # Log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # Update states
            if self.env.num_envs > 1:
                states = next_states
                critic_states = next_critic_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                        critic_states = self._get_critic_states()
                else:
                    states = next_states
                    critic_states = next_critic_states

    def _get_critic_states(self) -> torch.Tensor | None:
        """env.state()를 호출하여 critic observation 반환."""
        if hasattr(self.env, "state") and callable(self.env.state):
            try:
                return self.env.state()
            except NotImplementedError:
                pass
        return None
