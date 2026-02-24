"""
Asymmetric Actor-Critic trainer for skrl 1.4.3.

skrl 1.4.3의 SequentialTrainer는 critic obs를 추적하지 않는다.
이 서브클래스는 env.state()를 통해 critic obs를 별도 추적하여
AAC_PPO agent에 전달한다.

Early stopping: 지정된 metric의 rolling average가 threshold 이상이면 학습 중단.
"""
from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass

import torch
import tqdm
from skrl.trainers.torch import SequentialTrainer


@dataclass
class EarlyStopCfg:
    """Early stopping configuration (rolling window average).

    metric: extras["log"]에서 추적할 키 (예: "direction_compliance")
    threshold: rolling average가 이 값 이상이면 수렴으로 판정
    metric2: (optional) 두 번째 조건 — rolling avg가 threshold2 이하일 때 만족
    threshold2: metric2의 상한 임계값
    window: rolling average 윈도우 크기 (rollout steps)
    min_timesteps: 최소 학습 timestep (너무 빠른 중단 방지)
    """
    metric: str = ""
    threshold: float = 0.93
    metric2: str = ""
    threshold2: float = 0.05
    window: int = 500
    min_timesteps: int = 2000


class AACSequentialTrainer(SequentialTrainer):
    """SequentialTrainer with Asymmetric Actor-Critic support.

    env.state()로 critic observation을 읽어 agent의
    _current_critic_states / _current_next_critic_states에 설정.
    """

    def __init__(self, *args, early_stop: EarlyStopCfg | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stop_cfg = early_stop
        self._es_buffer: deque = deque(maxlen=early_stop.window if early_stop else 1)
        self._es_buffer2: deque = deque(maxlen=early_stop.window if early_stop else 1)
        self._es_triggered = False

    def single_agent_train(self) -> None:
        assert self.num_simultaneous_agents == 1
        assert self.env.num_agents == 1

        es = self.early_stop_cfg

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

                # ── Early stopping check (rolling window average) ──
                if es and es.metric and timestep >= es.min_timesteps:
                    if self.environment_info in infos:
                        log_dict = infos[self.environment_info]
                        # Primary metric (>= threshold)
                        if es.metric in log_dict:
                            val = log_dict[es.metric]
                            if isinstance(val, torch.Tensor):
                                val = val.item()
                            self._es_buffer.append(val)
                        # Secondary metric (<= threshold2)
                        if es.metric2 and es.metric2 in log_dict:
                            val2 = log_dict[es.metric2]
                            if isinstance(val2, torch.Tensor):
                                val2 = val2.item()
                            self._es_buffer2.append(val2)
                        # Check convergence
                        if len(self._es_buffer) >= es.window:
                            avg = sum(self._es_buffer) / len(self._es_buffer)
                            cond1 = avg >= es.threshold
                            cond2 = True
                            avg2_str = ""
                            if es.metric2 and len(self._es_buffer2) >= es.window:
                                avg2 = sum(self._es_buffer2) / len(self._es_buffer2)
                                cond2 = avg2 <= es.threshold2
                                avg2_str = f", {es.metric2} avg = {avg2:.4f} <= {es.threshold2}"
                            elif es.metric2:
                                cond2 = False  # not enough data yet
                            if cond1 and cond2:
                                self._es_triggered = True
                                print(
                                    f"\n  Early stop: {es.metric} avg "
                                    f"= {avg:.4f} >= {es.threshold}"
                                    f"{avg2_str}"
                                    f" (window={es.window}, timestep {timestep}). "
                                    f"Saving final checkpoint."
                                )
                                self.agents.post_interaction(
                                    timestep=timestep, timesteps=self.timesteps,
                                )
                                return

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
