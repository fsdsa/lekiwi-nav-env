"""
Asymmetric Actor-Critic wrapper for skrl 1.4.3 + Isaac Lab.

skrl 1.4.3의 IsaacLabWrapper는 {"policy": 30D, "critic": 37D} dict obs에서
"policy"만 추출하고 "critic"은 버린다. 이 wrapper는 critic obs를 state()로
노출하여 AAC trainer/agent가 활용할 수 있게 한다.
"""
from __future__ import annotations

import torch
from skrl.envs.wrappers.torch import wrap_env


def wrap_env_aac(env):
    """Isaac Lab 환경을 AAC 지원 wrapper로 감싼다.

    내부적으로 skrl의 표준 IsaacLabWrapper를 생성한 뒤,
    state() 메서드를 monkey-patch하여 critic obs를 노출한다.
    """
    wrapped = wrap_env(env, wrapper="isaaclab")

    # 원본 env 참조 (DirectRLEnv)
    raw_env = wrapped._unwrapped

    # state_space를 올바르게 노출 (critic observation space)
    try:
        _state_space = raw_env.single_observation_space["critic"]
    except (KeyError, AttributeError):
        _state_space = None

    def state(self_wrapper):
        """현재 critic observation 반환.

        환경의 _get_observations()에서 self._critic_obs에 저장된 텐서를 반환.
        AAC trainer가 step()/reset() 직후에 호출한다.
        """
        critic_obs = getattr(raw_env, "_critic_obs", None)
        return critic_obs

    # Monkey-patch
    import types
    wrapped.state = types.MethodType(state, wrapped)
    wrapped._aac_state_space = _state_space

    return wrapped
