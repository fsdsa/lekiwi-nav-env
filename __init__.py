"""LeKiwi Navigation — Task Registration."""
import gymnasium as gym

gym.register(
    id="Isaac-LeKiwi-Fetch-Direct-v0",
    entry_point="lekiwi_nav_env:LeKiwiNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "lekiwi_nav_env:LeKiwiNavEnvCfg",
    },
)

gym.register(
    id="Isaac-LeKiwi-ApproachGrasp-Direct-v0",
    entry_point="skill2_approach_grasp_env:ApproachGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "skill2_approach_grasp_env:ApproachGraspEnvCfg",
    },
)

gym.register(
    id="Isaac-LeKiwi-CarryPlace-Direct-v0",
    entry_point="skill3_carry_place_env:CarryPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "skill3_carry_place_env:CarryPlaceEnvCfg",
    },
)
