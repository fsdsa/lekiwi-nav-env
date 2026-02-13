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
