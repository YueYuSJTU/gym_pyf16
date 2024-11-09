from gymnasium.envs.registration import register

register(
    id="gym_pyf16_env/GridWorld-v0",
    entry_point="gym_pyf16_env.envs:GridWorldEnv",
)
