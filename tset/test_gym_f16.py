import gymnasium
import gym_pyf16_env
from gymnasium.wrappers import FlattenObservation


env = gymnasium.make('gym_pyf16_env/GridWorld-v0')

init_obs, _ = env.reset()
print(f"Initial observation: {init_obs}")

action = {"thrust": 100, "elevator": 0.0, "aileron": 0.0, "rudder": 0.0}

for i in range(10000):
    obs, reward, terminated, _, _ = env.step(action)
    print(f"Step:{i}, position: {obs['npos']}, {obs['epos']}, {obs['altitude']}, Done: {terminated}")


env.close()