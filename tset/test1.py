# run_gymnasium_env.py

import gymnasium
import gym_pyf16_env
from gymnasium.wrappers import FlattenObservation


env = gymnasium.make('gym_pyf16_env/GridWorld-v0', size = 5, render_mode = 'human')

init_obs, _ = env.reset()
print(f"Initial observation: {init_obs}")
# Initial observation: {'agent': array([1, 1]), 'target': array([4, 1])}
steps = 0
while True:
    env.render()
    action = env.action_space.sample()
    # Action: 0 
    obs, reward, terminated, _, _ = env.step(action)
    print(f"Step: {steps}, Action: {action}, Observation: {obs}, Reward: {reward}, Done: {terminated}")
    # Step: 0, Action: 0, Observation: {'agent': array([2, 1]), 'target': array([4, 1])}, Reward: 0, Done: False
    steps += 1
    if terminated:
        break
print(f"Simulation ended after {steps} steps")


env.close()
