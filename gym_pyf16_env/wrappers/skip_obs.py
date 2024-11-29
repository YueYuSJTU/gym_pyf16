import gymnasium as gym
import numpy as np

class SkipObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, skip_step, skip_times):
        super().__init__(env)
        self.skip_step = skip_step
        self.skip_times = skip_times
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([self.observation_space.low]*skip_times),
            high=np.concatenate([self.observation_space.high]*skip_times),
            dtype=np.float64
        )

    def _get_obs(self):
        return np.concatenate([self._state_list[i] for i in range(0, len(self._state_list), self.skip_step + 1)])

    def reset(self, seed=None):
        self._state_list = []
        obs, info = self.env.reset()
        self._state_list.append(obs)
        for _ in range(self.skip_times + self.skip_step*(self.skip_times-1) - 1):
            skip_obs, _, _, _, info = self.env.step(np.zeros(self.action_space.shape))
            self._state_list.append(skip_obs)
        self._state = self._get_obs()
        return self._state, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._state_list.pop()
        self._state_list.insert(0, obs)
        self._state = self._get_obs()
        # 在info里面传递当前obs以方便画图
        return self._state, reward, terminated, truncated, info
