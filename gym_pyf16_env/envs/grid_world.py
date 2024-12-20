from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import pyf16


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.aero_model = pyf16.AerodynamicModel("./models/f16_model")
        self.aero_model.install("./models/f16_model/data")
        self.control_limits = self.aero_model.load_ctrl_limits()

        self.observation_space = spaces.Box(
            low=np.concatenate([
                np.array([-75000, -75000, 0, -np.inf, -np.inf, -np.inf, 0, -0.5*np.pi, -0.15*np.pi, -np.pi, -np.pi, -np.pi]),
                np.array([-40000, -40000, -40000])
            ]),
            high=np.concatenate([
                np.array([75000, 75000, 30000, np.inf, np.inf, np.inf, 1000, 0.5*np.pi, 0.15*np.pi, np.pi, np.pi, np.pi]),
                np.array([40000, 40000, 40000])
            ]),
            dtype=np.float64
        )

        # self.action_space = spaces.Box(
        #     low=np.array([
        #         self.control_limits.thrust_cmd_limit_bottom,
        #         self.control_limits.ele_cmd_limit_bottom,
        #         self.control_limits.ail_cmd_limit_bottom,
        #         self.control_limits.rud_cmd_limit_bottom
        #     ]),
        #     high=np.array([
        #         self.control_limits.thrust_cmd_limit_top,
        #         self.control_limits.ele_cmd_limit_top,
        #         self.control_limits.ail_cmd_limit_top,
        #         self.control_limits.rud_cmd_limit_top
        #     ]),
        #     dtype=np.float64
        # )
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1]),
            high=np.array([1, 1, 1, 1]),
            dtype=np.float64
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
    
    def reflect_action(self, action):
        return np.array([
            (action[0] + 1) / 2 * (self.control_limits.thrust_cmd_limit_top - self.control_limits.thrust_cmd_limit_bottom) + self.control_limits.thrust_cmd_limit_bottom,
            (action[1] + 1) / 2 * (self.control_limits.ele_cmd_limit_top - self.control_limits.ele_cmd_limit_bottom) + self.control_limits.ele_cmd_limit_bottom,
            (action[2] + 1) / 2 * (self.control_limits.ail_cmd_limit_top - self.control_limits.ail_cmd_limit_bottom) + self.control_limits.ail_cmd_limit_bottom,
            (action[3] + 1) / 2 * (self.control_limits.rud_cmd_limit_top - self.control_limits.rud_cmd_limit_bottom) + self.control_limits.rud_cmd_limit_bottom
        ])
    

    def _get_obs(self):
        return np.concatenate([self._agent_state, self._relative_location])

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_state[0:3] - self._target_location, ord=1
            )
        }
    
    def _cal_relative_location(self, location):
        return location - self._agent_state[0:3]

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.simTime = 0.0

        self.trim_target = pyf16.TrimTarget(15000, 500, None, None)
        self.trim_init = None
        self.trim_result = pyf16.trim(self.aero_model, self.trim_target, self.control_limits, self.trim_init)

        self.f16 = pyf16.PlaneBlock("1", self.aero_model, self.trim_result, [0, 0, 0], self.control_limits)

        self._agent_state = np.array(self.f16.state.state.to_list())
        self.current_action = np.zeros(self.action_space.shape)
        self.waypoints = self._set_waypoints()
        self._target_location = self.waypoints[-1]
        self._relative_location = self._cal_relative_location(self._target_location)
        # self.previous_actions = [np.zeros(4) for _ in range(10)]
        self.previous_actions = []

        # 奖励函数各项
        self.alpha = 0
        self.height = 0
        self.beta = 0
        self.action_penalty = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def _rewardFcn(self):
        """
        返回的奖励函数包含以下项：
        1.高度惩罚：当高度低于5000时，值为(height - 5000)/5000
        2.角速度惩罚：当p,q,r中的最大值大于0.1时，值为max(-(max-0.1)/0.1, -1)
        3.攻角惩罚：当攻角的绝对值大于20度时，值为(angle - 20)/20
        4.侧滑角惩罚：当侧滑角的绝对值大于5度时，值为(angle - 5)/5
        5.导航点奖励：当飞机到达导航点时，值为1
        6.动作惩罚：对当前时刻以及之前10个时刻的动作进行相邻两个动作的差分，然后对10个差分求和，但惩罚值不超过0.02
        """
        height = self._agent_state[2]
        phi, theta, psi = self._agent_state[3:6]
        p, q, r = self._agent_state[9:12]
        alpha = self._agent_state[7]
        beta = self._agent_state[8]

        # 高度和角速度惩罚
        reward = 0
        if height < 10000:
            self.height = (10000 - height) / 10000 * 0.03
        # if np.max(np.abs([p, q, r])) > 0.1:
        #     reward -= max(-(np.max(np.abs([p, q, r])) - 0.1) / 0.1, -1) * 0.01
        
        # 攻角和侧滑角惩罚
        # if np.abs(phi) > np.pi / 2:
        #     self.phi = (np.abs(phi) - np.pi / 2) / (np.pi / 2) * 0.02
        # if theta > np.pi / 6:
        #     reward -= np.abs((theta - np.pi / 6) / (np.pi / 6)) * 0.001
        # elif theta < -np.pi * 0.0278:
        #     reward -= np.abs((theta + np.pi * 0.0278) / (np.pi * 0.0278)) * 0.001
        # if np.abs(psi) > np.pi / 18:
        #     self.psi = np.abs((np.abs(psi) - np.pi / 18)) * 0.01

        if np.abs(alpha) > 0.349:
            self.alpha = (np.abs(alpha) - 0.349) / 0.349 * 0.03
        if np.abs(beta) > 0.0872:
            self.beta = (np.abs(beta) - 0.0872) / 0.0872 * 0.03

        # 动作惩罚
        if self.simTime > 0.5:
            actions = np.array(self.previous_actions)
            action_diffs = np.diff(actions, axis=0)
            action_penalty = np.sum(np.abs(action_diffs))*0.002
            self.action_penalty = action_penalty
        else:
            self.action_penalty = 0
        # print(f"Debug: action_penalty{action_penalty}")

        # reward = reward * np.exp(-self.simTime / 50)
        reward = -self.height -self.alpha - self.beta - self.action_penalty
        reward = reward * (150 - self.simTime) / 150
        
        # # 导航点奖励
        # reward = reward * 0.2
        # if np.linalg.norm(self._agent_state[0:3] - self._target_location, ord=2) < 100:
        #     reward += 50
        # reward -= np.linalg.norm(self._agent_state[-3:], ord=2) / 15000

        # 时间奖励
        reward += 0.08
        if self.simTime > 30:
            reward += 0.05
        return reward

    def step(self, action, time_step=0.01):
        self.simTime += time_step
        self.current_action = action
        action = self.reflect_action(action)
        self._agent_state = np.array(self.f16.update(
            pyf16.Control(
                thrust=action[0], 
                elevator=action[1], 
                aileron=action[2], 
                rudder=action[3]
            ), 
            self.simTime
        ).state.to_list())
        # print(f"Debug: {self._agent_state.dtype}")

        if np.linalg.norm(self._agent_state[0:3] - self._target_location, ord=2) < 100:
            self.waypoints.pop()
            if len(self.waypoints) > 0:
                self._target_location = self.waypoints[-1]

        reward = self._rewardFcn()
        observation = self._get_obs()
        info = self._get_info()
        self._relative_location = self._cal_relative_location(self._target_location)
        # print(f"Debug: {not self.observation_space.contains(observation)}, {len(self.waypoints) == 0}")
        terminated = (not self.observation_space.contains(observation)) or len(self.waypoints) == 0
        if self.simTime > 150:
            terminated = True
        # terminated = self.simTime > 50 or len(self.waypoints) == 0

        if len(self.previous_actions) >= 10:
            self.previous_actions.pop(0)
        self.previous_actions.append(self.current_action)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def _set_waypoints(self):
        """
        在observation space 的前三个维度内随机选取三个点作为导航点
        """
        waypoints = []
        for _ in range(3):
            waypoints.append(
                np.array(
                    [
                        self.np_random.uniform(-10000, 10000),
                        self.np_random.uniform(-10000, 10000),
                        self.np_random.uniform(10000, 25000),
                    ]
                )
            )
        waypoints[2] = np.array([5000, 5000, 10000])
        waypoints[1] = np.array([5000, -5000, 15000])
        waypoints[0] = np.array([-5000, -5000, 13000])
        return waypoints

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_state + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        self.f16.delete_model()
        self.aero_model.uninstall()
