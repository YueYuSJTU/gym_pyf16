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
                np.array([-15000, -15000, 0, -np.pi, -np.pi, -np.pi, 0, -0.5*np.pi, -0.5*np.pi, -np.pi, -np.pi, -np.pi]),
                np.array([-15000, -15000, 0])
            ]),
            high=np.concatenate([
                np.array([15000, 15000, 30000, np.pi, np.pi, np.pi, 1000, 0.5*np.pi, 0.5*np.pi, np.pi, np.pi, np.pi]),
                np.array([15000, 15000, 30000])
            ]),
            dtype=np.float64
        )

        self.action_space = spaces.Box(
            low=np.array([
                self.control_limits.thrust_cmd_limit_bottom,
                self.control_limits.ele_cmd_limit_bottom,
                self.control_limits.ail_cmd_limit_bottom,
                self.control_limits.rud_cmd_limit_bottom
            ]),
            high=np.array([
                self.control_limits.thrust_cmd_limit_top,
                self.control_limits.ele_cmd_limit_top,
                self.control_limits.ail_cmd_limit_top,
                self.control_limits.rud_cmd_limit_top
            ]),
            dtype=np.float64
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return np.concatenate([self._agent_state, self._target_location])

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_state[0:3] - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.simTime = 0.0

        self.trim_target = pyf16.TrimTarget(15000, 500, None, None)
        self.trim_init = None
        self.trim_result = pyf16.trim(self.aero_model, self.trim_target, self.control_limits, self.trim_init)

        self.f16 = pyf16.PlaneBlock("1", self.aero_model, self.trim_result, [0, 0, 0], self.control_limits)

        self._agent_state = np.array(self.f16.state.state.to_list())
        self.waypoints = self._set_waypoints()
        self._target_location = self.waypoints[-1]

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
        """
        height = self._agent_state[2]
        phi, theta, psi = self._agent_state[3:6]
        p, q, r = self._agent_state[9:12]
        alpha = self._agent_state[7]
        beta = self._agent_state[8]

        reward = 0
        if height < 5000:
            reward -= (5000 - height) / 5000
        if np.max(np.abs([p, q, r])) > 0.1:
            reward -= max(-(np.max(np.abs([p, q, r])) - 0.1) / 0.1, -1)
        
        if np.abs(phi) > np.pi / 2:
            reward -= (np.abs(phi) - np.pi / 2) / (np.pi / 2)
        if theta > np.pi / 6:
            reward -= (theta - np.pi / 6) / (np.pi / 6)
        elif theta < -np.pi * 0.0278:
            reward -= (theta + np.pi * 0.0278) / (np.pi * 0.0278)
        if np.abs(psi) > np.pi / 18:
            reward -= (abs(psi) - np.pi / 18) / (np.pi / 18)

        if np.abs(alpha) > 0.349:
            reward -= (np.abs(alpha) - 0.349) / 0.349
        if np.abs(beta) > 0.0872:
            reward -= (np.abs(beta) - 0.0872) / 0.0872 * 3
        if np.linalg.norm(self._agent_state[-3:] - self._target_location, ord=1) < 100:
            reward += 5000
        if not self.observation_space.contains(self._get_obs()):
            reward -= 2000
        return reward

    def step(self, action, time_step=0.01):
        self.simTime += time_step
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

        if np.linalg.norm(self._agent_state[-3:] - self._target_location, ord=1) < 100:
            self.waypoints.pop()
            if len(self.waypoints) > 0:
                self._target_location = self.waypoints[-1]

        reward = self._rewardFcn()
        observation = self._get_obs()
        info = self._get_info()
        # print(f"Debug: {not self.observation_space.contains(observation)}, {len(self.waypoints) == 0}")
        terminated = (not self.observation_space.contains(observation)) or len(self.waypoints) == 0

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
                        self.np_random.uniform(-15000, 15000),
                        self.np_random.uniform(-15000, 15000),
                        self.np_random.uniform(0, 30000),
                    ]
                )
            )
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
