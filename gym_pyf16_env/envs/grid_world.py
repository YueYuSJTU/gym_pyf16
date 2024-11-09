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

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                # "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                # "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "npos": spaces.Box(-15000, 15000, shape=(1,), dtype=np.float32),
                "epos": spaces.Box(-15000, 15000, shape=(1,), dtype=np.float32),
                "altitude": spaces.Box(0, 30000, shape=(1,), dtype=np.float32),
                "phi": spaces.Box(-np.pi, np.pi, shape=(1,), dtype=np.float32),
                "theta": spaces.Box(-np.pi, np.pi, shape=(1,), dtype=np.float32),
                "psi": spaces.Box(-np.pi, np.pi, shape=(1,), dtype=np.float32),
                "velocity": spaces.Box(0, 1000, shape=(1,), dtype=np.float32),
                "alpha": spaces.Box(-0.5*np.pi, 0.5*np.pi, shape=(1,), dtype=np.float32),
                "beta": spaces.Box(-0.5*np.pi, 0.5*np.pi, shape=(1,), dtype=np.float32),
                "p": spaces.Box(-np.pi, np.pi, shape=(1,), dtype=np.float32),
                "q": spaces.Box(-np.pi, np.pi, shape=(1,), dtype=np.float32),
                "r": spaces.Box(-np.pi, np.pi, shape=(1,), dtype=np.float32),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        # self.action_space = spaces.Discrete(4)
        self.action_space = spaces.Dict(
            {
                "thrust": spaces.Box(self.control_limits.thrust_cmd_limit_bottom, self.control_limits.thrust_cmd_limit_top, shape=(1,), dtype=np.float32),
                "elevator": spaces.Box(self.control_limits.ele_cmd_limit_bottom, self.control_limits.ele_cmd_limit_top, shape=(1,), dtype=np.float32),
                "aileron": spaces.Box(self.control_limits.ail_cmd_limit_bottom, self.control_limits.ail_cmd_limit_top, shape=(1,), dtype=np.float32),
                "rudder": spaces.Box(self.control_limits.rud_cmd_limit_bottom, self.control_limits.rud_cmd_limit_top, shape=(1,), dtype=np.float32),
            }
        )

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        # self._action_to_direction = {
        #     Actions.right.value: np.array([1, 0]),
        #     Actions.up.value: np.array([0, 1]),
        #     Actions.left.value: np.array([-1, 0]),
        #     Actions.down.value: np.array([0, -1]),
        # }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        # return {"agent": self._agent_state, "target": self._target_location}
        return {
            "npos": self._agent_state[0:1],
            "epos": self._agent_state[1:2],
            "altitude": self._agent_state[2:3],
            "phi": self._agent_state[3:4],
            "theta": self._agent_state[4:5],
            "psi": self._agent_state[5:6],
            "velocity": self._agent_state[6:7],
            "alpha": self._agent_state[7:8],
            "beta": self._agent_state[8:9],
            "p": self._agent_state[9:10],
            "q": self._agent_state[10:11],
            "r": self._agent_state[11]
        }

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
        self._target_location = np.array([10000, 500, 20000])     #只是一个示例导航点

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action, time_step=0.01):
        self.simTime += time_step
        self._agent_state = np.array(self.f16.update(
            pyf16.Control(thrust=action["thrust"], elevator=action["elevator"], aileron=action["aileron"], rudder=action["rudder"]), self.simTime
        ).state.to_list())

        # terminated = np.array_equal(self._agent_state[0:3], self._target_location)
        ## 如果_agent_state[0:3]的任意一项超出了observation_space的定义，则中止
        terminated = self.observation_space["npos"].contains(self._agent_state[0:1]) and self.observation_space["epos"].contains(self._agent_state[1:2]) and self.observation_space["altitude"].contains(self._agent_state[2:3])
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

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
