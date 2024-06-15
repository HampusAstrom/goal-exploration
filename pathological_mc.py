from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
import numpy as np
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.envs.classic_control import utils

from gymnasium.envs.registration import register
# Example for the CartPole environment
register(
    # unique identifier for the env `name-version`
    id="PathologicalMountainCar-v1.1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="pathological_mc:PathologicalMountainCarEnv",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    max_episode_steps=200,
)

class PathologicalMountainCarEnv(MountainCarEnv):

    def __init__(self,
                 shift_inclination = -0.15,
                 terminate = False,
                 render_mode: Optional[str] = None,
                 goal_velocity=0):
        super().__init__(render_mode, goal_velocity)
        self.min_position = -1.7
        self.shift_inclination = shift_inclination
        self.terminate = terminate

        self.goal_position_left = -1.6

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)


    def step(self, action: int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        position, velocity = self.state
        velocity += (action - 1) * self.force + \
            (np.cos(3 * position) + self.shift_inclination) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        self.counter += 1

        terminated = False
        reward = -1.0
        if position >= self.goal_position and velocity >= self.goal_velocity:
            reward = 10
            if self.terminate:
                terminated = True
        if position <= self.goal_position_left and -velocity >= self.goal_velocity:
            reward = 500
            if self.terminate:
                terminated = True

        self.state = (position, velocity)
        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        ret = super().reset(seed=seed, options=options)
        self.counter = 0
        return ret

    def _height(self, xs):
        return (np.sin(3 * xs) + 3*self.shift_inclination*xs) * 0.45 + 0.55

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        carwidth = 40
        carheight = 20

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(np.cos(3 * pos) + self.shift_inclination)
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(np.cos(3 * pos) + self.shift_inclination)
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )

            gfxdraw.aacircle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )

        # right flag stuff
        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        # left flag stuff
        flagx_2 = int((self.goal_position_left - self.min_position) * scale)
        flagy1_2 = int(self._height(self.goal_position_left) * scale)
        flagy2_2 = flagy1_2 + 50
        gfxdraw.vline(self.surf, flagx_2, flagy1_2, flagy2_2, (0, 0, 0))

        gfxdraw.aapolygon(
            self.surf,
            [(flagx_2, flagy2_2), (flagx_2, flagy2_2 - 10), (flagx_2 + 50, flagy2_2 - 5)],
            (204, 0, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx_2, flagy2_2), (flagx_2, flagy2_2 - 10), (flagx_2 + 50, flagy2_2 - 5)],
            (204, 0, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )