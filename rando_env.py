from __future__ import absolute_import, print_function

import random

import gym
import numpy as np

from spaces import ActionSpace, ObservationSpace


class Gazebo(gym.Env):
    @property
    def reward_range(self):
        return -np.inf, np.inf

    def __init__(self,
                 action_shape):
        self.shape = (72, 32, 1)
        self._action_space = ActionSpace(shape=action_shape)
        self._observation_space = ObservationSpace(shape=self.shape)

    def _step(self, action):
        done = not random.randint(0, 10)
        state = np.random.random(self.shape)
        return state, 1, done, {}

    def _reset(self):
        state = np.random.random(self.shape)
        return state

    def _render(self, **kwargs):
        pass

    def _close(self):
        pass

    def _configure(self):
        pass

    def _seed(self, **kwargs):
        pass

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def reward_range(self):
        return -np.inf, np.inf

