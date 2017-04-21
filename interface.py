from __future__ import absolute_import

from abc import ABCMeta, abstractproperty, abstractmethod

import gym.spaces


class Space:
    __metaclass__ = ABCMeta

    @abstractproperty
    def is_discrete(self):
        pass

    @abstractproperty
    def is_continuous(self):
        pass

    @abstractmethod
    def dim(self):
        """
        :return: dimension of action (for Box, this is the number of elements in self.shape())
        """
        pass

    @abstractproperty
    def n(self):
        """
        :return: size of policy network output
        """
        pass


class Box(Space):
    __metaclass__ = ABCMeta

    @property
    def is_discrete(self):
        return False

    @property
    def is_continuous(self):
        return True

    @abstractproperty
    def low(self):
        """
        :return: array of lower bounds for each dimension of space
        """
        pass

    @abstractproperty
    def high(self):
        """
        :return: array of upper bounds for each dimension of space
        """
        pass

    @abstractmethod
    def dim(self):
        """
        :return: array of dimensions
        """
        pass


class Discrete(Space):
    __metaclass__ = ABCMeta

    @property
    def is_discrete(self):
        return True

    @property
    def is_continuous(self):
        return False

    @abstractmethod
    def dim(self):
        pass


class Env:
    """
    superclass of gym env and Gazebo env
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def action_space(self):
        pass

    @abstractproperty
    def observation_space(self):
        pass

    @abstractproperty
    def max_time(self):
        """
        simulation automatically terminates after this many steps. Int or None.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        resets environment to initial state
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        :returns: new_state [np.array], reward [float], done [bool], debugging info
        """
        pass


Env.register(gym.Env)
Space.register(gym.Space)
Box.register(gym.spaces.Box)

gym.Env.is_gazebo = False

# Box
gym.spaces.Box.is_discrete = False
gym.spaces.Box.is_continuous = True
gym.spaces.Box.dim = lambda self: self.low.size
# gym.spaces.Box.shape = lambda self: tuple(int(.5 * x) for x in self.low.shape[:2]) + (3,)
# gym.spaces.Box.shape = eval('self.low.shape')

# gym.spaces.Box.f = lambda self: print("WTF")
# gym.spaces.Box.g = get_shape

# Discrete
gym.spaces.Discrete.is_discrete = True
gym.spaces.Discrete.is_continuous = False
gym.spaces.Discrete.dim = lambda _: 1  # for some reason, gym[atari] allow only one button press per time step

gym.Env.max_time = lambda self: self.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    assert issubclass(gym.Env, Env)
    assert env.action_space.is_discrete
