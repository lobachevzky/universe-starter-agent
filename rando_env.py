from __future__ import absolute_import, print_function

import copy
import os
import random
from Queue import Queue
from threading import Lock

import gym
import numpy as np
import re

from gym.envs.registration import EnvSpec
from gym.wrappers import TimeLimit

from interface import Env

import rospy
import tf
from cv_bridge import CvBridge
from gazebo_msgs.msg import ContactsState, ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist, Point, Quaternion, Pose
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from spaces import ActionSpace, ObservationSpace
from std_msgs import msg
from std_srvs import srv


def call_service(service, message_type=srv.Empty, args=None):
    if args is None:
        args = []
    rospy.wait_for_service(service)
    return rospy.ServiceProxy(service, message_type)(*args)


class Gazebo(gym.Env):
    @property
    def reward_range(self):
        return -np.inf, np.inf

    def __init__(self,
                 observation_range,
                 action_range,
                 action_shape,
                 reward_file='reward.csv'):
        self.shape = (72, 32, 1)
        self._takeoff_publisher = rospy.Publisher('ardrone/takeoff', msg.Empty, queue_size=1)
        self._action_space = ActionSpace(*action_range, shape=action_shape)
        self._observation_space = ObservationSpace(*observation_range, shape=self.shape)

    def _step(self, action):
        done = not random.randint(0, 10)
        state = np.random.random(self.shape)
        return state, 1, done, {}

    def _reset(self):
        state = np.random.random(self.shape)
        return state

    def _render(self, close, **kwargs):
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

