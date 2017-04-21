#!/usr/bin/env python
import rospy
import numpy as np
from gazebo import Gazebo

env = Gazebo(observation_range=(-1, 1), action_range=(-1, 1), action_shape=(3,))

env.reset()
while not rospy.is_shutdown():
    _, _, done, _ = env.step(*list(np.random.rand(3)))
    if done:
        env.reset()

