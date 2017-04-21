#!/usr/bin/env python
import rospy
import numpy as np
from gazebo import Gazebo

env = Gazebo(observation_range=(-1, 1), action_range=(-1, 1), action_shape=(3,))

env.reset()
while not rospy.is_shutdown():
    action = np.random.rand(3)
    _, _, done, _ = env.step(action)
    if done:
        env.reset()

