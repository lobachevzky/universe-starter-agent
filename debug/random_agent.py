#!/usr/bin/env python
import rospy
import numpy as np
import rospy
from gazebo_progress import Gazebo
from std_srvs import srv

env = Gazebo(observation_range=(-1, 1), action_range=(-1, 1), action_shape=(3,))
env.reset()

# rospy.loginfo('resetting...')
# rospy.wait_for_service('gazebo/reset_world')
# rospy.ServiceProxy('gazebo/reset_world', srv.Empty)()
# rospy.loginfo('reset')
while not rospy.is_shutdown():
    # pass
    action = np.random.rand(3)
    _, reward, done, _ = env.step(action)
    if done:
        env.reset()

