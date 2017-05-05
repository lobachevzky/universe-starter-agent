#! /usr/bin/env python

import rospy
from std_msgs import msg
rospy.init_node('env')
pub = rospy.Publisher('ardrone/takeoff', msg.Empty, queue_size=1)
while not rospy.is_shutdown():
    pub.publish(msg.Empty())
