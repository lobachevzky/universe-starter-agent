#! /usr/bin/env python

import rospy
from std_srvs import srv
rospy.init_node('reset')
service = 'gazebo/reset_world'
rospy.wait_for_service(service)
rospy.ServiceProxy(service, srv.Empty)()
