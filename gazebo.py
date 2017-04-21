from __future__ import absolute_import, print_function

import copy
import os
from Queue import Queue
from threading import Lock

import numpy as np
import re
from interface import Env

import rospy
import tf
from cv_bridge import CvBridge
from gazebo_msgs.msg import ContactsState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from spaces import ActionSpace, ObservationSpace
from std_msgs import msg
from std_srvs import srv


def calculate_progress(tf_listener):
    target_frame = '/base_link'
    source_frame = '/nav'
    if tf_listener.frameExists(target_frame) and tf_listener.frameExists(source_frame):
        time = tf_listener.getLatestCommonTime(target_frame, source_frame)
        (x, y, z), _ = tf_listener.lookupTransform(target_frame, source_frame, time)
        return np.linalg.norm(np.array([x, y, z]), ord=2)
    else:
        return 0


def concat_images(images, cv_bridge):
    return np.concatenate(map(cv_bridge.imgmsg_to_cv2, images))


def call_service(service, message_type=srv.Empty, args=None):
    if args is None:
        args = []
    rospy.wait_for_service(service)
    return rospy.ServiceProxy(service, message_type)(*args)


def action_msg(lx, ly, az):
    linear = Vector3(x=lx, y=ly)
    angular = Vector3(z=az)
    return Twist(linear, angular)


def calculate_reward(new_progress, old_progress):
    return new_progress - old_progress


def get_debug_num(contact_msg):
    first_match = re.findall('Debug:\s+i:\((\d+)',
                             contact_msg.states[0].info)[0]
    return int(first_match)


class Gazebo(Env):

    def __init__(self,
                 observation_range,
                 action_range,
                 action_shape,
                 reward_file='reward.csv'):

        rospy.init_node('environment')

        self._done = False
        self._tf_listener = tf.TransformListener()
        self._cv_bridge = CvBridge()

        # image variables
        self._num_images = rospy.get_param('num_images')
        self._image_queue = Queue(self._num_images)
        self._images_to_skip = rospy.get_param('images_to_skip')
        self._skipped_images = 0

        # locks
        self._done_lock = Lock()
        self._images_lock = Lock()
        self._skipped_images_lock = Lock()

        # subscriptions
        rospy.Subscriber('ardrone/image_raw', Image, callback=self._update_image)
        rospy.Subscriber('ardrone_crash_bumper', ContactsState, callback=self._contact)

        # publications
        self._action_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self._takeoff_publisher = rospy.Publisher('ardrone/takeoff', msg.Empty, queue_size=1)
        self._land_publisher = rospy.Publisher('ardrone/land', msg.Empty, queue_size=1)

        # get observation shape
        rospy.loginfo('Waiting for first image...')
        while True:
            with self._images_lock:
                if self._image_queue.full():
                    observation_shape = self._images.shape
                    print(observation_shape)
                    break
            # prevent CPU burnout
            rospy.wait_for_message('ardrone/image_raw', Image)

        # spaces
        self._observation_space = ObservationSpace(*observation_range, shape=observation_shape)
        self._action_space = ActionSpace(*action_range, shape=action_shape)
        self._progress = 0  # updated at each call to step
        self._reward_file = reward_file

        try:
            os.remove(reward_file)
            rospy.loginfo('Deleted old rewards file.')
        except OSError:
            rospy.loginfo('No previous rewards file found.')
        with open(reward_file, 'a') as fp:
            fp.write('time, steps, reward\n')

    # crash_bumper callback
    def _contact(self, contact_msg):
        if contact_msg.states:
            with self._done_lock:
                self._done = True

        # image_raw callback
    def _update_image(self, image_msg):
        with self._skipped_images_lock, self._images_lock:
            if self._skipped_images >= self._images_to_skip:

                # add new image to queue
                if self._image_queue.full():
                    self._image_queue.get()
                self._image_queue.put(image_msg)

                self._skipped_images = 0
            else:
                self._skipped_images += 1

    def step(self, action):
        # self._action_publisher.publish(action_msg(*action))  # do_action
        self._action_publisher.publish(action_msg(1, 1, 1))  # do_action
        rospy.wait_for_message('ardrone/image_raw', Image)  # take at most one step per image
        with self._done_lock, self._images_lock:
            progress = calculate_progress(self._tf_listener)
            reward = calculate_reward(progress, self._progress)
            self._progress = progress
            return self._images, reward, self._done, None

    def reset(self):
        with self._done_lock:
            call_service('gazebo/reset_world')
            self._takeoff_publisher.publish(msg.Empty())
            self._progress = 0
            self._done = False
        with self._images_lock:
            return self._images

    def pause(self):
        self._land_publisher.publish(msg.Empty())

    @property
    def _images(self):
        assert self._image_queue.full()
        images = concat_images(list(self._image_queue.queue), self._cv_bridge)
        return np.expand_dims(images, 3)

    def max_time(self):
        return 300

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def is_gazebo(self):
        return True