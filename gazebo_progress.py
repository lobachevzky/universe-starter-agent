from __future__ import absolute_import, print_function

import copy
import os
import random
from Queue import Queue
from threading import Lock

import gym
import numpy as np
import re
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


def set_random_pos():
    call_service('/gazebo/set_model_state', SetModelState, [ModelState(
        model_name='quadrotor',
        pose=Pose(position=Point(
            x=random.uniform(0, 50),
            y=random.uniform(0, 50),
            z=1
        ))
    )])


class Gazebo(gym.Env):
    def __init__(self, action_shape, reward_file='reward.csv'):

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
        rospy.loginfo('Waiting for first images...')
        while True:
            with self._images_lock:
                if self._image_queue.full():
                    image_shape = self._images.shape
                    rospy.loginfo(image_shape)
                    break
            # prevent CPU burnout
            rospy.wait_for_message('ardrone/image_raw', Image)
        rospy.loginfo('Got image dimensions.')

        # spaces
        image_size = np.array(image_shape).prod(dtype=int)
        action_size = 3
        observation_size = image_size + action_size
        self._observation_space = ObservationSpace((observation_size,))
        self._action_space = ActionSpace(action_shape)

        self._observation_space.subsections = [image_size, action_size]
        self._observation_space.subsection_shapes = [image_shape, (action_size,)]

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

    def _step(self, action):
        self._action_publisher.publish(action_msg(*action))  # do_action
        rospy.wait_for_message('ardrone/image_raw', Image)  # take at most one step per image
        with self._done_lock, self._images_lock:
            progress = calculate_progress(self._tf_listener)
            reward = calculate_reward(progress, self._progress)
            self._progress = progress
            new_state = np.concatenate((self._images.flatten(), action), 1)
            print("shape of new state: {}".format(new_state.shape))
            print("shape of substates: {}".format(self._observation_space.subspace_shapes))
            return new_state, reward, self._done, {}

    def _takeoff(self):
        self._takeoff_publisher.publish(msg.Empty())

    def _reset(self):
        with self._done_lock:
            call_service('gazebo/reset_world')
            # set_random_pos()
            self._takeoff()
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
        return np.expand_dims(images, 3)  # for convolutions

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def reward_range(self):
        return -np.inf, np.inf

    # dummy methods to meet interface

    def _render(self, **kwargs):
        pass

    def _close(self):
        pass

    def _configure(self):
        pass

    def _seed(self, **kwargs):
        pass
