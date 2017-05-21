from __future__ import print_function
import tensorflow as tf
import numpy as np
import time

EPSILON = 1e-9
step_size = 2
map_height, map_width = 6, 9
half_height, half_width = [x / 2.0 for x in [map_height, map_width]]
meshgrid = tf.to_float(tf.meshgrid(tf.range(map_width), tf.range(map_height - 1, -1, -1)))
print(meshgrid.get_shape())
# hidden_map = np.random.operator((map_height, map_width), maxval=10)
hidden_map = np.random.randint(0, 9, size=(step_size, map_height, map_width))
print('map')
print(hidden_map)

hidden_map = tf.constant(hidden_map, dtype=tf.float32)
lidar_size = 6
lidar = tf.random_uniform((step_size, lidar_size))
# alpha = 1
alpha = tf.zeros(step_size)

add = tf.zeros_like(hidden_map)
mask = tf.ones_like(hidden_map)
for y in range(map_height):
    for x in range(map_width):
        angle = np.arctan2(float(y), np.maximum(EPSILON, float(x)))
        lidar_index = int(angle / (np.pi / 2) * lidar_size)
        lidar_distances = lidar[:, lidar_index]
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(lidar_distances), [step_size])
        ]):
            lidar_point = tf.stack([lidar_distances * component for component
                                    in [np.cos(angle), np.sin(angle)]], axis=1)
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(lidar_point), [step_size, 2]),
            tf.assert_equal(tf.shape(meshgrid), [2, map_height, map_width]),
        ]):
            lidar_point_ = tf.reshape(lidar_point, [-1, 2, 1, 1])
            meshgrid_ = tf.expand_dims(meshgrid, 0)

        with tf.control_dependencies([
            tf.assert_equal(tf.shape(meshgrid_), [1, 2, map_height, map_width]),
            tf.assert_equal(tf.shape(lidar_point_), [step_size, 2, 1, 1])
        ]):
            norm = tf.norm(lidar_point_ - meshgrid_, axis=1)

        lidar_distances = tf.reshape(lidar_distances, [-1, 1, 1, 1])
        mask = tf.clip_by_value(tf.norm(meshgrid_, axis=1) - lidar_distances, 0, 1)

        with tf.control_dependencies([
            tf.assert_equal(tf.shape(norm), [step_size, map_height, map_width])
        ]):
            add = tf.maximum(0.0, 1 - norm)


        def in_zone_function(i, j):
            angle = np.arctan2(map_height - i - 1, j)
            arc = lidar_size * angle / (np.pi / 2)
            return np.bitwise_and(lidar_index <= arc, arc <= lidar_index + 1)


        in_zone = np.fromfunction(in_zone_function, (map_height, map_width))
        alpha = tf.reshape(alpha, [-1, 1, 1])
        in_zone = np.expand_dims(in_zone, 0)
        change_value = alpha * in_zone


        with tf.control_dependencies([
            tf.assert_equal(tf.shape(change_value), [step_size, map_height, map_width]),
            tf.assert_equal(tf.shape(meshgrid_), [1, 2, map_height, map_width]),
            tf.assert_equal(tf.shape(add), [step_size, map_height, map_width]),
            tf.assert_equal(tf.shape(mask), [step_size, 1, map_height, map_width])
        ]):
            hidden_map *= tf.exp(change_value * tf.log(tf.maximum(EPSILON, mask)))
            hidden_map += change_value * add


with tf.Session() as sess:
    def print_val(x, f=None, name=None):
        if type(x) is str:
            print(x + ': ')
            x = eval(x)
        val = sess.run(x)
        if name is not None:
            print(name + ': ')
        if f is None:
            def f(x):
                return x
        try:
            for elt in val:
                print(f(elt))
        except TypeError:
            print(f(val))
        print()


    def print_shape(x, name=None):
        print_val(x, lambda x: x.shape, name)


    print('map')
    print_val(hidden_map, f=lambda x: np.round(x, 0))

    # time.sleep(.1)
    # print('add')
    # print_val(value_add,f=lambda x: np.round(x, 0))
    # print('mask')
    # print_val(exp,f=lambda x: np.round(x, 0))
    # print_val(mask, f=lambda x: np.round(x, 0))
