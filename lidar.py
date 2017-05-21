from __future__ import print_function
import tensorflow as tf
import numpy as np
import time

EPSILON = 1e-9
step_size = 1
height, width = 6, 9
meshgrid = tf.to_float(tf.meshgrid(tf.range(width), tf.range(height - 1, -1, -1)))
print(meshgrid.get_shape())
# hidden_map = np.random.operator((map_height, map_width), maxval=10)
hidden_map = np.random.randint(0, 9, size=(step_size, height, width))
print('map')
print(hidden_map)

hidden_map = tf.constant(hidden_map, dtype=tf.float32)
lidar_size = 6
lidar = 10 * tf.random_uniform((step_size, lidar_size))
alpha = tf.ones(step_size)

add = tf.zeros_like(hidden_map)
mask = tf.ones_like(hidden_map)
for y in range(height):
    for x in range(width):
        angle = np.arctan2(float(y), np.maximum(EPSILON, float(x)))
        lidar_index = int(angle / (np.pi / 2) * lidar_size)
        lidar_distances = lidar[:, lidar_index]
        lidar_point = tf.stack([lidar_distances * component for component
                                in [np.cos(angle), np.sin(angle)]], axis=1)
        lidar_point_ = tf.reshape(lidar_point, [-1, 2, 1, 1])
        meshgrid_ = tf.expand_dims(meshgrid, 0)
        norm = tf.norm(lidar_point_ - meshgrid_, axis=1)
        lidar_distances = tf.reshape(lidar_distances, [-1, 1, 1, 1])
        mask = tf.clip_by_value(tf.norm(meshgrid_, axis=1) - lidar_distances, 0, 1)
        add = tf.maximum(0.0, 1 - norm)


        def in_zone_function(i, j):
            angle = np.arctan2(height - i - 1, j)
            arc = lidar_size * angle / (np.pi / 2)
            return np.bitwise_and(lidar_index <= arc, arc <= lidar_index + 1)


        in_zone = np.fromfunction(in_zone_function, (height, width))
        alpha = tf.reshape(alpha, [-1, 1, 1])
        in_zone = np.expand_dims(in_zone, 0)
        change_value = alpha * in_zone
        exp = tf.exp(change_value * tf.log(tf.maximum(EPSILON, mask)))
        hidden_map *= exp
        value_add = change_value * add
        hidden_map += value_add

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

    print('lidar')
    print_val(lidar, f=lambda x: np.round(x, 0))

    print('map')
    print_val(hidden_map, f=lambda x: np.round(x, 0))

    # time.sleep(.1)
    # print('add')
    # print_val(value_add,f=lambda x: np.round(x, 0))
    # print('mask')
    # print_val(exp,f=lambda x: np.round(x, 0))
    # print_val(mask, f=lambda x: np.round(x, 0))
