from __future__ import print_function
import tensorflow as tf
import numpy as np
import time

from gaussian_log import NormalWithLogScale

EPSILON = 1e-9
step_size = 2
height, width = 6, 9
meshgrid = tf.to_float(tf.meshgrid(tf.range(width), tf.range(height - 1, -1, -1)))
print(meshgrid.get_shape())
# hidden_map = np.random.operator((map_height, map_width), maxval=10)
hidden_map = np.random.randint(0, 9, size=(step_size, height, width))
print('map')
print(hidden_map)

hidden_map = tf.constant(hidden_map, dtype=tf.float32)
lidar_size = 6
lidar = np.random.randint(0, 9, size=(step_size, 2, lidar_size)).astype(np.float32)
alpha = tf.ones((step_size, height, width))

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

add = tf.zeros_like(hidden_map)
mask = tf.ones_like(hidden_map)
for lidar_index, lidar_params in enumerate(tf.unstack(lidar, axis=2)):
    loc, log_scale = tf.unstack(lidar_params, axis=1)
    grid_distances = tf.norm(meshgrid, axis=0)
    grid_distances = tf.expand_dims(grid_distances, 0)
    with tf.control_dependencies([
        tf.assert_equal(tf.shape(grid_distances), [1, height, width]),
        tf.assert_equal(tf.shape(loc), [2])
    ]):
        loc = tf.reshape(loc, [-1, 1, 1])
        log_scale = tf.reshape(log_scale, [-1, 1, 1])
        dist = NormalWithLogScale(loc, log_scale)
        mask = dist.cdf(grid_distances)
    # mask = tf.sigmoid(grid_distances - lidar_distances)
    # add = tf.maximum(0.0, 1 - tf.abs(grid_distances - lidar_distances))

    add = dist.prob(grid_distances)
    print_val('add', f=lambda x: np.round(x, 0))
    # add = dist.cdf()


    def in_zone_function(i, j):
        angle = np.arctan2(height - i - 1, j)
        arc = lidar_size * angle / (np.pi / 2)
        return np.bitwise_and(lidar_index <= arc, arc <= lidar_index + 1)


    in_zone = np.fromfunction(in_zone_function, (height, width))
    in_zone = np.expand_dims(in_zone, 0)
    alpha = tf.reshape(alpha, [-1, height, width])
    change_value = in_zone * alpha
    exp = tf.exp(change_value * tf.log(tf.maximum(EPSILON, mask)))
    hidden_map *= exp
    value_add = change_value * add
    hidden_map += value_add

print('lidar')
print(lidar)

print_val(hidden_map, name='hidden_map', f=lambda x: np.round(x, 0))

    # print_val(mask, name='mask', f=lambda x: np.round(x, 0))
    # print_val(exp, name='mask2', f=lambda x: np.round(x, 0))
    # exit
    # print_val(add, name='add', f=lambda x: np.round(x, 0))

    # time.sleep(.1)
    # print('add')
    # print_val(value_add,f=lambda x: np.round(x, 0))
    # print('mask')
    # print_val(exp,f=lambda x: np.round(x, 0))
    # print_val(mask, f=lambda x: np.round(x, 0))
