import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.rnn import RNNCell
import collections
from model import use_tf100_api, EPSILON

from gaussian_log import NormalWithLogScale
from spatial_transformer import transformer

_MapperStateTuple = collections.namedtuple('MapperStateTuple', ('c', 'h', 'm'))

input_size = 5
hidden_map_size = 8, 10
lidar_size = 10


# noinspection PyClassHasNoInit
class MapperStateTuple(_MapperStateTuple):
    """
    Tuple used by Mapper Cells for `state_size`, `zero_state`, and output state.
    Stores three elements: `(c, h, m)`, in that order.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (c, h, m) = self
        if not c.dtype == h.dtype == m.dtype:
            raise TypeError('Inconsistent internal state: {} vs {} vs {}'
                            .format(c.dtype, h.dtype, m.dtype))
        return c.dtype


class Mapper(RNNCell):
    def __init__(self, hidden_map_size, lidar_size):
        """
        :type splits: list[int]
        :type state_size: (float, float)
        :type hidden_map_size: (float, float)
        """
        self._lidar_size = lidar_size
        self._splits = [1, 2, lidar_size, 2 * lidar_size]
        self._lstm_size = lstm_size = sum(self._splits)
        if use_tf100_api:
            self._lstm = rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)
        else:
            self._lstm = rnn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)
        self._hidden_map_size = hidden_map_size

    @property
    def output_size(self):
        return self._hidden_map_size

    @property
    def state_size(self):
        c, h = self._lstm_size
        return MapperStateTuple(c, h, self._hidden_map_size)

    def __call__(self, inputs, state, scope=None):
        content, hidden, hidden_map = state

        with tf.control_dependencies([
            tf.assert_equal(tf.shape(content), [1, self._lstm_size]),
            tf.assert_equal(tf.shape(hidden), [1, self._lstm_size]),
            tf.assert_equal(tf.shape(hidden_map), [1] + list(hidden_map_size)),
            tf.assert_equal(tf.shape(inputs), [1, input_size]),
        ]):
            # these are the parameters that determine the mutation of the hidden_map
            lstm_state = rnn.LSTMStateTuple(content, hidden)
            lstm_outputs, (content_, hidden_) = self._lstm(inputs, lstm_state)

        with tf.control_dependencies([
            tf.assert_equal(tf.shape(lstm_outputs), [1, self._lstm_size]),
            tf.assert_equal(tf.shape(hidden), [1, self._lstm_size]),
            tf.assert_equal(tf.shape(hidden_map), [1] + list(hidden_map_size)),
        ]):
            height, width = self._hidden_map_size
            angle, translation, alphas, lidar_ = [tf.squeeze(tensor, 0) for tensor in
                                                  tf.split(lstm_outputs, self._splits, axis=1)]

        with tf.control_dependencies([
            tf.assert_equal(tf.shape(angle), [1]),
            tf.assert_equal(tf.shape(translation), [2]),
            tf.assert_equal(tf.shape(alphas), [lidar_size]),
            tf.assert_equal(tf.shape(lidar_), [2 * lidar_size]),
        ]):
            # linear transform (rotation and translation) of map
            concat = tf.concat([tf.cos(angle), tf.sin(angle)], 0)
            tf_concat = tf.concat([-tf.sin(angle), tf.cos(angle)], 0)

        with tf.control_dependencies([
            tf.assert_equal(tf.shape(concat), [2]),
            tf.assert_equal(tf.shape(tf_concat), [2]),
            tf.assert_equal(tf.shape(translation), [2]),
        ]):
            theta = tf.stack([
                concat,
                tf_concat,
                translation
            ], axis=1)

        hidden_map_ = tf.expand_dims(hidden_map, 3)
        theta_ = tf.expand_dims(theta, 0)
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(hidden_map_), [1, height, width, 1]),
            tf.assert_equal(tf.shape(theta_), [1, 2, 3]),
        ]):

            hidden_map__ = transformer(hidden_map_, theta_, (height, width))
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(hidden_map__), [1, height, width, 1]),
        ]):
            hidden_map___ = tf.squeeze(hidden_map__, axis=[0, 3])

        lidar = tf.reshape(lidar_, [self._lidar_size, 2])
        meshgrid = tf.to_float(
            tf.meshgrid(tf.range(width / 2),
                        tf.range(height / 2 - 1, -1, -1))
        )

        def pad(quarter, value):
            half = tf.concat([quarter, value * tf.ones_like(quarter)], 0)
            whole = tf.concat([half, value * tf.ones_like(half)], 1)
            return whole

        # this loop transforms the "lidar readings" produced from the LSTM into a top-down 2d map
        for lidar_index, (lidar_params, alpha) in enumerate(zip(
                tf.unstack(lidar, axis=0),
                tf.unstack(alphas, axis=0)
        )):
            def in_zone_function(i, j):
                angle = np.arctan2(height / 2 - i - 1, j)
                arc = self._lidar_size * angle / (np.pi / 2)
                return np.bitwise_and(lidar_index <= arc, arc <= lidar_index + 1)

            # defines the gaussian for the reading

            with tf.control_dependencies([
                tf.assert_equal(tf.shape(lidar_params), [2]),
            ]):
                loc, log_scale = tf.unstack(lidar_params, axis=0)
            grid_distances_ = tf.norm(meshgrid, axis=0)
            # grid_distances = tf.expand_dims(grid_distances_, 0)
            dist = NormalWithLogScale(loc, log_scale)
            with tf.control_dependencies([
                tf.assert_equal(tf.shape(grid_distances_), [height / 2, width / 2]),
            ]):

                # `mask` zeroes out values between the sensor and lidar reading
                mask_ = dist.cdf(grid_distances_)
                # alternate versions:
                # mask_ = tf.clip_by_value(grid_distances - loc, 0, 1)
                # mask_ = tf.sigmoid(grid_distances - loc)

                # `add` contains new values to be added to the map
                add_ = dist.prob(grid_distances_)
                # alternate version:
                # add = tf.maximum(0.0, 1 - tf.abs(grid_distances - loc))

            # `in_zone` ensures that a lidar reading only
            # mutates the associated arc of the map
            in_zone = np.fromfunction(in_zone_function, (height / 2, width / 2))

            # alpha determines the interpolation between old and new
            with tf.control_dependencies([
                tf.assert_equal(tf.shape(alpha), [1]),
            ]):
                change_value = in_zone * alpha

            mask = tf.exp(change_value * tf.log(tf.maximum(EPSILON, mask_)))
            add = change_value * add_
            hidden_map___ *= pad(mask, 1)
            hidden_map___ += pad(add, 0)
        return hidden_map___, MapperStateTuple(content_, hidden_, hidden_map___)


if __name__ == '__main__':
    mapper = Mapper(hidden_map_size, lidar_size)
    inputs = tf.random_uniform((1, input_size))
    splits = [1, 2, lidar_size, 2 * lidar_size]
    lstm_size = sum(splits)
    c = tf.zeros((1, lstm_size), tf.float32)
    h = tf.zeros((1, lstm_size), tf.float32)
    m = tf.zeros([1] + list(hidden_map_size), tf.float32)
    state = MapperStateTuple(c, h, m)
    hidden_map, (c_, h_, m_) = mapper(inputs, state)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for name, x in zip('hidden_map content hidden maps'.split(),
                           sess.run([hidden_map, c_, h_, m_])):
            print(name)
            print(x.shape)
