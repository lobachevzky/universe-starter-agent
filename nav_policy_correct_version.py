from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version
import abc
from gaussian_log import NormalWithLogScale
from spatial_transformer import transformer
from model import use_tf100_api, EPSILON
from model import Policy


# todo: add multidimensional information to the map?
# noinspection PyAttributeOutsideInit
class NavPolicy2(Policy):
    def pass_through_network(self, x):

        # hyperparameters
        height = width = 8
        lidar_size = 10

        # LSTM
        splits = [1, 2, height / 2 * width / 2, 2 * lidar_size]
        lstm_size = sum(splits)
        step_size = tf.shape(self.x)[0]
        if use_tf100_api:
            lstm = rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)
        else:
            lstm = rnn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)
        self.state_size = lstm.state_size

        # initial values
        m_shape = height, width
        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        m_init = np.zeros(m_shape, np.float32)
        self.state_init = [c_init, h_init, m_init]

        # updated state gets passed to these placeholders every time-step
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        m_in = tf.placeholder(tf.float32, m_shape)

        # this is not really the 'initial_state of the LSTM --
        # it actually gets passed in every time-step from the previous time-step
        self.state_in = [c_in, h_in, m_in]
        if use_tf100_api:
            state_in = rnn.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        x = tf.expand_dims(x, 0)  # for LSTM
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=[step_size],
            time_major=False)  # lstm_outputs 1 x step_size x lstm_size
        lstm_outputs = tf.squeeze(lstm_outputs, 0)

        # these are the parameters that determine the mutation of the hidden_map
        angle, translation, alpha, lidar_ = tf.split(lstm_outputs, splits, axis=1)

        # I need to think about this: might be a bug. Usually, only one observation gets
        # processed by `pass_through_network`, but in the case of `train_op`, an entire
        # batch of observations get passed through. Here, we replicate the hidden_map
        # for each one. What should happen, is each observation should have its own
        # distinct hidden_map. Not sure how exactly to accomplish that
        hidden_map = tf.tile(m_in, [step_size, 1])  # step_size * height, width
        hidden_map = tf.reshape(hidden_map, [step_size, height, width, 1])

        # linear transform (rotation and translation) of map
        theta = tf.stack([
            tf.concat([tf.cos(angle), tf.sin(angle)], 1),
            tf.concat([-tf.sin(angle), tf.cos(angle)], 1),
            translation
        ], axis=2)
        hidden_map = transformer(hidden_map, theta, (height, width))

        lidar = tf.reshape(lidar_, [-1, 2, lidar_size])
        meshgrid = tf.to_float(
            tf.meshgrid(tf.range(width / 2),
                        tf.range(height / 2 - 1, -1, -1))
        )

        def pad(quarter, value):
            half = tf.concat([quarter, value * tf.ones_like(quarter)], 1)
            whole = tf.concat([half, value * tf.ones_like(half)], 2)
            return tf.expand_dims(whole, 3)

        # this loop transforms the "lidar readings" produced from the LSTM into a top-down 2d map
        for lidar_index, lidar_params in enumerate(tf.unstack(lidar, axis=2)):
            def in_zone_function(i, j):
                angle = np.arctan2(height / 2 - i - 1, j)
                arc = lidar_size * angle / (np.pi / 2)
                return np.bitwise_and(lidar_index <= arc, arc <= lidar_index + 1)

            # defines the gaussian for the reading
            loc, log_scale = [tf.reshape(tensor, [-1, 1, 1]) for tensor
                              in tf.unstack(lidar_params, axis=1)]
            grid_distances_ = tf.norm(meshgrid, axis=0)
            grid_distances = tf.expand_dims(grid_distances_, 0)
            dist = NormalWithLogScale(loc, log_scale)

            # `mask` zeroes out values between the sensor and lidar reading
            mask_ = dist.cdf(grid_distances)
            # alternate versions:
            # mask_ = tf.clip_by_value(grid_distances - loc, 0, 1)
            # mask_ = tf.sigmoid(grid_distances - loc)

            # `add` contains new values to be added to the map
            add_ = dist.prob(grid_distances)
            # alternate version:
            # add = tf.maximum(0.0, 1 - tf.abs(grid_distances - loc))

            # `in_zone` ensures that a lidar reading only
            # mutates the associated arc of the map
            in_zone = np.fromfunction(in_zone_function, (height / 2, width / 2))
            in_zone = np.expand_dims(in_zone, 0)

            # alpha determines the interpolation between old and new
            alpha = tf.reshape(alpha, [-1, height / 2, width / 2])
            change_value = in_zone * alpha

            mask = tf.exp(change_value * tf.log(tf.maximum(EPSILON, mask_)))
            add = change_value * add_
            hidden_map *= pad(mask, 1)
            hidden_map += pad(add, 0)

        lstm_c, lstm_h = state_in  # lstm_state, both 1 x size

        self.state_out = [lstm_c[:1, :], lstm_h[:1, :], m_in]
        return tf.reshape(hidden_map, [step_size, height * width])

    def get_initial_features(self, last_features=None):
        if last_features is None:
            c, h, m = self.state_init
        else:
            c, h, m = last_features
        return [c, h, m]

    def act(self, ob, c, h, m):
        sess = tf.get_default_session()
        return sess.run([self.action, self.vf] + self.state_out,
                        {self.x: [ob],
                         self.state_in[0]: c,
                         self.state_in[1]: h,
                         self.state_in[2]: m,
                         })

    def value(self, ob, c, h, m):
        sess = tf.get_default_session()
        return sess.run(self.vf,
                        {self.x: [ob],
                         self.state_in[0]: c,
                         self.state_in[1]: h,
                         self.state_in[2]: m,
                         })[0]
class NavPolicy2(Policy):
    def pass_through_network(self, x):

        # hyperparameters
        height = width = 8
        lidar_size = 10

        # LSTM
        splits = [1, 1, 2, 2 * lidar_size]
        lstm_size = sum(splits)
        step_size = tf.shape(self.x)[0]
        if use_tf100_api:
            lstm = rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)
        else:
            lstm = rnn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)
        self.state_size = lstm.state_size

        # initial values
        m_shape = height, width
        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        m_init = np.zeros(m_shape, np.float32)
        self.state_init = [c_init, h_init, m_init]

        # updated state gets passed to these placeholders every time-step
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        m_in = tf.placeholder(tf.float32, m_shape)

        # this is not really the 'initial_state of the LSTM --
        # it actually gets passed in every time-step from the previous time-step
        self.state_in = [c_in, h_in, m_in]
        if use_tf100_api:
            state_in = rnn.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        x = tf.expand_dims(x, 0)  # for LSTM
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=[step_size],
            time_major=False)  # lstm_outputs 1 x step_size x lstm_size
        lstm_outputs = tf.squeeze(lstm_outputs, 0)

        # these are the parameters that determine the mutation of the hidden_map
        alpha, angle, translation, lidar_ = tf.split(lstm_outputs, splits, axis=1)

        # I need to think about this: might be a bug. Usually, only one observation gets
        # processed by `pass_through_network`, but in the case of `train_op`, an entire
        # batch of observations get passed through. Here, we replicate the hidden_map
        # for each one. What should happen, is each observation should have its own
        # distinct hidden_map. Not sure how exactly to accomplish that
        hidden_map = tf.tile(m_in, [step_size, 1])  # step_size * height, width
        hidden_map = tf.reshape(hidden_map, [step_size, height, width, 1])

        # linear transform (rotation and translation) of map
        theta = tf.stack([
            tf.concat([tf.cos(angle), tf.sin(angle)], 1),
            tf.concat([-tf.sin(angle), tf.cos(angle)], 1),
            translation
        ], axis=2)
        hidden_map = transformer(hidden_map, theta, (height, width))

        lidar = tf.reshape(lidar_, [-1, 2, lidar_size])
        meshgrid = tf.to_float(
            tf.meshgrid(tf.range(width / 2),
                        tf.range(height / 2 - 1, -1, -1))
        )

        def pad(quarter, value):
            half = tf.concat([quarter, value * tf.ones_like(quarter)], 1)
            whole = tf.concat([half, value * tf.ones_like(half)], 2)
            return tf.expand_dims(whole, 3)

        # this loop transforms the "lidar readings" produced from the LSTM into a top-down 2d map
        for lidar_index, lidar_params in enumerate(tf.unstack(lidar, axis=2)):
            def in_zone_function(i, j):
                angle = np.arctan2(height / 2 - i - 1, j)
                arc = lidar_size * angle / (np.pi / 2)
                return np.bitwise_and(lidar_index <= arc, arc <= lidar_index + 1)

            # defines the gaussian for the reading
            loc, log_scale = [tf.reshape(tensor, [-1, 1, 1]) for tensor
                              in tf.unstack(lidar_params, axis=1)]
            grid_distances_ = tf.norm(meshgrid, axis=0)
            grid_distances = tf.expand_dims(grid_distances_, 0)
            dist = NormalWithLogScale(loc, log_scale)

            # `mask` zeroes out values between the sensor and lidar reading
            mask_ = dist.cdf(grid_distances)
            # alternate versions:
            # mask_ = tf.clip_by_value(grid_distances - loc, 0, 1)
            # mask_ = tf.sigmoid(grid_distances - loc)

            # `add` contains new values to be added to the map
            add_ = dist.prob(grid_distances)
            # alternate version:
            # add = tf.maximum(0.0, 1 - tf.abs(grid_distances - loc))

            # `in_zone` ensures that a lidar reading only
            # mutates the associated arc of the map
            in_zone = np.fromfunction(in_zone_function, (height / 2, width / 2))
            in_zone = np.expand_dims(in_zone, 0)

            # alpha determines the interpolation between old and new
            alpha = tf.reshape(alpha, [-1, 1, 1])
            change_value = in_zone * alpha

            mask = tf.exp(change_value * tf.log(tf.maximum(EPSILON, mask_)))
            add = change_value * add_
            hidden_map *= pad(mask, 1)
            hidden_map += pad(add, 0)

        lstm_c, lstm_h = state_in  # lstm_state, both 1 x size

        self.state_out = [lstm_c[:1, :], lstm_h[:1, :], m_in]

        hidden_map = tf.sigmoid(hidden_map)  # try with and without this line
        return tf.reshape(hidden_map, [step_size, height * width])

    def get_initial_features(self, last_features=None):
        if last_features is None:
            c, h, m = self.state_init
        else:
            c, h, m = last_features
        return [c, h, m]

    def act(self, ob, c, h, m):
        sess = tf.get_default_session()
        return sess.run([self.action, self.vf] + self.state_out,
                        {self.x: [ob],
                         self.state_in[0]: c,
                         self.state_in[1]: h,
                         self.state_in[2]: m,
                         })

    def value(self, ob, c, h, m):
        sess = tf.get_default_session()
        return sess.run(self.vf,
                        {self.x: [ob],
                         self.state_in[0]: c,
                         self.state_in[1]: h,
                         self.state_in[2]: m,
                         })[0]
