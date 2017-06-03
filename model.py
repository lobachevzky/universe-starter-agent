from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version
import abc
from gaussian_log import NormalWithLogScale
from spatial_transformer import transformer

use_tf100_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('1.0.0')
EPSILON = 0.00001


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


def normal_dist(param_tensor):
    return NormalWithLogScale(*tf.unstack(param_tensor, axis=2))


def categorical_dist(param_tensor):
    max = tf.reduce_max(param_tensor, axis=2, keep_dims=True)  # [bsize, 1, 1]
    return tf.contrib.distributions.Categorical(logits=(param_tensor - max))


def get_action(param_tensor, ac_shape, continuous):
    dist = normal_dist(param_tensor) if continuous else categorical_dist(param_tensor)
    return tf.reshape(dist.sample(), ac_shape)


def log_prob(actions, dist, discrete):
    if discrete:
        actions = tf.to_int32(actions)
    return tf.reduce_sum(dist.log_prob(actions))


class Policy(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, ob_space, ac_space):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space.shape))

        self.ac_space = ac_space
        self.ob_space = ob_space

        # pull out 'subsections' if specified by the environment. This allows the environment to pass
        # in multiple inputs with different shapes (e.g. an image along with the previous action)
        try:
            subspaces = tf.split(x, ob_space.subspaces, axis=1)
            obs = []
            for subsection, shape in zip(subspaces, ob_space.subspace_shapes):
                obs += [tf.reshape(subsection, [-1] + list(shape))]
        except AttributeError:
            obs = [x]
        for i, ob in enumerate(obs):
            if len(list(ob.shape)) == 4:
                for j in range(4):
                    obs[i] = tf.nn.elu(conv2d(obs[i], 32, "l{}".format(j + 1), [3, 3], [2, 2]))
            obs[i] = flatten(obs[i])
        obs = tf.concat(obs, axis=1)

        h = self.pass_through_network(obs)

        if ac_space.is_continuous:
            n_dist_params = 2
        else:
            n_dist_params = ac_space.n

        self.dist_params = linear(h, ac_space.dim() * n_dist_params, "action", normalized_columns_initializer(0.01))
        self.dist_params = tf.reshape(self.dist_params, [-1, ac_space.dim(), n_dist_params])
        self.vf = tf.reshape(linear(h, 1, "value", normalized_columns_initializer(1.0)), [-1])

        if ac_space.is_continuous:
            mean, stdev = tf.unstack(self.dist_params, axis=2)
            self.dist = NormalWithLogScale(mean, stdev)
            self.action = tf.reshape(self.dist.sample(), ac_space.shape)
        else:
            max = tf.reduce_max(self.dist_params, axis=2, keep_dims=True)  # [bsize, 1, 1]
            logits = self.dist_params - max
            self.dist = tf.contrib.distributions.Categorical(logits=logits)
            self.action = tf.squeeze(self.dist.sample())  # [bsize, ac_space.dim]

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def log_prob(self, actions):
        if self.ac_space.is_discrete:
            actions = tf.to_int32(actions)
        return tf.reduce_sum(self.dist.log_prob(actions), axis=1)

    @abc.abstractmethod
    def get_initial_features(self, last_features=None):
        pass

    @abc.abstractmethod
    def act(self, *args):
        pass

    @abc.abstractmethod
    def value(self, *args):
        pass

    @abc.abstractmethod
    def pass_through_network(self, x):
        pass


class MLPpolicy(Policy):
    def pass_through_network(self, x):
        size1, size2 = 60, 60
        h = tf.nn.elu(linear(x, size1, 'h1'))
        return tf.nn.elu(linear(h, size2, 'h2'))

    def get_initial_features(self, last_features=None):
        return []

    def act(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.action, self.vf],
                        {self.x: [ob]})

    def value(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob]})[0]


# params taken from paper
class LSTMpolicy(Policy):
    def pass_through_network(self, x):
        x = tf.nn.elu(linear(x, 200, 'h0'))
        x = tf.expand_dims(x, [0])

        # size = 256
        size = 128
        if use_tf100_api:
            lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
        else:
            lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)

        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]

        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        if use_tf100_api:
            state_in = rnn.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)

        lstm_c, lstm_h = lstm_state
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        return tf.reshape(lstm_outputs, [-1, size])

    def get_initial_features(self, last_features=None):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.action, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]


# noinspection PyAttributeOutsideInit
class NavPolicy1(Policy):
    def pass_through_network(self, x):
        hidden_size = 50
        height = width = 4
        lstm_size = 4 + (height * width * hidden_size) / 4
        step_size = tf.shape(self.x)[0]

        if use_tf100_api:
            lstm = rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)
        else:
            lstm = rnn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)

        self.state_size = lstm.state_size

        m_shape = height, width, hidden_size
        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        m_init = np.zeros(m_shape, np.float32)

        self.state_init = [c_init, h_init, m_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        m_in = tf.placeholder(tf.float32, m_shape)

        self.state_in = [c_in, h_in, m_in]

        if use_tf100_api:
            state_in = rnn.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)

        x = tf.expand_dims(x, 0)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=[step_size],
            time_major=False)  # lstm_outputs 1 x step_size x lstm_size

        lstm_outputs = tf.squeeze(lstm_outputs, 0)

        alpha, angle, translation, add = tf.split(lstm_outputs, [1, 1, 2, -1], axis=1)
        alpha = tf.reshape(alpha, [step_size, 1, 1, 1])
        add = tf.reshape(add, [step_size, height / 2, width / 2, hidden_size])
        add = tf.concat([add, tf.zeros_like(add)], 1)
        add = tf.concat([add, tf.zeros_like(add)], 2)

        theta = tf.stack([
            tf.concat([tf.cos(angle), tf.sin(angle)], 1),
            tf.concat([-tf.sin(angle), tf.cos(angle)], 1),
            translation
        ], axis=2)

        hidden_map = tf.tile(m_in, [step_size, 1, 1])  # step_size * height, width, hidden_size
        hidden_map = tf.reshape(hidden_map, [step_size, height, width, hidden_size])

        transformed = transformer(hidden_map, theta, (height, width))
        transformed = alpha * transformed + (1 - alpha) * add

        lstm_c, lstm_h = lstm_state  # , both 1 x size

        self.state_out = [lstm_c[:1, :], lstm_h[:1, :], m_in]
        return tf.reduce_sum(transformed, axis=[1, 2])

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

