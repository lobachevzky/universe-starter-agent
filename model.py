import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version
import abc

use_tf100_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('1.0.0')


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


def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


class Policy(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, ob_space, ac_space):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        if len(list(ob_space)) > 1:
            for i in range(4):
                x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))

        h = self.network(x)

        if ac_space.is_continuous:
            n_dist_params = 2
        else:
            n_dist_params = ac_space.n

        self.dist_params = linear(h, ac_space.dim() * n_dist_params, "action", normalized_columns_initializer(0.01))
        self.dist_params = tf.reshape(self.dist_params, [-1, ac_space.dim(), n_dist_params])
        self.vf = tf.reshape(linear(h, 1, "value", normalized_columns_initializer(1.0)), [-1])

        if ac_space.is_continuous:
            print('GOING WITH CONTINUOUS')
            dp = self.dist_params
            self.dist_params = tf.nn.softplus(self.dist_params + 1)  # alpha and beta are positive
            with tf.control_dependencies(
                    [tf.assert_greater_equal(self.dist_params, 1.0, message="dist_params vs 1"),
                     tf.assert_equal(tf.shape(dp), tf.shape(self.dist_params))]
            ):
                split = tf.unstack(self.dist_params, axis=2)
                self.dist = tf.contrib.distributions.Beta(*split)
            self.action = tf.squeeze(self.dist.sample(), axis=1)  # [bsize, ac_space.dim]
        else:
            max = tf.reduce_max(self.dist_params, axis=2, keep_dims=True)  # [bsize, 1, 1]
            logits = self.dist_params - max
            self.dist = tf.contrib.distributions.Categorical(logits=logits)
            self.action = tf.squeeze(self.dist.sample())  # [bsize, ac_space.dim]
            # self.dist.sample() gives # [bsize, ac_space.dim]

        with tf.control_dependencies([tf.check_numerics(self.action, message="action")]):

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
            self.ac_space = ac_space

    def log_prob(self, action):
        if self.ac_space.is_discrete:
            action = tf.to_int32(action)
        return tf.reduce_prod(self.dist.log_prob(action), axis=1)

    @abc.abstractmethod
    def get_initial_features(self):
        pass

    @abc.abstractmethod
    def act(self, *args):
        pass

    @abc.abstractmethod
    def value(self, *args):
        pass

    @abc.abstractmethod
    def network(self, x):
        pass


class MLPpolicy(Policy):
    def network(self, x):
        size1, size2 = 60, 60
        h = tf.nn.elu(linear(x, size1, 'h1'))
        return tf.nn.elu(linear(h, size2, 'h2'))

    def get_initial_features(self):
        return []

    def act(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.action, self.vf],
                        {self.x: [ob]})

    def value(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob]})[0]


class LSTMpolicy(Policy):
    def network(self, x):
        x = tf.expand_dims(flatten(x), [0])
        size = 256
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

        lstm_c, lstm_h = state_in  # lstm_state
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        return tf.reshape(lstm_outputs, [-1, size])

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        # ob = ob.reshape(-1, *self.x.get_shape()[1:])
        return sess.run([self.action, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]
