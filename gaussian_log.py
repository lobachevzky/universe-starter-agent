import tensorflow as tf
from tensorflow.contrib.distributions import Normal
import math

class NormalWithLogScale(Normal):
    """Normal with log scales."""

    def __init__(self,
                 loc,
                 log_scale,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="NormalWithLogScale"):
        parameters = locals()
        with tf.name_scope(name, values=[log_scale]):
            super(NormalWithLogScale, self).__init__(
                loc=loc,
                scale=tf.exp(log_scale),
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                name=name)
            self._parameters = parameters
            self._log_scale = log_scale

    @property
    def log_scale(self):
        return self._log_scale

    def _z(self, x):
        return (x - self.loc) * tf.exp(-self.log_scale)

    def _log_normalization(self):
        return 0.5 * math.log(2.0 * math.pi) + self.log_scale

    def _entropy(self):
        # Use broadcasting rules to calculate the full broadcast scale.
        log_scale = self.log_scale * tf.ones_like(self.loc)
        return 0.5 * math.log(2. * math.pi * math.e) + log_scale


