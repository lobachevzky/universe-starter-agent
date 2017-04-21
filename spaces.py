import numpy as np
from interface import Box


class Continuous(Box):
    def __init__(self, low, high, shape=None):

        def init(bound):
            if hasattr(bound, 'shape'):
                return bound
            else:
                assert shape is not None, \
                    "if either `low` or `high` are floats, then `shape` must be specified"
                return bound + np.zeros(shape)

        self._low = init(low)
        self._high = init(high)
        assert (self._high.shape == self._low.shape)

    def dim(self):
        return self._low.size

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def n(self):
        return 2  # for mean and stddev

    @property
    def shape(self):
        return self._low.shape


class ActionSpace(Continuous):
    pass


class ObservationSpace(Continuous):
    pass
