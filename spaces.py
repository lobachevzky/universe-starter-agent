import numpy as np
from interface import Box


class Continuous(Box):
    def __init__(self, shape=None):
        self._shape = shape

    def dim(self):
        size = np.array(self._shape).prod()
        assert size == 3, 'size == {}'.format(size)
        return size

    @property
    def n(self):
        return 2  # for mean and stddev

    @property
    def shape(self):
        assert self._shape in [(3,), (72, 32, 1)], 'shape == {}'.format(self._shape)
        return self._shape


class ActionSpace(Continuous):
    pass


class ObservationSpace(Continuous):
    pass
