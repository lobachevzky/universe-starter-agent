import numpy as np
from interface import Box


class Continuous(Box):
    def __init__(self, shape=None):
        self._shape = shape

    def dim(self):
        return np.array(self._shape).prod()

    @property
    def n(self):
        return 2  # for mean and stddev

    @property
    def shape(self):
        return self._shape


class ActionSpace(Continuous):
    pass


class ObservationSpace(Continuous):
    pass
