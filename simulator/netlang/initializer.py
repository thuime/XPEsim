import numpy

from netlang import functional


class Initializer(object):
    def __call__(self, shape, dtype):
        raise NotImplementedError


class Constant(Initializer):
    def __init__(self, value=0.0):
        self._value = value

    def __call__(self, shape, dtype=functional.floatX):
        if self._value == 0.0:
            return numpy.zeros(shape=shape, dtype=dtype)
        elif self._value == 1.0:
            return numpy.ones(shape=shape, dtype=dtype)
        else:
            return numpy.ones(shape=shape, dtype=dtype) * self._value


class Normal(Initializer):
    def __init__(self, mean=0.0, std=1.0):
        self._mean = mean
        self._std = std

    def __call__(self, shape, dtype=functional.floatX):
        return numpy.random.normal(loc=self._mean, scale=self._std, size=shape).astype(dtype=dtype)


class Uniform(Initializer):
    def __init__(self, low=0.0, high=1.0):
        self._low = low
        self._high = high

    def __call__(self, shape, dtype=functional.floatX):
        return numpy.random.uniform(low=self._low, high=self._high, size=shape).astype(dtype=dtype)


class Xavier(Initializer):
    def __init__(self, distribution='uniform'):
        self._distribution = distribution

    def __call__(self, shape, dtype=functional.floatX):
        fan_in = shape[0] if len(shape) is 2 else numpy.prod(shape[1:])
        fan_out = shape[1] if len(shape) is 2 else shape[0]
        if self._distribution is 'uniform':
            scale = numpy.sqrt(6.0 / (fan_in + fan_out))
            return numpy.random.uniform(low=-scale, high=scale, size=shape).astype(dtype=dtype)
        elif self._distribution is 'normal':
            scale = numpy.sqrt(2.0 / (fan_in + fan_out))
            return numpy.random.normal(scale=scale, size=shape).astype(dtype=dtype)
        else:
            raise ValueError('Unknown distribution %s' % self._distribution)
