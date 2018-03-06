import numpy

from netlang import functional
from netlang.core import Scope, wired
from netlang.module.concept import Forwardable, Parameterable


class NoiseLinear(Forwardable, Parameterable):
    @Scope.scope_args()
    def __init__(self, linear, weight_bits, noise, name=None):
        if not(name is None):
            name = linear.name
        super(NoiseLinear, self).__init__(name)
        w_value = linear.parameter('weight')
        b_value = linear.parameter('bias')
        self._w_max = numpy.abs(w_value).max()
        self._w = self.add_parameter(value=w_value, name='weight')
        self._b = self.add_parameter(value=b_value, name='bias')

        self._weight_bits = weight_bits
        self._noise = noise

    @wired
    def forward(self, x):
        x = functional.reshape(x, shape=(x.shape[0], -1))

        maximum = 2 ** self._weight_bits - 1
        w = self._w / self._w_max
        w *= maximum
        w = w.round()
        w = functional.where(w > maximum, maximum, w)
        w = functional.where(w < -maximum, -maximum, w)
        w /= maximum
        w += functional.random_normal(size=w.shape, mean=0.0, std=self._noise)
        w *= self._w_max

        return functional.dot(x, w) + self._b


class NoiseConv2d(Forwardable, Parameterable):
    @Scope.scope_args()
    def __init__(self, conv2d, weight_bits, noise, name=None):
        if name is None:
            name = conv2d.name
        super(NoiseConv2d, self).__init__(name)
        k_value = conv2d.parameter('kernel')
        b_value = conv2d.parameter('bias')
        self._k_max = numpy.abs(k_value).max()
        self._k = self.add_parameter(value=k_value, name='weight')
        self._b = self.add_parameter(value=b_value, name='bias')

        self._weight_bits = weight_bits
        self._noise = noise

        self._stride = conv2d._stride
        self._padding = conv2d._padding

    @wired
    def forward(self, x):
        maximum = 2 ** self._weight_bits - 1
        w = self._k / self._k_max
        w *= maximum
        w = w.round()
        w = functional.where(w > maximum, maximum, w)
        w = functional.where(w < -maximum, -maximum, w)
        w /= maximum
        w += functional.random_normal(size=w.shape, mean=0.0, std=self._noise)
        w *= self._k_max

        return functional.conv2d(x, w, self._stride, self._padding) + functional.reshape(self._b, (1, -1, 1, 1))
