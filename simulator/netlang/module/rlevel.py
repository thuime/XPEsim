from netlang import functional
from netlang.core import Scope, wired
from netlang.module.concept import Forwardable, Parameterable


class RLevelLinear(Forwardable, Parameterable):
    @Scope.scope_args()
    def __init__(self, linear, noise, num_bits, factor, name=None):
        super(RLevelLinear, self).__init__(name)
        w_value = linear.parameter('weight')
        b_value = linear.parameter('bias')
        factor_value = linear.parameter('factor')
        self._w = self.add_parameter(value=w_value, name='weight')
        self._b = self.add_parameter(value=b_value, name='bias')
        self._noise = noise
        self._num_bits = num_bits
        self._one_cell_factor = factor
        self._cells_factor = factor_value
        self._factor = (factor * factor_value).astype('float32')

    @wired
    def forward(self, x):
        x = functional.reshape(x, shape=(x.shape[0], -1))

        maximum = 2 ** self._num_bits - 1
        w = self._w * self._factor
        w = functional.where(w < maximum, w, maximum)
        w = functional.where(w > -maximum, w, -maximum)
        w = w.round()

        w += functional.random_normal(w.shape, 0, self._noise * self._one_cell_factor, w.dtype)

        w /= self._factor

        return functional.dot(x, w) + self._b
