from netlang import functional
from netlang.core import Scope, wired
from netlang.module.concept import Forwardable, Parameterable
from netlang.transfer.quantize import quantize_factor, quantize


class QuantizeLinear(Forwardable, Parameterable):
    @Scope.scope_args()
    def __init__(self, linear, num_bits, name=None):
        super(QuantizeLinear, self).__init__(name)
        w_value = linear.parameter('weight')
        b_value = linear.parameter('bias')
        self._w = self.add_parameter(value=w_value, name='weight')
        self._b = self.add_parameter(value=b_value, name='bias')

        factor = quantize_factor(w_value, num_bits)
        factor = factor.astype('float32')
        self._factor = self.add_parameter(value=factor, name='factor')
        self._num_bits = num_bits

    @wired
    def forward(self, x):
        x = functional.reshape(x, shape=(x.shape[0], -1))

        maximum = 2 ** self._num_bits - 1
        w = self._w * self._factor
        w = w.round()
        w = functional.where(w < maximum, w, maximum)
        w = functional.where(w > -maximum, w, -maximum)
        w /= self._factor

        return functional.dot(x, w) + self._b

    def pack(self):
        w_value = self.parameter('weight')
        factor = self.parameter('factor')
        w_value = quantize(w_value, factor, self._num_bits)
        functional.set_value(self._w, w_value)
        return super(QuantizeLinear, self).pack()
