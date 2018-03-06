from netlang import functional
from netlang.core import Scope, wired

from netlang.initializer import Xavier, Constant
from netlang.module.concept import Forwardable, Parameterable
from netlang.module.differentiator import Gradient
from netlang.module.regularizer import NoRegularize


class Linear(Forwardable, Parameterable):
    @Scope.scope_args()
    def __init__(self, w_shape, w_init=Xavier(), b_init=Constant(0.0), w_reg=NoRegularize(), b_reg=NoRegularize(),
                 w_diff=Gradient(), b_diff=Gradient(), name=None):
        super(Linear, self).__init__(name)
        with Scope(name):
            self._w = self.add_parameter(value=w_init(w_shape), regularizer=w_reg, differentiator=w_diff, name='weight')
            self._b = self.add_parameter(value=b_init((w_shape[1],)), regularizer=b_reg, differentiator=b_diff, name='bias')

    @wired
    def forward(self, x):
        x = functional.reshape(x, shape=(x.shape[0], -1))
        return functional.dot(x, self._w) + self._b