from netlang import functional
from netlang.core import Scope, Wire
from netlang.initializer import Constant
from netlang.module.concept import Forwardable, Stateful, Updateful


class Round(Forwardable, Stateful, Updateful):
    @Scope.scope_args()
    def __init__(self, io_bits, name=None):
        super(Round, self).__init__(name)
        self._io_bits = io_bits
        self.factor = self.add_states(Constant(0.0)(shape=()), name='factor')

    def forward(self, x):
        def apply(x_, is_training):
            maximum = 2 ** self._io_bits - 1

            if is_training:
                factor = functional.maximum(self.factor, functional.abs(x_).max())
                self.add_update((self.factor, factor))
            else:
                factor = self.factor

            x_ /= factor
            x_ *= maximum
            x_ = x_.round()
            x_ = functional.where(x_ > maximum, maximum, x_)
            x_ = functional.where(x_ < 0., 0., x_)
            x_ /= maximum
            x_ *= factor

            return x_

        return Wire(
            inputs=x,
            apply_function=apply,
            owner=self,
            branch=True
        )
