from netlang.initializer import Constant

from netlang import functional
from netlang.core import Module, apply, Wire
from netlang.module.concept import Stateful


class Optimizer(Module):
    def __init__(self, name):
        super(Optimizer, self).__init__(name)

    def updates(self, parameters, gradients):
        raise NotImplementedError


class SGD(Optimizer, Stateful):
    def __init__(self, learning_rate, momentum, name=None):
        super(SGD, self).__init__(name)
        self._learning_rate = learning_rate
        self._momentum = momentum

    def updates(self, parameters, gradients):
        parameters_cg = apply(parameters)
        for parameter in parameters_cg:
            value = functional.get_value(parameter)
            self.add_states(value=Constant(0.0)(shape=value.shape, dtype=value.dtype), name='%s-momentum' % parameter.name)

        def sgd_apply(ps, gs, ms):
            updates = []
            for p, g, m in zip(ps, gs, ms):
                new_m = self._momentum * m - self._learning_rate * g
                new_p = p + new_m
                updates.append((m, new_m))
                updates.append((p, new_p))
            return updates

        return Wire(
            inputs=[parameters, gradients, self._states],
            apply_function=sgd_apply,
            owner=self
        )