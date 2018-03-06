from netlang import functional
from netlang.core import Module, wired


class Differentiator(Module):
    def __init__(self, name):
        super(Differentiator, self).__init__(name)

    def __call__(self, loss, variable):
        raise NotImplementedError

class NoDifferentiator(Differentiator):
    def __init__(self, name=None):
        super(NoDifferentiator, self).__init__(name)

    @wired
    def __call__(self, loss, variable):
        return 0

class Gradient(Differentiator):
    def __init__(self, scale=1.0, name=None):
        super(Gradient, self).__init__(name)
        self._scale = scale

    @wired
    def __call__(self, loss, variable):
        return functional.gradient(loss, variable) * self._scale