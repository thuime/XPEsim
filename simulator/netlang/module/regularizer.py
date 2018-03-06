from netlang import functional
from netlang.core import Module, wired


class Regularizer(Module):
    def __init__(self, name):
        super(Regularizer, self).__init__(name)

    def __call__(self, variable):
        raise NotImplementedError


class NoRegularize(Regularizer):
    def __init__(self, name=None):
        super(NoRegularize, self).__init__(name)
    @wired
    def __call__(self, variable):
        return 0

class L1Norm(Regularizer):
    def __init__(self, scale=1.0, name=None):
        super(L1Norm, self).__init__(name)
        self._scale = scale

    @wired
    def __call__(self, variable):
        return functional.sum(functional.abs(variable)) * self._scale

class L2Norm(Regularizer):
    def __init__(self, scale=1.0, name=None):
        super(L2Norm, self).__init__(name)
        self._scale = scale

    @wired
    def __call__(self, variable):
        return functional.sum(functional.square(variable)) * self._scale