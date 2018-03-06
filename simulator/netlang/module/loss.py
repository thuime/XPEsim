from netlang import functional
from netlang.core import Module, wired


class Loss(Module):
    def __init__(self, name):
        super(Loss, self).__init__(name)

    def minimize(self, predict, target):
        raise NotImplementedError

class CrossEntropy(Loss):
    def __init__(self, name=None):
        super(CrossEntropy, self).__init__(name)

    @wired
    def minimize(self, predict, target):
        return functional.mean(functional.crossentropy(predict, target))

class MeanSquareError(Loss):
    def __init__(self, name):
        super(MeanSquareError, self).__init__(name)

    @wired
    def minimize(self, predict, target):
        return functional.mean(functional.square(predict - target))