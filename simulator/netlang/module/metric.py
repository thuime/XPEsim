from netlang import functional
from netlang.core import Module, wired


class OneHot(Module):
    def __init__(self, num_classes, name=None):
        super(OneHot, self).__init__(name)
        self._num_classes = num_classes

    @wired
    def turn(self, x):
        return functional.one_hot(x, self._num_classes)


class Accuracy(Module):
    def __init__(self, name=None):
        super(Accuracy, self).__init__(name)

    @wired
    def measure(self, predict, label):
        return functional.mean(functional.equal(functional.argmax(predict, axis=-1), label))

class TopKAccuracy(Module):
    def __init__(self, k=5, name=None):
        super(TopKAccuracy, self).__init__(name)
        self._k = k

    @wired
    def measure(self, predict, label):
        predict_k = functional.sort(predict)[:, -self._k]
        target_v = predict[functional.arange(label.shape[0]), label]
        return functional.mean(functional.greater_equal(target_v, predict_k))