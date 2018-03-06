from netlang import functional
from netlang.core import wired, Scope
from netlang.module.concept import Forwardable


class LRN2d(Forwardable):
    @Scope.scope_args()
    def __init__(self, alpha=1e-4, k=1.0, beta=0.75, n=5, name=None):
        super(LRN2d, self).__init__(name)
        self._alpha = alpha
        self._k = k
        self._beta = beta
        self._n = n

    @wired
    def forward(self, x):
        x_sqr = functional.square(x)
        batch, channel, rows, cols = x.shape
        half_n = self._n // 2
        extra_channels = functional.zeros(shape=(batch, half_n, rows, cols), dtype=x.dtype)
        x_sqr = functional.concatenate([extra_channels, x_sqr, extra_channels], axis=1)
        scale = self._k
        for i in xrange(self._n):
            scale += self._alpha * x_sqr[:, i:i + channel] / self._n
        scale = functional.power(scale, self._beta)
        return x / scale