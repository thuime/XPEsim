from netlang import functional
from netlang.core import Scope, wired, Wire
from netlang.module.concept import Forwardable, Parameterable, Stateful
from netlang.transfer.sparse import block_sparse


class SparseLinear(Forwardable, Parameterable, Stateful):
    @Scope.scope_args()
    def __init__(self, linear, block_size, pruning_rate, name=None):
        super(SparseLinear, self).__init__(name)
        w_value = linear.parameter('weight')
        b_value = linear.parameter('bias')
        self._w = self.add_parameter(value=w_value, name='weight')
        self._b = self.add_parameter(value=b_value, name='bias')

        mask, row_order, col_order = block_sparse(w_value, block_size, pruning_rate)

        self._mask = self.add_states(value=mask, name='mask')

    @wired
    def forward(self, x):
        x = functional.reshape(x, shape=(x.shape[0], -1))
        w = functional.where(self._mask, self._w, 0)
        return functional.dot(x, w) + self._b


class SparseConv2d(Forwardable, Parameterable, Stateful):
    @Scope.scope_args()
    def __init__(self, conv2d, block_size, pruning_rate, name=None):
        super(SparseConv2d, self).__init__(name)
        k_value = conv2d.parameter('kernel')
        b_value = conv2d.parameter('bias')
        self._k = self.add_parameter(value=k_value, name='kernel')
        self._b = self.add_parameter(value=b_value, name='bias')

        self._stride = conv2d._stride
        self._padding = conv2d._padding

        oc, ic, w, h = k_value.shape
        flatten_k = k_value.reshape((oc, ic * w * h))
        mask, row_order, col_order = block_sparse(flatten_k, block_size, pruning_rate)
        mask = mask.reshape((oc, ic, w, h))
        self._mask = self.add_states(value=mask, name='mask')

    @wired
    def forward(self, x):
        k = functional.where(self._mask, self._k, 0)
        return functional.conv2d(x, k, self._stride, self._padding) + functional.reshape(self._b, (1, -1, 1, 1))


class SparseGroupConv2d(Forwardable, Parameterable, Stateful):
    @Scope.scope_args()
    def __init__(self, group_conv2d, block_size, pruning_rate, name=None):
        super(SparseGroupConv2d, self).__init__(name)
        for layer in group_conv2d._submodules:
            self.add_module(SparseConv2d(layer, block_size, pruning_rate))
        self._k_shape = group_conv2d._k_shape
        self._num_groups = group_conv2d._num_groups

    def forward(self, x):
        wires = [
            m.forward(x[:, self._k_shape[1] * i / self._num_groups:self._k_shape[1] * (i + 1) / self._num_groups])
            for i, m in enumerate(self._submodules)
        ]
        return Wire(
            inputs=wires,
            apply_function=lambda *inputs: functional.concatenate(inputs, axis=1),
            owner=self
        )
