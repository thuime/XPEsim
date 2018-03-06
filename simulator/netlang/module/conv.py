from netlang import functional
from netlang.core import Scope, wired, Wire

from netlang.initializer import Xavier, Constant
from netlang.module.concept import Forwardable, Parameterable
from netlang.module.differentiator import Gradient
from netlang.module.regularizer import NoRegularize


class Conv2d(Forwardable, Parameterable):
    @Scope.scope_args()
    def __init__(self, k_shape, stride=1, padding='half', k_init=Xavier(), b_init=Constant(0.0), k_reg=NoRegularize(),
                 b_reg=NoRegularize(), k_diff=Gradient(), b_diff=Gradient(), name=None):
        super(Conv2d, self).__init__(name)

        self._k_shape = k_shape

        if isinstance(stride, int):
            self._stride = (stride, stride)
        else:
            self._stride = stride

        if isinstance(padding, str):
            if padding is 'valid':
                self._padding = (0, 0)
            elif padding is 'full':
                self._padding = (k_shape[2] - 1, k_shape[3] - 1)
            elif padding is 'half':
                self._padding = (k_shape[2] // 2, k_shape[3] // 2)
            else:
                raise ValueError('Unknown padding %s' % padding)
        elif isinstance(padding, int):
            self._padding = (padding, padding)
        else:
            self._padding = padding

        with Scope(name):
            self._k = self.add_parameter(value=k_init(k_shape), regularizer=k_reg, differentiator=k_diff, name='kernel')
            self._b = self.add_parameter(value=b_init((k_shape[0],)), regularizer=b_reg, differentiator=b_diff,
                                         name='bias')

    @wired
    def forward(self, x):
        return functional.conv2d(x, self._k, self._stride, self._padding) + functional.reshape(self._b, (1, -1, 1, 1))


class GroupConv2d(Forwardable, Parameterable):
    @Scope.scope_args()
    def __init__(self, num_groups, k_shape, stride=1, padding='half', k_init=Xavier(), b_init=Constant(0.0),
                 k_reg=NoRegularize(), b_reg=NoRegularize(), k_diff=Gradient(), b_diff=Gradient(), name=None):
        super(GroupConv2d, self).__init__(name)

        self._num_groups = num_groups
        self._k_shape = k_shape
        with Scope(name):
            for i in xrange(num_groups):
                output_channels, input_channels, rows, cols = k_shape
                i_output_channels = output_channels * (i + 1) / num_groups - output_channels * i / num_groups
                i_input_channels = input_channels * (i + 1) / num_groups - input_channels * i / num_groups
                i_k_shape = (i_output_channels, i_input_channels, rows, cols)
                self.add_module(
                    Conv2d(k_shape=i_k_shape, stride=stride, padding=padding, k_init=k_init, b_init=b_init, k_reg=k_reg,
                           b_reg=b_reg, k_diff=k_diff, b_diff=b_diff, name='group%d' % i))

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

class MaxPool2d(Forwardable):
    @Scope.scope_args()
    def __init__(self, k_shape, stride, padding='valid', name=None):
        super(MaxPool2d, self).__init__(name)

        if isinstance(k_shape, int):
            self._k_shape = (k_shape, k_shape)
        else:
            self._k_shape = k_shape

        if isinstance(stride, int):
            self._stride = (stride, stride)
        else:
            self._stride = stride

        if isinstance(padding, str):
            if padding is 'valid':
                self._padding = (0, 0)
            elif padding is 'full':
                self._padding = (self._k_shape[0] - 1, self._k_shape[1] - 1)
            elif padding is 'half':
                self._padding = (self._k_shape[0] // 2, self._k_shape[1] // 2)
            else:
                raise ValueError('Unknown padding %s' % padding)
        elif isinstance(padding, int):
            self._padding = (padding, padding)
        else:
            self._padding = padding

    @wired
    def forward(self, x):
        return functional.pool2d(x, self._k_shape, self._stride, self._padding, mode='max')