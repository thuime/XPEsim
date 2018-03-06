from netlang import functional
from netlang.core import wired, Scope
from netlang.module.concept import Forwardable


class ReLU(Forwardable):
    @Scope.scope_args()
    def __init__(self, name=None):
        super(ReLU, self).__init__(name)

    @wired
    def forward(self, x):
        return functional.relu(x)


class Sigmoid(Forwardable):
    @Scope.scope_args()
    def __init__(self, name=None):
        super(Sigmoid, self).__init__(name)

    @wired
    def forward(self, x):
        return functional.sigmoid(x)


class Softmax(Forwardable):
    @Scope.scope_args()
    def __init__(self, name=None):
        super(Softmax, self).__init__(name)

    @wired
    def forward(self, x):
        return functional.softmax(x)


class Tanh(Forwardable):
    @Scope.scope_args()
    def __init__(self, name=None):
        super(Tanh, self).__init__(name)

    @wired
    def forward(self, x):
        return functional.tanh(x)
