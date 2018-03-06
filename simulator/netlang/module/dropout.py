from netlang import functional
from netlang.core import Wire, Scope
from netlang.module.concept import Forwardable


class Dropout(Forwardable):
    @Scope.scope_args()
    def __init__(self, p=0.5, name=None):
        super(Dropout, self).__init__(name)
        self._p = p

    def forward(self, x):
        return Wire(
            inputs=x,
            apply_function=lambda x, is_training: functional.dropout(x, self._p) if is_training else x,
            owner=self,
            branch=True
        )