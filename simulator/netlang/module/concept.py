from netlang import functional
from netlang.core import Module, recurse_wired
from netlang.module.differentiator import Gradient
from netlang.module.regularizer import NoRegularize


class Forwardable(Module):
    def __init__(self, name):
        super(Forwardable, self).__init__(name)

    def forward(self, *args):
        raise NotImplementedError

class Parameterable(Module):
    def __init__(self, name):
        super(Parameterable, self).__init__(name)
        self._parameter_dict = {}
        self._parameters = []
        self._differentiators = []
        self._regularizers = []

    def pack(self):
        package = super(Parameterable, self).pack()
        for p in self._parameters:
            package[p.name] = functional.get_value(p)
        return package

    def unpack(self, package):
        super(Parameterable, self).unpack(package)
        for p in self._parameters:
            if p.name in package:
                functional.set_value(p, package[p.name])

    def add_parameter(self, value, differentiator=Gradient(), regularizer=NoRegularize(), name=None):
        parameter = functional.variable(value=value, name=name)
        self._parameters.append(parameter)
        self._differentiators.append(differentiator)
        self._regularizers.append(regularizer)
        self._parameter_dict[name] = parameter
        return parameter

    def parameter(self, name):
        return functional.get_value(self._parameter_dict[name])


    @recurse_wired([])
    def parameters(self):
        return self._parameters

    @recurse_wired([])
    def differentiate(self, loss):
        return [d(loss, p) for d, p in zip(self._differentiators, self._parameters)]

    @recurse_wired(0)
    def regularize(self):
        return sum([r(p) for r, p in zip(self._regularizers, self._parameters)], 0)


class Updateful(Module):
    def __init__(self, name):
        super(Updateful, self).__init__(name)
        self._updates = []

    def add_update(self, update):
        self._updates.append(update)

    @recurse_wired([])
    def updates(self):
        return self._updates

class Stateful(Module):
    def __init__(self, name):
        super(Stateful, self).__init__(name)
        self._states = []
        self._state_dict = {}

    def add_states(self, value, name=None):
        state = functional.variable(value=value, name=name)
        self._states.append(state)
        self._state_dict[name] = state
        return state

    def state(self, name):
        return functional.get_value(self._state_dict[name])
