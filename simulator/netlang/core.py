import functools
import inspect

import numpy

from netlang import functional


class Args(object):
    def __init__(self, ops=None, **kwargs):
        self.ops = ops
        self.kwargs = kwargs


class Scope(object):
    _stack = []

    def __init__(self, *args):
        self.operator_args = {}
        self.general_args = {}
        for arg in args:
            if isinstance(arg, Args):
                self._register_args(arg)

    def __enter__(self):
        Scope._stack.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        Scope._stack.pop()

    @classmethod
    def args(cls, op):
        args = {}
        for scope in cls._stack:
            args.update(scope.general_args)
            if op in scope.operator_args:
                args.update(scope.operator_args[op])
        return args

    @classmethod
    def scope_args(cls):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                op = func.im_class if hasattr(func, 'im_class') and func == func.im_class.__init__ else wrapper

                final_args = {}

                spec = inspect.getargspec(func)
                arg_keys = spec.args
                if spec.defaults is not None:
                    for i in xrange(len(spec.defaults)):
                        final_args[arg_keys[-i]] = spec.defaults[-i]

                for k, v in cls.args(op).items():
                    if k in arg_keys:
                        final_args[k] = v

                for i, v in enumerate(args):
                    final_args[arg_keys[i]] = v

                for k, v in kwargs.items():
                    final_args[k] = v

                return func(**final_args)

            return wrapper

        return decorator

    def _register_args(self, args):
        if args.ops is None:
            self._register_general_args(args.kwargs)
        else:
            for op in args.ops:
                self._register_operator_args(op, args.kwargs)

    def _register_general_args(self, kwargs):
        self.general_args.update(kwargs)

    def _register_operator_args(self, op, kwargs):
        if op in self.operator_args:
            self.operator_args[op].update(kwargs)
        else:
            self.operator_args[op] = kwargs


class Wire(object):
    def __init__(self, inputs=None, apply_function=None, owner=None, branch=False):
        if inputs is None:
            self._inputs = []
        elif isinstance(inputs, list):
            self._inputs = inputs
        elif isinstance(inputs, tuple):
            self._inputs = inputs
        else:
            self._inputs = [inputs]

        self._apply_function = apply_function
        self._owner = owner
        self._branch = branch

        self._training_output = self._apply(is_training=True)
        self._inference_output = self._apply(is_training=False)

    def _apply(self, is_training=True):
        applied_inputs = [apply(wire, is_training=is_training) for wire in self._inputs]
        if self._branch:
            return self._apply_function(*applied_inputs, is_training=is_training)
        else:
            return self._apply_function(*applied_inputs)

    def apply(self, is_training=True):
        if is_training:
            return self._training_output
        else:
            return self._inference_output
	def __len__(self):
		return len(self.inputs)

    def __add__(self, other):
        return Wire(inputs=[self, other], apply_function=lambda x, y: x + y)

    def __radd__(self, other):
        return Wire(inputs=[other, self], apply_function=lambda x, y: x + y)

    def __sub__(self, other):
        return Wire(inputs=[self, other], apply_function=lambda x, y: x - y)

    def __rsub__(self, other):
        return Wire(inputs=[other, self], apply_function=lambda x, y: x - y)

    def __mul__(self, other):
        return Wire(inputs=[self, other], apply_function=lambda x, y: x * y)

    def __rmul__(self, other):
        return Wire(inputs=[other, self], apply_function=lambda x, y: x * y)

    def __div__(self, other):
        return Wire(inputs=[self, other], apply_function=lambda x, y: x / y)

    def __rdiv__(self, other):
        return Wire(inputs=[other, self], apply_function=lambda x, y: x / y)

    def __floordiv__(self, other):
        return Wire(inputs=[self, other], apply_function=lambda x, y: x // y)

    def __rfloordiv__(self, other):
        return Wire(inputs=[other, self], apply_function=lambda x, y: x // y)

    def __mod__(self, other):
        return Wire(inputs=[self, other], apply_function=lambda x, y: x % y)

    def __rmod__(self, other):
        return Wire(inputs=[other, self], apply_function=lambda x, y: x % y)

    def __pow__(self, other):
        return Wire(inputs=[self, other], apply_function=lambda x, y: x ** y)

    def __rpow__(self, other):
        return Wire(inputs=[other, self], apply_function=lambda x, y: x ** y)

    def __and__(self, other):
        return Wire(inputs=[self, other], apply_function=lambda x, y: x & y)

    def __rand__(self, other):
        return Wire(inputs=[other, self], apply_function=lambda x, y: x & y)

    def __xor__(self, other):
        return Wire(inputs=[self, other], apply_function=lambda x, y: x ^ y)

    def __rxor__(self, other):
        return Wire(inputs=[other, self], apply_function=lambda x, y: x ^ y)

    def __or__(self, other):
        return Wire(inputs=[self, other], apply_function=lambda x, y: x | y)

    def __ror__(self, other):
        return Wire(inputs=[other, self], apply_function=lambda x, y: x | y)

    def __neg__(self):
        return Wire(inputs=[self], apply_function=lambda x: -x)

    def __invert__(self):
        return Wire(inputs=[self], apply_function=lambda x: ~x)

    def __lt__(self, other):
        return Wire(inputs=[self, other], apply_function=lambda x, y: x < y)

    def __le__(self, other):
        return Wire(inputs=[self, other], apply_function=lambda x, y: x <= y)

    def __gt__(self, other):
        return Wire(inputs=[self, other], apply_function=lambda x, y: x > y)

    def __ge__(self, other):
        return Wire(inputs=[self, other], apply_function=lambda x, y: x >= y)

    def __eq__(self, other):
        return Wire(inputs=[self, other], apply_function=lambda x, y: x == y)

    def __ne__(self, other):
        return Wire(inputs=[self, other], apply_function=lambda x, y: x != y)

    def __getitem__(self, other):
        return Wire(inputs=[self, other], apply_function=lambda x, y: x[y])


def apply(expr, is_training=True):
    if isinstance(expr, list):
        return [apply(ele, is_training) for ele in expr]
    elif isinstance(expr, tuple):
        return tuple(apply(ele, is_training) for ele in expr)
    elif isinstance(expr, dict):
        return {apply(key, is_training): apply(val, is_training) for key, val in expr.items()}
    elif isinstance(expr, Wire):
        return expr.apply(is_training)
    else:
        return expr


def wired(func):
    @functools.wraps(func)
    def wrapper(self, *args):
        return Wire(
            inputs=args,
            apply_function=lambda *inputs: func(self, *inputs),
            owner=self
        )

    return wrapper


def recurse_wired(reduce_start):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args):
            sub_wires = [
                wrapper(m, *args)
                for m in self._submodules if hasattr(m, func.__name__)
            ]
            this_wire = func(self, *args)
            return Wire(
                inputs=[this_wire] + sub_wires,
                apply_function=lambda *inputs: sum(inputs, reduce_start),
                owner=self
            )

        return wrapper

    return decorator


class Module(object):
    def __init__(self, name):
        self.name = name

        self._submodules = []
        self._submodule_dict = {}
        self.name_count = {}

    def add_module(self, module_):
        self._submodules.append(module_)
        if module_.name is None:
            class_name = module_.__class__.__name__
            if class_name not in self.name_count:
                self.name_count[class_name] = 0

            module_.name = "%s%d" % (class_name, self.name_count[class_name])
            self.name_count[class_name] += 1
        self._submodule_dict[module_.name] = module_

    def submodule(self, name):
        return self._submodule_dict[name]

    def pack(self):
        package = {}
        for m in self._submodules:
            package[m.name] = m.pack()
        return package

    def unpack(self, package):
        for m in self._submodules:
            m.unpack(package[m.name])

    def save(self, filename):
        package = self.pack()
        numpy.savez_compressed(filename, package)

    def load(self, filename):
        package = numpy.load(filename)['arr_0'].item()
        self.unpack(package)


def compile(inputs, outputs, updates=None):
    is_training = False if updates is None else True
    inputs = [apply(x, is_training=is_training) for x in inputs]
    outputs = [apply(x, is_training=is_training) for x in outputs]
    updates = apply(updates)
    return functional.compile(inputs, outputs, updates)
