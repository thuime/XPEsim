import numpy
import theano
import theano.tensor
import theano.tensor.signal.pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

floatX = theano.config.floatX


def compile(inputs, outputs, updates=None, givens=None):
    return theano.function(inputs=inputs, outputs=outputs, updates=updates, givens=givens)


def variable(value=None, shape=None, dtype=floatX, name=None):
    if value is None and shape is None:
        raise ValueError('Specify value or shape & dtype.')
    if value is None:
        value = numpy.ones(shape=shape, dtype=dtype) * numpy.nan

    return theano.shared(value=value, name=name)


def gradient(loss, var):
    return theano.tensor.grad(loss, var)


def sum(x, axis=None):
    return theano.tensor.sum(input=x, axis=axis)


def abs(x):
    return theano.tensor.abs_(x)


def square(x):
    return theano.tensor.sqr(x)


def relu(x):
    return theano.tensor.nnet.relu(x)


def sigmoid(x):
    return theano.tensor.nnet.sigmoid(x)


def softmax(x):
    return theano.tensor.nnet.softmax(x)


def tanh(x):
    return theano.tensor.tanh(x)


def conv2d(x, kernel, stride, padding):
    return theano.tensor.nnet.conv2d(input=x, filters=kernel, border_mode=padding, subsample=stride)


def reshape(x, shape):
    return theano.tensor.reshape(x, shape)


def concatenate(tensors, axis=0):
    return theano.tensor.concatenate(tensors, axis)


def pool2d(x, kernel, stride, padding, mode):
    return theano.tensor.signal.pool.pool_2d(input=x, ws=kernel, ignore_border=True, stride=stride, pad=padding,
                                             mode=mode)


def dropout(x, p, seed=12345):
    rng = RandomStreams(seed=seed)
    random_tensor = rng.binomial(size=x.shape, p=1 - p, dtype=x.dtype)
    return x * random_tensor / (1 - p)


def dot(x, y):
    return theano.tensor.dot(x, y)


def crossentropy(predict, target):
    return theano.tensor.nnet.categorical_crossentropy(predict, target)


def mean(x, axis=None):
    return theano.tensor.mean(x, axis=axis)


def one_hot(x, num_classes):
    return theano.tensor.extra_ops.to_one_hot(x, num_classes)


def argmax(predict, axis):
    return theano.tensor.argmax(predict, axis=axis)


def equal(x, y):
    return theano.tensor.eq(x, y)


def sort(x, axis=-1):
    return theano.tensor.sort(x, axis=axis)


def arange(start, stop=None, step=1):
    return theano.tensor.arange(start, stop, step)


def greater_equal(x, y):
    return theano.tensor.ge(x, y)


def zeros(shape, dtype=None):
    return theano.tensor.zeros(shape=shape, dtype=dtype)


def power(x, y):
    return theano.tensor.power(x, y)


def get_value(x):
    return x.get_value()


def placeholder(name, dims, dtype=floatX):
    return theano.tensor.TensorType(dtype=dtype, broadcastable=[False] * dims)(name=name)


def set_value(var, value):
    return var.set_value(value)


def where(condition, if_true, if_false):
    return theano.tensor.where(condition, if_true, if_false)


def random_normal(size, mean, std, dtype=floatX):
    rng = RandomStreams()
    return rng.normal(size=size, avg=mean, std=std, dtype=dtype)


def maximum(x, y):
    return theano.tensor.maximum(x, y)
