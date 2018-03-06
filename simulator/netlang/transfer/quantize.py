import numpy


def quantize(x, factor, num_bits):
    # type: (numpy.ndarray, numpy.ndarray, int) -> numpy.ndarray
    maximum = numpy.ones(shape=x.shape, dtype='float32') * (2 ** num_bits - 1)
    x_ = x * factor
    x_ = x_.round()
    x_ = numpy.where(x_ < maximum, x_, maximum)
    x_ = numpy.where(x_ > -maximum, x_, -maximum)
    x_ /= factor
    return x_


def quantize_factor(x, num_bits):
    # type: (numpy.ndarray, int) -> numpy.ndarray

    factor = 1.0
    minimum_error = float('inf')
    # EM iteration
    while True:
        # E step, quantize x
        x_ = quantize(x, factor, num_bits)

        error = ((x - x_) ** 2).sum()

        if error < minimum_error:
            minimum_error = error
        else:
            break

        # M step, choose best factor
        if len(x[x_ != 0]) is 0:
            factor /= 2
        else:
            factor = (x_[x_ != 0] / x[x_ != 0]).mean()

    return factor
