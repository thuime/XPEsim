import random

from netlang import functional

def cast(dtype=functional.floatX):
    return lambda x:x.astype(dtype=dtype)

def crop(size, crop_type):
    def func(x):
        row = x.shape[2]
        col = x.shape[3]
        margin_row = row - size
        margin_col = col - size
        if crop_type is 'center':
            row_start = margin_row / 2
            col_start = margin_col / 2
            return x[:, :, row_start:row_start + size, col_start:col_start + size]
        elif crop_type is 'random':
            row_start = random.randint(0, margin_row - 1)
            col_start = random.randint(0, margin_col - 1)
            return x[:, :, row_start:row_start + size, col_start:col_start + size]

    return func


def mirror(x):
    if bool(random.getrandbits(1)):
        return x[:, :, ::-1, ::-1]
    else:
        return x

def zero_mean(r, g, b):
    def func(x):
        x[:, 0, :, :] -= r
        x[:, 1, :, :] -= g
        x[:, 2, :, :] -= b
        return x
    return func