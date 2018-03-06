import os

from netlang.data_provider import Memory


def subset(subset_name):
    npy_dir = os.environ.get('CIFAR_NPY_DIR')
    if subset_name.lower() == 'train':
        return Memory(os.path.join(npy_dir, 'train.npy'))
    elif subset_name.lower() == 'test':
        return Memory(os.path.join(npy_dir, 'test.npy'), max_epoch=1)
    else:
        raise ValueError('Unknown subset %s' % subset_name)
