import glob
import os

from netlang.data_provider import Disk


def subset(subset_name):
    npy_dir = os.environ.get('IMAGENET_NPY_DIR')
    if subset_name.lower() == 'train':
        return Disk(
            filenames=glob.glob(os.path.join(npy_dir, 'train-*')),
            shuffle=True,
            capacity=4
        )
    elif subset_name.lower() == 'validation':
        return Disk(
            filenames=glob.glob(os.path.join(npy_dir, 'val-*')),
            max_epoch=1,
            shuffle=False,
            capacity=4
        )
    else:
        raise ValueError('Unknown subset %s' % subset_name)
