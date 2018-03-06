import Queue
import random
import threading

import numpy

__all__ = [
    'DataProvider',
    'Disk', 'Batch',
]


class DataProvider(object):
    def get(self, size):
        raise NotImplemented


class Memory(DataProvider):
    def __init__(self, filename, max_epoch=None, shuffle=False):
        self._filename = filename
        self._shuffle = shuffle
        self._buffer = numpy.load(filename)
        if shuffle:
            numpy.random.shuffle(self._buffer)
        self._index = 0

    def get(self, size):
        if self._index + size < self._buffer.shape[0]:
            samples = self._buffer[self._index:self._index + size]
            self._index += size
            return samples
        else:
            remain = self._buffer.shape[0] - self._index
            remain_samples = self._buffer[self._index:]
            if self._shuffle:
                numpy.random.shuffle(self._buffer)
            self._index = size - remain
            if remain > 0:
                return numpy.concatenate([remain_samples, self._buffer[:self._index]])
            else:
                return self._buffer[:self._index]



class Disk(DataProvider):
    def __init__(self, filenames, max_epoch=None, shuffle=False, capacity=4):
        self._filenames = filenames
        self._shuffle = shuffle
        self._capacity = capacity
        self._max_epoch = max_epoch

        self._queue = Queue.Queue(maxsize=capacity)
        self._empty = False
        self._process = threading.Thread(target=self._worker)
        self._process.setDaemon(True)
        self._process.start()
        self._buffer = self._queue.get()
        self._index = 0


    def _worker(self):
        for _ in xrange(self._max_epoch) if self._max_epoch is not None else iter(int, 1):
            if self._shuffle:
                random.shuffle(self._filenames)
            for filename in self._filenames:
                self._queue.put(numpy.load(filename))
        self._empty = True

    def _queue_get(self):
        if not self._empty or not self._queue.empty():
            return self._queue.get()
        else:
            raise StopIteration

    def get(self, size):
        sample_buffers = []
        if self._buffer.shape[0] - self._index == 0:
            self._buffer = self._queue_get()
            self._index = 0
        while self._buffer.shape[0] - self._index < size:
            sample_buffers.append(self._buffer[self._index:])
            size -= self._buffer.shape[0] - self._index
            self._buffer = self._queue_get()
            self._index = 0
        if size > 0:
            sample_buffers.append(self._buffer[self._index:self._index + size])
            self._index += size

        return numpy.concatenate(sample_buffers)


class Batch(DataProvider):
    def __init__(self, provider, batch_size, x_preprocess=None, y_preprocess=None, capacity=4):
        self._provider = provider
        self._batch_size = batch_size

        if x_preprocess is None:
            self._x_preprocess = []
        elif isinstance(x_preprocess, list):
            self._x_preprocess = x_preprocess
        else:
            self._x_preprocess = [x_preprocess]

        if y_preprocess is None:
            self._y_preprocess = []
        elif isinstance(y_preprocess, list):
            self._y_preprocess = y_preprocess
        else:
            self._y_preprocess = [y_preprocess]

        self._queue = Queue.Queue(maxsize=capacity)
        self._empty = False
        self._process = threading.Thread(target=self._worker)
        self._process.setDaemon(True)
        self._process.start()


    def _worker(self):
        while True:
            try:
                samples = self._provider.get(self._batch_size)
                x = numpy.asarray([samples[i, 0] for i in xrange(self._batch_size)])
                y = numpy.asarray([samples[i, 1] for i in xrange(self._batch_size)])
                for func in self._x_preprocess:
                    x = func(x)
                for func in self._y_preprocess:
                    y = func(y)
                self._queue.put((x, y))
            except StopIteration:
                break
        self._empty = True

    def _queue_get(self):
        if not self._empty or not self._queue.empty():
            return self._queue.get()
        else:
            raise StopIteration

    def get(self, size=1):
        if size == 1:
            return self._queue_get()
        else:
            return [self._queue_get() for _ in xrange(size)]
