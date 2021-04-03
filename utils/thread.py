#!/usr/bin/env/python
import queue
import threading
from threading import Thread


class ThreadedIterator:
    """An iterator object that computes its elements in a parallel thread to be ready to be consumed.
    The iterator should *not* return None"""

    def __init__(self, original_iterator, max_queue_size: int = 2):
        self.__queue = queue.Queue(maxsize=max_queue_size)
        self.__thread = threading.Thread(target=lambda: self.worker(original_iterator))
        self.__thread.start()

    def worker(self, original_iterator):
        for element in original_iterator:
            assert element is not None, 'By convention, iterator elements must not be None'
            self.__queue.put(element, block=True)
        self.__queue.put(None, block=True)

    def __iter__(self):
        next_element = self.__queue.get(block=True)
        while next_element is not None:
            yield next_element
            next_element = self.__queue.get(block=True)
        self.__thread.join()


class NThreadedIterators(ThreadedIterator):
    """An iterator object that computes its elements in a parallel thread to be ready to be consumed.
    The iterator should *not* return None"""

    def __init__(self, original_iterator, max_queue_size: int = 2, number_th: int = 1):
        self.__queue = queue.Queue(maxsize=max_queue_size)
        self.threads = []
        for i in range(number_th):
            __thread = threading.Thread(target=lambda: self.worker(original_iterator))
            __thread.start()
            self.threads.append(__thread)

    def worker(self, original_iterator):
        for element in original_iterator:
            assert element is not None, 'By convention, iterator elements must not be None'
            self.__queue.put(element, block=True)
        self.__queue.put(None, block=True)

    def __iter__(self):
        next_element = self.__queue.get(block=True)
        while next_element is not None:
            yield next_element
            next_element = self.__queue.get(block=True)
        for i in self.threads:
            i.__thread.join()


class ThreadWithReturnValue(object):
    def __init__(self, target=None, args=(), **kwargs):
        self._que = queue.Queue()
        self._t = Thread(target=lambda q, arg1, kwargs1: q.put(target(*arg1, **kwargs1)),
                         args=(self._que, args, kwargs), )
        self._t.start()

    def join(self):
        self._t.join()
        return self._que.get()
