import multiprocessing
import concurrent.futures
import numpy as np

class Multitask:
    def __init__(self, workers=None, func=None, arr=None, **kw):
        if workers is None:
            workers = multiprocessing.cpu_count()
        self.workers = workers
        self.func = func
        self.arr = arr
        n = arr.size
        if len(kw) > 0: self.kw = kw

        #self.executor = concurrent.futures.ProcessPoolExecutor(workers)
        self.executor = concurrent.futures.ThreadPoolExecutor(workers)
    
        self.step = np.ceil(n / workers).astype(np.int_)
        self.task()
        

    def task(self):
        def _task(_func, _arr, first, last, **kwargs):
            [_func(i, **kwargs) for i in _arr[first:last]]
            #_func(_arr[first:last], **kwargs)

        futures = {}
        for i in range(self.workers):
            try:
                args = (_task,
                        self.func,
                        self.arr,
                        i * self.step,
                        (i + 1) * self.step,
                        self.kw)
            except AttributeError:
                args = (_task,
                        self.func,
                        self.arr,
                        i * self.step,
                        (i + 1) * self.step)
            
            futures[self.executor.submit(*args)] = i
            #futures[self.executor.map(*args)] = i
        concurrent.futures.wait(futures)
        

    def __del__(self):
        self.executor.shutdown(False)









'''  
    
from numpy.random import default_rng, SeedSequence
import multiprocessing
import concurrent.futures

class MultithreadedRNG:
    def __init__(self, n, seed=None, threads=None):
        if threads is None:
            threads = multiprocessing.cpu_count()
        self.threads = threads

        seq = SeedSequence(seed)
        self._random_generators = [default_rng(s)
                                   for s in seq.spawn(threads)]

        self.n = n
        self.executor = concurrent.futures.ThreadPoolExecutor(threads)
        self.values = np.empty(n)
        self.step = np.ceil(n / threads).astype(np.int_)

    def fill(self):
        def _fill(random_state, out, first, last):
            random_state.standard_normal(out=out[first:last])

        futures = {}
        for i in range(self.threads):
            args = (_fill,
                    self._random_generators[i],
                    self.values,
                    i * self.step,
                    (i + 1) * self.step)
            futures[self.executor.submit(*args)] = i
        concurrent.futures.wait(futures)

    def __del__(self):
        self.executor.shutdown(False)'''


    


'''mrng = MultithreadedRNG(10000000, seed=12345)
print(mrng.values[-1])
mrng.fill()
print(mrng.values[-1])
from IPython import embed; embed()'''