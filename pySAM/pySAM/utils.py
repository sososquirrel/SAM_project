"""huk"""

import numpy as np
from multiprocess import Pool


def make_parallel(function, nprocesses):
    """
    Works similar to a decorator to paralelize "stupidly parallel"
    problems. Decorators and multiprocessing don't play nicely because
    of naming issues.

    Inputs
    ======
    function : the function that will be parallelized. The FIRST
        argument is the one to be iterated on (in parallel). The other
        arguments are the same in all the parallel runs of the function
        (they can be named or unnamedarguments).
    nprocesses : int, the number of processes to run. Default is None.
        It is passed to multiprocessing.Pool (see that for details).

    Output
    ======
    A paralelized function. DO NOT NAME IT THE SAME AS THE INPUT
    FUNCTION.

    SEE EXAMPLE BELOW

    """

    def apply(iterable_values, *args, **kwargs):
        args = list(args)
        processes_pool = Pool(nprocesses)
        result = [
            processes_pool.apply_async(function, args=[value] + args, kwds=kwargs)
            for value in iterable_values
        ]
        processes_pool.close()
        return [r.get() for r in result]

    # [p.apply_async(function, args=[value] + args, kwds=kwargs) for value in iterable_values]
    # p.close()

    return apply
