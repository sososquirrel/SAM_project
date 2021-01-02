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


def expand_array_to_tzyx_array(
    time_dependence: bool, input_array: np.array, final_shape: np.array
) -> np.array:

    if len(final_shape) != 4:
        raise ValueError("Output must be (t,z,y,x) type")

    if time_dependence == False:
        if len(input_array.shape) != 1:
            raise ValueError("Input array with no time dependence must be one-dimensionnal")

        if input_array.shape[0] != final_shape[1]:
            raise ValueError(
                "z length of final shape must be equal to the length of input array"
            )

        output_array = input_array[None, :, None, None]

        output_array.repeat(final_shape[0], axis=0)
        output_array.repeat(final_shape[2], axis=2)
        output_array.repeat(final_shape[3], axis=3)

    else:
        if len(input_array.shape) != 2:
            raise ValueError("Input array with time dependence must be 2-dimensionnal")

        if (input_array.shape != final_shape[:2]).all():
            raise ValueError(
                "time and z lengths of final shape must be equal to the length of input array"
            )

        output_array = input_array[:, :, None, None]

        output_array.repeat(final_shape[2], axis=2)
        output_array.repeat(final_shape[3], axis=3)

    return output_array
