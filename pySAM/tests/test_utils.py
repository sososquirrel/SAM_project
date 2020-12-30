import time

import numpy as np
import pySAM
import pytest
from pySAM.utils import make_parallel


def test_make_parallel():
    def yoo(a, b):
        for i in range(1000000):
            a + b
        return 1

    list_iterate = np.array(
        [1, 3, 5, 8, 10, 3, 5, 4, 12, 23, 45, 56, 76, 89, 98, 89, 87, 65, 45, 34]
    )
    constant_params = 5
    parallel_yoo = make_parallel(yoo, nprocesses=8)

    a = time.time()
    parallel_yoo(list_iterate, constant_params)
    print("TIME WITH PARALLEL ", time.time() - a)

    a = time.time()
    for i in list_iterate:
        yoo(i, 7)
    print("TIME WITHOUT PARALLEL", time.time() - a)
