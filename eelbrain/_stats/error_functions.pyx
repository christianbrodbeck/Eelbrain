# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#cython: boundscheck=False, wraparound=False

cimport cython
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np


def l1(double [::1] x):
    cdef:
        double out = 0.
        size_t i

    for i in range(x.shape[0]):
        out += abs(x[i])

    return out


def l2(double [::1] x):
    cdef:
        double out = 0.
        size_t i

    for i in range(x.shape[0]):
        out += x[i]**2

    return out
