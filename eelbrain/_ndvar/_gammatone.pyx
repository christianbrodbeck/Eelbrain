# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
cimport numpy


ctypedef numpy.float64_t FLOAT64
ctypedef numpy.int64_t INT64


def aggregate_left(
        FLOAT64 [:] xf,
        int n_samples,
        float step,
        int n_window,
        FLOAT64 [:] out,
):
    cdef:
        Py_ssize_t i, j
        Py_ssize_t n_xf = xf.shape[0]

    for i in range(n_samples):
        start = int(round(i * step))
        stop = start + n_window
        if stop > n_xf:
            stop = n_xf
        for j in range(start, start + n_window):
            out[i] += xf[j]


def aggregate_right(
        FLOAT64 [:] xf,
        int n_samples,
        float step,
        int n_window,
        FLOAT64 [:] out,
):
    cdef:
        Py_ssize_t i, j

    for i in range(n_samples):
        stop = int(round(i * step)) + 1
        start = stop - n_window
        if start < 0:
            start = 0
        for j in range(start, stop):
            out[i] += xf[j]
