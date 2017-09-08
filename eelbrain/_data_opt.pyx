# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt


ctypedef np.int64_t INT64
ctypedef np.float64_t FLOAT64


def gaussian_smoother(np.ndarray[FLOAT64, ndim=2] dist, double std):
    """Create a gaussian smoothing matrix

    Parameters
    ----------
    dist : array (float64)
        Distances; dist[i, j] should provide the distance for any vertex pair i,
         j. Distances < 0 indicate absence of a connection.
    std : float
        The standard deviation of the kernel.

    Returns
    -------
    kernel : array (float64)
        Gaussian smoothing kernel, with same shape as dist.
    """
    cdef INT64 source, target
    cdef long n_vertices = len(dist)
    cdef double a = 1. / (std * sqrt(2 * np.pi))
    cdef double weight
    cdef double target_sum
    cdef np.ndarray out = np.empty((n_vertices, n_vertices), np.float64)

    if dist.shape[1] != n_vertices:
        raise ValueError("dist needs to be rectangular, got shape")

    for target in range(n_vertices):
        target_sum = 0
        for source in range(n_vertices):
            if dist[target, source] < 0:
                out[target, source] = 0
            else:
                weight = a * exp(- (dist[target, source] / std) ** 2 / 2)
                out[target, source] = weight
                target_sum += weight

        # normalize values for each target
        for source in range(n_vertices):
            out[target, source] /= target_sum

    return out
