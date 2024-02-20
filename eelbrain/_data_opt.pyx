# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from libc.math cimport exp, sqrt, M_PI
import numpy as np

cimport numpy as np

ctypedef np.float64_t FLOAT64


def gaussian_smoother(FLOAT64[:,:] dist, double std):
    """Create a gaussian smoothing matrix

    Parameters
    ----------
    dist : array (float)
        Distances; dist[i, j] should provide the distance for any vertex pair i,
         j. Distances < 0 indicate absence of a connection.
    std : float
        The standard deviation of the kernel.

    Returns
    -------
    kernel : array (float64)
        Gaussian smoothing kernel, with same shape as dist.
    """
    cdef:
        Py_ssize_t source, target
        Py_ssize_t n_vertices = len(dist)
        double weight, target_sum
        double a = 1. / (std * sqrt(2 * M_PI))
        FLOAT64[:,:] out = np.empty((n_vertices, n_vertices))

    if dist.shape[1] != n_vertices:
        raise ValueError(f"dist needs to be rectangular, got shape {dist.shape}")

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

    return np.asarray(out)
