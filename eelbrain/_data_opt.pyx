# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
# cython: language_level=3, boundscheck=False, wraparound=False

from libc.math cimport exp, sqrt
import numpy as np


def gaussian_smoother(const double[:, :] dist, double std):
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
    cdef size_t source, target
    cdef long n_vertices = len(dist)
    cdef double weight, target_sum
    cdef double a = 1. / (std * sqrt(2 * np.pi))

    if dist.shape[1] != n_vertices:
        raise ValueError(f"dist needs to be rectangular, got shape {dist.shape}")

    out = np.empty((n_vertices, n_vertices), np.double)
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
