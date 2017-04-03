# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#cython: boundscheck=False, wraparound=False

from libc.stdlib cimport malloc, free
cimport numpy as np


ctypedef np.uint32_t UINT32
ctypedef np.float64_t FLOAT64


def tfce_increment(np.ndarray[UINT32, ndim=1] labels,
                   np.ndarray[UINT32, ndim=1] label_image,
                   np.ndarray[FLOAT64, ndim=1] image,
                   double e, double h_factor):
    cdef unsigned int i, cid
    cdef unsigned int n = image.shape[0]
    cdef unsigned int n_labels = labels.max() + 1

    cdef double* area = <double*> malloc(sizeof(double) * n_labels)

    # initialize area
    for i in range(n_labels):
        area[i] = 0.

    # determine area
    for i in range(n):
        cid = label_image[i]
        if cid > 0:
            area[cid] += 1.

    # determine TFCE value
    for cid in labels:
        area[cid] = area[cid] ** e * h_factor

    # update TFCE image
    for i in range(n):
        cid = label_image[i]
        if cid > 0:
            image[i] += area[cid]

    free(area)
