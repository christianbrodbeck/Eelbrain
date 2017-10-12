# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#cython: boundscheck=False, wraparound=False

cimport cython
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free
from libc.math cimport fabs
import numpy as np
cimport numpy as np

ctypedef np.int8_t INT8
ctypedef np.int64_t INT64
ctypedef np.float64_t FLOAT64


def l1(
        FLOAT64 [:] x,
        INT64 [:,:] indexes,
    ):
    cdef:
        double out = 0.
        size_t i, seg_i

    with nogil:
        for seg_i in range(indexes.shape[0]):
            for i in range(indexes[seg_i, 0], indexes[seg_i, 1]):
                out += fabs(x[i])

    return out


def l2(
        FLOAT64 [:] x,
        INT64 [:,:] indexes,
    ):
    cdef:
        double out = 0.
        size_t i, seg_i

    with nogil:
        for seg_i in range(indexes.shape[0]):
            for i in range(indexes[seg_i, 0], indexes[seg_i, 1]):
                out += x[i] ** 2

    return out


cdef void l1_for_delta(
        FLOAT64 [:] y_error,
        FLOAT64 [:] x,
        INT64 [:,:] indexes,  # training segment indexes
        double delta,
        size_t shift,
        double* e_add,
        double* e_sub,
    ) nogil:
    cdef:
        double d, temp_sum
        size_t i, seg_i, seg_start

    e_add[0] = 0.
    e_sub[0] = 0.

    for seg_i in range(indexes.shape[0]):
        seg_start = indexes[seg_i, 0]
        # start of the segment before the shift-delay
        temp_sum = 0.
        for i in range(seg_start, seg_start + shift):
            temp_sum += fabs(y_error[i])
        e_add[0] += temp_sum
        e_sub[0] += temp_sum
        for i in range(seg_start + shift, indexes[seg_i, 1]):
            d = delta * x[i - shift]
            e_add[0] += fabs(y_error[i] - d)
            e_sub[0] += fabs(y_error[i] + d)


cdef void l2_for_delta(
        FLOAT64 [:] y_error,
        FLOAT64 [:] x,
        INT64 [:,:] indexes,  # training segment indexes
        double delta,
        size_t shift,
        double* e_add,
        double* e_sub,
    ) nogil:
    cdef:
        double d, temp_sum
        size_t i, seg_i, seg_start

    e_add[0] = 0.
    e_sub[0] = 0.

    for seg_i in range(indexes.shape[0]):
        seg_start = indexes[seg_i, 0]
        # start of the segment before the shift-delay
        temp_sum = 0.
        for i in range(seg_start, seg_start + shift):
            temp_sum += y_error[i] ** 2
        e_add[0] += temp_sum
        e_sub[0] += temp_sum
        for i in range(seg_start + shift, indexes[seg_i, 1]):
            d = delta * x[i - shift]
            e_add[0] += (y_error[i] - d) ** 2
            e_sub[0] += (y_error[i] + d) ** 2


def generate_options(
        FLOAT64 [:] y_error,
        FLOAT64 [:,:] x,  # (n_stims, n_times)
        INT64 [:,:] indexes,  # training segment indexes
        int error,
        double delta,
        # buffers
        FLOAT64 [:,:] new_error,  # (n_stims, n_times_trf)
        INT8 [:,:] new_sign,
    ):
    cdef:
        double e_add
        double e_sub
        size_t n_stims = new_error.shape[0]
        size_t n_times_trf = new_error.shape[1]
        size_t i_stim, i_time
        FLOAT64 [:] x_stim

    if error != 1 and error != 2:
        raise RuntimeError("error=%r" % (error,))

    with nogil:
        for i_stim in range(n_stims):
            x_stim = x[i_stim]
            for i_time in range(n_times_trf):
                # +/- delta
                if error == 1:
                    l1_for_delta(y_error, x_stim, indexes, delta, i_time, &e_add, &e_sub)
                else:
                    l2_for_delta(y_error, x_stim, indexes, delta, i_time, &e_add, &e_sub)

                if e_add > e_sub:
                    new_error[i_stim, i_time] = e_sub
                    new_sign[i_stim, i_time] = -1
                else:
                    new_error[i_stim, i_time] = e_add
                    new_sign[i_stim, i_time] = 1


def update_error(
        FLOAT64 [:] y_error,
        FLOAT64 [:] x,
        INT64 [:,:] indexes,  # training segment indexes
        double delta,
        size_t shift,
    ):
    cdef:
        size_t i, seg_i

    with nogil:
        for seg_i in range(indexes.shape[0]):
            for i in range(indexes[seg_i, 0] + shift, indexes[seg_i, 1]):
                y_error[i] -= delta * x[i - shift]
