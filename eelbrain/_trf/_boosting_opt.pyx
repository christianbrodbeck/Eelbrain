# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#cython: boundscheck=False, wraparound=False

cimport cython
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np

ctypedef np.int8_t INT8
ctypedef np.int64_t INT64
ctypedef np.float64_t FLOAT64


def l1(
        np.ndarray[FLOAT64, ndim=1] x,
        np.ndarray[INT64, ndim=2] indexes,
    ):
    cdef:
        double out = 0.
        size_t i, seg_i

    for seg_i in range(indexes.shape[0]):
        for i in range(indexes[seg_i, 0], indexes[seg_i, 1]):
            out += abs(x[i])

    return out


def l2(
        np.ndarray[FLOAT64, ndim=1] x,
        np.ndarray[INT64, ndim=2] indexes,
    ):
    cdef:
        double out = 0.
        size_t i, seg_i

    for seg_i in range(indexes.shape[0]):
        for i in range(indexes[seg_i, 0], indexes[seg_i, 1]):
            out += x[i] ** 2

    return out


cdef l1_for_delta(
        np.ndarray[FLOAT64, ndim=1] y_error,
        np.ndarray[FLOAT64, ndim=1] x,
        np.ndarray[INT64, ndim=2] indexes,  # training segment indexes
        double delta,
        size_t shift,
        double* e_add,
        double* e_sub,
    ):
    cdef:
        double out_pos = 0.
        double out_neg = 0.
        double d, temp_sum
        size_t i, seg_i, seg_start

    for seg_i in range(indexes.shape[0]):
        seg_start = indexes[seg_i, 0]
        # start of the segment before the shift-delay
        temp_sum = 0.
        for i in range(seg_start, seg_start + shift):
            temp_sum += abs(y_error[i])
        out_pos += temp_sum
        out_neg += temp_sum
        for i in range(seg_start + shift, indexes[seg_i, 1]):
            d = delta * x[i - shift]
            out_pos += abs(y_error[i] - d)
            out_neg += abs(y_error[i] + d)


cdef l2_for_delta(
        np.ndarray[FLOAT64, ndim=1] y_error,
        np.ndarray[FLOAT64, ndim=1] x,
        np.ndarray[INT64, ndim=2] indexes,  # training segment indexes
        double delta,
        size_t shift,
        double *e_add,
        double *e_sub,
    ):
    cdef:
        double out_pos = 0.
        double out_neg = 0.
        double d, temp_sum
        size_t i, seg_i, seg_start

    for seg_i in range(indexes.shape[0]):
        seg_start = indexes[seg_i, 0]
        # start of the segment before the shift-delay
        temp_sum = 0.
        for i in range(seg_start, seg_start + shift):
            temp_sum += y_error[i] ** 2
        out_pos += temp_sum
        out_neg += temp_sum
        for i in range(seg_start + shift, indexes[seg_i, 1]):
            d = delta * x[i - shift]
            out_pos += (y_error[i] - d) ** 2
            out_neg += (y_error[i] + d) ** 2

        e_add[0] = out_pos
        e_sub[0] = out_neg


def generate_options(
        np.ndarray[FLOAT64, ndim=1] y_error,
        np.ndarray[FLOAT64, ndim=2] x,  # (n_stims, n_times)
        np.ndarray[INT64, ndim=2] indexes,  # training segment indexes
        int error,
        double delta,
        # buffers
        np.ndarray[FLOAT64, ndim=2] new_error,  # (n_stims, n_times_trf)
        np.ndarray[INT8, ndim=2] new_sign,
    ):
    cdef:
        double e_add
        double e_sub
        size_t n_stims = new_error.shape[0]
        size_t n_times_trf = new_error.shape[1]
        size_t i_stim, i_time
        np.ndarray[FLOAT64, ndim=1] x_stim


    for i_stim in range(n_stims):
        x_stim = x[i_stim]
        for i_time in range(n_times_trf):
            # +/- delta
            if error == 1:
                l1_for_delta(y_error, x_stim, indexes, delta, i_time, &e_add, &e_sub)
            elif error == 2:
                l2_for_delta(y_error, x_stim, indexes, delta, i_time, &e_add, &e_sub)
            else:
                raise RuntimeError("error=%r" % (error,))

            if e_add > e_sub:
                new_error[i_stim, i_time] = e_sub
                new_sign[i_stim, i_time] = -1
            else:
                new_error[i_stim, i_time] = e_add
                new_sign[i_stim, i_time] = 1


def update_error(
        np.ndarray[FLOAT64, ndim=1] error,
        np.ndarray[FLOAT64, ndim=1] x,
        np.ndarray[INT64, ndim=2] indexes,  # training segment indexes
        double delta,
        size_t shift,
    ):
    cdef:
        size_t i, seg_i

    for seg_i in range(indexes.shape[0]):
        for i in range(indexes[seg_i, 0] + shift, indexes[seg_i, 1]):
            error[i] -= delta * x[i - shift]
