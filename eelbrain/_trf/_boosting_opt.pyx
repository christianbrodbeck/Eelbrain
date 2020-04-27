# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
# cython: language_level=3, boundscheck=False, wraparound=False
from libc.math cimport fabs
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
        double x_pad,  # pad x outside valid convolution area
        INT64 [:,:] indexes,  # training segment indexes
        double delta,
        int shift,  # TRF element offset
        double* e_add,
        double* e_sub,
    ) nogil:
    cdef:
        double d
        size_t i, seg_i, seg_start, seg_stop, conv_start, conv_stop

    e_add[0] = 0.
    e_sub[0] = 0.

    for seg_i in range(indexes.shape[0]):
        seg_start = indexes[seg_i, 0]
        seg_stop = indexes[seg_i, 1]
        # determine valid convolution segment
        conv_start = seg_start
        conv_stop = seg_stop
        if shift > 0:
            conv_start += shift
        elif shift < 0:
            conv_stop += shift
        # padding
        d = delta * x_pad
        # pre-
        for i in range(seg_start, conv_start):
            e_add[0] += fabs(y_error[i] - d)
            e_sub[0] += fabs(y_error[i] + d)
        # post-
        for i in range(conv_stop, seg_stop):
            e_add[0] += fabs(y_error[i] - d)
            e_sub[0] += fabs(y_error[i] + d)
        # valid segment
        for i in range(conv_start, conv_stop):
            d = delta * x[i - shift]
            e_add[0] += fabs(y_error[i] - d)
            e_sub[0] += fabs(y_error[i] + d)


cdef void l2_for_delta(
        FLOAT64 [:] y_error,
        FLOAT64 [:] x,
        double x_pad,  # pad x outside valid convolution area
        INT64 [:,:] indexes,  # training segment indexes
        double delta,
        int shift,
        double* e_add,
        double* e_sub,
    ) nogil:
    cdef:
        double d
        size_t i, seg_i, seg_start, seg_stop, conv_start, conv_stop

    e_add[0] = 0.
    e_sub[0] = 0.

    for seg_i in range(indexes.shape[0]):
        seg_start = indexes[seg_i, 0]
        seg_stop = indexes[seg_i, 1]
        # determine valid convolution segment
        conv_start = seg_start
        conv_stop = seg_stop
        if shift > 0:
            conv_start += shift
        elif shift < 0:
            conv_stop += shift
        # padding
        d = delta * x_pad
        # pre-
        for i in range(seg_start, conv_start):
            e_add[0] += (y_error[i] - d) ** 2
            e_sub[0] += (y_error[i] + d) ** 2
        # post-
        for i in range(conv_stop, seg_stop):
            e_add[0] += (y_error[i] - d) ** 2
            e_sub[0] += (y_error[i] + d) ** 2
        # part of the segment that is affected
        for i in range(conv_start, conv_stop):
            d = delta * x[i - shift]
            e_add[0] += (y_error[i] - d) ** 2
            e_sub[0] += (y_error[i] + d) ** 2


def generate_options(
        FLOAT64 [:] y_error,
        FLOAT64 [:,:] x,  # (n_stims, n_times)
        FLOAT64 [:] x_pads,  # (n_stims,)
        INT8 [:] x_active,  # for each predictor whether it is still used
        INT64 [:,:] indexes,  # training segment indexes
        int i_start,  # kernel start index (time axis offset)
        INT64 [:] i_start_by_x,  # (n_stims,) kernel start index
        INT64 [:] i_stop_by_x, # (n_stims,) kernel stop index
        size_t error,  # ID of the error function (l1/l2)
        double delta,
        # buffers
        FLOAT64 [:,:] new_error,  # (n_stims, n_times_trf)
        INT8 [:,:] new_sign,  # (n_stims, n_times_trf)
    ):
    cdef:
        double e_add, e_sub, x_pad
        size_t n_stims = new_error.shape[0]
        size_t i_stim
        int i_time
        FLOAT64 [:] x_stim

    if error != 1 and error != 2:
        raise RuntimeError("error=%r" % (error,))

    with nogil:
        for i_stim in range(n_stims):
            if x_active[i_stim] == 0:
                continue
            x_stim = x[i_stim]
            x_pad = x_pads[i_stim]
            for i_time in range(i_start_by_x[i_stim], i_stop_by_x[i_stim]):
                # +/- delta
                if error == 1:
                    l1_for_delta(y_error, x_stim, x_pad, indexes, delta, i_time, &e_add, &e_sub)
                else:
                    l2_for_delta(y_error, x_stim, x_pad, indexes, delta, i_time, &e_add, &e_sub)

                i_time -= i_start
                if e_add > e_sub:
                    new_error[i_stim, i_time] = e_sub
                    new_sign[i_stim, i_time] = -1
                else:
                    new_error[i_stim, i_time] = e_add
                    new_sign[i_stim, i_time] = 1


def update_error(
        FLOAT64 [:] y_error,
        FLOAT64 [:] x,
        double x_pad,  # pad x outside valid convolution area
        INT64 [:,:] indexes,  # segment indexes
        double delta,
        int shift,
    ):
    cdef:
        size_t i, seg_i, seg_start, seg_stop, conv_start, conv_stop

    with nogil:
        for seg_i in range(indexes.shape[0]):
            seg_start = indexes[seg_i, 0]
            seg_stop = indexes[seg_i, 1]
            conv_start = seg_start
            conv_stop = seg_stop
            if shift > 0:
                conv_start += shift
            elif shift < 0:
                conv_stop += shift
            # padding
            d = delta * x_pad
            # pre-
            for i in range(seg_start, conv_start):
                y_error[i] -= d
            # post-
            for i in range(conv_stop, seg_stop):
                y_error[i] -= d
            # part of the segment that is affected
            for i in range(conv_start, conv_stop):
                y_error[i] -= delta * x[i - shift]
