# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
# cython: boundscheck=False, wraparound=False, cdivision=True

cimport cython
from cython.parallel cimport prange
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np


ctypedef np.float64_t FLOAT64
ctypedef np.int64_t INT64


def convolve_2d(
        FLOAT64 [:,:,:] h_flat,  # (n_h_only, n_shared, n_h_times)
        FLOAT64 [:,:,:] x_flat,  # (n_x_only, n_shared, n_x_times)
        FLOAT64 [:,:] x_pads,  # (n_x_only, n_shared)
        int h_i_start,  # offset of the first sample of h
        INT64 [:,:] segments,  # (n_segments, 2)
        FLOAT64 [:,:,:] out_flat,  # (n_x_only, n_h_only, n_x_times)
):
    cdef:
        Py_ssize_t i_h, i_x, i_out
        Py_ssize_t n_x_only = x_flat.shape[0]
        Py_ssize_t n_h_only = h_flat.shape[0]

    # loop through x and h dimensions
    for i_out in prange(n_h_only * n_x_only , nogil=True, schedule='guided'):
        i_h = i_out // n_x_only
        i_x = i_out % n_x_only
        convolve_1d(h_flat[i_h], x_flat[i_x], x_pads[i_x], h_i_start, segments, out_flat[i_x, i_h])


cpdef int convolve_1d(
        FLOAT64 [:,:] h,  # (n_x, n_h_times)
        FLOAT64 [:,:] x,  # (n_x, n_x_times)
        FLOAT64 [:] x_pads,  # (n_x,)
        int h_i_start,  # offset of the first sample of h
        INT64 [:,:] segments,  # (n_segments, 2)
        FLOAT64 [:] out,  # (n_x_times,)
) noexcept nogil:
    cdef:
        Py_ssize_t i, i_h, i_t, i_tau, i_x, start, stop
        Py_ssize_t n_x = h.shape[0]
        Py_ssize_t h_n_times = h.shape[1]
        Py_ssize_t h_i_stop = h_i_start + h_n_times
        Py_ssize_t pad_head_n_times = max(0, h_n_times + h_i_start)
        Py_ssize_t pad_tail_n_times = -min(0, h_i_start)
        FLOAT64 * out_pad
        FLOAT64 * pad_head
        FLOAT64 * pad_tail

    # padding: sum(h * x_pads[:, None], 0)
    if pad_head_n_times or pad_tail_n_times:
        # h_pad = pad * h
        out_pad = <FLOAT64*> malloc(sizeof(FLOAT64) * h_n_times)
        for i_t in range(h_n_times):
            out_pad[i_t] = 0
            for i_x in range(n_x):
                out_pad[i_t] += x_pads[i_x] * h[i_x, i_t]
        # padding for pre-
        if pad_head_n_times:
            pad_head = <FLOAT64*> malloc(sizeof(FLOAT64) * pad_head_n_times)
            for i_t in range(pad_head_n_times):
                for i_tau in range(min(pad_head_n_times - i_t, h_n_times)):
                    pad_head[i_t] += out_pad[h_n_times - i_tau - 1]
        # padding for post-
        if pad_tail_n_times:
            pad_tail = <FLOAT64*> malloc(sizeof(FLOAT64) * pad_tail_n_times)
            for i_t in range(pad_tail_n_times):
                for i_tau in range(min(i_t, h_n_times)):
                    pad_tail[i_t] += out_pad[i_tau]

    for i in range(len(segments)):
        start = segments[i, 0]
        stop = segments[i, 1]
        out[start: stop] = 0
        if pad_head_n_times:
            for i_tau in range(pad_head_n_times):
                out[start + i_tau] += pad_head[i_tau]
        if pad_tail_n_times:
            for i_tau in range(pad_tail_n_times):
                out[stop - pad_tail_n_times + i_tau] += pad_tail[i_tau]
        convolve_segment(h, x[:, start:stop], out[start:stop], h_i_start, h_i_stop)

    if pad_head_n_times or pad_tail_n_times:
        free(out_pad)
        if pad_head_n_times:
            free(pad_head)
        if pad_tail_n_times:
            free(pad_tail)

    return 0


cdef int convolve_segment(
        FLOAT64 [:,:] h,  # n_x, n_h_times
        FLOAT64 [:,:] x,  # n_x, n_x_times
        FLOAT64 [:] out,  # n_x_times
        Py_ssize_t i_start,
        Py_ssize_t i_stop,
) noexcept nogil:
    cdef:
        Py_ssize_t i_x, i_t, i_tau, i_t_tau
        Py_ssize_t n_times = x.shape[1]
        Py_ssize_t n_x = x.shape[0]
        Py_ssize_t n_tau = i_stop - i_start

    for i_t in range(n_times):
        for i_tau in range(n_tau):
            for i_x in range(n_x):
                i_t_tau = i_t + i_start + i_tau
                if i_t_tau < 0 or i_t_tau >= n_times:
                    continue
                out[i_t_tau] += h[i_x, i_tau] * x[i_x, i_t]
