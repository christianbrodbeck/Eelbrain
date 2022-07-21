# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy
from libc.math cimport fabs
from cython.parallel import prange
from libc.stdlib cimport malloc, free
import numpy as np

cimport numpy as np

ctypedef np.int8_t INT8
ctypedef np.int64_t INT64
ctypedef np.float64_t FLOAT64


cdef double inf = float('inf')


cdef double square(double x) nogil:
    # cf. https://github.com/scikit-image/scikit-image/issues/3026
    return x * x


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
        Py_ssize_t i, seg_i, seg_start, seg_stop, conv_start, conv_stop

    e_add[0] = 0.
    e_sub[0] = 0.

    for seg_i in range(indexes.shape[0]):
        seg_start = indexes[seg_i, 0]
        if seg_start == -1:
            break
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
        Py_ssize_t i, seg_i, seg_start, seg_stop, conv_start, conv_stop

    e_add[0] = 0.
    e_sub[0] = 0.

    for seg_i in range(indexes.shape[0]):
        seg_start = indexes[seg_i, 0]
        if seg_start == -1:
            break
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
            e_add[0] += square(y_error[i] - d)
            e_sub[0] += square(y_error[i] + d)
        # post-
        for i in range(conv_stop, seg_stop):
            e_add[0] += square(y_error[i] - d)
            e_sub[0] += square(y_error[i] + d)
        # part of the segment that is affected
        for i in range(conv_start, conv_stop):
            d = delta * x[i - shift]
            e_add[0] += square(y_error[i] - d)
            e_sub[0] += square(y_error[i] + d)


cpdef double error_for_indexes(
        FLOAT64[:] x,
        INT64[:,:] indexes,  # (n_segments, 2)
        int error,  # 1 --> l1; 2 --> l2
) nogil:
    cdef:
        Py_ssize_t seg_i, i
        double out = 0

    if error == 1:
        for seg_i in range(indexes.shape[0]):
            if indexes[seg_i, 0] == -1:
                break
            for i in range(indexes[seg_i, 0], indexes[seg_i, 1]):
                out += fabs(x[i])
    else:
        for seg_i in range(indexes.shape[0]):
            if indexes[seg_i, 0] == -1:
                break
            for i in range(indexes[seg_i, 0], indexes[seg_i, 1]):
                out += x[i] * x[i]
    return out


def boosting_runs(
        FLOAT64[:,:] y,  # (n_y, n_times)
        FLOAT64[:,:] x,  # (n_stims, n_times)
        FLOAT64 [:] x_pads,  # (n_stims,)
        INT64[:,:,:] split_train,
        INT64[:,:,:] split_validate,
        INT64[:,:,:] split_train_and_validate,
        INT64 [:] i_start_by_x,  # (n_stims,) kernel start index
        INT64 [:] i_stop_by_x, # (n_stims,) kernel stop index
        double delta,
        double mindelta,
        int error,
        int selective_stopping,
):
    """Estimate multiple filters with boosting"""
    cdef:
        Py_ssize_t n_y = len(y)
        Py_ssize_t n_x = len(x)
        Py_ssize_t n_splits = len(split_train)
        Py_ssize_t n_total = n_splits * n_y
        Py_ssize_t n_times_h = np.max(i_stop_by_x) - np.min(i_start_by_x)
        FLOAT64[:,:,:,:] hs = np.empty((n_splits, n_y, n_x, n_times_h))
        INT8[:,:] hs_failed = np.zeros((n_splits, n_y), 'int8')
        Py_ssize_t i, i_y, i_split
        BoostingRunResult *result

    for i in prange(n_total, nogil=True):
        i_y = i // n_splits
        i_split = i % n_splits
        result = boosting_run(y[i_y], x, x_pads, hs[i_split, i_y], split_train[i_split], split_validate[i_split], split_train_and_validate[i_split], i_start_by_x, i_stop_by_x, delta, mindelta, error, selective_stopping)
        hs_failed[i_split, i_y] = result.failed
        free_history(result)
    return hs.base, np.asarray(hs_failed, 'bool')


ctypedef struct BoostingStep:
    Py_ssize_t i_step
    Py_ssize_t i_stim
    Py_ssize_t i_time
    double delta
    double e_test
    double e_train
    BoostingStep *previous


cdef BoostingStep * boosting_step(
        Py_ssize_t i_step,
        Py_ssize_t i_stim,
        Py_ssize_t i_time,
        double delta,
        double e_test,
        double e_train,
        BoostingStep *previous,
) nogil:
    step = <BoostingStep*> malloc(sizeof(BoostingStep))
    step.i_step = i_step
    step.i_stim = i_stim
    step.i_time = i_time
    step.delta = delta
    step.e_test = e_test
    step.e_train = e_train
    step.previous = previous
    return step



ctypedef struct BoostingRunResult:
    int failed
    BoostingStep *history


cdef BoostingRunResult * boosting_run_result(int failed, BoostingStep *history) nogil:
    result = <BoostingRunResult*> malloc(sizeof(BoostingRunResult))
    result.failed = failed
    result.history = history
    return result


cdef void free_history(
        BoostingRunResult *result,
) nogil:
    cdef:
        BoostingStep *step
        BoostingStep *step_i

    step = result.history
    while step.previous != NULL:
        step_i = step.previous
        free(step)
        step = step_i
    free(result)


cdef BoostingRunResult * boosting_run(
        FLOAT64 [:] y,  # (n_times,)
        FLOAT64 [:,:] x,  # (n_stims, n_times)
        FLOAT64 [:] x_pads,  # (n_stims,)
        FLOAT64 [:,:] h,  # (n_stims, n_times_h)
        INT64[:,:] split_train,  # Training data index
        INT64[:,:] split_validate,  # Validation data index
        INT64[:,:] split_train_and_validate,  # Training and validation data index
        INT64 [:] i_start_by_x,  # (n_stims,) kernel start index
        INT64 [:] i_stop_by_x, # (n_stims,) kernel stop index
        double delta,
        double mindelta,
        int error,
        int selective_stopping,
) nogil:
    cdef:
        int out
        Py_ssize_t n_x = x.shape[0]
        Py_ssize_t n_x_active = n_x
        Py_ssize_t n_times = x.shape[1]
        Py_ssize_t n_times_h = h.shape[1]
        Py_ssize_t i_start
        Py_ssize_t n_times_trf
        BoostingStep *step
        BoostingStep *step_i
        BoostingStep *history = NULL

        # buffers
        FLOAT64[:] y_error
        FLOAT64[:,:] new_error
        INT8[:,:] new_sign
        INT8[:] x_active

    with gil:
        i_start = np.min(i_start_by_x)
        n_times_trf = np.max(i_stop_by_x) - i_start
        y_error = y.copy()
        new_error = np.empty((n_x, n_times_h))
        new_sign = np.empty((n_x, n_times_h), np.int8)
        x_active = np.ones(n_x, np.int8)

    h[...] = 0
    new_error[...] = inf  # ignore values outside TRF

    # history
    cdef:
        Py_ssize_t i_stim = -1
        Py_ssize_t i_time = -1
        double delta_signed = 0.
        double new_train_error
        double best_test_error = inf
        Py_ssize_t best_iteration = 0
        int n_bad, undo
        long argmin
        Py_ssize_t i_step, i

    # pre-assign iterators
    for i_step in range(999999):
        # evaluate current h
        e_train = error_for_indexes(y_error, split_train, error)
        e_test = error_for_indexes(y_error, split_validate, error)
        step = boosting_step(i_step, i_stim, i_time, delta_signed, e_test, e_train, history)
        history = step

        # print(i_step, 'error:', e_test)

        # evaluate stopping conditions
        if e_test < best_test_error:
            # print(' ', e_test, '<', best_test_error)
            best_test_error = e_test
            best_iteration = i_step
        elif i_step >= 2:
            step_i = step.previous
            # print(' ', e_test, '>', step_i.e_test, '? (', step.i_step, step_i.i_step, ')')
            if e_test > step_i.e_test:
                if selective_stopping:
                    if selective_stopping == 1:
                        undo = 1
                    else:
                        # only stop if the predictor overfits n times without intermittent improvement
                        n_bad = 1
                        undo = 1
                        while step_i.previous != NULL:
                            if step_i.e_test > e_test:
                                undo = 0
                                break  # the error improved
                            undo += 1
                            if step_i.i_stim == i_stim:
                                if step_i.e_test > step_i.previous[0].e_test:
                                    # the same stimulus caused an error increase
                                    n_bad += 1
                                    if n_bad == selective_stopping:
                                        break
                                else:
                                    undo = 0
                                    break
                            step_i = step_i.previous

                    if undo:
                        # print(' undo')
                        # revert changes
                        for i in range(undo):
                            h[step.i_stim, step.i_time] -= step.delta
                            update_error(y_error, x[step.i_stim], x_pads[step.i_stim], split_train_and_validate, -step.delta, step.i_time + i_start)
                            step_i = step.previous
                            free(step)
                            step = step_i
                        history = step
                        # disable predictor
                        x_active[i_stim] = False
                        n_x_active -= 1
                        if n_x_active == 0:
                            break
                        new_error[i_stim, :] = inf
                # Basic
                # -----
                # stop the iteration if all the following requirements are met
                # 1. more than 10 iterations are done
                # 2. The testing error in the latest iteration is higher than that in
                #    the previous two iterations
                elif i_step > 10 and e_test > step_i.previous[0].e_test:
                    # print("error(test) not improving in 2 steps")
                    break

        # generate possible movements -> training error
        argmin = generate_options(y_error, x, x_pads, x_active, split_train, i_start, i_start_by_x, i_stop_by_x, error, delta, new_error, new_sign)
        i_stim = argmin // n_times_trf
        i_time = argmin % n_times_trf
        new_train_error = new_error[i_stim, i_time]
        delta_signed = new_sign[i_stim, i_time] * delta
        # print(new_train_error, end=', ')

        # If no improvements can be found reduce delta
        if new_train_error > step.e_train:
            delta *= 0.5
            if delta >= mindelta:
                i_stim = i_time = -1
                delta_signed = 0.
                # print("new delta: %s" % delta)
                continue
            else:
                # print("No improvement possible for training data")
                break

        # abort if we're moving in circles
        if step.delta and i_stim == step.i_stim and i_time == step.i_time and delta_signed == -step.delta:
            # print("Moving in circles")
            break

        # update h with best movement
        h[i_stim, i_time] += delta_signed
        update_error(y_error, x[i_stim], x_pads[i_stim], split_train_and_validate, delta_signed, i_time + i_start)
    else:
        with gil:
            raise RuntimeError("Boosting: maximum number of iterations exceeded")

    # reverse changes after best iteration
    if best_iteration:
        while step.i_step > best_iteration:
            if step.delta:
                h[step.i_stim, step.i_time] -= step.delta
            step = step.previous
        return boosting_run_result(0, history)
    else:
        # print(' failed')
        return boosting_run_result(1, history)


cdef Py_ssize_t generate_options(
        FLOAT64 [:] y_error,
        FLOAT64 [:,:] x,  # (n_stims, n_times)
        FLOAT64 [:] x_pads,  # (n_stims,)
        INT8 [:] x_active,  # for each predictor whether it is still used
        INT64 [:,:] indexes,  # training segment indexes
        int i_start,  # kernel start index (time axis offset)
        INT64 [:] i_start_by_x,  # (n_stims,) kernel start index
        INT64 [:] i_stop_by_x, # (n_stims,) kernel stop index
        int error,  # ID of the error function (l1/l2)
        double delta,
        # buffers
        FLOAT64 [:,:] new_error,  # (n_stims, n_times_trf)
        INT8 [:,:] new_sign,  # (n_stims, n_times_trf)
    ) nogil:
    cdef:
        double e_add, e_sub, e_new, x_pad
        Py_ssize_t n_stims = new_error.shape[0]
        Py_ssize_t n_times = new_error.shape[1]
        Py_ssize_t i_stim, i_time, i_stim_min, i_time_min
        FLOAT64 [:] x_stim
        double e_min = inf

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
                e_new = e_sub
                new_sign[i_stim, i_time] = -1
            else:
                e_new = e_add
                new_sign[i_stim, i_time] = 1
            new_error[i_stim, i_time] = e_new

            # find smallest error
            if e_new < e_min:
                e_min = e_new
                i_stim_min = i_stim
                i_time_min = i_time

    return i_stim_min * n_times + i_time_min


cdef void update_error(
        FLOAT64 [:] y_error,
        FLOAT64 [:] x,
        double x_pad,  # pad x outside valid convolution area
        INT64 [:,:] indexes,  # segment indexes
        double delta,
        Py_ssize_t shift,
    ) nogil:
    cdef:
        Py_ssize_t i, seg_i, seg_start, seg_stop, conv_start, conv_stop

    for seg_i in range(indexes.shape[0]):
        seg_start = indexes[seg_i, 0]
        if seg_start == -1:
            break
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


cdef class Step:
    cdef readonly:
        long i_step, i_stim, i_time
        double delta, e_test, e_train

    def __init__(self, i_step, i_stim, i_time, delta, e_test, e_train):
        self.i_step = i_step
        self.i_stim = i_stim
        self.i_time = i_time
        self.delta = delta
        self.e_test = e_test
        self.e_train = e_train

    def __repr__(self):
        return f"{self.i_step:4}: {self.i_stim:4}, {self.i_time:4} {self.delta:+} --> {self.e_train:10f} / {self.e_test:10f}"


def boosting_fit(
        FLOAT64 [:] y,  # (n_times,)
        FLOAT64 [:,:] x,  # (n_stims, n_times)
        FLOAT64 [:] x_pads,  # (n_stims,)
        INT64[:,:] split_train,
        INT64[:,:] split_validate,
        INT64[:,:] split_train_and_validate,
        INT64 [:] i_start_by_x,  # (n_stims,) kernel start index
        INT64 [:] i_stop_by_x, # (n_stims,) kernel stop index
        double delta,
        double mindelta,
        int error,
        int selective_stopping = 0,
):
    """Single model fit using boosting

    Parameters
    ----------
    y : array (n_times,)
        Dependent signal, time series to predict.
    x : array (n_stims, n_times)
        Stimulus.
    x_pads : array (n_stims,)
        Padding for x.
    split_train
        Training data index.
    split_validate
        Validation data index.
    split_train_and_validate
        Training and validation data index.
    i_start_by_x : ndarray
        Array of i_start for trfs.
    i_stop_by_x : ndarray
        Array of i_stop for TRF.
    delta : scalar
        Step of the adjustment.
    mindelta : scalar
        Smallest delta to use. If no improvement can be found in an iteration,
        the first step is to divide delta in half, but stop if delta becomes
        smaller than ``mindelta``.
    error: int
        Error function to use (1 for l1, 2 for l2).
    selective_stopping : int
        Selective stopping.
    """
    cdef:
        BoostingRunResult *result
        BoostingStep *step
        Step step2
        Py_ssize_t n_x = len(x)
        Py_ssize_t n_times_h = np.max(i_stop_by_x) - np.min(i_start_by_x)
        FLOAT64[:, :] h = np.empty((n_x, n_times_h))

    result = boosting_run(y, x, x_pads, h, split_train, split_validate, split_train_and_validate, i_start_by_x, i_stop_by_x, delta, mindelta, error, selective_stopping)
    out = []
    step = result.history
    while step != NULL:
        step2 = Step(step.i_step, step.i_stim, step.i_time, step.delta, step.e_test, step.e_train)
        out.insert(0, step2)
        step = step.previous
    free_history(result)
    return np.asarray(h), out
