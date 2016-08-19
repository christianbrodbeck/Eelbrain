from __future__ import division
from itertools import izip
import logging
from math import floor
import time

import numpy as np
from numpy import newaxis
from scipy.stats import spearmanr

from .._data_obj import NDVar, UTS


class BoostingResult(object):
    """Result from boosting temporal response function"""
    _attr = ('h', 'corr', 'isnan', 't_run')

    def __init__(self, h, corr, isnan, t_run):
        self.h = h
        self.corr = corr
        self.isnan = isnan
        self.t_run = t_run

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in self._attr}

    def __setstate__(self, state):
        self.__init__(*(state[attr] for attr in self._attr))


def boosting(y, x, tstart, tstop, delta=0.005):
    """Estimate a temporal response function of ``x`` through boosting

    Parameters
    ----------
    y : NDVar
        Signal to predict.
    x : NDVar | sequence of NDVar
        Signal to use to predict ``y``. Can be sequence of NDVars to include
        multiple predictors.
    tstart : float
        Start of the TRF in seconds.
    tstop : float
        Stop of the TRF in seconds.
    """
    if isinstance(x, NDVar):
        x = (x,)
        multiple_x = False
    else:
        multiple_x = True
    time_dim = y.get_dim('time')
    if any(x_.get_dim('time') != time_dim for x_ in x):
        raise ValueError("Not all NDVars have the same time dimension")

    # x:  predictor x time
    x_dims = []
    x_data = []
    x_slices = []
    i = 0
    for x_ in x:
        if x_.ndim == 1:
            xdim = None
            data = x_.x[newaxis, :]
            index = i
        elif x_.ndim == 2:
            xdim = x_.dims[not x_.dimnames.index('time')]
            data = x_.get_data((xdim.name, 'time'))
            index = slice(i, i + len(data))
        else:
            raise NotImplementedError("x with more than 2 dimensions")
        x_dims.append(xdim)
        x_data.append(data)
        x_slices.append(index)
        i += len(data)

    if len(x_data) == 1:
        x_data = x_data[0]
    else:
        x_data = np.vstack(x_data)

    # y
    if y.ndim == 1:
        ydim = None
        y_data = y.x[None, :]
    elif y.ndim == 2:
        ydim = y.dims[not y.dimnames.index('time')]
        y_data = y.get_data((ydim.name, 'time'))
    else:
        raise NotImplementedError("y with more than 2 dimensions")

    # trf
    i_start = int(round(tstart / y.time.tstep))
    i_stop = int(round(tstop / y.time.tstep))
    trf_length = i_stop - i_start
    if i_start < 0:
        x_data = x_data[:, -i_start:]
        y_data = y_data[:, :i_start]
    elif i_start > 0:
        raise NotImplementedError("start > 0")

    t0 = time.time()
    hs = []
    corrs = []
    for y_ in y_data:
        h, corr = boosting_continuous(x_data, y_, trf_length, delta)
        hs.append(h)
        corrs.append(corr)
    dt = time.time() - t0

    # correlation
    if ydim is None:
        corr = corrs[0]
        isnan = np.isnan(corr)
    else:
        corrs = np.array(corrs)
        isnan = np.isnan(corrs)
        corrs[isnan] = 0
        corr = NDVar(corrs, (ydim,))

    # TRF
    h_time = UTS(tstart, y.time.tstep, trf_length)
    h_x = np.array(hs)
    hs = []
    for dim, index in izip(x_dims, x_slices):
        h_x_ = h_x[:, index, :]
        if dim is None:
            dims = (h_time,)
        else:
            dims = (dim, h_time)
        if ydim is None:
            h_x_ = h_x_[0]
        else:
            dims = (ydim,) + dims
        hs.append(NDVar(h_x_, dims))

    if not multiple_x:
        hs = hs[0]

    return BoostingResult(hs, corr, isnan, dt)


def boosting_continuous(x, y, trf_length, delta, mindelta=None, maxiter=10000, nsegs=10):
    """Boosting for a continuous data segment, cycle through even splits for
    test segment"""
    logger = logging.getLogger('eelbrain.boosting')
    if mindelta is None:
        mindelta = delta
    hs = []
    for i in xrange(nsegs):
        h, test_sse_history, msg = boost_1seg(x, y, trf_length, delta, maxiter,
                                              nsegs, i, mindelta)
        logger.debug(msg)
        if np.any(h):
            hs.append(h)

    if hs:
        h = np.mean(hs, 0)
        corr = corr_for_kernel(y, x, h, False)
    else:
        h = np.zeros((len(x), trf_length))
        corr = 0
    return h, corr


def boost_1seg(x, y, trf_length, delta, maxiter, nsegs, segno, mindelta):
    """Basic port of svdboostV4pred

    Parameters
    ----------
    x : array (n_stims, n_times)
        Stimulus.
    y : array (n_times,)
        Dependent signal, time series to predict.
    trf_length : int
        Length of the TRF (in time samples).
    delta : scalar
        Step of the adjustment.
    maxiter : int
        Maximum number of iterations.
    nsegs : int
        Number of segments
    segno : int [0, nsegs-1]
        which segment to use for testing
    mindelta : scalar
        Smallest delta to use. If no improvement can be found in an iteration,
        the first step is to divide delta in half, but stop if delta becomes
        smaller than ``mindelta``.

    Returns
    -------
    history[best_iter] : array like h
        Winning kernel.
    test_corr[best_iter] : scalar
        Test data correlation for winning kernel.
    test_rcorr[best_iter] : scalar
        Test data rank correlation for winning kernel.
    test_sse_history : list of len n_iterations
        SSE for test data at each iteration
    train_corr : list of len n_iterations
        Correlation for training data at each iteration.
    """
    n_stims, n_times = x.shape
    assert y.shape == (n_times,)

    h = np.zeros((n_stims, trf_length))

    # separate training and testing signal
    test_seg_len = int(floor(x.shape[1] / nsegs))
    testing_range = np.arange(test_seg_len, dtype=int) + test_seg_len * segno
    training_range = np.setdiff1d(np.arange(x.shape[1], dtype=int), testing_range)
    x_test = x[:, testing_range]
    y_test = y[testing_range]
    x = x[:, training_range]
    y = y[training_range]

    # buffers
    ypred_now = np.empty(y.shape)
    ypred_next_step = np.empty(y.shape)
    ypred_test = np.empty(y_test.shape)
    y_test_error = np.empty(y_test.shape)
    new_error = np.empty(h.shape)
    new_sign = np.empty(h.shape, np.int8)
    y_delta = np.empty(y.shape)

    # history lists
    history = []
    test_sse_history = []
    for i_boost in xrange(maxiter):
        history.append(h.copy())

        # evaluate current h
        if np.any(h):
            # predict
            apply_kernel(x, h, ypred_now)
            apply_kernel(x_test, h, ypred_test)

            # Compute predictive power on testing data
            np.subtract(y_test, ypred_test, y_test_error)
            test_sse_history.append(np.dot(y_test_error, y_test_error[:, None])[0])
        else:
            ypred_now.fill(0)
            test_sse_history.append(np.dot(y_test, y_test[:, None])[0])

        # stop the iteration if all the following requirements are met
        # 1. more than 10 iterations are done
        # 2. The testing error in the latest iteration is higher than that in
        #    the previous two iterations
        if (i_boost > 10 and test_sse_history[-1] > test_sse_history[-2] and
                test_sse_history[-1] > test_sse_history[-3]):
            reason = "SSE(test) not improving in 2 steps"
            break

        # generate possible movements
        new_sign.fill(0)
        for ind1 in xrange(h.shape[0]):
            for ind2 in xrange(h.shape[1]):
                # y_delta = change in y from delta change in h
                y_delta[:ind2] = 0.
                y_delta[ind2:] = x[ind1, :-ind2 or None]
                y_delta *= delta

                # ypred = ypred_now + y_delta
                # error = SS(y - ypred)
                np.add(ypred_now, y_delta, ypred_next_step)
                np.subtract(y, ypred_next_step, ypred_next_step)
                e1 = np.dot(ypred_next_step, ypred_next_step[:, None])

                # ypred = y_pred_now - y_delta
                # error = SS(y - ypred)
                np.subtract(ypred_now, y_delta, ypred_next_step)
                np.subtract(y, ypred_next_step, ypred_next_step)
                e2 = np.dot(ypred_next_step, ypred_next_step[:, None])

                if e1 > e2:
                    new_error[ind1, ind2] = e2
                    new_sign[ind1, ind2] = -1
                else:
                    new_error[ind1, ind2] = e1
                    new_sign[ind1, ind2] = 1

        # If no improvements can be found reduce delta
        if new_error.min() > np.sum((y - ypred_now) ** 2):
            if delta < mindelta:
                reason = ("No improvement possible for training data, "
                          "stopping...")
                break
            else:
                delta *= 0.5
                # print("No improvement, new delta=%s..." % delta)
                continue

        # update h with best movement
        bestfil = np.unravel_index(np.argmin(new_error), h.shape)
        h[bestfil] += new_sign[bestfil] * delta

        # abort if we're moving in circles
        if len(history) >= 2 and np.array_equal(h, history[-2]):
            reason = "Same h after 2 iterations"
            break
        elif len(history) >= 3 and np.array_equal(h, history[-3]):
            reason = "Same h after 3 iterations"
            break
    else:
        reason = "maxiter exceeded"

    best_iter = np.argmin(test_sse_history)

    # Keep predictive power as the correlation for the best iteration
    return (history[best_iter], test_sse_history,
            reason + ' (%i iterations)' % len(test_sse_history))


def apply_kernel(x, h, out=None):
    """Predict ``y`` by applying kernel ``h`` to ``x``"""
    if out is None:
        out = np.zeros(x.shape[1])
    else:
        out.fill(0)

    for ind in xrange(len(h)):
        out += np.convolve(h[ind], x[ind])[:len(out)]

    return out


def corr_for_kernel(y, x, h, skip_beginning=True, out=None):
    """Correlation of ``y`` and the prediction with kernel ``h``"""
    y_pred = apply_kernel(x, h)
    if skip_beginning:
        i0 = h.shape[1] - 1
        y = y[i0:]
        y_pred = y_pred[i0:]

    if out is None:
        return np.corrcoef(y, y_pred)[0, 1]
    elif out == 'rank':
        return spearmanr(y, y_pred)[0]
    elif out == 'both':
        return np.corrcoef(y, y_pred)[0, 1], spearmanr(y, y_pred)[0]
    else:
        raise ValueError("out=%s" % repr(out))
