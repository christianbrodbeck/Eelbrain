from __future__ import division
from itertools import chain, izip, product
import logging
from math import floor
import time

import numpy as np
from numpy import newaxis
from scipy.stats import spearmanr
from tqdm import tqdm

from .. import _colorspaces as cs
from .._data_obj import NDVar, UTS


VERSION = 4


class BoostingResult(object):
    """Result from boosting a temporal response function

    Attributes
    ----------
    h : NDVar | list of NDVar
        The temporal response function. Whether ``h`` is an NDVar or a list of
        NDVars depends on whether the ``x`` parameter to :func:`boosting` was
        an NDVar or a list.
    corr : float | NDVar
        Correlation between the measured response and the response predicted
        with ``h``. Type depends on the ``y`` parameter to :func:`boosting`.
    t_run : float
        Time it took to run the boosting algorithm (in seconds).
    error : str
        The error evaluation method used.
    """
    _attr = ('h', 'corr', 'isnan', 't_run', 'version', 'error')

    def __init__(self, h, corr, isnan, t_run, version, error='SS'):
        self.error = error
        self.h = h
        self.corr = corr
        self.isnan = isnan
        self.t_run = t_run
        self.version = version

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in self._attr}

    def __setstate__(self, state):
        self.__init__(**state)


def boosting(y, x, tstart, tstop, delta=0.005, error='SScentered'):
    """Estimate a temporal response function through boosting

    Parameters
    ----------
    y : NDVar
        Signal to predict.
    x : NDVar | sequence of NDVar
        Signal to use to predict ``y``. Can be sequence of NDVars to include
        multiple predictors. Time dimension must correspond to ``y``.
    tstart : float
        Start of the TRF in seconds.
    tstop : float
        Stop of the TRF in seconds.
    error : 'SS' | 'SScentered' | 'sum(abs)' | 'sum(abs centered)'
        Error function to use (default is ``SScentered``).

    Returns
    -------
    result : BoostingResult
        Object containig results from the boosting estimation (see
        :class:`BoostingResult`).
    """
    # check y and x
    if isinstance(x, NDVar):
        x = (x,)
        multiple_x = False
    else:
        multiple_x = True
    time_dim = y.get_dim('time')
    if any(x_.get_dim('time') != time_dim for x_ in x):
        raise ValueError("Not all NDVars have the same time dimension")

    # x_data:  predictor x time array
    x_data = []
    x_meta = []
    i = 0
    for x_ in x:
        if x_.ndim == 1:
            xdim = None
            data = x_.x[newaxis, :]
            index = i
        elif x_.ndim == 2:
            xdim = x_.dims[not x_.get_axis('time')]
            data = x_.get_data((xdim.name, 'time'))
            index = slice(i, i + len(data))
        else:
            raise NotImplementedError("x with more than 2 dimensions")
        x_data.append(data)
        x_meta.append((x_.name, xdim, index))
        i += len(data)

    if len(x_data) == 1:
        x_data = x_data[0]
    else:
        x_data = np.vstack(x_data)

    # y_data:  ydim x time array
    if y.ndim == 1:
        ydim = None
        y_data = y.x[None, :]
    elif y.ndim == 2:
        ydim = y.dims[not y.get_axis('time')]
        y_data = y.get_data((ydim.name, 'time'))
    else:
        raise NotImplementedError("y with more than 2 dimensions")

    # prepare trf (by cropping data)
    i_start = int(round(tstart / time_dim.tstep))
    i_stop = int(round(tstop / time_dim.tstep))
    trf_length = i_stop - i_start
    if i_start < 0:
        x_data = x_data[:, -i_start:]
        y_data = y_data[:, :i_start]
    elif i_start > 0:
        x_data = x_data[:, :-i_start]
        y_data = y_data[:, i_start:]

    # do boosting
    n_responses = len(y_data)
    pbar = tqdm(desc="Boosting %i response" % n_responses + 's' * (n_responses > 1),
                total=n_responses * 10)
    res = [boosting_continuous(x_data, y_, trf_length, delta, error, pbar=pbar)
           for y_ in y_data]
    pbar.close()
    hs, corrs = zip(*res)
    dt = time.time() - pbar.start_t

    # correlation
    if ydim is None:
        corr = corrs[0]
        isnan = np.isnan(corr)
    else:
        corrs = np.array(corrs)
        isnan = np.isnan(corrs)
        corrs[isnan] = 0
        corr = NDVar(corrs, (ydim,), cs.stat_info('r'), 'Correlation')

    # TRF
    h_time = UTS(tstart, time_dim.tstep, trf_length)
    h_x = np.array(hs)
    hs = []
    for name, dim, index in x_meta:
        h_x_ = h_x[:, index, :]
        if dim is None:
            dims = (h_time,)
        else:
            dims = (dim, h_time)
        if ydim is None:
            h_x_ = h_x_[0]
        else:
            dims = (ydim,) + dims
        hs.append(NDVar(h_x_, dims, y.info.copy(), name))

    if multiple_x:
        hs = tuple(hs)
    else:
        hs = hs[0]

    return BoostingResult(hs, corr, isnan, dt, VERSION, error)


def boosting_continuous(x, y, trf_length, delta, error, mindelta=None,
                        maxiter=10000, nsegs=10, pbar=None):
    """Boosting for a continuous data segment, cycle through even splits for
    test segment

    Parameters
    ----------
    ...
    error : 'SS' | 'Sabs'
        Error function to use.
    """
    logger = logging.getLogger('eelbrain.boosting')
    if mindelta is None:
        mindelta = delta
    hs = []
    for i in xrange(nsegs):
        h, test_sse_history, msg = boost_1seg(x, y, trf_length, delta, maxiter,
                                              nsegs, i, mindelta, error)
        logger.debug(msg)
        if np.any(h):
            hs.append(h)
        if pbar is not None:
            pbar.update()

    if hs:
        h = np.mean(hs, 0)
        corr = corr_for_kernel(y, x, h, False)
    else:
        h = np.zeros((len(x), trf_length))
        corr = 0
    return h, corr


def boost_1seg(x, y, trf_length, delta, maxiter, nsegs, segno, mindelta, error):
    """boosting with one test segment determined by regular division

    Based on port of svdboostV4pred

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
    error : 'SS' | 'Sabs'
        Error function to use.

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
    assert x.ndim == 2
    assert y.shape == (x.shape[1],)

    # separate training and testing signal
    test_seg_len = int(floor(x.shape[1] / nsegs))
    test_index = slice(test_seg_len * segno, test_seg_len * (segno + 1))
    if segno == 0:
        train_index = (slice(test_seg_len, None),)
    elif segno == nsegs-1:
        train_index = (slice(None, -test_seg_len),)
    elif segno < 0 or segno >= nsegs:
        raise ValueError("segno=%r" % segno)
    else:
        train_index = (slice(None, test_seg_len * segno),
                       slice(test_seg_len * (segno + 1), None))

    y_train = [y[i] for i in train_index]
    y_test = (y[test_index],)
    x_train = [x[:, i] for i in train_index]
    x_test = (x[:, test_index],)

    return boost_segs(y_train, y_test, x_train, x_test, trf_length, delta,
                      maxiter, mindelta, error)


def boost_segs(y_train, y_test, x_train, x_test, trf_length, delta, maxiter,
               mindelta, error):
    """Boosting supporting multiple array segments

    Parameters
    ----------
    y_train, y_test : tuple of array (n_times,)
        Dependent signal, time series to predict.
    x_train, x_test : array (n_stims, n_times)
        Stimulus.
    trf_length : int
        Length of the TRF (in time samples).
    delta : scalar
        Step of the adjustment.
    maxiter : int
        Maximum number of iterations.
    mindelta : scalar
        Smallest delta to use. If no improvement can be found in an iteration,
        the first step is to divide delta in half, but stop if delta becomes
        smaller than ``mindelta``.
    error : 'SS' | 'Sabs'
        Error function to use.
    """
    n_stims = len(x_train[0])
    if any(len(x) != n_stims for x in chain(x_train, x_test)):
        raise ValueError("Not all x have same number of stimuli")
    n_times = [len(y) for y in chain(y_train, y_test)]
    if any(x.shape[1] != n for x, n in izip(chain(x_train, x_test), n_times)):
        raise ValueError("y and x have inconsistent number of time points")
    error = ERROR_FUNC[error]

    h = np.zeros((n_stims, trf_length))

    # buffers
    y_train_pred = [np.empty(y.shape) for y in y_train]
    y_train_pred_next = [np.empty(y.shape) for y in y_train]
    y_delta = [np.empty(y.shape) for y in y_train]
    y_test_pred = [np.empty(y.shape) for y in y_test]
    # y_train_error = [np.empty(y.shape) for y in y_train]
    y_test_error = [np.empty(y.shape) for y in y_test]
    new_error = np.empty(h.shape)
    new_sign = np.empty(h.shape, np.int8)

    # history lists
    history = []
    test_error_history = []
    for i_boost in xrange(maxiter):
        history.append(h.copy())

        # evaluate current h
        if np.any(h):
            # predict
            for x, y in izip(x_train, y_train_pred):
                apply_kernel(x, h, y)
            for x, y in izip(x_test, y_test_pred):
                apply_kernel(x, h, y)

            # Compute predictive power on testing data
            e_test = sum(error(np.subtract(y, pred, err), err) for y, pred, err in
                         izip(y_test, y_test_pred, y_test_error))
        else:
            for y in y_train_pred:
                y.fill(0)
            e_test = sum(error(err) for err in y_test)

        e_train = sum(error(y - ynow) for y, ynow in izip(y_train, y_train_pred))

        test_error_history.append(e_test)

        # stop the iteration if all the following requirements are met
        # 1. more than 10 iterations are done
        # 2. The testing error in the latest iteration is higher than that in
        #    the previous two iterations
        if (i_boost > 10 and test_error_history[-1] > test_error_history[-2] and
                test_error_history[-1] > test_error_history[-3]):
            reason = "error(test) not improving in 2 steps"
            break

        # generate possible movements -> training error
        new_sign.fill(0)
        for ind1, ind2 in product(xrange(h.shape[0]), xrange(h.shape[1])):
            # y_delta = change in y from delta change in h
            for y, x in izip(y_delta, x_train):
                y[:ind2] = 0.
                y[ind2:] = x[ind1, :-ind2 or None]
                y *= delta

            # +/- delta
            e_add = 0
            e_sub = 0
            for y, ynow, dy, ynext in izip(y_train, y_train_pred, y_delta, y_train_pred_next):
                # + delta
                np.add(ynow, dy, ynext)
                np.subtract(y, ynext, ynext)
                e_add += error(ynext, ynext)
                # - delta
                np.subtract(ynow, dy, ynext)
                np.subtract(y, ynext, ynext)
                e_sub += error(ynext, ynext)

            if e_add > e_sub:
                new_error[ind1, ind2] = e_sub
                new_sign[ind1, ind2] = -1
            else:
                new_error[ind1, ind2] = e_add
                new_sign[ind1, ind2] = 1

        # If no improvements can be found reduce delta
        if new_error.min() > e_train:
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

    best_iter = np.argmin(test_error_history)

    # Keep predictive power as the correlation for the best iteration
    return (history[best_iter], test_error_history,
            reason + ' (%i iterations)' % len(test_error_history))


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


# Error functions
def ss(error, buf=None):
    "Sum squared error"
    return np.dot(error, error[:, None])[0]


# Error functions
def ss_centered(error, buf=None):
    "Sum squared of the centered error"
    error = np.subtract(error, error.mean(), buf)
    return np.dot(error, error[:, None])[0]


def sum_abs(error, buf=None):
    "Sum of absolute error"
    return np.abs(error, buf).sum()


def sum_abs_centered(error, buf=None):
    "Sum of absolute centered error"
    error = np.subtract(error, error.mean(), buf)
    return np.abs(error, buf).sum()


ERROR_FUNC = {'SS': ss, 'SScentered': ss_centered,
              'sum(abs)': sum_abs, 'sum(abs centered)': sum_abs}
