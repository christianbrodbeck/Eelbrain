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


VERSION = 5


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
    train_method : str
        The training method used.
    """
    _attr = ('h', 'corr', 'isnan', 'delta', 't_run', 'version', 'error',
             'train_method', 'forward')

    def __init__(self, h, corr, isnan, t_run, version, delta=None, error='SS',
                 train_method='best', forward=None):
        self.h = h
        self.corr = corr
        self.isnan = isnan
        self.t_run = t_run
        self.version = version
        self.delta = delta
        self.error = error
        self.train_method = train_method
        self.forward = forward

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in self._attr}

    def __setstate__(self, state):
        self.__init__(**state)


def boosting(y, x, tstart, tstop, delta=0.005, forward=None, error='SScentered',
             train_method='best'):
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
    forward : NDVar
        Transform from h to y.
    error : 'SS' | 'SScentered' | 'sum(abs)' | 'sum(abs centered)'
        Error function to use (default is ``SScentered``).
    train_method : 'best' | 'stepdown'
        Kernel training method. At each training step: pick the ``best`` kernel
        modification for the training data, or ``stepdown`` until finding a
        modification that also helps the test data.

    Returns
    -------
    result : BoostingResult
        Object containig results from the boosting estimation (see
        :class:`BoostingResult`).
    """
    if train_method not in ('best', 'stepdown'):
        raise ValueError("train_method=%s" % repr(train_method))
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

    # determine forward model
    if forward is None:
        forward_m = None
        trf_dim = ydim
    else:
        assert forward.ndim == 2
        assert ydim is not None
        trf_dim, fwd_ydim = forward.get_dims((None, ydim.name))
        if fwd_ydim != ydim:
            forward = forward[ydim]
            trf_dim = forward.get_dim(trf_dim.name)
        forward_m = forward.get_data((ydim.name, trf_dim.name))

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
    if forward is None:
        n_responses = len(y_data)
        desc = "Boosting %i response" % n_responses + 's' * (n_responses > 1)
        total = n_responses * 10
    else:
        n_src = len(trf_dim)
        desc = "Boosting %i sources" % n_src + 's' * (n_src > 1)
        total = 10
    pbar = tqdm(desc=desc, total=total)
    if forward is None:
        res = [boosting_continuous(x_data, y_, trf_length, delta, error,
                                   train_method=train_method, pbar=pbar)
               for y_ in y_data]
        hs, corrs = zip(*res)
        h_x = np.array(hs)
    else:
        h_x, corrs = boosting_continuous(x_data, y_data, trf_length, delta,
                                         error, train_method=train_method,
                                         forward=forward_m, pbar=pbar)
    pbar.close()
    dt = time.time() - pbar.start_t

    # correlation
    if ydim is None:
        corr = corrs[0]
        isnan = np.isnan(corr)
    else:
        corrs = np.asarray(corrs)
        isnan = np.isnan(corrs)
        corrs[isnan] = 0
        corr = NDVar(corrs, (ydim,), cs.stat_info('r'), 'Correlation')

    # TRF
    h_time = UTS(tstart, time_dim.tstep, trf_length)
    hs = []
    for name, dim, index in x_meta:
        h_x_ = h_x[:, index, :]
        if dim is None:
            dims = (h_time,)
        else:
            dims = (dim, h_time)
        if trf_dim is None:
            h_x_ = h_x_[0]
        else:
            dims = (trf_dim,) + dims
        hs.append(NDVar(h_x_, dims, y.info.copy(), name))

    if multiple_x:
        hs = tuple(hs)
    else:
        hs = hs[0]

    return BoostingResult(hs, corr, isnan, dt, VERSION, delta, error,
                          train_method, forward)


def boosting_continuous(x, y, trf_length, delta, error, mindelta=None,
                        maxiter=10000, nsegs=10, train_method='best',
                        forward=None, pbar=None):
    """Boosting for a continuous data segment, cycle through even splits for
    test segment

    Parameters
    ----------
    ...
    error : 'SS' | 'Sabs'
        Error function to use.
    ...
    """
    logger = logging.getLogger('eelbrain.boosting')
    if mindelta is None:
        mindelta = delta
    hs = []

    for i in xrange(nsegs):
        h, test_sse_history, msg = boost_1seg(x, y, trf_length, delta, maxiter,
                                              nsegs, i, mindelta, error,
                                              train_method, forward)
        logger.debug(msg)
        if np.any(h):
            hs.append(h)
        if pbar is not None:
            pbar.update()

    if hs:
        h = np.mean(hs, 0)
        corr = corr_for_kernel(y, x, h, False, forward=forward)
    else:
        h = np.zeros(h.shape)
        if forward is None:
            corr = 0
        else:
            corr = np.zeros(len(forward))
    return h, corr


def boost_1seg(x, y, trf_length, delta, maxiter, nsegs, segno, mindelta, error,
               train_method, forward):
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
    train_method : 'best' | 'stepdown'
        Kernel training method.
    forward : array (optional)
        Forward operator, transform h to y.

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
    if forward is None:
        assert y.shape == (x.shape[1],)
    else:
        assert y.shape == (len(forward), x.shape[1])

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

    y_train = tuple(y[..., i] for i in train_index)
    y_test = (y[..., test_index],)
    x_train = tuple(x[:, i] for i in train_index)
    x_test = (x[:, test_index],)

    if forward is None:
        return boost_segs(y_train, y_test, x_train, x_test, trf_length, delta,
                          maxiter, mindelta, error, train_method)
    else:
        if train_method != 'best':
            raise NotImplementedError("train_method=%s for forward-boosting" %
                                      train_method)
        return boost_segs_fwd(y_train, y_test, x_train, x_test, trf_length,
                              delta, maxiter, mindelta, error, forward)


def boost_segs(y_train, y_test, x_train, x_test, trf_length, delta, maxiter,
               mindelta, error, train_method):
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
    error = ERROR_FUNC[error]
    n_stims = len(x_train[0])
    if any(len(x) != n_stims for x in chain(x_train, x_test)):
        raise ValueError("Not all x have same number of stimuli")
    n_times = [len(y) for y in chain(y_train, y_test)]
    if any(x.shape[1] != n for x, n in izip(chain(x_train, x_test), n_times)):
        raise ValueError("y and x have inconsistent number of time points")

    h = np.zeros((n_stims, trf_length))

    # buffers
    y_train_error = tuple(y.copy() for y in y_train)
    y_train_buf = tuple(np.empty(y.shape) for y in y_train)
    y_test_error = tuple(y.copy() for y in y_test)
    y_test_buf = tuple(np.empty(y.shape) for y in y_test)

    ys_error = y_train_error + y_test_error
    ys_delta = tuple(np.empty(y.shape) for y in ys_error)
    y_test_delta = ys_delta[len(y_train):]
    xs = x_train + x_test

    new_error = np.empty(h.shape)
    new_sign = np.empty(h.shape, np.int8)

    # history lists
    history = []
    test_error_history = []
    for i_boost in xrange(maxiter):
        history.append(h.copy())

        # evaluate current h
        e_test = sum(error(y, buf) for y, buf in izip(y_test_error, y_test_buf))
        e_train = sum(error(y, buf) for y, buf in izip(y_train_error, y_train_buf))

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
        for i_stim, i_time in product(xrange(h.shape[0]), xrange(h.shape[1])):
            # y_delta = change in y from delta change in h
            for yd, x in izip(ys_delta, x_train):
                yd[:i_time] = 0.
                yd[i_time:] = x[i_stim, :-i_time or None]
                yd *= delta

            # +/- delta
            e_add = 0
            e_sub = 0
            for y_err, dy, buf in izip(y_train_error, ys_delta, y_train_buf):
                # + delta
                np.subtract(y_err, dy, buf)
                e_add += error(buf, buf)
                # - delta
                np.add(y_err, dy, buf)
                e_sub += error(buf, buf)

            if e_add > e_sub:
                new_error[i_stim, i_time] = e_sub
                new_sign[i_stim, i_time] = -1
            else:
                new_error[i_stim, i_time] = e_add
                new_sign[i_stim, i_time] = 1

        while True:
            i_stim, i_time = np.unravel_index(np.argmin(new_error), h.shape)
            new_train_error = new_error[i_stim, i_time]
            delta_signed = new_sign[i_stim, i_time] * delta
            if new_train_error > e_train or train_method == 'best':
                break

            # predict new test error
            new_test_error = 0
            for dy, yerr, x in izip(y_test_delta, y_test_error, x_test):
                dy[:i_time] = 0.
                dy[i_time:] = x[i_stim, :-i_time or None]
                dy *= delta_signed
                np.subtract(yerr, dy, dy)
                new_test_error += error(dy, dy)

            if new_test_error < e_test:
                break

            new_error[i_stim, i_time] = e_train + 1

        # If no improvements can be found reduce delta
        if new_train_error > e_train:
            if delta < mindelta:
                reason = ("No improvement possible for training data, "
                          "stopping...")
                break
            else:
                delta *= 0.5
                # print("No improvement, new delta=%s..." % delta)
                continue

        # update h with best movement
        h[i_stim, i_time] += delta_signed

        if train_method == 'best':
            # abort if we're moving in circles
            if len(history) >= 2 and np.array_equal(h, history[-2]):
                reason = "Same h after 2 iterations"
                break
            elif len(history) >= 3 and np.array_equal(h, history[-3]):
                reason = "Same h after 3 iterations"
                break

        # update error
        for err, yd, x in izip(ys_error, ys_delta, xs):
            yd[:i_time] = 0.
            yd[i_time:] = x[i_stim, :-i_time or None]
            yd *= delta_signed
            err -= yd

    else:
        reason = "maxiter exceeded"

    best_iter = np.argmin(test_error_history)

    # Keep predictive power as the correlation for the best iteration
    return (history[best_iter], test_error_history,
            reason + ' (%i iterations)' % len(test_error_history))


def boost_segs_fwd(y_train, y_test, x_train, x_test, trf_length, delta, maxiter,
                   mindelta, error, forward):
    """Boosting supporting multiple array segments

    Parameters
    ----------
    y_train, y_test : tuple of array (n_resp, n_times)
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
    error = ERROR_FUNC[error]
    n_stims = len(x_train[0])
    if any(len(x) != n_stims for x in chain(x_train, x_test)):
        raise ValueError("Not all x have same number of stimuli")
    n_times = [y.shape[1] for y in chain(y_train, y_test)]
    if any(x.shape[1] != n for x, n in izip(chain(x_train, x_test), n_times)):
        raise ValueError("y and x have inconsistent number of time points")
    n_responses, n_sources = forward.shape

    h = np.zeros((n_sources, n_stims, trf_length))
    fwd_buf = np.empty((n_responses, 1))

    # buffers
    y_train_error = tuple(y.copy() for y in y_train)
    y_train_error_flat = tuple(y.ravel() for y in y_train_error)
    y_train_buf = tuple(np.empty(y.shape) for y in y_train)
    y_train_buf_flat = tuple(y.ravel() for y in y_train_buf)
    y_train_buf2_flat = tuple(np.empty(y.shape) for y in y_train_buf_flat)
    y_test_error = tuple(y.copy() for y in y_test)
    y_test_error_flat = tuple(y.ravel() for y in y_test_error)
    y_test_buf_flat = tuple(np.empty(y.shape) for y in y_test_error_flat)

    ys_error = y_train_error + y_test_error
    ys_delta = tuple(np.empty(y.shape) for y in ys_error)
    xs = x_train + x_test

    new_error = np.empty(h.shape)
    new_sign = np.empty(h.shape, np.int8)

    # history lists
    history = []
    test_error_history = []
    for i_boost in xrange(maxiter):
        history.append(h.copy())

        # evaluate current h
        e_test = sum(error(y, buf) for y, buf in izip(y_test_error_flat, y_test_buf_flat))
        e_train = sum(error(y, buf) for y, buf in izip(y_train_error_flat, y_train_buf_flat))

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
        for i_src in xrange(n_sources):
            fwd_buf[:, 0] = forward[:, i_src]
            fwd_buf *= delta
            for i_stim in xrange(n_stims):
                for dy, x in izip(ys_delta, x_train):
                    np.multiply(fwd_buf, x[i_stim], dy)  # impulse response

                # y_buf will contain the new error; initialize the early part:
                for err, buf in izip(y_train_error, y_train_buf):
                    buf[:, :trf_length] = err[:, :trf_length]

                for i_time in xrange(trf_length - 1, -1, -1):
                    # +/- delta
                    e_add = 0
                    e_sub = 0
                    for err, dy, buf, buf_flat, buf2_flat in \
                            izip(y_train_error, ys_delta, y_train_buf,
                                 y_train_buf_flat, y_train_buf2_flat):
                        if i_time:
                            err = err[:, i_time:]
                            dy = dy[:, :-i_time]
                            buf = buf[:, i_time:]

                        # + delta
                        np.subtract(err, dy, buf)
                        e_add += error(buf_flat, buf2_flat)
                        # - delta
                        np.add(err, dy, buf)
                        e_sub += error(buf_flat, buf2_flat)

                    if e_add > e_sub:
                        new_error[i_src, i_stim, i_time] = e_sub
                        new_sign[i_src, i_stim, i_time] = -1
                    else:
                        new_error[i_src, i_stim, i_time] = e_add
                        new_sign[i_src, i_stim, i_time] = 1

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

        i_src, i_stim, i_time = np.unravel_index(np.argmin(new_error), h.shape)
        delta_signed = new_sign[i_src, i_stim, i_time] * delta
        h[i_src, i_stim, i_time] += delta_signed

        # abort if we're moving in circles
        if len(history) >= 2 and np.array_equal(h, history[-2]):
            reason = "Same h after 2 iterations"
            break
        elif len(history) >= 3 and np.array_equal(h, history[-3]):
            reason = "Same h after 3 iterations"
            break

        # update error
        fwd_buf[:, 0] = forward[:, i_src]
        fwd_buf *= delta_signed
        for err, dy, x in izip(ys_error, ys_delta, xs):
            dy = dy[:, i_time:]
            np.multiply(fwd_buf, x[i_stim, :-i_time or None], dy)
            err[:, i_time:] -= dy

    else:
        reason = "maxiter exceeded"

    best_iter = np.argmin(test_error_history)

    # Keep predictive power as the correlation for the best iteration
    return (history[best_iter], test_error_history,
            reason + ' (%i iterations)' % len(test_error_history))


def apply_kernel(x, h, out=None):
    """Predict ``y`` by applying kernel ``h`` to ``x``

    x.shape is (n_stims, n_samples)
    h.shape is (n_stims, n_trf_samples)
    """
    if out is None:
        out = np.zeros(x.shape[1])
    else:
        out.fill(0)

    for ind in xrange(len(h)):
        out += np.convolve(h[ind], x[ind])[:len(out)]

    return out


def apply_kernel_3d(x, h, out=None):
    """Predict ``y`` by applying kernel ``h`` to ``x``

    x.shape is (n_stims, n_samples)
    h.shape is (n_responses, n_stims, n_trf_samples)
    y.shape is (n_responses, n_samples)
    """
    n_responses = len(h)
    n_stims, n_samples = x.shape
    if out is None:
        out = np.zeros((n_responses, n_samples))
    else:
        out.fill(0)

    for i_resp, i_stim in product(xrange(n_responses), xrange(n_stims)):
        out[i_resp] += np.convolve(h[i_resp, i_stim], x[i_stim])[:n_samples]

    return out


def corr_for_kernel(y, x, h, skip_beginning=True, out=None, forward=None):
    """Correlation of ``y`` and the prediction with kernel ``h``"""
    if forward is not None:
        assert h.ndim == 3
        y_pred = apply_kernel_3d(x, np.tensordot(forward, h, 1))
    else:
        y_pred = apply_kernel(x, h)

    if skip_beginning:
        i0 = h.shape[-1] - 1
        y = y[..., i0:]
        y_pred = y_pred[..., i0:]

    if forward is not None:
        if out is not None:
            raise NotImplementedError("out != None for forward")
        return [np.corrcoef(y[i], y_pred[i])[0, 1] for i in xrange(len(y))]

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
