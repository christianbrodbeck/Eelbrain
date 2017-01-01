"""
Boosting as described by David et al. (2007).


Profiling
---------
ds = datasets._get_continuous()
y = ds['y']
x1 = ds['x1']
x2 = ds['x2']

%prun -s cumulative res = boosting(y, x1, 0, 1)

"""
from __future__ import division
from inspect import getargspec
from itertools import chain, izip, product
from math import floor
import time

import numpy as np
from numpy import newaxis
from scipy.stats import spearmanr
from tqdm import tqdm

from .. import _colorspaces as cs
from .._data_obj import NDVar, UTS
from .._stats.error_functions import (l1, l2, l1_for_delta, l2_for_delta,
                                      update_error)


# BoostingResult version
VERSION = 6

ERROR_FUNC = {'l2': l2, 'l1': l1}
DELTA_ERROR_FUNC = {'l2': l2_for_delta, 'l1': l1_for_delta}


class BoostingResult(object):
    """Result from boosting a temporal response function

    Attributes
    ----------
    h : NDVar | tuple of NDVar
        The temporal response function. Whether ``h`` is an NDVar or a tuple of
        NDVars depends on whether the ``x`` parameter to :func:`boosting` was
        an NDVar or a sequence of NDVars.
    r : float | NDVar
        Correlation between the measured response and the response predicted
        with ``h``. Type depends on the ``y`` parameter to :func:`boosting`.
    spearmanr : float | NDVar
        As ``r``, the Spearman rank correlation.
    t_run : float
        Time it took to run the boosting algorithm (in seconds).
    error : str
        The error evaluation method used.
    fit_error : float | NDVar
        The fit error, i.e. the result of the ``error`` error function on the
        final fit.
    delta : scalar
        Kernel modification step used.
    mindelta : None | scalar
        Mindelta parameter used.
    scale_data : bool
        Scale_data parameter used.
    y_mean : NDVar | scalar
        Mean that was subtracted from ``y``.
    y_scale : NDVar | scalar
        Scale by which ``y`` was divided.
    x_mean : NDVar | scalar | tuple
        Mean that was subtracted from ``x``.
    x_scale : NDVar | scalar | tuple
        Scale by which ``x`` was divided.
    """
    def __init__(self, h, r, isnan, t_run, version, delta, mindelta, error,
                 spearmanr, fit_error, scale_data, y_mean, y_scale, x_mean,
                 x_scale, y=None, x=None, tstart=None, tstop=None):
        self.h = h
        self.r = r
        self.isnan = isnan
        self.t_run = t_run
        self.version = version
        self.delta = delta
        self.mindelta = mindelta
        self.error = error
        self.spearmanr = spearmanr
        self.fit_error = fit_error
        self.scale_data = scale_data
        self.y_mean = y_mean
        self.y_scale = y_scale
        self.x_mean = x_mean
        self.x_scale = x_scale
        self.y = y
        self.x = x
        self.tstart = tstart
        self.tstop = tstop

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in
                getargspec(self.__init__).args[1:]}

    def __setstate__(self, state):
        self.__init__(**state)

    def __repr__(self):
        if self.x is None or isinstance(self.x, basestring):
            x = self.x
        else:
            x = ' + '.join(map(str, self.x))
        items = ['boosting %s ~ %s' % (self.y, x),
                 '%g - %g' % (self.tstart, self.tstop)]
        argspec = getargspec(boosting)
        names = argspec.args[-len(argspec.defaults):]
        for name, default in izip(names, argspec.defaults):
            value = getattr(self, name)
            if value != default:
                items.append('%s=%r' % (name, value))
        return '<%s>' % ', '.join(items)


def boosting(y, x, tstart, tstop, scale_data=True, delta=0.005, mindelta=None,
             error='l2'):
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
    scale_data : bool | 'inplace'
        Scale ``y`` and ``x`` before boosting: subtract the mean and divide by
        the standard deviation (when ``error='l2'``) or the mean absolute
        value (when ``error='l1'``). With ``scale_data=True`` (default) the
        original ``y`` and ``x`` are left untouched; use ``'inplace'`` to save
        memory by scaling the original ``y`` and ``x``.
    delta : scalar
        Step for changes in the kernel.
    mindelta : scalar
        If set, ``delta`` is divided in half after each unsuccessful iteration
        until ``delta < mindelta``. The default is ``mindelta = delta``.
    error : 'l2' | 'l1'
        Error function to use (default is ``l2``).

    Returns
    -------
    result : BoostingResult
        Object containing results from the boosting estimation (see
        :class:`BoostingResult`).
    """
    # check arguments
    mindelta_ = delta if mindelta is None else mindelta

    # check y and x
    if isinstance(x, NDVar):
        x_name = x.name
        x = (x,)
        multiple_x = False
    else:
        x = tuple(x)
        assert all(isinstance(x_, NDVar) for x_ in x)
        x_name = tuple(x_.name for x_ in x)
        multiple_x = True
    y_name = y.name
    time_dim = y.get_dim('time')
    if any(x_.get_dim('time') != time_dim for x_ in x):
        raise ValueError("Not all NDVars have the same time dimension")

    # scale y and x appropriately for error function
    if scale_data:
        data = (y,) + x
        data_mean = tuple(d.mean('time') for d in data)
        if isinstance(scale_data, int):
            data = tuple(d - d_mean for d, d_mean in izip(data, data_mean))
        elif isinstance(scale_data, str):
            if scale_data == 'inplace':
                for d, d_mean in izip(data, data_mean):
                    d -= d_mean
            else:
                raise ValueError("scale_data=%r" % scale_data)
        else:
            raise TypeError("scale_data=%r" % (scale_data,))

        if error == 'l1':
            data_scale = tuple(d.abs().mean('time') for d in data)
        elif error == 'l2':
            data_scale = tuple(d.std('time') for d in data)
        else:
            raise ValueError("error=%r; needs to be 'l1' or 'l2' if "
                             "scale_data=True." % (error,))
        for d, d_scale in izip(data, data_scale):
            d /= d_scale
        y = data[0]
        x = data[1:]
    else:
        data_mean = data_scale = (None,) * (len(x) + 1)

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

    # TRF extra dimension
    trf_dim = ydim

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

    # progress bar
    n_responses = len(y_data)
    pbar = tqdm(desc="Boosting %i signals" % n_responses if n_responses > 1 else
                "Boosting", total=n_responses * 10)
    # boosting
    res = []
    for y_ in y_data:
        hs = []
        for i in xrange(10):
            h, test_sse_history, msg = boost_1seg(x_data, y_, trf_length, delta,
                                                  10, i, mindelta_, error)
            if np.any(h):
                hs.append(h)
            if pbar is not None:
                pbar.update()

        if hs:
            h = np.mean(hs, 0)
            r, rr, err = evaluate_kernel(y_, x_data, h, error)
        else:
            h = np.zeros(h.shape)
            r = rr = err = 0.
        res.append((h, r, rr, err))
    hs, rs, rrs, errs = zip(*res)
    h_x = np.array(hs)
    pbar.close()
    dt = time.time() - pbar.start_t

    # correlation
    if ydim is None:
        r = rs[0]
        rr = rrs[0]
        err = errs[0]
        isnan = np.isnan(r)
    else:
        rs = np.asarray(rs)
        isnan = np.isnan(rs)
        rs[isnan] = 0
        r = NDVar(rs, (ydim,), cs.stat_info('r'), 'correlation')
        rr = NDVar(np.asarray(rrs), (ydim,), cs.stat_info('r'), 'rank correlation')
        err = NDVar(np.asarray(errs), (ydim,), y.info.copy(), 'fit error')

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
        idx = slice(1, None)
    else:
        hs = hs[0]
        idx = 1

    return BoostingResult(hs, r, isnan, dt, VERSION, delta, mindelta, error, rr,
                          err, scale_data, data_mean[0], data_scale[0],
                          data_mean[idx], data_scale[idx], y_name, x_name,
                          tstart, tstop)


def boost_1seg(x, y, trf_length, delta, nsegs, segno, mindelta, error):
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
    nsegs : int
        Number of segments
    segno : int [0, nsegs-1]
        which segment to use for testing
    mindelta : scalar
        Smallest delta to use. If no improvement can be found in an iteration,
        the first step is to divide delta in half, but stop if delta becomes
        smaller than ``mindelta``.
    error : 'l2' | 'Sabs'
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
        train_index = (slice(0, -test_seg_len),)
    elif segno < 0 or segno >= nsegs:
        raise ValueError("segno=%r" % segno)
    else:
        train_index = (slice(0, test_seg_len * segno),
                       slice(test_seg_len * (segno + 1), None))

    y_train = tuple(y[..., i] for i in train_index)
    y_test = (y[..., test_index],)
    x_train = tuple(x[:, i] for i in train_index)
    x_test = (x[:, test_index],)

    return boost_segs(y_train, y_test, x_train, x_test, trf_length, delta,
                      mindelta, error)


def boost_segs(y_train, y_test, x_train, x_test, trf_length, delta, mindelta,
               error):
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
    mindelta : scalar
        Smallest delta to use. If no improvement can be found in an iteration,
        the first step is to divide delta in half, but stop if delta becomes
        smaller than ``mindelta``.
    error : str
        Error function to use.
    """
    delta_error = DELTA_ERROR_FUNC[error]
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
    y_test_error = tuple(y.copy() for y in y_test)

    ys_error = y_train_error + y_test_error
    xs = x_train + x_test

    new_error = np.empty(h.shape)
    new_sign = np.empty(h.shape, np.int8)

    # history lists
    history = []
    test_error_history = []
    # pre-assign iterators
    iter_h = tuple(product(xrange(h.shape[0]), xrange(h.shape[1])))
    iter_train_error = zip(y_train_error, x_train)
    iter_error = zip(ys_error, xs)
    for i_boost in xrange(999999):
        history.append(h.copy())

        # evaluate current h
        e_test = sum(error(y) for y in y_test_error)
        e_train = sum(error(y) for y in y_train_error)

        test_error_history.append(e_test)

        # stop the iteration if all the following requirements are met
        # 1. more than 10 iterations are done
        # 2. The testing error in the latest iteration is higher than that in
        #    the previous two iterations
        if (i_boost > 10 and e_test > test_error_history[-2] and
                e_test > test_error_history[-3]):
            reason = "error(test) not improving in 2 steps"
            break

        # generate possible movements -> training error
        for i_stim, i_time in iter_h:
            # +/- delta
            e_add = e_sub = 0.
            for y_err, x in iter_train_error:
                e_add_, e_sub_ = delta_error(y_err, x[i_stim], delta, i_time)
                e_add += e_add_
                e_sub += e_sub_

            if e_add > e_sub:
                new_error[i_stim, i_time] = e_sub
                new_sign[i_stim, i_time] = -1
            else:
                new_error[i_stim, i_time] = e_add
                new_sign[i_stim, i_time] = 1

        i_stim, i_time = np.unravel_index(np.argmin(new_error), h.shape)
        new_train_error = new_error[i_stim, i_time]
        delta_signed = new_sign[i_stim, i_time] * delta

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

        # abort if we're moving in circles
        if i_boost >= 2 and h[i_stim, i_time] == history[-2][i_stim, i_time]:
            reason = "Same h after 2 iterations"
            break
        elif i_boost >= 3 and h[i_stim, i_time] == history[-3][i_stim, i_time]:
            reason = "Same h after 3 iterations"
            break

        # update error
        for err, x in iter_error:
            update_error(err, x[i_stim], delta_signed, i_time)

    else:
        reason = "maxiter exceeded"

    best_iter = np.argmin(test_error_history)

    # Keep predictive power as the correlation for the best iteration
    return (history[best_iter], test_error_history,
            reason + ' (%i iterations)' % (i_boost + 1))


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


def evaluate_kernel(y, x, h, error):
    """Fit quality statistics

    Returns
    -------
    r : float | array
        Pearson correlation.
    rank_r : float | array
        Spearman rank correlation.
    error : float | array
        Error corresponding to error_func.
    """
    y_pred = apply_kernel(x, h)

    # discard onset (length of kernel)
    i0 = h.shape[-1] - 1
    y = y[..., i0:]
    y_pred = y_pred[..., i0:]

    error_func = ERROR_FUNC[error]
    return (np.corrcoef(y, y_pred)[0, 1],
            spearmanr(y, y_pred)[0],
            error_func(y - y_pred))
