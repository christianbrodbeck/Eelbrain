"""
Boosting as described by David et al. (2007).

Versions
--------
7: Accept segmented data, respect segmentation (don't concatenate data)


Profiling
---------
ds = datasets._get_continuous()
y = ds['y']
x1 = ds['x1']
x2 = ds['x2']

%prun -s cumulative res = boosting(y, x1, 0, 1)

"""
import inspect
from itertools import product
from multiprocessing import Process, Queue
from multiprocessing.sharedctypes import RawArray
import os
import signal
import time
from threading import Event, Thread

import numpy as np
import scipy.signal
from scipy.stats import spearmanr
from tqdm import tqdm

from .._config import CONFIG
from .._data_obj import NDVar
from .._utils import LazyProperty, user_activity
from ._boosting_opt import l1, l2, generate_options, update_error
from .shared import RevCorrData


# BoostingResult version
VERSION = 7

# process messages
JOB_TERMINATE = -1

# error functions
ERROR_FUNC = {'l2': l2, 'l1': l1}
DELTA_ERROR_FUNC = {'l2': 2, 'l1': 1}

DELTA_REDUCTION_STEP = (None, None, None)


class BoostingResult(object):
    """Result from boosting a temporal response function

    Attributes
    ----------
    h : NDVar | tuple of NDVar
        The temporal response function. Whether ``h`` is an NDVar or a tuple of
        NDVars depends on whether the ``x`` parameter to :func:`boosting` was
        an NDVar or a sequence of NDVars.
    h_scaled : NDVar | tuple of NDVar
        ``h`` scaled such that it applies to the original input ``y`` and ``x``.
        If boosting was done with ``scale_data=False``, ``h_scaled`` is the same
        as ``h``.
    h_time : UTS
        Time dimension of the kernel.
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
                 x_scale, y=None, x=None, tstart=None, tstop=None,
                 n_segments=None, model=None, **experimental_parameters):
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
        self.n_segments = n_segments
        self.model = model
        self._experimental_parameters = experimental_parameters

    def __getstate__(self):
        state = {attr: getattr(self, attr) for attr, param in
                 inspect.signature(self.__class__).parameters.items()
                 if param.kind is not inspect.Parameter.VAR_KEYWORD}
        state.update(self._experimental_parameters)
        return state

    def __setstate__(self, state):
        self.__init__(**state)

    def __repr__(self):
        if self.x is None or isinstance(self.x, str):
            x = self.x
        else:
            x = ' + '.join(map(str, self.x))
        items = ['boosting %s ~ %s' % (self.y, x),
                 '%g - %g' % (self.tstart, self.tstop)]
        for name, param in inspect.signature(boosting).parameters.items():
            if param.default is inspect.Signature.empty or name == 'ds':
                continue
            value = getattr(self, name)
            if value != param.default:
                items.append('%s=%r' % (name, value))
        return '<%s>' % ', '.join(items)

    @LazyProperty
    def h_scaled(self):
        if self.y_scale is None:
            return self.h
        elif isinstance(self.h, NDVar):
            return self.h * (self.y_scale / self.x_scale)
        else:
            return tuple(h * (self.y_scale / sx) for h, sx in
                         zip(self.h, self.x_scale))

    @LazyProperty
    def h_time(self):
        if isinstance(self.h, NDVar):
            return self.h.time
        else:
            return self.h[0].time

    def _set_parc(self, parc):
        """Change the parcellation of source-space result
         
        Notes
        -----
        No warning for missing sources!
        """
        from .._ndvar import set_parc

        if not self.r.has_dim('source'):
            raise RuntimeError('BoostingResult does not have source-space data')

        def sub_func(obj):
            if obj is None:
                return None
            elif isinstance(obj, tuple):
                return tuple(sub_func(obj_) for obj_ in obj)
            obj_new = set_parc(obj, parc)
            index = np.invert(obj_new.source.parc.startswith('unknown-'))
            return obj_new.sub(source=index)

        for attr in ('h', 'r', 'spearmanr', 'fit_error', 'y_mean', 'y_scale'):
            setattr(self, attr, sub_func(getattr(self, attr)))


@user_activity
def boosting(y, x, tstart, tstop, scale_data=True, delta=0.005, mindelta=None,
             error='l2', n_segments=None, model=None, ds=None):
    """Estimate a filter with boosting

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
        If the error for the training data can't be reduced, divide ``delta``
        in half until ``delta < mindelta``. The default is ``mindelta = delta``,
        i.e. ``delta`` is constant.
    error : 'l2' | 'l1'
        Error function to use (default is ``l2``).
    n_segments : int
        Divide the data into ``n_segments`` for early stopping based on
        cross-validation. In each iteration, ``n - 1`` segments are used for
        training, and the remaining segment is used for validation.
        If data is continuous, data are divided into ``n_segments`` contiguous
        segments of equal length (default 10).
        If data has cases, cases are divided with ``[::n_segments]`` slices
        (default ``min(n_cases, 10)``, where if ``model`` is specified,
        ``n_cases`` is the lowest number of cases in any cell of the model).
    model : Categorial
        If data has cases, divide cases into different categories (division
        for crossvalidation is done separately for each cell).
    ds : Dataset
        If provided, other parameters can be specified as string for items in
        ``ds``.

    Returns
    -------
    result : BoostingResult
        Object containing results from the boosting estimation (see
        :class:`BoostingResult`).

    Notes
    -----
    The boosting algorithm is described in [1]_.

    References
    ----------
    .. [1] David, S. V., Mesgarani, N., & Shamma, S. A. (2007). Estimating
        sparse spectro-temporal receptive fields with natural stimuli. Network:
        Computation in Neural Systems, 18(3), 191-212.
        `10.1080/09548980701609235 <https://doi.org/10.1080/09548980701609235>`_.
    """
    # check arguments
    mindelta_ = delta if mindelta is None else mindelta

    data = RevCorrData(y, x, error, scale_data, ds)
    data.initialize_cross_validation(n_segments, model, ds)
    n_y = len(data.y)
    n_x = len(data.x)

    # TRF extent in indices
    tstep = data.time.tstep
    i_start = int(round(tstart / tstep))
    i_stop = int(round(tstop / tstep))
    trf_length = i_stop - i_start

    if data.segments is None:
        i_skip = trf_length - 1
    else:
        i_skip = 0

    # progress bar
    n_cv = len(data.cv_segments)
    pbar = tqdm(desc=f"Boosting{f' {n_y} signals' if n_y > 1 else ''}", total=n_y * n_cv, disable=CONFIG['tqdm'])
    t_start = time.time()
    # result containers
    res = np.empty((3, n_y))  # r, rank-r, error
    h_x = np.empty((n_y, n_x, trf_length))
    # boosting
    if CONFIG['n_workers']:
        # Make sure cross-validations are added in the same order, otherwise
        # slight numerical differences can occur
        job_queue, result_queue = setup_workers(data, i_start, trf_length, delta, mindelta_, error)
        stop_jobs = Event()
        thread = Thread(target=put_jobs, args=(job_queue, n_y, n_cv, stop_jobs))
        thread.start()

        # collect results
        try:
            h_segs = {}
            for _ in range(n_y * n_cv):
                y_i, seg_i, h = result_queue.get()
                pbar.update()
                if y_i in h_segs:
                    h_seg = h_segs[y_i]
                    h_seg[seg_i] = h
                    if len(h_seg) == n_cv:
                        del h_segs[y_i]
                        hs = [h for h in (h_seg[i] for i in range(n_cv)) if
                              h is not None]
                        if hs:
                            h = np.mean(hs, 0, out=h_x[y_i])
                            res[:, y_i] = evaluate_kernel(data.y[y_i], data.x, h, i_start, error, i_skip, data.segments)
                        else:
                            h_x[y_i] = 0
                            res[:, y_i] = 0.
                else:
                    h_segs[y_i] = {seg_i: h}
        except KeyboardInterrupt:
            stop_jobs.set()
            raise
    else:
        for y_i, y_ in enumerate(data.y):
            hs = []
            for segments, train, test in data.cv_segments:
                h = boost(y_, data.x, segments, train, test, i_start, trf_length, delta,
                          mindelta_, error)
                if h is not None:
                    hs.append(h)
                pbar.update()

            if hs:
                h = np.mean(hs, 0, out=h_x[y_i])
                res[:, y_i] = evaluate_kernel(y_, data.x, h, i_start, error, i_skip, data.segments)
            else:
                h_x[y_i].fill(0)
                res[:, y_i].fill(0.)

    pbar.close()
    dt = time.time() - t_start

    # fit-evaluation statistics
    rs, rrs, errs = res
    isnan = np.isnan(rs)
    rs[isnan] = 0
    r = data.package_statistic(rs, 'r', 'correlation')
    rr = data.package_statistic(rrs, 'r', 'rank correlation')
    err = data.package_value(errs, 'fit error')

    y_mean, y_scale, x_mean, x_scale = data.data_scale_ndvars()

    return BoostingResult(
        data.package_kernel(h_x, tstart), r, isnan, dt, VERSION, delta,
        mindelta, error, rr, err, scale_data, y_mean, y_scale, x_mean, x_scale,
        data.y_name, data.x_name, tstart, tstop)


def boost(y, x, all_index, train_index, test_index, i_start, trf_length, delta, mindelta,
          error, return_history=False):
    """Estimate one filter with boosting

    Parameters
    ----------
    y : array (n_times,)
        Dependent signal, time series to predict.
    x : array (n_stims, n_times)
        Stimulus.
    train_index : array of (start, stop)
        Time sample index of training segments.
    test_index : array of (start, stop)
        Time sample index of test segments.
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
    return_history : bool
        Return error history as second return value.

    Returns
    -------
    history[best_iter] : None | array
        Winning kernel, or None if 0 is the best kernel.
    test_sse_history : list (only if ``return_history==True``)
        SSE for test data at each iteration.
    """
    delta_error_func = DELTA_ERROR_FUNC[error]
    error = ERROR_FUNC[error]
    n_stims, n_times = x.shape
    assert y.shape == (n_times,)

    h = np.zeros((n_stims, trf_length))

    # buffers
    y_error = y.copy()
    new_error = np.empty(h.shape)
    new_sign = np.empty(h.shape, np.int8)

    # history
    best_test_error = np.inf
    history = []
    test_error_history = []
    # pre-assign iterators
    for i_boost in range(999999):
        # evaluate current h
        e_test = error(y_error, test_index)
        e_train = error(y_error, train_index)

        if e_test < best_test_error:
            best_test_error = e_test
            best_iteration = i_boost

        test_error_history.append(e_test)

        # stop the iteration if all the following requirements are met
        # 1. more than 10 iterations are done
        # 2. The testing error in the latest iteration is higher than that in
        #    the previous two iterations
        if (i_boost > 10 and e_test > test_error_history[-2] and
                e_test > test_error_history[-3]):
            # print("error(test) not improving in 2 steps")
            break

        # generate possible movements -> training error
        generate_options(y_error, x, train_index, i_start, delta_error_func, delta, new_error, new_sign)

        i_stim, i_time = np.unravel_index(np.argmin(new_error), h.shape)
        new_train_error = new_error[i_stim, i_time]
        delta_signed = new_sign[i_stim, i_time] * delta

        # If no improvements can be found reduce delta
        if new_train_error > e_train:
            delta *= 0.5
            if delta >= mindelta:
                # print("new delta: %s" % delta)
                history.append(DELTA_REDUCTION_STEP)
                continue
            else:
                # print("No improvement possible for training data")
                break

        # abort if we're moving in circles
        if i_boost >= 2 and (i_stim, i_time, -delta_signed) == history[-1]:
            # print("Same h after 2 iterations")
            break
        elif i_boost >= 4 and history[-3] is DELTA_REDUCTION_STEP:
            step = (i_stim, i_time, -delta_signed / 2.)
            if history[-1] == step and history[-2] == step:
                # print("Same h after 3 iterations")
                break

        # update h with best movement
        h[i_stim, i_time] += delta_signed
        history.append((i_stim, i_time, delta_signed))
        update_error(y_error, x[i_stim], all_index, delta_signed, i_time + i_start)
    else:
        raise RuntimeError("Maximum number of iterations exceeded")
    # print('  (%i iterations)' % (i_boost + 1))

    # reverse changes after best iteration
    if best_iteration:
        for i_stim, i_time, delta_signed in history[-1: best_iteration - 1: -1]:
            if delta_signed is not None:
                h[i_stim, i_time] -= delta_signed
    else:
        h = None

    if return_history:
        return h, test_error_history
    else:
        return h


def setup_workers(data, i_start, trf_length, delta, mindelta, error):
    n_y, n_times = data.y.shape
    n_x, _ = data.x.shape

    y_buffer = RawArray('d', n_y * n_times)
    y_buffer[:] = data.y.ravel()
    x_buffer = RawArray('d', n_x * n_times)
    x_buffer[:] = data.x.ravel()

    job_queue = Queue(200)
    result_queue = Queue(200)

    args = (y_buffer, x_buffer, n_y, n_times, n_x, data.cv_segments,
            i_start, trf_length, delta, mindelta, error,
            job_queue, result_queue)
    for _ in range(CONFIG['n_workers']):
        process = Process(target=boosting_worker, args=args)
        process.start()

    return job_queue, result_queue


def boosting_worker(y_buffer, x_buffer, n_y, n_times, n_x, cv_segments,
                    i_start, trf_length, delta, mindelta, error,
                    job_queue, result_queue):
    if CONFIG['nice']:
        os.nice(CONFIG['nice'])
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    y = np.frombuffer(y_buffer, np.float64, n_y * n_times).reshape((n_y, n_times))
    x = np.frombuffer(x_buffer, np.float64, n_x * n_times).reshape((n_x, n_times))

    while True:
        y_i, seg_i = job_queue.get()
        if y_i == JOB_TERMINATE:
            return
        all_index, train_index, test_index = cv_segments[seg_i]
        h = boost(y[y_i], x, all_index, train_index, test_index,
                  i_start, trf_length, delta, mindelta, error)
        result_queue.put((y_i, seg_i, h))


def put_jobs(queue, n_y, n_segs, stop):
    "Feed boosting jobs into a Queue"
    for job in product(range(n_y), range(n_segs)):
        queue.put(job)
        if stop.isSet():
            while not queue.empty():
                queue.get()
            break
    for _ in range(CONFIG['n_workers']):
        queue.put((JOB_TERMINATE, None))


def convolve(h, x, i_start, segments=None, out=None):
    "h * x with time axis matching x"
    n_times = x.shape[1]
    if out is None:
        out = np.zeros(n_times)
    else:
        out.fill(0)

    if segments is None:
        segments = ((0, n_times),)

    for start, stop in segments:
        out_index = slice(start + max(0, i_start), stop + min(0, i_start))
        y_index = slice(max(0, -i_start), stop - start + min(0, -i_start))
        for ind in range(len(h)):
            out[out_index] += scipy.signal.convolve(h[ind], x[ind, start:stop])[y_index]

    return out


def evaluate_kernel(y, x, h, i_start, error, i_skip, segments=None):
    """Fit quality statistics

    Parameters
    ----------
    y : array, (n_samples)
        Y.
    x : array, (n_stims, n_samples)
        X.
    h : array, (n_stims, h_n_samples)
        H.
    i_start : int
        Time shift of the first sample of ``h``.
    error : str
        Error metric.
    segments : array (n_segnents, 2)
        Data segments.
    i_skip : int
        Skip this many samples for evaluating model fit.

    Returns
    -------
    r : float | array
        Pearson correlation.
    rank_r : float | array
        Spearman rank correlation.
    error : float | array
        Error corresponding to error_func.
    """
    y_pred = convolve(h, x, i_start, segments)

    # discard onset
    if i_skip:
        assert segments is None, "Not implemented"
        y = y[i_skip:]
        y_pred = y_pred[i_skip:]

    error_func = ERROR_FUNC[error]
    index = np.array(((0, len(y)),), np.int64)
    return (np.corrcoef(y, y_pred)[0, 1],
            spearmanr(y, y_pred)[0],
            error_func(y - y_pred, index))
