# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
'''
Statistics functions that work on numpy arrays.

'''

import numpy as np
import scipy.stats

from .._data_obj import asfactor, asmodel, Model
from . import opt


def confidence_interval(y, x=None, match=None, confidence=.95):
    """Confidence interval based on the inverse t-test

    Parameters
    ----------
    y : array [n, ...]
        Data, first dimension reflecting cases.
    x : Categorial
        Categorial predictor for using pooled variance.
    match : Factor
        Specifies which cases are related.
    confidence : scalar
        Confidence in the interval (i.e., .95 for 95% CI).

    Returns
    -------
    ci : array [...]
        Confidence interval (i.e., the mean of y lies within m +- ci with the
        specified confidence).

    Notes
    -----
    See
    `<http://en.wikipedia.org/wiki/Confidence_interval#Statistical_hypothesis_testing>`_
    """
    if x is None:
        n = len(y)
        df = n - 1
    else:
        x = asfactor(x)
        n = x._cellsize()
        if n < 0:
            raise NotImplementedError()

        if match is None:
            x = Model(x)
        else:
            x = x + match
        df = x.df_error
    out = residual_mean_square(y, x)
    np.sqrt(out, out)
    t = scipy.stats.t.isf((1 - confidence) / 2, df)
    out *= t / np.sqrt(n)
    return out


def residual_mean_square(y, x=None):
    """Mean square of the residuals

    Parameters
    ----------
    y : array [n, ...]
        Dependent measure.
    x : Model
        Model to account for a part of the variance.

    Returns
    -------
    mean_square : array [...]
        Estimate of the mean square within x.
    """
    n = len(y)
    out = np.empty(y.shape[1:])
    out_ = out.ravel()
    y_ = y.reshape((n, -1))
    if x is None:
        opt.ss(y_, out_)
        out_ /= n - 1
    else:
        x = asmodel(x)
        res = residuals(y_, x)
        opt.sum_square(res, out_)
        out_ /= x.df_error
    return out


def residuals(y, x):
    """Calculate residuals of y regressed on x (over the first axis)

    Parameters
    ----------
    y : array
        Data
    x : Model
        Predictors
    """
    n = len(y)
    x = asmodel(x)
    res = np.empty(y.shape)
    y_ = y.reshape((n, -1))
    res_ = res.reshape((n, -1))
    opt.lm_res(y_, x.full, x.xsinv, res_)
    return res


def rms(a, axis=None):
    """Root mean square

    Parameters
    ----------
    a : array_like
        Data.
    axis : None | int | tuple of ints
        Axis or axes over which to calculate the RMS.
        The default (`axis` = `None`) is the RMS over all the dimensions of
        the input array.
    """
    square = np.square(a)
    out = square.mean(axis)
    if np.isscalar(out):
        return np.sqrt(out)
    else:
        return np.sqrt(out, out)


def rmssd(Y):
    """Root mean square of successive differences

    Used for heart rate variance analysis.
    """
    assert np.ndim(Y) == 1
    N = len(Y)
    assert N > 1
    dY = np.diff(Y)
    X = N / (N - 1) * np.sum(dY ** 2)
    return np.sqrt(X)


def ftest_f(p, df_num, df_den):
    "F values for given probabilities."
    p = np.asanyarray(p)
    f = scipy.stats.f.isf(p, df_num, df_den)
    return f


def ftest_p(f, df_num, df_den):
    "P values for given f values."
    f = np.asanyarray(f)
    p = scipy.stats.f.sf(f, df_num, df_den)
    return p
