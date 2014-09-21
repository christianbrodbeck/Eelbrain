# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
'''
Statistics functions that work on numpy arrays.

'''

import numpy as np
import scipy.stats



def confidence_interval(y, confidence=.95):
    """Confidence interval based on the inverse t-test

    Parameters
    ----------
    y : array [n, ...]
        Data, first dimension reflecting cases.
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
    n = len(y)
    df = n - 1
    out = y.std(0, ddof=1)
    t = scipy.stats.t.isf((1 - confidence) / 2, df)
    out *= t / np.sqrt(n)
    return out


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
