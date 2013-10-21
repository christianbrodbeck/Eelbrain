# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
'''
Statistics functions that work on numpy arrays.

'''

import numpy as np
import scipy.stats

__all__ = ['ci', 'cihw', 'rms', 'rmssd', 'ftest_f', 'ftest_p']


def ci(x, p=.95):
    """
    :returns: list with the endpoints of the confidence interval based on the
        inverse t-test
        (`<http://en.wikipedia.org/wiki/Confidence_interval#Statistical_hypothesis_testing>`_).

    :arg array x: data
    :arg float p: p value for confidence interval

    """
    M = np.mean(x)
    c = cihw(x, p)
    return [M - c, M + c]


def cihw(x, p=.95):
    """
    :returns: half-width of the confidence interval based on the inverse t-test
        (`<http://en.wikipedia.org/wiki/Confidence_interval#Statistical_hypothesis_testing>`_).

    :arg array x: data
    :arg float p: p value for confidence interval

    """
    N = len(x)
    t = scipy.stats.t.isf((1 - p) / 2, N - 1)
    c = (np.std(x, ddof=1) * t) / np.sqrt(N)
    return c


def rms(Y, axis=-1, rm_mean=False):
    """Root mean square

    Parameters
    ----------
    Y : array_like
        Data.
    axis : int
        Axis over which to calculate the RMS.
    rm_mean : bool
        Remove the mean over axis before calculating the RMS (= average
        reference).

    Notes
    -----
    Used as 'Global Field Power' (Murray et al., 2008).

    Murray, M. M., Brunet, D., and Michel, C. M. (2008). Topographic ERP
            analyses: a step-by-step tutorial review. Brain Topogr, 20(4),
            249-64.
    """
    if rm_mean:  # avg reference
        shape = list(Y.shape)
        shape[axis] = 1
        Y = Y - Y.mean(axis).reshape(shape)

    # root mean square
    rms = np.sqrt(np.mean(Y ** 2, axis))
    return rms


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
