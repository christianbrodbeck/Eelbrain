# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Statistics functions that work on numpy arrays."""
import re

import numpy as np
import scipy.stats
from scipy.linalg import inv

from .._data_obj import asfactor, asmodel, Model
from . import opt


def betas(y, x):
    """Regression coefficients

    Parameters
    ----------
    y : array  [n_cases, ...]
        Dependent measure.
    x : Model
        Predictors

    Returns
    -------
    betas : array  [n_predictors, ...]
        Beta coefficients for the regression. The first beta is always the
        intercept (see Model).
    """
    n = len(y)
    x = asmodel(x)
    shape = (x.df,) + y.shape[1:]
    y_ = y.reshape((n, -1))
    out = np.empty(shape)
    out_ = out.reshape((x.df, -1))
    opt.lm_betas(y_, x.full, x.xsinv, out_)
    return out


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


def corr(y, x, out=None, perm=None):
    """Correlation parameter map

    Parameters
    ----------
    y : array_like, shape = (n_cases, ...)
        Dependent variable with case in the first axis and case mean zero.
    x : array_like, shape = (n_cases, )
        Covariate.

    Returns
    -------
    r : array, shape = (...)
        The correlation. Occurrence of NaN due to 0 variance in either y or x
        are replaced with 0.
    """
    if out is None and y.ndim > 1:
        out = np.empty(y.shape[1:])
    if perm is not None:
        x = x[perm]

    z_x = scipy.stats.zscore(x, ddof=1)
    z_x.shape = (len(x),) + (1,) * (y.ndim - 1)
    z_y = scipy.stats.zscore(y, ddof=1)
    z_y *= z_x
    out = z_y.sum(0, out=out)
    out /= len(x) - 1

    # replace NaN values
    isnan = np.isnan(out)
    if np.any(isnan):
        if np.isscalar(out):
            out = 0
        else:
            out.place(isnan, 0)
    return out


def lm_t(y, x):
    """Calculate t-values for regression coefficients

    Parameters
    ----------
    y : array  [n_cases, ...]
        Dependent measure.
    x : Model
        Predictors
    """
    # t = beta / se(beta)
    # se(beta) = sqrt(ms_e / a)
    # t = sqrt(a) * (beta / sqrt(ms_e))
    x = asmodel(x)

    # calculate a
    a = np.empty(x.df)
    x_ = np.empty((len(x), x.df - 1))
    for i in xrange(x.df):
        y_ = x.full[:, i:i+1]
        x_[:, :i] = x.full[:, :i]
        x_[:, i:] = x.full[:, i + 1:]
        opt.lm_res_ss(y_, x_, xsinv(x_), a[i:i+1])
    np.sqrt(a, a)

    y_ = y.reshape((len(y), -1))
    out = np.empty((x.df,) + y.shape[1:])
    out_ = out.reshape((x.df, -1))
    opt.lm_t(y_, x.full, x.xsinv, a, out_)
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


def rmssd(y):
    """Root mean square of successive differences

    Used for heart rate variance analysis.
    """
    assert np.ndim(y) == 1
    N = len(y)
    assert N > 1
    dy = np.diff(y)
    X = N / (N - 1) * np.sum(dy ** 2)
    return np.sqrt(X)


def standard_error_of_the_mean(y, x=None, match=None):
    """Standard error of the mean

    Parameters
    ----------
    y : array [n, ...]
        Data, first dimension reflecting cases.
    x : Categorial
        Categorial predictor for using pooled variance.
    """
    if x is None:
        n = len(y)
        if match is not None:
            x = match
    else:
        x = asfactor(x)
        n = x._cellsize()
        if n < 0:
            raise NotImplementedError()

        if match is not None:
            x = x + match
    out = residual_mean_square(y, x)
    out /= n
    np.sqrt(out, out)
    return out


def t_1samp(y, out=None):
    "T-value for 1-sample t-test"
    n_cases = len(y)
    if out is None:
        out = np.empty(y.shape[1:])

    if out.ndim == 1:
        out_ = out
    else:
        out_ = out.ravel()

    opt.t_1samp(y.reshape(n_cases, -1), out_)
    return out


def t_ind(x, n1, n2, equal_var=True, out=None, perm=None):
    "Based on scipy.stats.ttest_ind"
    if out is None:
        out = np.empty(x.shape[1:])

    if perm is None:
        a = x[:n1]
        b = x[n1:]
    else:
        cat = np.zeros(n1 + n2)
        cat[n1:] = 1
        cat_perm = cat[perm]
        a = x[cat_perm == 0]
        b = x[cat_perm == 1]
    v1 = np.var(a, 0, ddof=1)
    v2 = np.var(b, 0, ddof=1)

    if equal_var:
        df = n1 + n2 - 2
        svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / float(df)
        denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    else:
        vn1 = v1 / n1
        vn2 = v2 / n2
        denom = np.sqrt(vn1 + vn2)

    d = np.mean(a, 0) - np.mean(b, 0)
    t = np.divide(d, denom, out)
    return t


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


def rtest_p(r, df):
    # http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Inference
    r = np.asanyarray(r)
    t = r * np.sqrt(df / (1 - r ** 2))
    p = ttest_p(t, df)
    return p


def rtest_r(p, df):
    # http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Inference
    p = np.asanyarray(p)
    t = ttest_t(p, df)
    r = t / np.sqrt(df + t ** 2)
    return r


def ttest_p(t, df, tail=0):
    """Two tailed probability

    Parameters
    ----------
    t : array_like
        T values.
    df : int
        Degrees of freedom.
    tail : 0 | 1 | -1
        Which tail of the t-distribution to consider:
        0: both (two-tailed);
        1: upper tail (one-tailed);
        -1: lower tail (one-tailed).
    """
    t = np.asanyarray(t)
    if tail == 0:
        t = np.abs(t)
    elif tail == -1:
        t = -t
    elif tail != 1:
        raise ValueError("tail=%r" % tail)
    p = scipy.stats.t.sf(t, df)
    if tail == 0:
        p *= 2
    return p


def ttest_t(p, df, tail=0):
    """Positive t value for a given probability

    Parameters
    ----------
    p : array_like
        Probability.
    df : int
        Degrees of freedom.
    tail : 0 | 1 | -1
        One- or two-tailed t-distribution (the return value is always positive):
        0: two-tailed;
        1 or -1: one-tailed).
    """
    p = np.asanyarray(p)
    if tail == 0:
        p = p / 2
    t = scipy.stats.t.isf(p, df)
    return t


def variability(y, x, match, spec, pool):
    """Calculate data variability

    Parameters
    ----------
    y : array
        Dependent measure.
    x : None | Categorial
        Cells for pooling variance.
    match : Factor
        Calculate variability for related measures (Loftus & Masson 1994).
    spec : str
        Specification of the kind of variability estimate. Contains an optional
        number, an optional percent-sign, and a kind ('ci' or 'sem'). Examples:
        'ci': 95% confidence interval;
        '99%ci': 99% confidence interval (default);
        '2sem': 2 standard error of the mean.
    pool : bool
        Pool the variability to create a single estimate (as opposed to one for
        each cell in x).

    Returns
    -------
    var : scalar | array
        Variability estimate. A single estimate if errors are pooled, otherwise
        an estimate for every cell in x.
    """
    try:
        m = re.match("^([.\d]*)(\%?)(ci|sem)$", spec.lower())
        scale, perc, kind = m.groups()
        if scale:
            scale = float(scale)
            if perc:
                scale /= 100
        elif kind == 'ci':
            scale = .95
        else:
            scale = 1
    except:
        raise ValueError("Invalid variability specification: %r" % spec)

    if x is None and match is not None:
        if match.df == len(match) - 1:
            raise ValueError("Can't calculate within-subject error because the "
                             "match predictor explains all variability")

    if kind == 'ci':
        if pool or x is None:
            out = confidence_interval(y, x, match, scale)
        else:
            cis = [confidence_interval(y[x == cell], confidence=scale) for cell
                                       in x.cells]
            out = np.array(cis)
    elif kind == 'sem':
        if pool or x is None:
            out = standard_error_of_the_mean(y, x, match)
        else:
            sems = [standard_error_of_the_mean(y[x == cell]) for cell in x.cells]
            out = np.array(sems)

        if scale != 1:
            out *= scale
    else:
        raise RuntimeError

    # return scalars for 1d-arrays
    if out.ndim == 0:
        return out.item()
    else:
        return out


def xsinv(x):
    xt = x.T
    return inv(xt.dot(x)).dot(xt)
