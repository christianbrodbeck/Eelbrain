'''Statistical tests for ndvars'''
from __future__ import division

from itertools import izip
from math import ceil, floor
from time import time as current_time

import numpy as np
import scipy.stats
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ... import fmtxt
from .. import colorspaces as _cs
from ..data_obj import (ascategorial, asmodel, asndvar, asvar, assub, Dataset,
                        Factor, NDVar, Var, Celltable, cellname, combine)
from .glm import LMFitter
from .permutation import resample, _resample_params
from .stats import ftest_f, ftest_p
from .test import star_factor


__all__ = ['ttest_1samp', 'ttest_ind', 'ttest_rel', 'anova', 'corr',
           'clean_time_axis']
__test__ = False


def clean_time_axis(pmap, dtmin=0.02, below=None, above=None, null=0):
    """
    Clean a parameter map by requiring a threshold value for a minimum duration

    Parameters
    ----------
    pmap : NDVar
        Parameter map with time axis.
    dtmin : scalar
        Minimum duration required
    below : scalar | None
        Threshold value for finding clusters: find clusters of values below
        this threshold.
    above : scalar | None
        As ``below``, but for finding clusters above a threshold.
    null : scalar
        Value to substitute outside of clusters.

    Returns
    -------
    cleaned_map : NDVar
        A copy of pmap with all values that do not belong to a cluster set to
        null.
    """
    if below is None and above is None:
        raise TypeError("Need to specify either above or below.")
    elif below is None:
        passes_t = pmap.x >= above
    elif above is None:
        passes_t = pmap.x <= below
    else:
        passes_t = np.logical_and(pmap.x >= above, pmap.x <= below)

    ax = pmap.get_axis('time')
    di_min = int(ceil(dtmin / pmap.time.tstep))
    struct_shape = (1,) * ax + (di_min,) + (1,) * (pmap.ndim - ax - 1)
    struct = np.ones(struct_shape, dtype=int)

    cores = binary_erosion(passes_t, struct)
    keep = binary_dilation(cores, struct)
    x = np.where(keep, pmap.x, null)

    info = pmap.info.copy()
    cleaned = NDVar(x, pmap.dims, info, pmap.name)
    return cleaned


class corr:
    """Correlation

    Attributes
    ----------
    r : NDVar
        Correlation (with threshold contours).
    """
    def __init__(self, Y, X, norm=None, sub=None, ds=None, samples=0, pmin=0.1,
                 tstart=None, tstop=None, match=None, **criteria):
        """Correlation.

        Parameters
        ----------
        Y : NDVar
            Dependent variable.
        X : continuous
            The continuous predictor variable.
        norm : None | categorial
            Categories in which to normalize (z-score) X.
        sub : None | index-array
            Perform the test with a subset of the data.
        ds : None | Dataset
            If a Dataset is specified, all data-objects can be specified as
            names of Dataset variables.
        samples : None | int
            Number of samples for permutation cluster test. For None, no
            clusters are formed. Use 0 to compute clusters without performing
            any permutations.
        pmin : scalar (0 < pmin < 1)
            Threshold p value for forming clusters in permutation cluster test.
        tstart, tstop : None | scalar
            Restrict time window for permutation cluster test.
        match : None | categorial
            When permuting data, only shuffle the cases within the categories
            of match.
        mintime : scalar
            Minimum duration for clusters (in seconds).
        minsource : int
            Minimum number of sources per cluster.
        """
        sub = assub(sub, ds)
        Y = asndvar(Y, sub=sub, ds=ds)
        if not Y.has_case:
            msg = ("Dependent variable needs case dimension")
            raise ValueError(msg)
        X = asvar(X, sub=sub, ds=ds)
        if norm is not None:
            norm = ascategorial(norm, sub, ds)
        if match is not None:
            match = ascategorial(match, sub, ds)

        name = "%s corr %s" % (Y.name, X.name)

        # Normalize by z-scoring the data for each subject
        # normalization is done before the permutation b/c we are interested in
        # the variance associated with each subject for the z-scoring.
        Y = Y.copy()
        if norm is not None:
#             Y.x = Y.x.reshape((n, -1))
            for cell in norm.cells:
                idx = (norm == cell)
                Y.x[idx] = scipy.stats.zscore(Y.x[idx], None)

        # subtract the mean from Y and X so that this can be omitted during
        # permutation
        Y -= Y.summary('case')
        X = X - X.mean()

        n = len(Y)
        df = n - 2

        rmap = _corr(Y.x, X.x)

        if samples is not None:
            # calculate r threshold for clusters
            threshold = _rtest_r(pmin, df)


            cdist = _ClusterDist(Y, samples, threshold, -threshold, 'r', name,
                                 tstart, tstop, criteria)
            cdist.add_original(rmap)
            if cdist.n_clusters and samples:
                for Y_ in resample(cdist.Y_perm, samples, unit=match):
                    rmap_ = _corr(Y_.x, X.x)
                    cdist.add_perm(rmap_)
            info = _cs.stat_info('r', threshold)
        else:
            r0, r1, r2 = _rtest_r((.05, .01, .001), df)
            info = _cs.stat_info('r', r0, r1, r2)

        # compile results
        dims = Y.dims[1:]
        r = NDVar(rmap, dims, info, name)

        # store attributes
        self.name = name
        self.df = df
        self.r = r
        self.r_p = [[r, r]]
        if samples is None:
            self.all = [[r, r]]
        else:
            self.cdist = cdist
            self.clusters = cdist.clusters
            if cdist.n_clusters:
                self.r_cl = [[r, cdist.cpmap]]
                self.all = [[r, cdist.cpmap]]
            else:
                self.r_cl = [[r]]
                self.all = [[r]]


def _corr(y, x):
    """Correlation parameter map

    Parameters
    ----------
    y : array_like, shape = (n_cases, ...)
        Dependent variable with case in the first axis and case mean zero.
    x : array_like, shape = (n_cases, )
        Covariate.
    """
    x = x.reshape((len(x),) + (1,) * (y.ndim - 1))
    r = np.sum(y * x, axis=0) / (np.sqrt(np.sum(y ** 2, axis=0)) *
                                 np.sqrt(np.sum(x ** 2, axis=0)))
    return r

def _corr_alt(y, x):
    n = len(y)
    cov = np.sum(x * y, axis=0) / (n - 1)
    r = cov / (np.std(x, axis=0) * np.std(y, axis=0))
    return r


def _rtest_p(r, df):
    # http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Inference
    r = np.asanyarray(r)
    t = r * np.sqrt(df / (1 - r ** 2))
    p = _ttest_p(t, df)
    return p


def _rtest_r(p, df):
    # http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Inference
    p = np.asanyarray(p)
    t = _ttest_t(p, df)
    r = t / np.sqrt(df + t ** 2)
    return r


class ttest_1samp:
    """Element-wise one sample t-test

    Attributes
    ----------
    all :
        c1, c0, [c0 - c1, P]
    p_val :
        [c0 - c1, P]
    """
    def __init__(self, Y, popmean=0, match=None, sub=None, ds=None, tail=0,
                 samples=None, pmin=0.1, tstart=None, tstop=None, **criteria):
        """Element-wise one sample t-test

        Parameters
        ----------
        Y : NDVar
            Dependent variable.
        popmean : scalar
            Value to compare Y against (default is 0).
        match : None | Factor
            Combine data for these categories before testing.
        sub : None | index-array
            Perform test with a subset of the data.
        ds : None | Dataset
            If a Dataset is specified, all data-objects can be specified as
            names of Dataset variables
        tail : 0 | 1 | -1
            Which tail of the t-distribution to consider:
            0: both (two-tailed);
            1: upper tail (one-tailed);
            -1: lower tail (one-tailed).
        samples : None | int
            Number of samples for permutation cluster test. For None, no
            clusters are formed. Use 0 to compute clusters without performing
            any permutations.
        pmin : scalar (0 < pmin < 1)
            Threshold p value for forming clusters in permutation cluster test.
        tstart, tstop : None | scalar
            Restrict time window for permutation cluster test.
        mintime : scalar
            Minimum duration for clusters (in seconds).
        minsource : int
            Minimum number of sources per cluster.
        """
        ct = Celltable(Y, match=match, sub=sub, ds=ds)

        n = len(ct.Y)
        df = n - 1
        tmap = _t_1samp(ct.Y.x, popmean)
        pmap = _ttest_p(tmap, df, tail)

        test_name = '1-Sample t-Test'
        y = ct.Y.summary()
        if popmean:
            diff = y - popmean
            if np.any(diff < 0):
                diff.info['cmap'] = 'xpolar'
        else:
            diff = y

        if samples is not None:
            t_threshold = _ttest_t(pmin, df, tail)
            t_upper = t_threshold if tail >= 0 else None
            t_lower = -t_threshold if tail <= 0 else None
            if popmean:
                y_perm = ct.Y - popmean
            else:
                y_perm = ct.Y
            n_samples, samples_ = _resample_params(len(y_perm), samples)
            cdist = _ClusterDist(y_perm, n_samples, t_upper, t_lower, 't',
                                 test_name, tstart, tstop, criteria)
            cdist.add_original(tmap)
            if cdist.n_clusters and samples:
                for Y_ in resample(cdist.Y_perm, samples_, sign_flip=True):
                    tmap_ = _t_1samp(Y_.x, 0)
                    cdist.add_perm(tmap_)

        dims = ct.Y.dims[1:]

        info = _cs.set_info_cs(ct.Y.info, _cs.sig_info())
        info['test'] = test_name
        p = NDVar(pmap, dims, info=info, name='p')

        t0, t1, t2 = _ttest_t((.05, .01, .001), df, tail)
        info = _cs.stat_info('t', t0, t1, t2, tail)
        info = _cs.set_info_cs(ct.Y.info, info)
        t = NDVar(tmap, dims, info=info, name='T')

        # store attributes
        self.popmean = popmean
        self.n = n
        self.df = df
        self.name = test_name
        self._samples = samples

        self.y = y
        self.diff = diff
        self.t = t
        self.p = p

        self.diffp = [[diff, t]]
        self.all = [y, [diff, t]] if popmean else [[diff, t]]
        if samples is not None:
            self._n_samples = n_samples
            self._all_permutations = samples_ < 0
            if cdist.n_clusters and samples:
                self.diff_cl = [[diff, cdist.cpmap]]
            else:
                self.diff_cl = [[diff]]
            self._cdist = cdist
            self.clusters = cdist.clusters

    def __repr__(self):
        parts = ["<%s against %g, n=%i" % (self.name, self.popmean, self.n)]
        if self._samples is not None:
            if self.clusters is None:
                parts.append(", no clusters found")
            else:
                parts.append(", %i clusters" % self._cdist.n_clusters)

                if self._samples:
                    if self._all_permutations:
                        parts.append(", %i permutations" % self._n_samples)
                    elif self._n_samples:
                        parts.append(", %i samples" % self._n_samples)
                    parts.append(", p >= %.3f" % self.clusters['p'].x.min())
        parts.append(">")
        return ''.join(parts)


class ttest_ind:
    """Element-wise independent samples t-test

    Attributes
    ----------
    all :
        c1, c0, [c0 - c1, P]
    p_val :
        [c0 - c1, P]
    """
    def __init__(self, Y, X, c1=None, c0=None, match=None, sub=None, ds=None,
                 tail=0, samples=None, pmin=0.1, tstart=None, tstop=None,
                 **criteria):
        """Element-wise t-test

        Parameters
        ----------
        Y : NDVar
            Dependent variable.
        X : categorial
            Model containing the cells which should be compared.
        c1 : str | tuple | None
            Test condition (cell of X). Can be None is X only contains two
            cells.
        c0 : str | tuple | None
            Control condition (cell of X). Can be None if X only contains two
            cells.
        match : None | categorial
            Combine cases with the same cell on X % match for testing.
        sub : None | index-array
            Perform the test with a subset of the data.
        ds : None | Dataset
            If a Dataset is specified, all data-objects can be specified as
            names of Dataset variables.
        tail : 0 | 1 | -1
            Which tail of the t-distribution to consider:
            0: both (two-tailed);
            1: upper tail (one-tailed);
            -1: lower tail (one-tailed).
        samples : None | int
            Number of samples for permutation cluster test. For None, no
            clusters are formed. Use 0 to compute clusters without performing
            any permutations.
        pmin : scalar (0 < pmin < 1)
            Threshold p value for forming clusters in permutation cluster test.
        tstart, tstop : None | scalar
            Restrict time window for permutation cluster test.
        mintime : scalar
            Minimum duration for clusters (in seconds).
        minsource : int
            Minimum number of sources per cluster.
        """
        ct = Celltable(Y, X, match, sub, cat=(c1, c0), ds=ds)
        c1, c0 = ct.cat

        test_name = 'Independent Samples t-Test'
        n1 = len(ct.data[c1])
        N = len(ct.Y)
        n0 = N - n1
        df = N - 2
        tmap = _t_ind(ct.Y.x, n1, n0)
        pmap = _ttest_p(tmap, df, tail)
        if samples is not None:
            t_threshold = _ttest_t(pmin, df, tail)
            t_upper = t_threshold if tail >= 0 else None
            t_lower = -t_threshold if tail <= 0 else None
            cdist = _ClusterDist(ct.Y, samples, t_upper, t_lower, 't',
                                 test_name, tstart, tstop, criteria)
            cdist.add_original(tmap)
            if cdist.n_clusters and samples:
                for Y_ in resample(cdist.Y_perm, samples):
                    tmap_ = _t_ind(Y_.x, n1, n0)
                    cdist.add_perm(tmap_)

        dims = ct.Y.dims[1:]

        info = _cs.set_info_cs(ct.Y.info, _cs.sig_info())
        info['test'] = test_name
        p = NDVar(pmap, dims, info=info, name='p')

        t0, t1, t2 = _ttest_t((.05, .01, .001), df, tail)
        info = _cs.stat_info('t', t0, t1, t2, tail)
        info = _cs.set_info_cs(ct.Y.info, info)
        t = NDVar(tmap, dims, info=info, name='T')

        c1_mean = ct.data[c1].summary(name=cellname(c1))
        c0_mean = ct.data[c0].summary(name=cellname(c0))
        diff = c1_mean - c0_mean
        if np.any(diff < 0):
            diff.info['cmap'] = 'xpolar'

        # store attributes
        self.n1 = n1
        self.n0 = n0
        self.df = df
        self.name = test_name
        self._c0 = c0
        self._c1 = c1
        self._samples = samples
        self._ct = ct

        self.c1 = c1_mean
        self.c0 = c0_mean
        self.diff = diff
        self.t = t
        self.p = p

        self.diffp = [[diff, t]]
        self.uncorrected = [c1_mean, c0_mean] + self.diffp
        if samples is None:
            self.all = self.uncorrected
            self._cdist = None
        else:
            self.diff_cl = [[diff, cdist.cpmap]]
            self.all = [c1_mean, c0_mean] + self.diff_cl
            self._cdist = cdist
            self.clusters = cdist.clusters

    def __repr__(self):
        parts = ["<%s %r-%r" % (self.name, self._c1, self._c0)]
        if self.n1 == self.n0:
            parts.append(", n1=n0=%i" % self.n1)
        else:
            parts.append(", n1=%i, n0=%i" % (self.n1, self.n0))
        if self._samples:
            if self.clusters is None:
                parts.append(", no clusters found")
            else:
                n = self._cdist.n_clusters
                parts.append(", %i samples: %i clusters" % (self._samples, n))
                parts.append(", p >= %.3f" % self.clusters['p'].x.min())
        parts.append('>')
        return ''.join(parts)


class ttest_rel:
    """Element-wise t-test

    Attributes
    ----------
    all :
        c1, c0, [c0 - c1, P]
    p_val :
        [c0 - c1, P]
    """
    def __init__(self, Y, X, c1=None, c0=None, match=None, sub=None, ds=None,
                 tail=0, samples=None, pmin=0.1, tstart=None, tstop=None,
                 **criteria):
        """Element-wise t-test

        Parameters
        ----------
        Y : NDVar
            Dependent variable.
        X : categorial
            Model containing the cells which should be compared.
        c1 : str | tuple | None
            Test condition (cell of X). Can be None is X only contains two
            cells.
        c0 : str | tuple | None
            Control condition (cell of X). Can be None if X only contains two
            cells.
        match : Factor
            Match cases for a repeated measures test.
        sub : None | index-array
            Perform the test with a subset of the data.
        ds : None | Dataset
            If a Dataset is specified, all data-objects can be specified as
            names of Dataset variables.
        tail : 0 | 1 | -1
            Which tail of the t-distribution to consider:
            0: both (two-tailed);
            1: upper tail (one-tailed);
            -1: lower tail (one-tailed).
        samples : None | int
            Number of samples for permutation cluster test. For None, no
            clusters are formed. Use 0 to compute clusters without performing
            any permutations.
        pmin : scalar (0 < pmin < 1)
            Threshold p value for forming clusters in permutation cluster test.
        tstart, tstop : None | scalar
            Restrict time window for permutation cluster test.
        mintime : scalar
            Minimum duration for clusters (in seconds).
        minsource : int
            Minimum number of sources per cluster.

        Notes
        -----
        In the permutation cluster test, permutations are done within the
        categories of ``match``.
        """
        ct = Celltable(Y, X, match, sub, cat=(c1, c0), ds=ds)
        c1, c0 = ct.cat
        if not ct.all_within:
            err = ("conditions %r and %r do not have the same values on "
                   "%r" % (c1, c0, ct.match.name))
            raise ValueError(err)

        test_name = 'Related Samples t-Test'
        n = len(ct.Y) / 2
        df = n - 1
        tmap = _t_rel(ct.Y.x)
        pmap = _ttest_p(tmap, df, tail)
        if samples is not None:
            t_threshold = _ttest_t(pmin, df, tail)
            t_upper = t_threshold if tail >= 0 else None
            t_lower = -t_threshold if tail <= 0 else None
            cdist = _ClusterDist(ct.Y, samples, t_upper, t_lower, 't',
                                 test_name, tstart, tstop, criteria)
            cdist.add_original(tmap)
            if cdist.n_clusters and samples:
                for Y_ in resample(cdist.Y_perm, samples, unit=ct.match):
                    tmap_ = _t_rel(Y_.x)
                    cdist.add_perm(tmap_)

        dims = ct.Y.dims[1:]

        info = _cs.set_info_cs(ct.Y.info, _cs.sig_info())
        info['test'] = test_name
        p = NDVar(pmap, dims, info=info, name='p')

        t0, t1, t2 = _ttest_t((.05, .01, .001), df, tail)
        info = _cs.stat_info('t', t0, t1, t2, tail)
        info = _cs.set_info_cs(ct.Y.info, info)
        t = NDVar(tmap, dims, info=info, name='T')

        c1_mean = ct.data[c1].summary(name=cellname(c1))
        c0_mean = ct.data[c0].summary(name=cellname(c0))
        diff = c1_mean - c0_mean
        if np.any(diff < 0):
            diff.info['cmap'] = 'xpolar'

        # store attributes
        self.n = n
        self.df = df
        self.name = test_name
        self._c0 = c0
        self._c1 = c1
        self._samples = samples
        self._ct = ct

        self.c1 = c1_mean
        self.c0 = c0_mean
        self.diff = diff
        self.t = t
        self.p = p

        self.diffp = [[diff, t]]
        self.uncorrected = [c1_mean, c0_mean] + self.diffp
        if samples is None:
            self.all = self.uncorrected
            self._cdist = None
        else:
            if cdist.n_clusters and samples:
                self.diff_cl = [[diff, cdist.cpmap]]
            else:
                self.diff_cl = [[diff]]
            self.all = [c1_mean, c0_mean] + self.diff_cl
            self._cdist = cdist
            self.clusters = cdist.clusters

    def __repr__(self):
        parts = ["<%s %r-%r" % (self.name, self._c1, self._c0)]
        parts.append(", n=%i" % self.n)
        if self._samples:
            if self.clusters is None:
                parts.append(", no clusters found")
            else:
                n = self._cdist.n_clusters
                parts.append(", %i samples: %i clusters" % (self._samples, n))
                parts.append(", p >= %.3f" % self.clusters['p'].x.min())
        parts.append('>')
        return ''.join(parts)


def _t_1samp(a, popmean):
    "Based on scipy.stats.ttest_1samp"
    n = len(a)
    if np.prod(a.shape) > 2 ** 25:
        a_flat = a.reshape((n, -1))
        n_samp = a_flat.shape[1]
        step = int(floor(2 ** 25 / n))
        t_flat = np.empty(n_samp)
        for i in xrange(0, n_samp, step):
            t_flat[i:i + step] = _t_1samp(a_flat[:, i:i + step], popmean)
        t = t_flat.reshape(a.shape[1:])
        return t

    d = np.mean(a, 0) - popmean
    v = np.var(a, 0, ddof=1)
    denom = np.sqrt(v / n)
    t = np.divide(d, denom)
    return t


def _t_ind(x, n1, n2, equal_var=True):
    "Based on scipy.stats.ttest_ind"
    a = x[:n1]
    b = x[n1:]
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
    t = np.divide(d, denom)
    return t


def _t_rel(Y):
    """
    Calculates the T statistic on two related samples.

    Parameters
    ----------
    Y : array_like, shape = (n_cases, ...)
        Dependent variable in right input format: The first half and second
        half of the data represent the two samples; in each subjects

    Returns
    -------
    t : array, shape = (...)
        t-statistic

    Notes
    -----
    Based on scipy.stats.ttest_rel
    df = n - 1
    """
    n_cases = len(Y)
    shape = Y.shape[1:]
    n_tests = np.product(shape)
    if np.log2(n_tests) > 13:
        Y = Y.reshape((n_cases, n_tests))
        t = np.empty(n_tests)
        step = 2 ** 13
        for i in xrange(0, n_tests, step):
            i1 = i + step
            t[i:i1] = _t_rel(Y[:, i:i1])
        t = t.reshape(shape)
        return t
    n = n_cases // 2
    a = Y[:n]
    b = Y[n:]
    d = (a - b).astype(np.float64)
    v = np.var(d, 0, ddof=1)
    dm = np.mean(d, 0)
    denom = np.sqrt(v / n)
    t = np.divide(dm, denom)
    return t


def _ttest_p(t, df, tail=0):
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


def _ttest_t(p, df, tail=0):
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


class anova:
    """Element-wise ANOVA

    Attributes
    ----------
    clusters : None | Dataset
        When performing a cluster permutation test, a Dataset listing all
        clusters.
    f : list
        Maps of f values with probability contours.
    p : list
        Maps of p values.
    """
    def __init__(self, Y, X, sub=None, ds=None, samples=None, pmin=0.1,
                 tstart=None, tstop=None, match=None, **criteria):
        """ANOVA with cluster permutation test

        Parameters
        ----------
        Y : NDVar
            Measurements (dependent variable)
        X : categorial
            Model
        sub : None | index-array
            Perform the test with a subset of the data.
        ds : None | Dataset
            If a Dataset is specified, all data-objects can be specified as
            names of Dataset variables.
        samples : None | int
            Number of samples for permutation cluster test. For None, no
            clusters are formed. Use 0 to compute clusters without performing
            any permutations.
        pmin : scalar (0 < pmin < 1)
            Threshold p value for forming clusters in permutation cluster test.
        replacement : bool
            whether random samples should be drawn with replacement or
            without
        tstart, tstop : None | scalar
            Restrict time window for permutation cluster test.
        match : None | categorial
            When permuting data, only shuffle the cases within the categories
            of match.
        mintime : scalar
            Minimum duration for clusters (in seconds).
        minsource : int
            Minimum number of sources per cluster.
        """
        sub = assub(sub, ds)
        Y = self.Y = asndvar(Y, sub, ds)
        X = self.X = asmodel(X, sub, ds)
        if match is not None:
            match = self.match = ascategorial(match, sub, ds)

        lm = LMFitter(X, Y.x.shape)
        effects = lm.effects
        df_den = lm.df_den
        fmaps = lm.map(Y.x)

        if samples is not None:
            # find F-thresholds for clusters
            fmins = [ftest_f(pmin, e.df, df_den[e]) for e in effects]
            cdists = [_ClusterDist(Y, samples, fmin, None, 'F', e.name,
                                   tstart, tstop, criteria)
                      for e, fmin in izip(effects, fmins)]

            # Find clusters in the actual data
            n_clusters = 0
            for cdist, fmap in izip(cdists, fmaps):
                cdist.add_original(fmap)
                n_clusters += cdist.n_clusters

            if n_clusters and samples:
                fmaps_ = lm.preallocate(cdist.Y_perm.shape)
                for Y_ in resample(cdist.Y_perm, samples, unit=match):
                    lm.map(Y_.x)
                    for cdist, fmap in izip(cdists, fmaps_):
                        cdist.add_perm(fmap)

        # create ndvars
        dims = Y.dims[1:]

        f = []
        p = []
        for e, fmap in izip(effects, fmaps):
            f0, f1, f2 = ftest_f((0.05, 0.01, 0.001), e.df, df_den[e])
            info = _cs.set_info_cs(Y.info, _cs.stat_info('f', f0, f1, f2))
            f_ = NDVar(fmap, dims, info, e.name)
            f.append(f_)

            info = _cs.set_info_cs(Y.info, _cs.sig_info())
            pmap = ftest_p(fmap, e.df, df_den[e])
            p_ = NDVar(pmap, dims, info, e.name)
            p.append(p_)

        if samples is None:
            clusters = None
        else:
            # f-maps with clusters
            f_and_clusters = []
            for e, fmap, cdist in izip(effects, fmaps, cdists):
                # create f-map with cluster threshold
                f0 = ftest_f(pmin, e.df, df_den[e])
                info = _cs.set_info_cs(Y.info, _cs.stat_info('f', f0))
                f_ = NDVar(fmap, dims, info, e.name)
                # add overlay with cluster
                if cdist.n_clusters and samples:
                    f_and_clusters.append([f_, cdist.cpmap])
                else:
                    f_and_clusters.append([f_])

            # create cluster table
            dss = []
            for cdist in cdists:
                ds = cdist.clusters
                if ds is None:
                    continue
                ds['effect'] = Factor([cdist.name], rep=ds.n_cases)
                dss.append(ds)

            if dss:
                clusters = combine(dss)
            else:
                clusters = None

        # store attributes
        self.name = "anova(%s, %s)" % (Y.name, X.name)
        self.clusters = clusters
        self.f = [[f_, f_] for f_ in f]
        self.p = p
        self.samples = samples
        if samples is None:
            self.all = self.f
        else:
            self.fmin = fmins
            self._cdists = cdists
            self.all = f_and_clusters

    def __repr__(self):
        parts = ["<%s" % (self.name)]
        if self.samples:
            if self.clusters is None:
                parts.append(" no clusters found")
            else:
                parts.append(" %i samples, clusters:" % self.samples)
                for e in self.clusters['effect'].cells:
                    idx = self.clusters['effect'] == e
                    p = np.min(self.clusters[idx, 'p'].x)
                    parts.append(" %r p >= %.3f" % (e, p))
        parts.append('>')
        return ''.join(parts)


class _ClusterDist:
    """Accumulate information on a cluster statistic.

    Notes
    -----
    Use of the _ClusterDist proceeds in 3 steps:

    - initialize the _ClusterDist object: ``cdist = _ClusterDist(...)``
    - use a copy of Y cropped to the time window of interest:
      ``Y = cdist.Y_perm``
    - add the actual statistical map with ``cdist.add_original(pmap)``
    - if any clusters are found (``if cdist.n_clusters``):

      - proceed to add statistical maps from permuted data with
        ``cdist.add_perm(pmap)``.
    """
    def __init__(self, Y, N, t_upper, t_lower=None, meas='?', name=None,
                 tstart=None, tstop=None, criteria={}):
        """Accumulate information on a cluster statistic.

        Parameters
        ----------
        Y : NDVar
            Dependent variable.
        N : int
            Number of permutations.
        t_upper, t_lower : None | scalar
            Positive and negative thresholds for finding clusters. If None,
            no clusters with the corresponding sign are counted.
        meas : str
            Label for the parameter measurement (e.g., 't' for t-values).
        name : None | str
            Name for the comparison.
        tstart, tstop : None | scalar
            Restrict the time window for finding clusters (None: use the whole
            epoch).
        criteria : dict
            Dictionary with threshold criteria for cluster size: 'mintime'
            (seconds) and 'minsource' (n_sources).
        """
        assert Y.has_case
        if t_lower is not None:
            if t_lower >= 0:
                raise ValueError("t_lower needs to be < 0; is %s" % t_lower)
        if t_upper is not None:
            if t_upper <= 0:
                raise ValueError("t_upper needs to be > 0; is %s" % t_upper)
        if (t_lower is not None) and (t_upper is not None):
            if t_lower != -t_upper:
                err = ("If t_upper and t_lower are defined, t_upper has to be "
                       "-t_lower")
                raise ValueError(err)

        # prepare time manipulation
        if Y.has_dim('time'):
            t_ax = Y.get_axis('time') - 1
        else:
            t_ax = None

        # prepare cropping
        if (tstart is None) and (tstop is None):
            self.crop = False
            Y_perm = Y
        else:
            self.crop = True
            Y_perm = Y.sub(time=(tstart, tstop))
            istart = 0 if tstart is None else Y.time.index(tstart, 'up')
            istop = istart + len(Y_perm.time)
            self._crop_idx = (slice(None),) * t_ax + (slice(istart, istop),)
            self._uncropped_shape = Y.shape[1:]

        # prepare adjacency
        adjacent = [d.adjacent for d in Y_perm.dims[1:]]
        self._all_adjacent = all_adjacent = all(adjacent)
        if not all_adjacent:
            if sum(adjacent) < len(adjacent) - 1:
                err = ("more than one non-adjacent dimension")
                raise NotImplementedError(err)
            self._nad_ax = ax = adjacent.index(False)
            self._conn = Y_perm.dims[ax + 1].connectivity()
            struct = ndimage.generate_binary_structure(2, 1)
            struct[::2] = False
            self._struct = struct
            # flattening and reshaping (cropped) p-maps with swapped axes
            shape = Y_perm.shape[1:]
            if ax:
                shape = list(shape)
                shape[0], shape[ax] = shape[ax], shape[0]
                shape = tuple(shape)
            self._orig_shape = shape
            self._flat_shape = (shape[0], np.prod(shape[1:]))

        # interpret cluster criteria
        criteria_ = {}
        for k, v in criteria.iteritems():
            if k == 'tmin' or k == 'mintime':
                criteria_['time'] = int(ceil(v / Y.time.tstep))
            elif k == 'minsource':
                criteria_['source'] = v
            else:
                raise ValueError("Unknown criterion: %r" % k)

            for k in criteria_:
                if not Y.has_dim(k):
                    err = ("Criterion specified for dimension that is not in "
                           "the data: %r" % k)
                    raise ValueError(err)

        N = int(N)

        self.Y = Y
        self.Y_perm = Y_perm
        self.N = N
        self.dist = np.zeros(N)
        self._i = N
        self.t_upper = t_upper
        self.t_lower = t_lower
        self.tstart = tstart
        self.tstop = tstop
        self._t_ax = t_ax
        self.meas = meas
        self.name = name
        self.criteria = criteria_
        self.has_original = False

    def __repr__(self):
        items = []
        if self.has_original:
            dt = self.dt_original / 60.
            items.append("%i clusters (%.2f min)" % (self.n_clusters, dt))

            if self.N > 0 and self.n_clusters > 0:
                if self._i == 0:
                    dt = self.dt_perm / 60.
                    item = "%i permutations (%.2f min)" % (self.N, dt)
                else:
                    item = "%i of %i permutations" % (self.N - self._i, self.N)
                items.append(item)
        else:
            items.append("no data")

        return "<ClusterDist: %s>" % ', '.join(items)

    def _crop(self, im):
        if self.crop:
            return im[self._crop_idx]
        else:
            return im

    def _finalize(self):
        if self._i < 0:
            raise RuntimeError("Too many permutations added to _ClusterDist")

        if not self.n_clusters:
            self.clusters = None
            return

        # retrieve original clusters
        pmap = self._original_pmap  # parameter map
        pmap_cropped = self._crop(pmap)
        cmap = self._cluster_im
        cids = self._cids

        # prepare container for clusters
        self.clusters = ds = Dataset()
        dims = self.Y.dims

        # measure original clusters
        cluster_v = ndimage.sum(pmap_cropped, cmap, cids)
        ds['v'] = Var(cluster_v)

        # p-values: "the proportion of random partitions that resulted in a
        # larger test statistic than the observed one" (179)
        if self.N:
            n_larger = np.sum(self.dist > np.abs(cluster_v[:, None]), 1)
            cluster_p = n_larger / self.N
            ds['p'] = p = Var(cluster_p)
            ds['*'] = star_factor(p)

        # create cluster ndvars
        cpmap = np.ones_like(pmap_cropped)  # cluster probability
        cmaps = np.empty((self.n_clusters,) + pmap.shape, dtype=pmap.dtype)
        for i in xrange(self.n_clusters):
            # cluster index
            cid = cids[i]
            # cluster extent
            c_mask = (cmap == cid)
            # cluster value map
            cmaps[i] = self._uncrop(pmap_cropped * c_mask)
            if self.N:
                # cluster probability map
                p = cluster_p[i]
                cpmap[c_mask] = p

        # store cluster NDVar
        contours = {}
        if self.t_lower is not None:
            contours[self.t_lower] = (0.7, 0, 0.7)
        if self.t_upper is not None:
            contours[self.t_upper] = (0.7, 0.7, 0)
        info = _cs.stat_info(self.meas, contours=contours, summary_func=np.sum)
        ds['cluster'] = NDVar(cmaps, dims=dims, info=info)

        # add cluster info
        for axis, dim in enumerate(dims[1:], 1):
            properties = dim._cluster_properties(cmaps, axis)
            if properties is not None:
                ds.update(properties)

        # cluster probability map
        cpmap = self._uncrop(cpmap, 1)
        info = _cs.cluster_pmap_info()
        self.cpmap = NDVar(cpmap, dims=dims[1:], name=self.name, info=info)

        # statistic parameter map
        info = _cs.stat_info(self.meas, contours=contours)
        self.pmap = NDVar(pmap, dims=dims[1:], name=self.name, info=info)

        self.all = [[self.pmap, self.cpmap]]
        self.dt_perm = current_time() - self._t0

    def _label_clusters(self, pmap):
        """Find clusters on a statistical parameter map

        Parameters
        ----------
        pmap : array
            Statistical parameter map (flattened if the data contains
            non-adjacent dimensions).

        Returns
        -------
        cluster_map : array
            Array of same shape as pmap with clusters labeled.
        cluster_ids : tuple
            Identifiers of the clusters that survive the minimum duration
            criterion.
        """
        if self.t_upper is not None:
            bin_map_above = (pmap > self.t_upper)
            cmap, cids = self._label_clusters_1tailed(bin_map_above)

        if self.t_lower is not None:
            bin_map_below = (pmap < self.t_lower)
            if self.t_upper is None:
                cmap, cids = self._label_clusters_1tailed(bin_map_below)
            else:
                cmap_l, cids_l = self._label_clusters_1tailed(bin_map_below)
                x = int(cmap.max())  # apparently np.uint64 + int makes a float
                cmap_l[bin_map_below] += x
                cmap += cmap_l
                cids.update(c + x for c in cids_l)

        return cmap, tuple(cids)

    def _label_clusters_1tailed(self, bin_map):
        """
        Parameters
        ----------
        bin_map : array
            Binary map of where the parameter map exceeds the threshold for a
            cluster. (flattened if the data contains non-adjacent dimensions).

        Returns
        -------
        cluster_map : array
            Array of same shape as bin_map with clusters labeled.
        cluster_ids : iterator over int
            Identifiers of the clusters that survive the minimum duration
            criterion.
        """
        # find clusters
        if self._all_adjacent:
            cmap, n = ndimage.label(bin_map)
            # n is 1 even when no cluster is found
            if n == 1 and cmap.max() == 0:
                n = 0
            cids = set(xrange(1, n + 1))
        else:
            c = self._conn
            cmap, n = ndimage.label(bin_map, self._struct)
            if n == 1 and cmap.max() == 0:
                n = 0
            cids = set(xrange(1, n + 1))
            n_chan = len(cmap)

            for i in xrange(bin_map.shape[1]):
                if len(np.setdiff1d(cmap[:, i], np.zeros(1), False)) <= 1:
                    continue

                idx = np.flatnonzero(cmap[:, i])
                c_idx = np.logical_and(np.in1d(c.row, idx), np.in1d(c.col, idx))
                row = c.row[c_idx]
                col = c.col[c_idx]
                data = c.data[c_idx]
                n = np.max(idx)
                c_ = coo_matrix((data, (row, col)), shape=c.shape)
                n_, lbl_map = connected_components(c_, False)
                if n_ == n_chan:
                    continue
                labels_ = np.flatnonzero(np.bincount(lbl_map) > 1)
                for lbl in labels_:
                    idx_ = lbl_map == lbl
                    merge = np.unique(cmap[idx_, i])

                    # merge labels
                    idx_ = reduce(np.logical_or, (cmap == m for m in merge))
                    cmap[idx_] = merge[0]
                    cids.difference_update(merge[1:])

                if len(cids) == 1:
                    break

        # apply minimum cluster size criteria
        criteria = self.criteria
        if criteria:
            # return cmap to proper shape
            if self._all_adjacent:
                cmap_ = cmap
            else:
                cmap_ = cmap.reshape(self._orig_shape)
                cmap_ = cmap_.swapaxes(0, self._nad_ax)

            for axis, dim in enumerate(self.Y_perm.dims[1:]):
                if dim.name not in criteria:
                    continue
                vmin = criteria[dim.name]
                size = dim._cluster_size_basic
                rm = tuple(i for i in cids if size(cmap_ == i, axis) < vmin)
                cids.difference_update(rm)

        return cmap, cids

    def _uncrop(self, im, background=0):
        if self.crop:
            im_ = np.empty(self._uncropped_shape, dtype=im.dtype)
            im_[:] = background
            im_[self._crop_idx] = im
            return im_
        else:
            return im

    def add_original(self, pmap):
        """Add the original statistical parameter map.

        Parameters
        ----------
        pmap : array
            Parameter map of the statistic of interest (uncropped).
        """
        if hasattr(self, '_cluster_im'):
            raise RuntimeError("Original pmap already added")

        t0 = current_time()
        pmap_ = self._crop(pmap)
        if not self._all_adjacent:
            pmap_ = pmap_.swapaxes(0, self._nad_ax)
            pmap_ = pmap_.reshape(self._flat_shape)
        cmap, cids = self._label_clusters(pmap_)
        if not self._all_adjacent:  # return cmap to proper shape
            cmap = cmap.reshape(self._orig_shape)
            cmap = cmap.swapaxes(0, self._nad_ax)

        self.has_original = True
        self.dt_original = current_time() - t0
        self._cluster_im = cmap
        self._original_pmap = pmap
        self._cids = cids
        self.n_clusters = len(cids)
        self._t0 = current_time()
        if self.n_clusters == 0 or self.N == 0:
            self._finalize()

    def add_perm(self, pmap):
        """Add the statistical parameter map from permuted data.

        Parameters
        ----------
        pmap : array
            Parameter map of the statistic of interest.
        """
        self._i -= 1

        if not self._all_adjacent:
            pmap = pmap.swapaxes(0, self._nad_ax)
            pmap = pmap.reshape(self._flat_shape)

        cmap, cids = self._label_clusters(pmap)
        if cids:
            clusters_v = ndimage.sum(pmap, cmap, cids)
            self.dist[self._i] = np.max(np.abs(clusters_v))

        if self._i == 0:
            self._finalize()

    def as_table(self, pmax=1.):
        cols = 'll'
        headings = ('#', 'p')
        time = self.Y.get_dim('time') if self.Y.has_dim('time') else None
        if time is not None:
            time_ax = self.Y.get_axis('time') - 1
            any_axes = tuple(i for i in xrange(self.Y.ndim - 1) if i != time_ax)
            cols += 'l'
            headings += ('time interval',)

        table = fmtxt.Table(cols)
        table.cells(*headings)
        table.midrule()

        i = 0
        for c in self.clusters:
            p = c.info['p']
            if p <= pmax:
                table.cell(i)
                i += 1
                table.cell(p)

                if time is not None:
                    nz = np.flatnonzero(np.any(c.x, axis=any_axes))
                    tstart = time[nz.min()]
                    tstop = time[nz.max()]
                    interval = '%.3f - %.3f s' % (tstart, tstop)
                    table.cell(interval)

        return table
