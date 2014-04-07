'''Statistical tests for ndvars'''
from __future__ import division

from itertools import izip
from math import ceil, floor
import re
from time import time as current_time

import numpy as np
import scipy.stats
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation

from ... import fmtxt
from .. import colorspaces as _cs
from ..data_obj import (ascategorial, asmodel, asndvar, asvar, assub, Dataset,
                        Factor, NDVar, Var, Celltable, cellname, combine)
from .glm import LMFitter
from .permutation import resample, _resample_params
from .stats import ftest_f, ftest_p
from .test import star_factor


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


class t_contrast_rel:
    def __init__(self, Y, X, contrast, match=None, sub=None, ds=None,
                 tail=0, samples=None, pmin=None, tstart=None, tstop=None,
                 **criteria):
        """Contrast with t-values from multiple comparisons

        Parameters
        ----------
        Y : NDVar
            Dependent variable.
        X : categorial
            Model containing the cells which are compared with the contrast.
        contrast : str
            Contrast specification: see Notes.
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
        pmin : None | scalar (0 < pmin < 1)
            Threshold p value for forming clusters: a t-value equivalent to p
            for a related samples t-test (with df = len(match.cells) - 1) is
            used. Alternatively, in order to directly specify the threshold as
            t-value you can supply ``tmin`` as keyword argument. This overrides
            the ``pmin`` parameter. None for threshold-free cluster
            enhancement.
        tstart, tstop : None | scalar
            Restrict time window for permutation cluster test.
        mintime : scalar
            Minimum duration for clusters (in seconds).
        minsource : int
            Minimum number of sources per cluster.

        Notes
        -----
        Contrast definitions can contain:

         - comparisons using ">" and "<", e.g. ``"cell1 > cell0"``.
         - numpy functions, e.g. ``min(...)``.
         - prefixing a function or comparison with ``+`` or ``-`` makes the
           relevant comparison one-tailed by setting all values of the opposite
           sign to zero (e.g., ```"+a>b"``` sets all data points where a<b to
           0.

        So for example, to find cluster where both of two pairwise comparisons
        are reliable, one could use ``"min(a1 > a0, b1 > b0)"``

        If X is an interaction, interaction cells are specified with "|", e.g.
        ``"a1 | b > a0 | b"``.
        """
        test_name = "t-contrast"
        ct = Celltable(Y, X, match, sub, ds=ds, coercion=asndvar)
        Y = ct.Y
        index = ct.data_indexes

        if 'tmin' in criteria:
            tmin = criteria.pop('tmin')
        elif pmin is None:
            tmin = None
        else:
            df = len(ct.match.cells) - 1
            tmin = _ttest_t(pmin, df, tail)
        contrast_ = _parse_t_contrast(contrast)

        # buffer memory allocation
        shape = ct.Y.shape[1:]
        buffers = _t_contrast_rel_setup(contrast_)
        buff = np.empty((buffers,) + shape)

        # original data
        tmap = _t_contrast_rel(contrast_, Y.x, index, buff)

        if samples is None:
            cdist = None
        else:
            cdist = _ClusterDist(Y, samples, tmin, tail, 't', test_name,
                                 tstart, tstop, criteria)
            cdist.add_original(tmap)
            if cdist.n_clusters and samples:
                # buffer memory allocation
                shape = cdist.Y_perm.shape[1:]
                buff = np.empty((buffers,) + shape)
                tmap_ = np.empty(shape)
                for Y_ in resample(cdist.Y_perm, samples, unit=ct.match):
                    _t_contrast_rel(contrast_, Y_.x, index, buff, tmap_)
                    cdist.add_perm(tmap_)

        # construct results
        dims = ct.Y.dims[1:]

        # store attributes
        self.name = test_name
        self.t = NDVar(tmap, dims, {}, 't')
        self._cdist = cdist
        self.contrast = contrast
        self._contrast = contrast_
        if samples is not None:
            self.clusters = cdist.clusters

    def __repr__(self):
        parts = ["<%s %r" % (self.name, self.contrast)]
        if self._cdist:
            parts.append(self._cdist._cluster_repr())
        parts.append('>')
        return ''.join(parts)


def _parse_cell(cell_name):
    cell = tuple(s.strip() for s in cell_name.split('|'))
    if len(cell) == 1:
        return cell[0]
    else:
        return cell


def _parse_t_contrast(contrast):
    depth = 0
    start = 0
    if not '(' in contrast:
        m = re.match("\s*([+-]*)\s*([\w\|]+)\s*([<>])\s*([\w\|]+)", contrast)
        if m:
            clip, c1, direction, c0 = m.groups()
            if direction == '<':
                c1, c0 = c0, c1
            c1 = _parse_cell(c1)
            c0 = _parse_cell(c0)
            return ('comp', clip or None, c1, c0)

    for i, c in enumerate(contrast):
        if c == '(':
            if depth == 0:
                prefix = contrast[start:i]
                i_open = i + 1
                items = []
            depth += 1
        elif c == ',':
            if depth == 0:
                raise
            elif depth == 1:
                item = _parse_t_contrast(contrast[i_open:i])
                items.append(item)
                i_open = i + 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                item = _parse_t_contrast(contrast[i_open:i])
                items.append(item)

                m = re.match("\s*([+-]*)\s*(\w+)", prefix)
                if m is None:
                    raise ValueError("uninterpretable prefix: %r" % prefix)
                clip, func_name = m.groups()
                func = getattr(np, func_name)

                return ('func', clip or None, func, items)
            elif depth == -1:
                err = "Invalid ')' at position %i of %r" % (i, contrast)
                raise ValueError(err)


def _t_contrast_rel_setup(item):
    """Setup t-contrast

    Parameters
    ----------
    item : tuple
        Contrast specification.

    Returns
    -------
    buffers : int
        Number of buffer maps needed.
    """
    if item[0] == 'func':
        _, _, _, items_ = item
        local_buffers = len(items_)
        for i, item_ in enumerate(items_):
            available_buffers = local_buffers - i - 1
            needed_buffers = _t_contrast_rel_setup(item_)
            additional_buffers = needed_buffers - available_buffers
            if additional_buffers > 0:
                local_buffers += additional_buffers
        return local_buffers
    else:
        return 0


def _t_contrast_rel(item, y, index, buff=None, out=None):
    if out is None:
        out = np.zeros(y.shape[1:])
    else:
        out[:] = 0

    if item[0] == 'func':
        _, clip, func, items_ = item
        tmaps = buff[:len(items_)]
        for i, item_ in enumerate(items_):
            if buff is None:
                buff_ = None
            else:
                buff_ = buff[i + 1:]
            _t_contrast_rel(item_, y, index, buff_, tmaps[i])
        tmap = func(tmaps, axis=0, out=out)
    else:
        _, clip, c1, c0 = item
        i1 = index[c1]
        i0 = index[c0]
        tmap = _t_rel(y[i1], y[i0], out)

    if clip is not None:
        if clip == '+':
            a_min = 0
            a_max = tmap.max() + 1
        elif clip == '-':
            a_min = tmap.min() - 1
            a_max = 0
        tmap.clip(a_min, a_max, tmap)

    return tmap


class corr:
    """Correlation

    Attributes
    ----------
    r : NDVar
        Correlation (with threshold contours).
    """
    def __init__(self, Y, X, norm=None, sub=None, ds=None, samples=None,
                 pmin=None, tstart=None, tstop=None, match=None, **criteria):
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
        pmin : None | scalar (0 < pmin < 1)
            Threshold p value for forming clusters. None for threshold-free
            cluster enhancement.
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
            if pmin is None:
                threshold = None
            else:
                threshold = _rtest_r(pmin, df)

            cdist = _ClusterDist(Y, samples, threshold, 0, 'r', name,
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
                self.r_cl = [[r, cdist.probability_map]]
                self.all = [[r, cdist.probability_map]]
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
                 samples=None, pmin=None, tstart=None, tstop=None, **criteria):
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
        pmin : None | scalar (0 < pmin < 1)
            Threshold p value for forming clusters. None for threshold-free
            cluster enhancement.
        tstart, tstop : None | scalar
            Restrict time window for permutation cluster test.
        mintime : scalar
            Minimum duration for clusters (in seconds).
        minsource : int
            Minimum number of sources per cluster.
        """
        ct = Celltable(Y, match=match, sub=sub, ds=ds, coercion=asndvar)

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

        if samples is None:
            cdist = None
        else:
            if pmin is None:
                threshold = None
            else:
                threshold = _ttest_t(pmin, df, tail)
            if popmean:
                y_perm = ct.Y - popmean
            else:
                y_perm = ct.Y
            n_samples, samples_ = _resample_params(len(y_perm), samples)
            cdist = _ClusterDist(y_perm, n_samples, threshold, tail, 't',
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
        self.tail = tail

        self.y = y
        self.diff = diff
        self.t = t
        self.p = p

        self.diffp = [[diff, t]]
        self.all = [y, [diff, t]] if popmean else [[diff, t]]
        self._cdist = cdist
        if samples is not None:
            self._n_samples = n_samples
            self._all_permutations = samples_ < 0
            if cdist.n_clusters and samples:
                self.diff_cl = [[diff, cdist.probability_map]]
            else:
                self.diff_cl = [[diff]]
            self.clusters = cdist.clusters

    def __repr__(self):
        parts = ["<%s against %g, n=%i" % (self.name, self.popmean, self.n)]
        if self.tail:
            parts.append(", tail=%i" % self.tail)
        if self._cdist:
            txt = self._cdist._cluster_repr(perm=self._all_permutations)
            parts.append(txt)
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
                 tail=0, samples=None, pmin=None, tstart=None, tstop=None,
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
        pmin : None | scalar (0 < pmin < 1)
            Threshold p value for forming clusters. None for threshold-free
            cluster enhancement.
        tstart, tstop : None | scalar
            Restrict time window for permutation cluster test.
        mintime : scalar
            Minimum duration for clusters (in seconds).
        minsource : int
            Minimum number of sources per cluster.
        """
        ct = Celltable(Y, X, match, sub, cat=(c1, c0), ds=ds, coercion=asndvar)
        c1, c0 = ct.cat

        test_name = 'Independent Samples t-Test'
        n1 = len(ct.data[c1])
        N = len(ct.Y)
        n0 = N - n1
        df = N - 2
        tmap = _t_ind(ct.Y.x, n1, n0)
        pmap = _ttest_p(tmap, df, tail)
        if samples is not None:
            if pmin is None:
                threshold = None
            else:
                threshold = _ttest_t(pmin, df, tail)

            cdist = _ClusterDist(ct.Y, samples, threshold, tail, 't',
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
        self.tail = tail
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
            self._cdist = cdist
            self.clusters = cdist.clusters
            if samples > 0:
                self.diff_cl = [[diff, cdist.probability_map]]
                self.all = [c1_mean, c0_mean] + self.diff_cl

    def __repr__(self):
        parts = ["<%s %r-%r" % (self.name, self._c1, self._c0)]
        if self.n1 == self.n0:
            parts.append(", n1=n0=%i" % self.n1)
        else:
            parts.append(", n1=%i, n0=%i" % (self.n1, self.n0))
        if self.tail:
            parts.append(", tail=%i" % self.tail)
        if self._cdist:
            parts.append(self._cdist._cluster_repr())
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
                 tail=0, samples=None, pmin=None, tstart=None, tstop=None,
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
        pmin : None | scalar (0 < pmin < 1)
            Threshold p value for forming clusters. None for threshold-free
            cluster enhancement.
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
        ct = Celltable(Y, X, match, sub, cat=(c1, c0), ds=ds, coercion=asndvar)
        c1, c0 = ct.cat
        if not ct.all_within:
            err = ("conditions %r and %r do not have the same values on "
                   "%r" % (c1, c0, ct.match.name))
            raise ValueError(err)

        test_name = 'Related Samples t-Test'
        n = len(ct.Y) // 2
        if n <= 2:
            raise ValueError("Not enough observations for t-test (n=%n)" % n)
        df = n - 1
        tmap = _t_rel(ct.Y.x[:n], ct.Y.x[n:])
        pmap = _ttest_p(tmap, df, tail)
        if samples is not None:
            if pmin is None:
                threshold = None
            else:
                threshold = _ttest_t(pmin, df, tail)

            cdist = _ClusterDist(ct.Y, samples, threshold, tail, 't',
                                 test_name, tstart, tstop, criteria)
            cdist.add_original(tmap)
            if cdist.n_clusters and samples:
                tmap_ = np.empty(cdist.Y_perm.shape[1:])
                for Y_ in resample(cdist.Y_perm, samples, unit=ct.match):
                    _t_rel(Y_.x[:n], Y_.x[n:], tmap_)
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
        self.tail = tail
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
                self.diff_cl = [[diff, cdist.probability_map]]
            else:
                self.diff_cl = [[diff]]
            self.all = [c1_mean, c0_mean] + self.diff_cl
            self._cdist = cdist
            self.clusters = cdist.clusters

    def __repr__(self):
        parts = ["<%s %r-%r" % (self.name, self._c1, self._c0)]
        parts.append(", n=%i" % self.n)
        if self.tail:
            parts.append(", tail=%i" % self.tail)
        if self._cdist:
            parts.append(self._cdist._cluster_repr())
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


def _t_rel(y1, y0, out=None, buff=None):
    """
    Calculates the T statistic on two related samples.

    Parameters
    ----------
    y1, y0 : array_like, shape (n_cases, ...)
        Dependent variable for the two samples.
    out : None | array, shape (...)
        array in which to place the result.
    buff : None | array, shape (n_cases, ...)
        Array to serve as buffer for the difference y1 - y0.

    Returns
    -------
    t : array, shape (...)
        t-statistic.

    Notes
    -----
    Based on scipy.stats.ttest_rel
    df = n - 1
    """
    assert(y1.shape == y0.shape)
    n_subjects = len(y1)
    shape = y1.shape[1:]
    if out is None:
        out = np.empty(shape)
    n_tests = np.product(shape)

    if np.log2(n_tests) > 13:
        y1 = y1.reshape((n_subjects, n_tests))
        y0 = y0.reshape((n_subjects, n_tests))
        out_flat = out.reshape(n_tests)
        step = 2 ** 13
        for i in xrange(0, n_tests, step):
            i1 = i + step
            if buff is None:
                buff_ = None
            else:
                buff_ = buff[:, i:i1]
            _t_rel(y1[:, i:i1], y0[:, i:i1], out_flat[i:i1], buff_)
        return out

    if buff is None:
        buff = np.empty(y1.shape)
    d = np.subtract(y1, y0, buff)
    # out = mean(d) / sqrt(var(d) / n_subjects)
    np.mean(d, 0, out=out)
    denom = np.var(d, 0, ddof=1)
    denom /= n_subjects
    np.sqrt(denom, out=denom)
    out /= denom
    return out


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
    def __init__(self, Y, X, sub=None, ds=None, samples=None, pmin=None,
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
        pmin : None | scalar (0 < pmin < 1)
            Threshold p value for forming clusters. None for threshold-free
            cluster enhancement.
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

        if samples is None:
            cdists = None
        else:
            # find F-thresholds for clusters
            if pmin is None:
                fmins = [None] * len(effects)
            else:
                fmins = [ftest_f(pmin, e.df, df_den[e]) for e in effects]
            cdists = [_ClusterDist(Y, samples, fmin, 1, 'F', e.name, tstart,
                                   tstop, criteria)
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
                    f_and_clusters.append([f_, cdist.probability_map])
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
        self._cdists = cdists
        if samples is None:
            self.all = self.f
        else:
            self.fmin = fmins
            self.all = f_and_clusters

    def __repr__(self):
        parts = [self.name]
        if self._cdists:
            if self.clusters is None:
                parts.append("no clusters")
            else:
                parts.append(self._cdists[0]._param_repr())
                for cdist in self._cdists:
                    name = cdist.name
                    clusters = cdist._cluster_repr(params=False)
                    parts.append("%s: %s" % (name, clusters))
        return "<%s>" % ', '.join(parts)


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
    def __init__(self, Y, N, threshold, tail=0, meas='?', name=None,
                 tstart=None, tstop=None, criteria={}):
        """Accumulate information on a cluster statistic.

        Parameters
        ----------
        Y : NDVar
            Dependent variable.
        N : int
            Number of permutations.
        threshold : None | scalar > 0
            Threshold for finding clusters. None for threshold free cluster
            evaluation.
        tail : 1 | 0 | -1
            Which tail(s) of the distribution to consider. 0 is two-tailed,
            whereas 1 only considers positive values and -1 only considers
            negative values.
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
        assert threshold is None or threshold > 0

        # prepare temporal cropping
        if (tstart is None) and (tstop is None):
            self.crop = False
            Y_perm = Y
        else:
            t_ax = Y.get_axis('time') - 1
            self.crop = True
            Y_perm = Y.sub(time=(tstart, tstop))
            istart = 0 if tstart is None else Y.time.index(tstart, 'up')
            istop = istart + len(Y_perm.time)
            self._crop_idx = (slice(None),) * t_ax + (slice(istart, istop),)
            self._uncropped_shape = Y.shape[1:]

        # cluster map properties
        ndim = Y_perm.ndim - 1
        shape = Y_perm.shape[1:]

        # prepare adjacency
        struct = ndimage.generate_binary_structure(ndim, 1)
        adjacent = [d.adjacent for d in Y_perm.dims[1:]]
        all_adjacent = all(adjacent)
        if all_adjacent:
            nad_ax = 0
        else:
            if sum(adjacent) < len(adjacent) - 1:
                err = ("more than one non-adjacent dimension")
                raise NotImplementedError(err)
            nad_ax = adjacent.index(False)
            struct[::2] = False
            # prepare flattening (cropped) maps with swapped axes
            if nad_ax:
                shape = list(shape)
                shape[0], shape[nad_ax] = shape[nad_ax], shape[0]
                shape = tuple(shape)
            self._flat_shape = (shape[0], np.prod(shape[1:]))

            # prepare connectivity
            coo = Y_perm.dims[nad_ax + 1].connectivity()
            pairs = set()
            for v0, v1, d in izip(coo.row, coo.col, coo.data):
                if not d or v0 == v1:
                    continue
                src = min(v0, v1)
                dst = max(v0, v1)
                pairs.add((src, dst))
            connectivity = np.array(sorted(pairs), dtype=np.int32)
            self._connectivity_src = connectivity[:, 0]
            self._connectivity_dst = connectivity[:, 1]

        # prepare cluster minimum size criteria
        if criteria:
            if threshold is None:
                err = ("Can not use cluster size criteria in doing threshold "
                       "free cluster evaluation")
                raise ValueError(err)

            criteria_ = []
            for k, v in criteria.iteritems():
                if k == 'mintime':
                    ax = Y.get_axis('time') - 1
                    v = int(ceil(v / Y.time.tstep))
                elif k == 'minsource':
                    ax = Y.get_axis('source') - 1
                else:
                    raise ValueError("Unknown criterion: %r" % k)

                if nad_ax:
                    if ax == 0:
                        ax = nad_ax
                    elif ax == nad_ax:
                        ax = 0

                axes = tuple(i for i in xrange(ndim) if i != ax)
                criteria_.append((axes, v))
        else:
            criteria_ = None

        N = int(N)

        self.Y_perm = Y_perm
        self.N = N
        self.dist = np.zeros(N)
        self._i = N
        self.threshold = threshold
        self.tail = tail
        self._all_adjacent = all_adjacent
        self._nad_ax = nad_ax
        self._struct = struct
        self.tstart = tstart
        self.tstop = tstop
        self.meas = meas
        self.name = name
        self.criteria = criteria_
        self._criteria_arg = criteria
        self.has_original = False

        # pre-allocate memory buffers
        self._bin_buff = np.empty(shape, dtype=np.bool8)
        self._int_buff = np.empty(shape, dtype=np.uint32)
        if threshold is None:
            self._original_cmap = np.empty(shape)
            self._float_buff = np.empty(shape)
        else:
            self._original_cmap = np.empty(shape, dtype=np.uint32)
            if tail == 0:
                self._int_buff2 = np.empty(shape, dtype=np.uint32)
        if not all_adjacent:
            self._slice_buff = np.empty(shape[0], dtype=np.bool8)
            self._bin_buff2 = np.empty(shape, dtype=np.bool8)
            self._bin_buff3 = np.empty(shape, dtype=np.bool8)

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

    def _cluster_repr(self, params=True, perm=False):
        """Repr fragment with cluster properties

        Parameters
        ----------
        params : bool
            Include information on input parameters.
        perm : bool
            Whether permutation rather than random resampling was used.
        """
        if self.threshold and self.n_clusters == 0:
            txt = ", no clusters"
        else:
            txt = []
            if params:
                txt.append(self._param_repr(perm))
                if self.threshold:
                    txt.append(": %i clusters" % self.n_clusters)
            elif self.threshold:
                txt.append("%i" % self.n_clusters)

            if self.N:
                if self.threshold:
                    minp = self.clusters['p'].min()
                else:
                    minp = self.probability_map.min()
                txt.append(", p >= %.3f" % minp)
            txt = ''.join(txt)
        return txt

    def _crop(self, im):
        if self.crop:
            return im[self._crop_idx]
        else:
            return im

    def _finalize(self):
        "Package results and delete temporary data"
        if self._i < 0:
            raise RuntimeError("Too many permutations added to _ClusterDist")

        # prepare container for clusters
        dims = self.Y_perm.dims
        ds = Dataset()
        param_contours = {}
        if self.threshold:
            if self.tail >= 0:
                param_contours[self.threshold] = (0.7, 0.7, 0)
            if self.tail <= 0:
                param_contours[-self.threshold] = (0.7, 0, 0.7)

        # original parameter-map
        param_map = self._original_param_map

        if self.threshold is None:
            tfce_map = self._original_cmap
            tfce_map_ = NDVar(tfce_map, dims[1:], {}, self.name)
        else:
            tfce_map_ = None

        if self.threshold is None and self.N:
            # construct probability map
            idx = self._bin_buff
            cpmap = self._float_buff
            cpmap.fill(0)
            for v in self.dist:
                np.greater(v, tfce_map, idx)
                cpmap[idx] += 1
            cpmap /= self.N

            # simplify: find clusters where p < 0.05; should find maxima
            np.less_equal(cpmap, 0.05, idx)
            cmap, cids = self._label_clusters_binary(idx, self._int_buff)
            cids = list(cids)
            # add trends
            np.less_equal(cpmap, 0.1, idx)
            cmap_, cids_ = self._label_clusters_binary(idx, np.empty_like(cmap))
            new_id = max(cids) + 1
            for cid in cids_:
                np.equal(cmap_, cid, idx)
                if not np.any(cmap[idx]):
                    cmap[idx] = new_id
                    cids.append(new_id)
                    new_id += 1

            self.n_clusters = len(cids)
            cluster_p = []
        elif self.threshold and self.n_clusters:
            cmap = self._original_cmap
            cids = self._cids

            # measure original clusters
            cluster_v = ndimage.sum(param_map, cmap, cids)
            ds['v'] = Var(cluster_v)

            # p-values: "the proportion of random partitions that resulted in a
            # larger test statistic than the observed one" (179)
            if self.N:
                n_larger = np.sum(self.dist > np.abs(cluster_v[:, None]), 1)
                cluster_p = n_larger / self.N

            # create cluster NDVar
            cpmap = np.ones(cmap.shape)  # cluster probability
        else:
            self.n_clusters = 0
            cpmap = None

        # expand clusters and find cluster properties
        if self.n_clusters:
            cmaps = np.empty((self.n_clusters,) + cmap.shape,
                             dtype=param_map.dtype)
            c_mask = self._bin_buff
            for i, cid in enumerate(cids):
                # cluster extent
                np.equal(cmap, cid, c_mask)
                # cluster value map
                np.multiply(param_map, c_mask, cmaps[i])
                if self.N:
                    if self.threshold is None:
                        cluster_p.append(cpmap[c_mask].min())
                    else:
                        cpmap[c_mask] = cluster_p[i]

            if self.N:
                ds['p'] = p = Var(cluster_p)
                ds['*'] = star_factor(p)

            # store cluster NDVar
            if self._nad_ax:
                cmaps = cmaps.swapaxes(1, self._nad_ax + 1)
            info = _cs.stat_info(self.meas, contours=param_contours,
                                 summary_func=np.sum)
            ds['cluster'] = NDVar(cmaps, dims=dims, info=info)

            # add cluster info
            for axis, dim in enumerate(dims[1:], 1):
                properties = dim._cluster_properties(cmaps, axis)
                if properties is not None:
                    ds.update(properties)

        # original parameter map
        info = _cs.stat_info(self.meas, contours=param_contours)
        if self._nad_ax:
            param_map = param_map.swapaxes(0, self._nad_ax)
        param_map_ = NDVar(param_map, dims[1:], info, self.name)

        # cluster probability map
        if cpmap is None:
            probability_map = None
            all_ = [[param_map_]]
        else:
            # revert to original shape
            if self._nad_ax:
                cpmap = cpmap.swapaxes(0, self._nad_ax)
            info = _cs.cluster_pmap_info()
            probability_map = NDVar(cpmap, dims[1:], info, self.name)
            all_ = [[param_map_, probability_map]]

        # remove memory buffers
        del self._bin_buff
        del self._int_buff
        if self.threshold is None:
            del self._float_buff
        elif self.tail == 0:
            del self._int_buff2
        if not self._all_adjacent:
            del self._slice_buff
            del self._bin_buff2
            del self._bin_buff3

        # store attributes
        self.clusters = ds
        self.parameter_map = param_map_
        self.tfce_map = tfce_map_
        self.probability_map = probability_map
        self.all = all_

        self.dt_perm = current_time() - self._t0

    def _label_clusters(self, pmap, out):
        """Find clusters on a statistical parameter map

        Parameters
        ----------
        pmap : array
            Statistical parameter map (non-adjacent dimension on the first
            axis).

        Returns
        -------
        cluster_map : array
            Array of same shape as pmap with clusters labeled.
        cluster_ids : tuple
            Identifiers of the clusters that survive the minimum duration
            criterion.
        """
        if self.tail >= 0:
            bin_map_above = np.greater(pmap, self.threshold, self._bin_buff)
            cmap, cids = self._label_clusters_binary(bin_map_above, out)

        if self.tail <= 0:
            bin_map_below = np.less(pmap, -self.threshold, self._bin_buff)
            if self.tail < 0:
                cmap, cids = self._label_clusters_binary(bin_map_below, out)
            else:
                cmap_l, cids_l = self._label_clusters_binary(bin_map_below,
                                                             self._int_buff2)
                x = int(cmap.max())  # apparently np.uint64 + int makes a float
                cmap_l[bin_map_below] += x
                cmap += cmap_l
                cids.update(c + x for c in cids_l)

        return cmap, tuple(cids)

    def _label_clusters_binary(self, bin_map, out):
        """
        Parameters
        ----------
        bin_map : array
            Binary map of where the parameter map exceeds the threshold for a
            cluster (non-adjacent dimension on the first axis).

        Returns
        -------
        cluster_map : array
            Array of same shape as bin_map with clusters labeled.
        cluster_ids : iterator over int
            Identifiers of the clusters that survive the selection criteria.
        """
        # find clusters
        n = ndimage.label(bin_map, self._struct, out)
        # n is 1 even when no cluster is found
        if n == 1 and out.max() == 0:
            n = 0
        cids = set(xrange(1, n + 1))
        if not self._all_adjacent:
            conn_src = self._connectivity_src
            conn_dst = self._connectivity_dst
            cidx = self._bin_buff2
            # reshape cluster map for iteration
            cmap_flat = out.reshape(self._flat_shape).swapaxes(0, 1)
            for cmap_slice in cmap_flat:
                if np.count_nonzero(np.unique(cmap_slice)) <= 1:
                    continue

                # find connectivity of True entries
                idx = cmap_slice.nonzero()[0]
                c_idx = np.in1d(conn_src, idx)
                c_idx *= np.in1d(conn_dst, idx)
                if np.count_nonzero(c_idx) == 0:
                    continue

                c_idx = c_idx.nonzero()[0]
                for i in c_idx:
                    # find corresponding cluster indices
                    id_src = cmap_slice[conn_src[i]]
                    id_dst = cmap_slice[conn_dst[i]]
                    if id_src == id_dst:
                        continue

                    # merge id_dst into id_src
                    np.equal(out, id_dst, cidx)
                    out[cidx] = id_src
                    cids.remove(id_dst)
                    if len(cids) == 1:
                        break
                if len(cids) == 1:
                    break

        # apply minimum cluster size criteria
        if self.criteria:
            for axes, v in self.criteria:
                rm = tuple(i for i in cids if
                           np.count_nonzero(np.equal(out, i).any(axes)) < v)
                cids.difference_update(rm)

        return out, cids

    def _param_repr(self, perm=False):
        "Repr fragment with clustering parameters"
        if perm:
            sampling = "permutations"
        elif self.N == 1:
            sampling = "sample"
        else:
            sampling = "samples"

        items = [", %i %s" % (self.N, sampling)]

        for item in self._criteria_arg.iteritems():
            items.append("%s=%s" % item)

        return ', '.join(items)

    def _tfce(self, p_map, out):
        dh = 0.1
        E = 0.5
        H = 2.0

        if self.tail <= 0:
            hs = np.arange(-dh, p_map.min(), -dh)
        if self.tail >= 0:
            upper = np.arange(dh, p_map.max(), dh)
            if self.tail == 0:
                hs = np.hstack((hs, upper))
            else:
                hs = upper

        # data buffers
        out.fill(0)
        bin_map = self._bin_buff
        cluster_map = self._int_buff

        # label clusters in slices at different heights
        # fill each cluster with total section value
        # each point's value is the vertical sum
        for h in hs:
            if h > 0:
                np.greater_equal(p_map, h, bin_map)
                h_factor = h ** H
            else:
                np.less_equal(p_map, h, bin_map)
                h_factor = (-h) ** H

            _, cluster_ids = self._label_clusters_binary(bin_map, cluster_map)
            for id_ in cluster_ids:
                np.equal(cluster_map, id_, bin_map)
                v = np.count_nonzero(bin_map) ** E * h_factor
                out[bin_map] += v

        return out

    def _uncrop(self, im, background=0):
        if self.crop:
            im_ = np.empty(self._uncropped_shape, dtype=im.dtype)
            im_[:] = background
            im_[self._crop_idx] = im
            return im_
        else:
            return im

    def add_original(self, param_map):
        """Add the original statistical parameter map.

        Parameters
        ----------
        param_map : array
            Parameter map of the statistic of interest (uncropped).
        """
        if self.has_original:
            raise RuntimeError("Original pmap already added")

        t0 = current_time()
        param_map = self._crop(param_map)
        if self._nad_ax:
            param_map = param_map.swapaxes(0, self._nad_ax)

        if self.threshold is None:
            self._tfce(param_map, self._original_cmap)
            self.n_clusters = True
        else:
            _, cids = self._label_clusters(param_map, self._original_cmap)
            self._cids = cids
            self.n_clusters = len(cids)

        self.has_original = True
        self.dt_original = current_time() - t0
        self._t0 = current_time()
        self._original_param_map = param_map
        if self.N == 0 or self.n_clusters == 0:
            self._finalize()

    def add_perm(self, pmap):
        """Add the statistical parameter map from permuted data.

        Parameters
        ----------
        pmap : array
            Parameter map of the statistic of interest.
        """
        self._i -= 1

        if self._nad_ax:
            pmap = pmap.swapaxes(0, self._nad_ax)

        if self.threshold is None:
            cmap = self._tfce(pmap, self._float_buff)
            self.dist[self._i] = cmap.max()
        else:
            cmap, cids = self._label_clusters(pmap, self._int_buff)
            if cids:
                clusters_v = ndimage.sum(pmap, cmap, cids)
                self.dist[self._i] = np.max(np.abs(clusters_v))

        if self._i == 0:
            self._finalize()
