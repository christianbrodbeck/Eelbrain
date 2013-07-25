'''Statistical tests for ndvars'''

from math import ceil

import numpy as np
import scipy.stats
from scipy.stats import percentileofscore
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_erosion, binary_dilation

from .. import fmtxt
from ..vessels.structure import celltable
from ..vessels import colorspaces as _cs
from ..vessels.data import ascategorial, asmodel, asndvar, asvar, ndvar

from ..test import glm as _glm
from ..test.test import _resample


__all__ = ['ttest', 'f_oneway', 'anova', 'cluster_anova', 'corr',
           'cluster_corr', 'clean_time_axis']
__test__ = False


def clean_time_axis(pmap, dtmin=0.02, below=None, above=None, null=0):
    """Clean a parameter map by requiring a threshold value for a minimum time
    window.

    Parameters
    ----------
    pmap : ndvar
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
    cleaned_map : ndvar
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
    cleaned = ndvar(x, pmap.dims, info, pmap.name)
    return cleaned



class corr:
    """
    Attributes
    ----------

    r : ndvar
        Correlation (with threshold contours).

    """
    def __init__(self, Y, X, norm=None, sub=None, ds=None,
                 contours={.05: (.8, .2, .0), .01: (1., .6, .0), .001: (1., 1., .0)}):
        """

        Y : ndvar
            Dependent variable.
        X : continuous | None
            The continuous predictor variable.
        norm : None | categorial
            Categories in which to normalize (z-score) X.

        """
        Y = asndvar(Y, sub=sub, ds=ds)
        X = asvar(X, sub=sub, ds=ds)

        if not Y.has_case:
            msg = ("Dependent variable needs case dimension")
            raise ValueError(msg)

        y = Y.x.reshape((len(Y), -1))
        if norm is not None:
            y = y.copy()
            for cell in norm.cells:
                idx = (norm == cell)
                y[idx] = scipy.stats.mstats.zscore(y[idx])

        n = len(X)
        x = X.x.reshape((n, -1))

        # covariance
        m_x = np.mean(x)
        if np.isnan(m_x):
            raise ValueError("np.mean(x) is nan")
        x -= m_x
        y -= np.mean(y, axis=0)
        cov = np.sum(x * y, axis=0) / (n - 1)

        # correlation
        r = cov / (np.std(x, axis=0) * np.std(y, axis=0))

        # p-value calculation
        # http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Inference
        pcont = {}
        df = n - 2
        for p, color in contours.iteritems():
            t = scipy.stats.distributions.t.isf(p, df)
            r_p = t / np.sqrt(n - 2 + t ** 2)
            pcont[r_p] = color
            pcont[-r_p] = color

        dims = Y.dims[1:]
        shape = Y.x.shape[1:]
        info = Y.info.copy()
        info.update(cmap='xpolar', vmax=1, contours=pcont)
        r = ndvar(r.reshape(shape), dims=dims, info=info)

        # store results
        self.name = "%s corr %s" % (Y.name, X.name)
        self.r = r
        self.all = r


class cluster_corr:
    """
    Attributes
    ----------

    r : ndvar
        Correlation (with threshold contours).

    """
    def __init__(self, Y, X, norm=None, sub=None, ds=None,
                 contours={.05: (.8, .2, .0), .01: (1., .6, .0), .001: (1., 1., .0)},
                 tp=.1, samples=1000, replacement=False,
                 tstart=None, tstop=None, close_time=0):
        """

        Y : ndvar
            Dependent variable.
        X : continuous | None
            The continuous predictor variable.
        norm : None | categorial
            Categories in which to normalize (z-score) X.

        """
        Y = asndvar(Y, sub=sub, ds=ds)
        X = asvar(X, sub=sub, ds=ds)

        self.name = name = "%s corr %s" % (Y.name, X.name)

        # calculate threshold
        # http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Inference
        self.n = n = len(X)
        df = n - 2
        tt = scipy.stats.distributions.t.isf(tp, df)
        tr = tt / np.sqrt(df + tt ** 2)

        cdist = ClusterDist(Y, samples, t_upper=tr, t_lower=-tr,
                            tstart=tstart, tstop=tstop, close_time=close_time,
                            meas='r', name=name)

        # normalization is done before the permutation b/c we are interested in the variance associated with each subject for the z-scoring.
        Y = Y.copy()
        Y.x = Y.x.reshape((n, -1))
        if norm is not None:
            for cell in norm.cells:
                idx = (norm == cell)
                Y.x[idx] = scipy.stats.mstats.zscore(Y.x[idx])

        x = X.x.reshape((n, -1))
        m_x = np.mean(x)
        if np.isnan(m_x):
            raise ValueError("np.mean(x) is nan")
        self.x = x - m_x

        for _, Yrs in _resample(Y, replacement=replacement, samples=samples):
            r = self._corr(Yrs)
            cdist.add_perm(r)

        r = self._corr(Y)
        cdist.add_original(r)

        self.r_map = cdist.P
        self.all = [[self.r_map] + cdist.clusters]
        self.clusters = cdist

    def _corr(self, Y):
        n = self.n
        x = self.x

        # covariance
        y = Y.x - np.mean(Y.x, axis=0)
        cov = np.sum(x * y, axis=0) / (n - 1)

        # correlation
        r = cov / (np.std(x, axis=0) * np.std(y, axis=0))
        return r

    def as_table(self, pmax=1.):
        table = self.clusters.as_table(pmax=pmax)
        return table


class ttest:
    """Element-wise t-test

    Attributes
    ----------
    all :
        c1, c0, [c0 - c1, P]
    p_val :
        [c0 - c1, P]
    """
    def __init__(self, Y='meg', X=None, c1=None, c0=0, match=None, sub=None,
                 ds=None):
        """Element-wise t-test

        Parameters
        ----------
        Y : ndvar
            Dependent variable.
        X : categorial | None
            Model; None if the grand average should be tested against a
            constant.
        c1 : str | None
            Test condition (cell of X).
        c0 : str | scalar
            Control condition (cell of X or constant to test against).
        match : factor
            Match cases for a repeated measures t-test.
        sub : index-array
            perform test with a subset of the data
        ds : dataset
            If a dataset is specified, all data-objects can be specified as
            names of dataset variables
        """
        ct = celltable(Y, X, match, sub, ds=ds)

        if len(ct) == 1:
            pass
        elif c1 is None:
            if len(ct) == 2:
                c1, c0 = ct.cell_labels()
            else:
                err = ("If X does not have exactly 2 categories (has %s), c1 and c0 "
                       "must be explicitly specified." % len(ct))
                raise ValueError(err)

        axis = ct.Y.get_axis('case')

        if isinstance(c0, (basestring, tuple)):  # two samples
            c1_mean = ct.data[c1].summary(name=str(c1))
            c0_mean = ct.data[c0].summary(name=str(c0))
            diff = c1_mean - c0_mean
            if match:
                if not ct.within[(c1, c0)]:
                    err = ("match kwarg: Conditions have different values on"
                           " <%r>" % ct.match.name)
                    raise ValueError(err)
                T, P = scipy.stats.ttest_rel(ct.data[c1].x, ct.data[c0].x,
                                             axis=axis)
                n = len(ct.data[c1])
                df = n - 1
                test_name = 'Related Samples t-Test'
            else:
                T, P = scipy.stats.ttest_ind(ct.data[c1].x, ct.data[c0].x,
                                             axis=axis)
                n1 = len(ct.data[c1])
                n0 = len(ct.data[c0])
                n = (n1, n0)
                df = n1 + n0 - 2
                test_name = 'Independent Samples t-Test'
        elif np.isscalar(c0):  # one sample
            c1_data = ct.data[c1]
            x = c1_data.x
            c1_mean = c1_data.summary()
            c0_mean = None

            # compute T and P
            if np.prod(x.shape) > 2 ** 25:
                ax = np.argmax(x.shape[1:]) + 1
                x = x.swapaxes(ax, 1)
                mod_len = x.shape[1]
                fix_shape = x.shape[0:1] + x.shape[2:]
                N = 2 ** 25 // np.prod(fix_shape)
                res = [scipy.stats.ttest_1samp(x[:, i:i + N], popmean=c0, axis=axis)
                       for i in xrange(0, mod_len, N)]
                T = np.vstack((v[0].swapaxes(ax, 1) for v in res))
                P = np.vstack((v[1].swapaxes(ax, 1) for v in res))
            else:
                T, P = scipy.stats.ttest_1samp(x, popmean=c0, axis=axis)

            n = len(c1_data)
            df = n - 1
            test_name = '1-Sample t-Test'
            if c0:
                diff = c1_mean - c0
            else:
                diff = c1_mean
        else:
            raise ValueError('invalid c0: %r. Must be string or scalar.' % c0)

        dims = ct.Y.dims[1:]

        info = _cs.set_info_cs(ct.Y.info, _cs.sig_info())
        info['test'] = test_name
        P = ndvar(P, dims, info=info, name='p')

        info = _cs.set_info_cs(ct.Y.info, _cs.default_info('T', vmin=0))
        T = ndvar(T, dims, info=info, name='T')

        # diff
        if np.any(diff < 0):
            diff.info['cmap'] = 'xpolar'

        # add Y.name to dataset name
        Yname = getattr(Y, 'name', None)
        if Yname:
            test_name = ' of '.join((test_name, Yname))

        # store attributes
        self.t = T
        self.p = P
        self.n = n
        self.df = df
        self.name = test_name
        self.c1_mean = c1_mean
        if c0_mean:
            self.c0_mean = c0_mean

        self.diff = diff
        self.p_val = [[diff, P]]

        if c0_mean:
            self.all = [c1_mean, c0_mean] + self.p_val
        elif c0:
            self.all = [c1_mean] + self.p_val
        else:
            self.all = self.p_val



class f_oneway:
    def __init__(self, Y='MEG', X='condition', sub=None, ds=None,
                 p=.05, contours={.01: '.5', .001: '0'}):
        """
        uses scipy.stats.f_oneway

        """
        Y = asndvar(Y, sub=sub, ds=ds)
        X = ascategorial(X, sub=sub, ds=ds)

        Ys = [Y[X == c] for c in X.cells]
        Ys = [y.x.reshape((y.x.shape[0], -1)) for y in Ys]
        N = Ys[0].shape[1]

        Ps = []
        for i in xrange(N):
            groups = (y[:, i] for y in Ys)
            F, p = scipy.stats.f_oneway(*groups)
            Ps.append(p)
        test_name = 'One-way ANOVA'

        dims = Y.dims[1:]
        Ps = np.reshape(Ps, tuple(len(dim) for dim in dims))

        info = _cs.set_info_cs(Y.info, _cs.sig_info(p, contours))
        info['test'] = test_name
        p = ndvar(Ps, dims, info=info, name=X.name)

        # store results
        self.name = "anova"
        self.p = p
        self.all = p



class anova:
    """
    Attributes
    ----------

    Y : ndvar
        Dependent variable.
    X : model
        Model.
    p_maps : {effect -> ndvar}
        Maps of p-values.
    all : [ndvar]
        List of all p-maps.

    """
    def __init__(self, Y='MEG', X='condition', sub=None, ds=None,
                 p=.05, contours={.01: '.5', .001: '0'}):
        self.name = "anova"
        Y = self.Y = asndvar(Y, sub=sub, ds=ds)
        X = self.X = asmodel(X, sub=sub, ds=ds)

        fitter = _glm.lm_fitter(X)

        info = _cs.set_info_cs(Y.info, _cs.sig_info(p, contours))
        kwargs = dict(dims=Y.dims[1:], info=info)

        self.all = []
        self.p_maps = {}
        for e, _, Ps in fitter.map(Y.x):
            name = e.name
            P = ndvar(Ps, name=name, **kwargs)
            self.all.append(P)
            self.p_maps[e] = P


class cluster_anova:
    """
    Attributes
    ----------

    Y : ndvar
        Dependent variable.
    X : model
        Model.

    all other attributes are dictionaries mapping effects from X.effects to
    results

    F_maps : {effect -> ndvar{
        Maps of F-values.

    """
    def __init__(self, Y, X, t=.1, samples=1000, replacement=False,
                 tstart=None, tstop=None, close_time=0, sub=None, ds=None):
        """ANOVA with cluster permutation test

        Parameters
        ----------
        Y : ndvar
            Measurements (dependent variable)
        X : categorial
            Model
        t : scalar
            Threshold (uncorrected p-value) to use for finding clusters
        samples : int
            Number of samples to estimate parameter distributions
        replacement : bool
            whether random samples should be drawn with replacement or
            without
        tstart, tstop : None | scalar
            Time window for clusters.
            **None**: use the whole epoch;
            **scalar**: use only a part of the epoch

            .. Note:: implementation is not optimal: F-values are still
                computed but ignored.

        close_time : scalar
            Close gaps in clusters that are smaller than this interval. Assumes
            that Y is a uniform time series.
        sub : index
            Apply analysis to a subset of cases in Y, X


        .. FIXME:: connectivity for >2 dimensional data. Currently, adjacent
            samples are connected.

        """
        Y = self.Y = asndvar(Y, sub=sub, ds=ds)
        X = self.X = asmodel(X, sub=sub, ds=ds)
        lm = _glm.lm_fitter(X)

        # get F-thresholds from p-threshold
        tF = {}
        if lm.full_model:
            for e in lm.E_MS:
                effects_d = lm.E_MS[e]
                if effects_d:
                    df_d = sum(ed.df for ed in effects_d)
                    tF[e] = scipy.stats.distributions.f.isf(t, e.df, df_d)
        else:
            df_d = X.df_error
            tF = {e: scipy.stats.distributions.f.isf(t, e.df, df_d) for e in X.effects}

        # Estimate statistic distributions from permuted Ys
        kwargs = dict(tstart=tstart, tstop=tstop, close_time=close_time, meas='F')
        dists = {e: ClusterDist(Y, samples, tF[e], name=e.name, **kwargs) for e in tF}
        self.cluster_dists = dists
        for _, Yrs in _resample(Y, replacement=replacement, samples=samples):
            for e, F in lm.map(Yrs.x, p=False):
                dists[e].add_perm(F)

        # Find clusters in the actual data
        test0 = lm.map(Y.x, p=False)
        self.effects = []
        self.clusters = {}
        self.F_maps = {}
        for e, F in test0:
            self.effects.append(e)
            dist = dists[e]
            dist.add_original(F)
            self.clusters[e] = dist
            self.F_maps[e] = dist.P

        self.name = "ANOVA Permutation Cluster Test"
        self.tF = tF

        self.all = [[self.F_maps[e]] + self.clusters[e].clusters
                    for e in self.X.effects if e in self.F_maps]

    def as_table(self, pmax=1.):
        tables = []
        for e in self.effects:
            dist = self.cluster_dists[e]
            table = dist.as_table(pmax=pmax)
            table.title(e.name)
            tables.append(table)
        return tables



class ClusterDist:
    def __init__(self, Y, N, t_upper, t_lower=None,
                 tstart=None, tstop=None, close_time=0, meas='?', name=None):
        """
        Parameters
        ----------
        Y : ndvar
            Dependent variable.
        N : int
            Number of permutations.
        t_upper, t_lower : None | scalar
            Positive and negative thresholds for finding clusters. If None,
            no clusters with the corresponding sign are counted.
        tstart, tstop : None | scalar
            Time window for clusters.
            **None**: use the whole epoch;
            **scalar**: use only a part of the epoch

            .. Note:: implementation is not optimal: F-values are still
                computed but ignored.

        close_time : scalar
            Close gaps in clusters that are smaller than this interval. Assumes
            that Y is a uniform time series.
        unit : str
            Label for the parameter.
        cs : None | dict
            Plotting parameters for info dict.
        """
        if t_lower is not None:
            if t_lower >= 0:
                raise ValueError("t_lower needs to be < 0; is %s" % t_lower)
        if t_upper is not None:
            if t_upper <= 0:
                raise ValueError("t_upper needs to be > 0; is %s" % t_upper)
        if (t_lower is not None) and (t_upper is not None):
            if t_lower != -t_upper:
                err = ("If t_upper and t_lower are defined, t_upp has to be "
                       "-t_lower")
                raise ValueError(err)

        # make sure we only get case by time data
        assert Y.ndim == 2
        assert Y.has_case
        assert Y.get_axis('time') == 1
        self._time_ax = Y.get_axis('time') - 1
        self.dims = Y.dims[1:]

        # prepare cluster merging
        if close_time:
            time = Y.get_dim('time')
            self.close_time_structure = np.ones(round(close_time / time.tstep))
        self.close_time = bool(close_time)

        # prepare morphology manipulation
        self.delim = (tstart is not None) or (tstop is not None)
        if self.delim:
            time = Y.get_dim('time')
            self.delim_idx = np.zeros(len(time), dtype=bool)
            if tstart is not None:
                self.delim_idx[time.times < tstart] = True
            if tstop is not None:
                self.delim_idx[time.times >= tstop] = True

        self.dist = np.zeros(N)
        self._i = 0
        self.t_upper = t_upper
        self.t_lower = t_lower
        self.meas = meas
        self.name = name

    def _find_clusters(self, P):
        "returns (clusters, n)"
        if self.t_upper is None:
            cmap_upper = None
        else:
            cmap_upper = (P > self.t_upper)
            clusters, n = self._find_clusters_1tailed(cmap_upper)

        if self.t_lower is not None:
            cmap_lower = (P < self.t_lower)
            if cmap_upper is None:
                clusters, n = self._find_clusters_1tailed(cmap_lower)
            else:
                clusters_l, n_l = self._find_clusters_1tailed(cmap_lower)
                clusters_l[cmap_lower] += n
                clusters += clusters_l
                n += n_l

        return clusters, n

    def _find_clusters_1tailed(self, cmap):
        "returns (clusters, n)"
        # manipulate morphology
        if self.delim:
            cmap[self.delim_idx] = False
        if self.close_time:
            cmap = cmap | binary_closing(cmap, self.close_time_structure)

        # find clusters
        return ndimage.label(cmap)

    def add_original(self, P):
        """
        P : array
            Parameter map of the statistic of interest.

        """
        self.clusters = []

        # find clusters
        clusters, n = self._find_clusters(P)
        clusters_v = ndimage.sum(P, clusters, xrange(1, n + 1))

        for i in xrange(n):
            v = clusters_v[i]
            p = 1 - percentileofscore(self.dist, np.abs(v), 'mean') / 100
            im = P * (clusters == i + 1)
            name = 'p=%.3f' % p
            threshold = self.t_upper if (v > 0) else self.t_lower
            info = _cs.cluster_info(self.meas, threshold, p)
            ndv = ndvar(im, dims=self.dims, name=name, info=info)
            self.clusters.append(ndv)

        contours = {self.t_lower: (0.7, 0, 0.7), self.t_upper: (0.7, 0.7, 0)}
        info = _cs.stat_info(self.meas, contours=contours)
        self.P = ndvar(P, dims=self.dims, name=self.name, info=info)

    def add_perm(self, P):
        """
        P : array
            Parameter map of the statistic of interest.

        """
        clusters, n = self._find_clusters(P)

        if n:
            clusters_v = ndimage.sum(P, clusters, xrange(1, n + 1))
            self.dist[self._i] = np.max(np.abs(clusters_v))

        self._i += 1

    def as_table(self, pmax=1.):
        cols = 'll'
        headings = ('#', 'p')
        if self._time_ax is not None:
            time = self.dims[self._time_ax]
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

                if self._time_ax is not None:
                    nz = c.x.nonzero()[self._time_ax]
                    tstart = time[nz.min()]
                    tstop = time[nz.max()]
                    interval = '%.3f - %.3f s' % (tstart, tstop)
                    table.cell(interval)

        return table
