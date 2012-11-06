'''
Statistical tests for ndvar objects.

Tests are defined as classes that provide aspects of their results as
attributes and methods::

    >>> res = testnd.ttest(Y, X, 'test', 'control')
    >>> res.p  # an ndvar object with an uncorrected p-value for each sample

Test result objects can be directly submitted to plotting functions. To plot
only part of the results, specific attributes can be submitted (for a
description of the attributes see the relevant class documentation)::

    >>> plot.uts.uts(res)  # plots values in both conditions as well as
    ... difference values with p-value thresholds
    >>> plot.uts.uts(res.p)  # plots only p-values

The way this is implemented is that plotting functions test for the presence
of a ``._default_plot_obj`` and a ``.all`` attribute (in that order) which
is expected to provide a default object for plotting. This is implemented in
:py:mod:`plot._base.unpack_epochs_arg`.





Created on Feb 22, 2012

@author: christian
'''

import numpy as np
import scipy.stats
import scipy.ndimage

from eelbrain import fmtxt
from eelbrain import vessels as _vsl
import eelbrain.vessels.colorspaces as _cs
from eelbrain.vessels.data import ascategorial, asmodel, asndvar, asvar, ndvar

import glm as _glm
from test import _resample


__all__ = ['anova', 'cluster_anova', 'cluster_corr', 'corr', 'ttest',
           'f_oneway']



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
        r_ps = {}
        df = n - 2
        for p, color in contours.iteritems():
            t = scipy.stats.distributions.t.isf(p, df)
            r_p = t / np.sqrt(n - 2 + t ** 2)
            pcont[r_p] = color
            r_ps[r_p] = p
            pcont[-r_p] = color
            r_ps[-r_p] = p

        dims = Y.dims[1:]
        shape = Y.x.shape[1:]
        properties = Y.properties.copy()
        cs = _cs.Colorspace(cmap=_cs.cm_xpolar, vmax=1, vmin= -1, contours=pcont, ps=r_ps)
        properties['colorspace'] = cs
        r = ndvar(r.reshape(shape), dims=dims, properties=properties)

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
                 tstart=None, tstop=None, close_time=0, pmax=1):
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

        cs = _cs.Colorspace(cmap=_cs.cm_xpolar, vmax=1, vmin= -1)
        cdist = cluster_dist(Y, N=samples, threshold=tr, tstart=tstart,
                             tstop=tstop, close_time=close_time, unit='r',
                             pmax=pmax, name=name, cs=cs)

        # store Y properties before manipulating it
        dims = Y.dims[1:]
        shape = Y.x.shape[1:]

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

    def _corr(self, Y):
        n = self.n
        x = self.x

        # covariance
        y = Y.x - np.mean(Y.x, axis=0)
        cov = np.sum(x * y, axis=0) / (n - 1)

        # correlation
        r = cov / (np.std(x, axis=0) * np.std(y, axis=0))
        return r



class ttest:
    """
    **Attributes:**

    all :
        c1, c0, [c0 - c1, P]
    diff :
        [c0 - c1, P]

    """
    def __init__(self, Y='MEG', X=None, c1=None, c0=0,
                 match=None, sub=None, ds=None,
                 contours={.05: (.8, .2, .0), .01: (1., .6, .0), .001: (1., 1., .0)}):
        """

        Y : var
            dependent variable
        X : categorial | None
            Model; None if the grand average should be tested against a
            constant.
        c1 : str | None
            Test condition (cell of X)
        c0 : str | scalar
            Control condition (cell of X or constant to test against)
        match : factor
            Match cases for a repeated measures t-test
        sub : index-array
            perform test with a subset of the data
        ds : dataset
            If a dataset is specified, all data-objects can be specified as
            names of dataset variables

        """
#        contours = { .05: (.8, .2, .0),  .01: (1., .6, .0),  .001: (1., 1., .0),
#                    -.05: (0., .2, 1.), -.01: (.4, .8, 1.), -.001: (.5, 1., 1.),
#                    }
#                    (currently, p values are not directional)
        ct = _vsl.structure.celltable(Y, X, match, sub, ds=ds)

        if len(ct) == 1:
            pass
        elif c1 is None:
            if len(ct) == 2:
                c1, c0 = ct.cell_labels()
            else:
                err = ("If X does not have exactly 2 categories (has %s), c1 and c0 "
                       "must be explicitly specified." % len(ct))
                raise ValueError(err)


        if isinstance(c0, basestring):
            c1_mean = ct.data[c1].summary(name=c1)
            c0_mean = ct.data[c0].summary(name=c0)
            diff = c1_mean - c0_mean
            if match:
                if not ct.within[(c1, c0)]:
                    err = ("match kwarg: Conditions have different values on"
                           " <%r>" % ct.match.name)
                    raise ValueError(err)
                T, P = scipy.stats.ttest_rel(ct.data[c1].x, ct.data[c0].x, axis=0)
                test_name = 'Related Samples t-Test'
            else:
                T, P = scipy.stats.ttest_ind(ct.data[c1].x, ct.data[c0].x, axis=0)
                test_name = 'Independent Samples t-Test'
        elif np.isscalar(c0):
            c1_data = ct.data[str(Y.name)]
            c1_mean = c1_data.summary()
            c0_mean = None
            T, P = scipy.stats.ttest_1samp(c1_data.x, popmean=c0, axis=0)
            test_name = '1-Sample t-Test'
            if c0:
                diff = c1_mean - c0
            else:
                diff = None
        else:
            raise ValueError('invalid c0: %r' % c0)

#        direction = np.sign(diff.x)
#        P = P * direction# + 1 # (1 - P)
#        for k in contours.copy():
#            contours[k+1] = contours.pop(k)

        dims = ct.Y.dims[1:]
        properties = ct.Y.properties.copy()

        properties['colorspace'] = _vsl.colorspaces.Colorspace(contours=contours)
        properties['test'] = test_name
        P = _vsl.data.ndvar(P, dims, properties=properties, name='p')

        properties['colorspace'] = _vsl.colorspaces.get_default()
        T = _vsl.data.ndvar(T, dims, properties=properties, name='T')

        # add Y.name to dataset name
        Yname = getattr(Y, 'name', None)
        if Yname:
            test_name = ' of '.join((test_name, Yname))

        # store attributes
        self.t = T
        self.p = P
        self.name = test_name
        self.c1_mean = c1_mean
        if c0_mean:
            self.c0_mean = c0_mean

        if diff:
            self.diff = diff
            self.p_val = [[diff, P]]
        else:
            self.p_val = [[c1_mean, P]]

        if c0_mean:
            self.all = [c1_mean, c0_mean] + self.p_val
        elif diff:
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

        properties = Y.properties.copy()
        properties['colorspace'] = _vsl.colorspaces.get_sig(p=p, contours=contours)
        properties['test'] = test_name
        p = ndvar(Ps, dims, properties=properties, name=X.name)

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

        properties = Y.properties.copy()
        properties['colorspace'] = _vsl.colorspaces.get_sig(p=p, contours=contours)
        kwargs = dict(dims=Y.dims[1:], properties=properties)

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
                 tstart=None, tstop=None, close_time=0,
                 pmax=1, sub=None, ds=None,
                 ):
        """

        Arguments
        ---------

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

        pmax : scalar <= 1
            Maximum cluster p-values to keep cluster.


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
        kwargs = dict(tstart=tstart, tstop=tstop, close_time=close_time, unit='F')
        dists = {e: cluster_dist(Y, samples, tF[e], name=e.name, **kwargs) for e in tF}
        for _, Yrs in _resample(Y, replacement=replacement, samples=samples):
            for e, F in lm.map(Yrs.x, p=False):
                dists[e].add_perm(F)

        # Find clusters in the actual data
        test0 = lm.map(Y.x, p=False)
        self.clusters = {}
        self.F_maps = {}
        for e, F in test0:
            dist = dists[e]
            dist.add_original(F)
            self.clusters[e] = dist.clusters
            self.F_maps[e] = dist.P

        self.name = "ANOVA Permutation Cluster Test"
        self.tF = tF

        self.all = [[self.F_maps[e]] + self.clusters[e] for e in self.X.effects
                    if e in self.F_maps]

    def as_table(self, pmax=1.):
        table = fmtxt.Table('ll')
        for e in self.X.effects:
            if e in self.F_maps:
                table.cell(e.name, width=2)
                cs = self.clusters[e]
                ps = [c.properties['p'] for c in cs]
                for i in np.argsort(ps):
                    c = cs[i]
                    p = c.properties['p']
                    interval = np.nonzero(c.x)[0]
                    c_start = format(c.time[interval[0]], '1.3f')
                    c_stop = format(c.time[interval[-1]], '1.3f')
                    if p > pmax:
                        break
                    table.cell(i)
                    table.cell((c_start,c_stop))
                    table.cell(p)

        return table



class cluster_dist:
    def __init__(self, Y, N, threshold, tstart=None, tstop=None, close_time=0, unit='T', cs=None, pmax=.5, name=None):
        """
        Y : ndvar
            Dependent variable.

        N : int
            Number of permutations.

        threshold : scalar
            Threshold for finding clusters.

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

        pmax : scalar
            For the original data, only retain clusters with a p-value
            smaller than pmax.

        """
        # make sure we only get case by time data
        assert Y.ndims == 2
        assert Y.has_case
        assert Y.get_axis('time') == 1
        self.dims = Y.dims[1:]

        # prepare cluster merging
        if close_time:
            T = Y.get_dim('time')
            dT = np.mean(np.diff(T.x))
            self.close_time_structure = np.ones(round(close_time / dT))
        self.close_time = bool(close_time)

        # prepare morphology manipulation
        self.delim = (tstart is not None) or (tstop is not None)
        if self.delim:
            T = Y.get_dim('time')
            self.delim_idx = np.zeros(len(T), dtype=bool)
            if tstart is not None:
                self.delim_idx[T < tstart] = True
            if tstop is not None:
                self.delim_idx[T >= tstop] = True

        self.dist = np.zeros(N)
        self._i = 0
        self.threshold = threshold
        self.unit = unit
        self.pmax = pmax
        self.name = name
        self.cs = cs

    def _find_clusters(self, P):
        "returns (clusters, n)"
        cmap = (P > self.threshold)

        # manipulate morphology
        if self.delim:
            cmap[self.delim_idx] = False
        if self.close_time:
            cmap = cmap | scipy.ndimage.binary_closing(cmap, self.close_time_structure)

        # find clusters
        return scipy.ndimage.label(cmap)

    def add_original(self, P):
        """
        P : array
            Parameter map of the statistic of interest.

        """
        self.clusters = []

        # find clusters
        clusters, n = self._find_clusters(P)
        clusters_v = scipy.ndimage.measurements.sum(P, clusters, xrange(1, n + 1))

        for i in xrange(n):
            v = clusters_v[i]
            p = 1 - scipy.stats.percentileofscore(self.dist, v, 'mean') / 100
            if p <= self.pmax:
                im = P * (clusters == i + 1)
                name = 'p=%.3f' % p
                properties = {'p': p, 'threshold': self.threshold, 'unit': self.unit}
                ndv = ndvar(im, dims=self.dims, name=name, properties=properties)
                self.clusters.append(ndv)

        props = {'tF': self.clusters, 'unit': self.unit, 'cs': self.cs}
        self.P = ndvar(P, dims=self.dims, name=self.name, properties=props)

    def add_perm(self, P):
        """
        P : array
            Parameter map of the statistic of interest.

        """
        clusters, n = self._find_clusters(P)

        if n:
            clusters_v = scipy.ndimage.measurements.sum(P, clusters, xrange(1, n + 1))
            self.dist[self._i] = max(clusters_v)

        self._i += 1







# - not functional - ------
def _test(ndvars, parametric=True, match=None, func=None, attr='data',
         name="{test}"):
    """
    use func (func) or attr (str) to customize data
    (func=abs for )

    """
    raise NotImplementedError
    if match is None:
        related = False
    else:
        raise NotImplementedError

    v0 = ndvars[0]

    # data
    data = [getattr(v, attr) for v in ndvars]
    if func != None:
        data = [func(d) for d in data]

    # test
    k = len(ndvars)  # number of levels
    if k == 0:
        raise ValueError("no segments provided")

    # perform test
    if parametric:  # simple tests
        if k == 1:
            statistic = 't'
            T, P = scipy.stats.ttest_1samp(*data, popmean=0, axis=0)
            test_name = '1-Sample $t$-Test'
        elif k == 2:
            statistic = 't'
            if related:
                T, P = scipy.stats.ttest_rel(*data, axis=0)
                test_name = 'Related Samples $t$-Test'
            else:
                T, P = scipy.stats.ttest_ind(*data, axis=0)
                test_name = 'Independent Samples $t$-Test'
        else:
            statistic = 'F'
            raise NotImplementedError("Use segframe for 1-way ANOVA")

    else:  # non-parametric:
        raise NotImplementedError("axis from -1 to 0")
        if k <= 2:
            if related:
                test_func = scipy.stats.wilcoxon
                statistic = 't'
                test_name = 'Wilcoxon'
            else:
                raise NotImplementedError()
        else:
            if related:
                test_func = scipy.stats.friedmanchisquare
                statistic = 'Chi**2'
                test_name = 'Friedman'
            else:
                raise NotImplementedError()

        shape = data[0].shape[:-1]
        # function to apply stat test to array
        def testField(*args):
            """
            will be executed for a grid, args will contain coordinates of shape
            assumes that subjects are on last axis

            """
            rargs = [np.ravel(a) for a in args]
            T = []
            P = []
            for indexes in zip(*rargs):
                index = tuple([slice(int(i), int(i + 1)) for i in indexes] + [slice(0, None)])
                testArgs = tuple([ d[index].ravel() for d in data])
                t, p = test_func(*testArgs)
                T.append(t)
                P.append(p)
            P = np.array(P).reshape(args[0].shape)
            T = np.array(T).reshape(args[0].shape)
            return T, P
        T, P = np.fromfunction(testField, shape)

    # Direction of the effect
    if len(data) == 2:
        direction = np.sign(data[0].mean(0) - data[1].mean(0))
        P = (1 - P) * direction
        cs = _vsl.colorspaces.get_symsig
    elif len(data) == 1:
        direction = np.sign(data[0].mean(0))
        cs = _vsl.colorspaces.get_symsig
        P = (1 - P) * direction
    else:
        cs = _vsl.colorspaces.get_sig

    properties = ndvars[0].properties.copy()
    properties['colorspace_func'] = cs
    properties['statistic'] = statistic
    properties['test'] = test_name

    # create test_segment
    name_fmt = name.format(**properties)
    stat = _vsl.data.epoch(v0.dims, T, properties=properties, name=name_fmt)
    name_fmt = 'P'
    P = _vsl.data.epoch(v0.dims, P, properties=properties, name=name_fmt)
    return stat, P
