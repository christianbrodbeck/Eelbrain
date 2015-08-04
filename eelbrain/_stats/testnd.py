'''Statistical tests for NDVars

Common Attributes
-----------------

The following attributes are always present. For ANOVA, they are lists with the
corresponding items for different effects.

t/f/... : NDVar
    Map of the statistical parameter.
p_uncorrected : NDVar
    Map of uncorrected p values.
p : NDVar | None
    Map of corrected p values (None if no correct was applied).
clusters : Dataset | None
    Table of all the clusters found (None if no clusters were found, or if no
    clustering was performed).
n_samples : None | int
    The actual number of permutations. If ``samples = -1``, i.e. a complete set
    or permutations is performed, then ``n_samples`` indicates the actual
    number of permutations that constitute the complete set.
'''
from __future__ import division

from datetime import datetime, timedelta
from itertools import chain, izip
from math import ceil
from multiprocessing import Process, cpu_count
from multiprocessing.queues import SimpleQueue
from multiprocessing.sharedctypes import RawArray
import operator
import re
import socket
from time import time as current_time

import numpy as np
import scipy.stats
from scipy import ndimage

from .. import fmtxt
from .. import _colorspaces as _cs
from .._data_obj import (ascategorial, asmodel, asndvar, asvar, assub, Dataset,
                         NDVar, Var, Celltable, cellname, combine, Categorial,
                         UTS)
from .._report import enumeration, format_timewindow, ms
from .._utils import logger, LazyProperty
from .._utils.numpy_utils import full_slice
from . import opt, stats
from .glm import _nd_anova
from .opt import merge_labels
from .permutation import _resample_params, permute_order, permute_sign_flip
from .t_contrast import TContrastRel
from .test import star_factor


__test__ = False

# toggle multiprocessing for _ClusterDist
MULTIPROCESSING = 1


class _Result(object):
    """Baseclass for testnd test results

    Attributes
    ----------
    p : NDVar | None
        Map of p-values corrected for multiple comparison (or None if no
        correction was performed).
    tfce_map : NDVar | None
        Map of the test statistic processed with the threshold-free cluster
        enhancement algorithm (or None if no TFCE was performed).
    """
    _state_common = ('Y', 'match', 'sub', 'samples', 'tfce', 'pmin', '_cdist',
                     'tstart', 'tstop', '_dims')
    _state_specific = ()

    @property
    def _attributes(self):
        return self._state_common + self._state_specific

    def __init__(self, Y, match, sub, samples, tfce, pmin, cdist, tstart,
                 tstop):
        self.Y = Y.name
        if match:
            self.match = match.name
        else:
            self.match = None
        self.sub = sub
        self.samples = samples
        self.tfce = tfce
        self.pmin = pmin
        self._cdist = cdist
        self.tstart = tstart
        self.tstop = tstop
        self._dims = Y.dims[1:]

    def __getstate__(self):
        state = {name: getattr(self, name, None) for name in self._attributes}
        return state

    def __setstate__(self, state):
        for k, v in state.iteritems():
            setattr(self, k, v)

        # backwards compatibility:
        if 'tstart' not in state:
            cdist = self._first_cdist
            self.tstart = cdist.tstart
            self.tstop = cdist.tstop
        if '_dims' not in state:  # 0.17
            if 't' in state:
                self._dims = state['t'].dims
            elif 'r' in state:
                self._dims = state['r'].dims
            elif 'f' in state:
                self._dims = state['f'][0].dims
            else:
                raise RuntimeError("Error recovering old test results dims")

        self._expand_state()

    def __repr__(self):
        temp = "<%s %%s>" % self.__class__.__name__

        args = self._repr_test_args()
        if self.sub:
            args.append(', sub=%r' % self.sub)
        if self._cdist:
            args += self._cdist._repr_test_args(self.pmin)
            args += self._cdist._repr_clusters()

        out = temp % ', '.join(args)
        return out

    def _repr_test_args(self):
        """List of strings describing parameters unique to the test, to be
        joined by comma
        """
        raise NotImplementedError()

    def _expand_state(self):
        "override to create secondary results"
        cdist = self._cdist
        if cdist is None:
            self.tfce_map = None
            self.p = None
            self._kind = None
        else:
            self.tfce_map = cdist.tfce_map
            self.p = cdist.probability_map
            self._kind = cdist.kind

    def _iter_cdists(self):
        yield (None, self._cdist)

    @property
    def _first_cdist(self):
        return self._cdist

    def _plot_model(self):
        "Determine x for plotting categories"
        return None

    def _plot_sub(self):
        if isinstance(self.sub, basestring) and self.sub == "<unsaved array>":
            raise RuntimeError("The sub parameter was not saved for previous "
                               "versions of Eelbrain. Please recompute this "
                               "result with the current version.")
        return self.sub

    def masked_parameter_map(self, pmin=0.05, **sub):
        """Create a copy of the parameter map masked by significance

        Parameters
        ----------
        pmin : None | scalar
            Threshold p-value for masking (default 0.05). For threshold-based
            cluster tests, pmin=None includes all clusters regardless of their
            p-value.

        Returns
        -------
        masked_map : NDVar
            NDVar with data from the original parameter map wherever p <= pmin
            and 0 everywhere else.
        """
        if self._cdist is None:
            err = "Method only applies to results with samples > 0"
            raise RuntimeError(err)
        return self._cdist.masked_parameter_map(pmin, **sub)

    def cluster(self, cluster_id):
        """Retrieve a specific cluster as NDVar

        Parameters
        ----------
        cluster_id : int
            Cluster id.

        Returns
        -------
        cluster : NDVar
            NDVar of the cluster, 0 outside the cluster.

        Notes
        -----
        Clusters only have stable ids for thresholded cluster distributions.
        """
        if self._cdist is None:
            err = "Method only applies to results with samples > 0"
            raise RuntimeError(err)
        return self._cdist.cluster(cluster_id)

    @LazyProperty
    def clusters(self):
        if self._cdist is None:
            return None
        else:
            return self.find_clusters(None, True)

    def find_clusters(self, pmin=None, maps=False, **sub):
        """Find significant regions as clusters

        Parameters
        ----------
        pmin : None | scalar, 1 >= p  >= 0
            Threshold p-value for clusters (for thresholded cluster tests the
            default is 1, for others 0.05).
        maps : bool
            Include in the output a map of every cluster (can be memory
            intensive if there are large statistical maps and/or many
            clusters; default False).

        Returns
        -------
        ds : Dataset
            Dataset with information about the clusters.
        """
        if self._cdist is None:
            err = ("Test results have no clustering (set samples to an int "
                   " >= 0 to find clusters")
            raise RuntimeError(err)
        return self._cdist.clusters(pmin, maps, **sub)

    def find_peaks(self):
        """Find peaks in a threshold-free cluster distribution

        Returns
        -------
        ds : Dataset
            Dataset with information about the peaks.
        """
        if self._cdist is None:
            err = "Method only applies to results with samples > 0"
            raise RuntimeError(err)
        return self._cdist.find_peaks()

    def compute_probability_map(self, **sub):
        """Compute a probability map

        Returns
        -------
        probability : NDVar
            Map of p-values.
        """
        if self._cdist is None:
            err = "Method only applies to results with samples > 0"
            raise RuntimeError(err)
        return self._cdist.compute_probability_map(**sub)

    def info_list(self, computation=True):
        "List with information about the test"
        out = fmtxt.List("Mass-univariate statistics:")
        out.add_item(self._name())
        dimnames = [dim.name for dim in self._dims]
        dimlist = out.add_sublist("Over %s" % enumeration(dimnames))
        if 'time' in dimnames:
            dimlist.add_item("Time interval: %s." % format_timewindow(self))

        cdist = self._first_cdist
        if cdist is None:
            out.add_item("No inferential statistics")
            return out

        # inference
        l = out.add_sublist("Inference:")
        if cdist.kind == 'raw':
            l.add_item("Based on maximum statistic")
        elif cdist.kind == 'tfce':
            l.add_item("Based on maximum statistic with threshold-"
                       "free cluster enhancement (Smith & Nichols, 2009)")
        elif cdist.kind == 'cluster':
            l.add_item("Based on maximum cluster sum statistic")
            sl = l.add_sublist("Cluster criteria:")
            for dim in dimnames:
                if dim == 'time':
                    sl.add_item("Minimum cluster duration:  %s ms"
                                % ms(cdist.criteria.get('mintime', 0)))
                elif dim == 'source':
                    sl.add_item("At least %s contiguous sources."
                                % cdist.criteria.get('minsource', 0))
                elif dim == 'sensor':
                    sl.add_item("At least %s contiguous sensors."
                                % cdist.criteria.get('minsensor', 0))
                else:
                    sl.add_item("Minimum number of contiguous elements in %s: "
                                "%s"
                                % (dim, cdist.criteria.get("min%s" % dim, 0)))
        # n samples
        if self.samples == -1:
            l.add_item("In all %s possible permutations" % self.n_samples)
        else:
            l.add_item("In %s random permutations" % self.samples)

        # computation
        if computation:
            out.add_item(cdist.info_list())

        return out

    @property
    def n_samples(self):
        if self.samples == -1:
            return self._first_cdist.samples
        else:
            return self.samples

    @property
    def _time_dim(self):
        for dim in self._first_cdist.dims:
            if isinstance(dim, UTS):
                return dim
        return None


class t_contrast_rel(_Result):
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
        Threshold for forming clusters:  use a t-value equivalent to an
        uncorrected p-value for a related samples t-test (with df =
        len(match.cells) - 1).
    tmin : None | scalar
        Threshold for forming clusters.
    tfce : bool
        Use threshold-free cluster enhancement (Smith & Nichols, 2009).
        Default is False.
    tstart, tstop : None | scalar
        Restrict time window for permutation cluster test.
    mintime : scalar
        Minimum duration for clusters (in seconds).
    minsource : int
        Minimum number of sources per cluster.

    Notes
    -----
    Contrast definitions can contain:

    - Comparisons using ">" or "<" and data cells,
      e.g. ``"cell1 > cell0"``. If the data is defined based on an
      interaction, cells are specified with "|",
      e.g. ``"a1 | b0 > a0 | b0"``.
    - Unary numpy functions ``abs`` and ``negative``, e.g.
      ``"abs(cell1 > cell0)"``.
    - Binary numpy functions ``subtract`` and ``add``, e.g.
      ``"add(a>b, a>c)"``
    - Numpy functions for multiple arrays ``min``, ``max`` and ``sum``,
      e.g. ``min(a>d, b>d, c>d)``.

    Examples
    --------
    To find cluster where both of two pairwise comparisons are reliable,
    i.e. an intersection of two effects, one could use
    ``"min(a > c, b > c)"``.

    To find a specific kind of interaction, where a is greater than b, and
    this difference is greater than the difference between c and d, one
    could use ``"subtract(a > b, abs(c > d))"``.
    """

    _state_specific = ('X', 'contrast', 't', 'tail')

    def __init__(self, Y, X, contrast, match=None, sub=None, ds=None, tail=0,
                 samples=None, pmin=None, tmin=None, tfce=False, tstart=None,
                 tstop=None, dist_dim=(), parc=(), dist_tstep=None,
                 **criteria):
        ct = Celltable(Y, X, match, sub, ds=ds, coercion=asndvar)

        # setup contrast
        t_contrast = TContrastRel(contrast, ct.cells, ct.data_indexes)

        # original data
        tmap = t_contrast.map(ct.Y.x)
        t = NDVar(tmap, ct.Y.dims[1:], {}, 't')

        if samples is None:
            cdist = None
        else:
            # threshold
            if sum((pmin is not None, tmin is not None, tfce)) > 1:
                msg = "Only one of pmin, tmin and tfce can be specified"
                raise ValueError(msg)
            elif pmin is not None:
                df = len(ct.match.cells) - 1
                threshold = stats.ttest_t(pmin, df, tail)
            elif tmin is not None:
                threshold = abs(tmin)
            elif tfce:
                threshold = 'tfce'
            else:
                threshold = None

            cdist = _ClusterDist(ct.Y, samples, threshold, tail, 't',
                                 "t-contrast", tstart, tstop, criteria,
                                 dist_dim, parc, dist_tstep)
            cdist.add_original(tmap)
            if cdist.do_permutation:
                iterator = permute_order(len(ct.Y), samples, unit=ct.match)
                run_permutation(t_contrast, cdist, iterator)

        # store attributes
        _Result.__init__(self, ct.Y, ct.match, sub, samples, tfce, pmin, cdist,
                         tstart, tstop)
        self.X = ct.X.name
        self.contrast = contrast
        self.tail = tail
        self.tmin = tmin
        self.t = t

        self._expand_state()

    def _name(self):
        if self.Y:
            return "T-Contrast:  %s ~ %s" % (self.Y, self.contrast)
        else:
            return "T-Contrast:  %s" % self.contrast

    def _plot_model(self):
        return self.X

    def _repr_test_args(self):
        args = [repr(self.Y), repr(self.X), repr(self.contrast)]
        if self.tail:
            args.append("tail=%r" % self.tail)
        if self.match:
            args.append('match=%r' % self.match)
        return args


class corr(_Result):
    """Correlation

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
        Threshold for forming clusters:  use an r-value equivalent to an
        uncorrected p-value.
    rmin : None | scalar
        Threshold for forming clusters.
    tfce : bool
        Use threshold-free cluster enhancement (Smith & Nichols, 2009).
        Default is False.
    tstart, tstop : None | scalar
        Restrict time window for permutation cluster test.
    match : None | categorial
        When permuting data, only shuffle the cases within the categories
        of match.
    mintime : scalar
        Minimum duration for clusters (in seconds).
    minsource : int
        Minimum number of sources per cluster.

    Attributes
    ----------
    r : NDVar
        Correlation (with threshold contours).
    """
    _state_specific = ('X', 'norm', 'n', 'df', 'r')

    def __init__(self, Y, X, norm=None, sub=None, ds=None, samples=None,
                 pmin=None, rmin=None, tfce=False, tstart=None, tstop=None,
                 match=None, dist_dim=(), parc=(), dist_tstep=None,
                 **criteria):
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
        x = X.x

        n = len(Y)
        df = n - 2

        rmap = stats.corr(Y.x, x)

        if samples is None:
            cdist = None
            r0, r1, r2 = stats.rtest_r((.05, .01, .001), df)
            info = _cs.stat_info('r', r0, r1, r2)
        else:
            # threshold
            if sum((pmin is not None, rmin is not None, tfce)) > 1:
                msg = "Only one of pmin, rmin and tfce can be specified"
                raise ValueError(msg)
            elif pmin is not None:
                threshold = stats.rtest_r(pmin, df)
            elif rmin is not None:
                threshold = abs(rmin)
            elif tfce:
                threshold = 'tfce'
            else:
                threshold = None
            info = _cs.stat_info('r', threshold)

            cdist = _ClusterDist(Y, samples, threshold, 0, 'r', name, tstart,
                                 tstop, criteria, dist_dim, parc, dist_tstep)
            cdist.add_original(rmap)
            if cdist.do_permutation:
                def test_func(y, out, perm):
                    return stats.corr(y, x, out, perm)
                iterator = permute_order(n, samples, unit=match)
                run_permutation(test_func, cdist, iterator)

        # compile results
        dims = Y.dims[1:]
        r = NDVar(rmap, dims, info, name)

        # store attributes
        _Result.__init__(self, Y, match, sub, samples, tfce, pmin, cdist,
                         tstart, tstop)
        self.X = X.name
        self.norm = None if norm is None else norm.name
        self.rmin = rmin
        self.n = n
        self.df = df
        self.r = r

        self._expand_state()

    def _expand_state(self):
        _Result._expand_state(self)

        r = self.r

        # uncorrected probability
        pmap = stats.rtest_p(r.x, self.df)
        info = _cs.sig_info()
        p_uncorrected = NDVar(pmap, r.dims, info, 'p_uncorrected')
        self.p_uncorrected = p_uncorrected

        self.r_p_uncorrected = [[r, r]]
        if self.samples:
            self.r_p = self._default_plot_obj = [[r, self.p]]
        else:
            self._default_plot_obj = self.r_p_uncorrected

    def _name(self):
        if self.Y and self.X:
            return "Correlation:  %s ~ %s" % (self.Y, self.X)
        else:
            return "Correlation"

    def _repr_test_args(self):
        args = [repr(self.Y), repr(self.X)]
        if self.norm:
            args.append('norm=%r' % self.norm)
        return args


class ttest_1samp(_Result):
    """Element-wise one sample t-test

    Parameters
    ----------
    Y : NDVar
        Dependent variable.
    popmean : scalar
        Value to compare Y against (default is 0).
    match : None | categorial
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
        Threshold for forming clusters:  use a t-value equivalent to an
        uncorrected p-value.
    tmin : None | scalar
        Threshold for forming clusters.
    tfce : bool
        Use threshold-free cluster enhancement (Smith & Nichols, 2009).
        Default is False.
    tstart, tstop : None | scalar
        Restrict time window for permutation cluster test.
    mintime : scalar
        Minimum duration for clusters (in seconds).
    minsource : int
        Minimum number of sources per cluster.

    Attributes
    ----------
    all : tuple
        c1, c0, [c0 - c1, P]
    diff : NDVar
        The difference value entering the test (`` y`` if popmean is 0).
    n : int
        Number of cases.
    p : NDVar | None
        Map of p-values corrected for multiple comparison (or None if no
        correction was performed).
    p_uncorrected : NDVar
        Map of p-values uncorrected for multiple comparison.
    t : NDVar
        Map of t-values.
    tfce_map : NDVar | None
        Map of the test statistic processed with the threshold-free cluster
        enhancement algorithm (or None if no TFCE was performed).
    """
    _state_specific = ('popmean', 'tail', 'n', 'df', 't', 'diff')

    def __init__(self, Y, popmean=0, match=None, sub=None, ds=None, tail=0,
                 samples=None, pmin=None, tmin=None, tfce=False, tstart=None,
                 tstop=None, dist_dim=(), parc=(), dist_tstep=None,
                 **criteria):
        ct = Celltable(Y, match=match, sub=sub, ds=ds, coercion=asndvar)

        n = len(ct.Y)
        df = n - 1
        y = ct.Y.summary()
        tmap = stats.t_1samp(ct.Y.x)
        if popmean:
            raise NotImplementedError("popmean != 0")
            diff = y - popmean
            if np.any(diff < 0):
                diff.info['cmap'] = 'xpolar'
        else:
            diff = y

        if samples is None:
            cdist = None
        else:
            # threshold
            if sum((pmin is not None, tmin is not None, tfce)) > 1:
                msg = "Only one of pmin, tmin and tfce can be specified"
                raise ValueError(msg)
            elif pmin is not None:
                threshold = stats.ttest_t(pmin, df, tail)
            elif tmin is not None:
                threshold = abs(tmin)
            elif tfce:
                threshold = 'tfce'
            else:
                threshold = None

            if popmean:
                y_perm = ct.Y - popmean
            else:
                y_perm = ct.Y
            n_samples, samples = _resample_params(len(y_perm), samples)
            cdist = _ClusterDist(y_perm, n_samples, threshold, tail, 't',
                                 '1-Sample t-Test', tstart, tstop, criteria,
                                 dist_dim, parc, dist_tstep)
            cdist.add_original(tmap)
            if cdist.do_permutation:
                iterator = permute_sign_flip(n, samples)
                run_permutation(opt.t_1samp_perm, cdist, iterator)

        # NDVar map of t-values
        dims = ct.Y.dims[1:]
        t0, t1, t2 = stats.ttest_t((.05, .01, .001), df, tail)
        info = _cs.stat_info('t', t0, t1, t2, tail)
        info = _cs.set_info_cs(ct.Y.info, info)
        t = NDVar(tmap, dims, info=info, name='T')

        # store attributes
        _Result.__init__(self, ct.Y, ct.match, sub, samples, tfce, pmin, cdist,
                         tstart, tstop)
        self.popmean = popmean
        self.tail = tail
        self.tmin = tmin

        self.n = n
        self.df = df

        self.diff = diff
        self.t = t

        self._expand_state()

    def _expand_state(self):
        _Result._expand_state(self)

        t = self.t
        pmap = stats.ttest_p(t.x, self.df, self.tail)
        info = _cs.set_info_cs(t.info, _cs.sig_info())
        p_uncorr = NDVar(pmap, t.dims, info=info, name='p')
        self.p_uncorrected = p_uncorr

        if self.samples:
            self._default_plot_obj = [[self.diff, self.p]]
        else:
            self._default_plot_obj = [[self.diff, t]]

    def _name(self):
        if self.Y:
            return "One-Sample T-Test:  %s" % self.Y
        else:
            return "One-Sample T-Test"

    def _repr_test_args(self):
        args = [repr(self.Y)]
        if self.popmean:
            args.append(repr(self.popmean))
        if self.match:
            args.append('match=%r' % self.match)
        if self.tail:
            args.append("tail=%i" % self.tail)
        return args


class ttest_ind(_Result):
    """Element-wise independent samples t-test

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

    Attributes
    ----------
    all :
        c1, c0, [c0 - c1, P]
    p_val :
        [c0 - c1, P]
    """
    _state_specific = ('X', 'c1', 'c0', 'tail', 't', 'n1', 'n0', 'df', 'c1_mean',
                       'c0_mean')

    def __init__(self, Y, X, c1=None, c0=None, match=None, sub=None, ds=None,
                 tail=0, samples=None, pmin=None, tmin=None, tfce=False,
                 tstart=None, tstop=None, dist_dim=(), parc=(),
                 dist_tstep=None, **criteria):
        ct = Celltable(Y, X, match, sub, cat=(c1, c0), ds=ds, coercion=asndvar)
        c1, c0 = ct.cat

        n1 = len(ct.data[c1])
        n = len(ct.Y)
        n0 = n - n1
        df = n - 2
        tmap = stats.t_ind(ct.Y.x, n1, n0)

        if samples is None:
            cdist = None
        else:
            # threshold
            if sum((pmin is not None, tmin is not None, tfce)) > 1:
                msg = "Only one of pmin, tmin and tfce can be specified"
                raise ValueError(msg)
            elif pmin is not None:
                threshold = stats.ttest_t(pmin, df, tail)
            elif tmin is not None:
                threshold = abs(tmin)
            elif tfce:
                threshold = 'tfce'
            else:
                threshold = None

            cdist = _ClusterDist(ct.Y, samples, threshold, tail, 't',
                                 'Independent Samples t-Test', tstart, tstop,
                                 criteria, dist_dim, parc, dist_tstep)
            cdist.add_original(tmap)
            if cdist.do_permutation:
                def test_func(y, out, perm):
                    return stats.t_ind(y, n1, n0, True, out, perm)
                iterator = permute_order(n, samples)
                run_permutation(test_func, cdist, iterator)

        dims = ct.Y.dims[1:]

        t0, t1, t2 = stats.ttest_t((.05, .01, .001), df, tail)
        info = _cs.stat_info('t', t0, t1, t2, tail)
        info = _cs.set_info_cs(ct.Y.info, info)
        t = NDVar(tmap, dims, info=info, name='T')

        c1_mean = ct.data[c1].summary(name=cellname(c1))
        c0_mean = ct.data[c0].summary(name=cellname(c0))

        # store attributes
        _Result.__init__(self, ct.Y, ct.match, sub, samples, tfce, pmin, cdist,
                         tstart, tstop)
        self.X = ct.X.name
        self.c0 = c0
        self.c1 = c1
        self.tail = tail
        self.tmin = tmin

        self.n1 = n1
        self.n0 = n0
        self.df = df

        self.c1_mean = c1_mean
        self.c0_mean = c0_mean
        self.t = t

        self._expand_state()

    def _expand_state(self):
        _Result._expand_state(self)

        c1_mean = self.c1_mean
        c0_mean = self.c0_mean
        t = self.t

        # difference
        diff = c1_mean - c0_mean
        if np.any(diff.x < 0):
            diff.info['cmap'] = 'xpolar'
        self.difference = diff

        # uncorrected p
        pmap = stats.ttest_p(t.x, self.df, self.tail)
        info = _cs.set_info_cs(t.info, _cs.sig_info())
        p_uncorr = NDVar(pmap, t.dims, info=info, name='p')
        self.p_uncorrected = p_uncorr

        # composites
        diff_p_uncorrected = [diff, t]
        self.diff_p_uncorrected = [diff_p_uncorrected]
        self.all_uncorrected = [c1_mean, c0_mean, diff_p_uncorrected]
        if self.samples:
            diff_p = [diff, self.p]
            self.diff_p = [diff_p]
            self.all = [c1_mean, c0_mean, diff_p]
            self._default_plot_obj = self.all
        else:
            self._default_plot_obj = self.all_uncorrected

    def _name(self):
        if self.tail == 0:
            comp = "%s == %s" % (self.c1, self.c0)
        elif self.tail > 0:
            comp = "%s > %s" % (self.c1, self.c0)
        else:
            comp = "%s < %s" % (self.c1, self.c0)

        if self.Y:
            return "Independent-Samples T-Test:  %s ~ %s" % (self.Y, comp)
        else:
            return "Independent-Samples T-Test:  %s" % comp

    def _plot_model(self):
        return self.X

    def _plot_sub(self):
        return "(%s).isin(%s)" % (self.X, (self.c1, self.c0))

    def _repr_test_args(self):
        args = [repr(self.Y), repr(self.X), "%r (n=%i)" % (self.c1, self.n1),
                "%r (n=%i)" % (self.c0, self.n0)]
        if self.match:
            args.append('match=%r' % self.match)
        if self.tail:
            args.append("tail=%i" % self.tail)
        return args


class ttest_rel(_Result):
    """Element-wise related samples t-test

    Parameters
    ----------
    Y : NDVar
        Dependent variable.
    X : categorial
        Model containing the cells which should be compared.
    c1 : str | tuple | None
        Test condition (cell of X). Can be omitted (or ``None``) if X only
        contains two cells.
    c0 : str | tuple | None
        Control condition (cell of X). Can be omitted (or ``None``) if X
        only contains two cells.
    match : categorial
        Units within which measurements are related (e.g. 'subject' in a
        within-subject comparison).
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
        Threshold for forming clusters:  use a t-value equivalent to an
        uncorrected p-value.
    tmin : None | scalar
        Threshold for forming clusters.
    tfce : bool
        Use threshold-free cluster enhancement (Smith & Nichols, 2009).
        Default is False.
    tstart, tstop : None | scalar
        Restrict time window for permutation cluster test.
    mintime : scalar
        Minimum duration for clusters (in seconds).
    minsource : int
        Minimum number of sources per cluster.

    Attributes
    ----------
    all :
        c1, c0, [c0 - c1, P]
    p_val :
        [c0 - c1, P]

    Notes
    -----
    In the permutation cluster test, permutations are done within the
    categories of ``match``.
    """
    _state_specific = ('X', 'c1', 'c0', 'tail', 't', 'n', 'df', 'c1_mean',
                       'c0_mean')

    def __init__(self, Y, X, c1=None, c0=None, match=None, sub=None, ds=None,
                 tail=0, samples=None, pmin=None, tmin=None, tfce=False,
                 tstart=None, tstop=None, dist_dim=(), parc=(),
                 dist_tstep=None, **criteria):
        if match is None:
            msg = ("The `match` argument needs to be specified for a related "
                   "samples t-test.")
            raise TypeError(msg)
        ct = Celltable(Y, X, match, sub, cat=(c1, c0), ds=ds, coercion=asndvar)
        c1, c0 = ct.cat
        if not ct.all_within:
            err = ("conditions %r and %r do not have the same values on "
                   "%r" % (c1, c0, ct.match.name))
            raise ValueError(err)

        n = len(ct.Y) // 2
        if n <= 2:
            raise ValueError("Not enough observations for t-test (n=%i)" % n)
        df = n - 1
        diff = ct.Y[:n] - ct.Y[n:]
        tmap = stats.t_1samp(diff.x)

        if samples is None:
            cdist = None
        else:
            # threshold
            if sum((pmin is not None, tmin is not None, tfce)) > 1:
                msg = "Only one of pmin, tmin and tfce can be specified"
                raise ValueError(msg)
            elif pmin is not None:
                threshold = stats.ttest_t(pmin, df, tail)
            elif tmin is not None:
                threshold = abs(tmin)
            elif tfce:
                threshold = 'tfce'
            else:
                threshold = None

            n_samples, samples = _resample_params(len(diff), samples)
            cdist = _ClusterDist(diff, n_samples, threshold, tail, 't',
                                 'Related Samples t-Test', tstart, tstop,
                                 criteria, dist_dim, parc, dist_tstep)
            cdist.add_original(tmap)
            if cdist.do_permutation:
                iterator = permute_sign_flip(n, samples)
                run_permutation(opt.t_1samp_perm, cdist, iterator)

        dims = ct.Y.dims[1:]
        t0, t1, t2 = stats.ttest_t((.05, .01, .001), df, tail)
        info = _cs.stat_info('t', t0, t1, t2, tail)
        t = NDVar(tmap, dims, info=info, name='T')

        c1_mean = ct.data[c1].summary(name=cellname(c1))
        c0_mean = ct.data[c0].summary(name=cellname(c0))

        # store attributes
        _Result.__init__(self, ct.Y, ct.match, sub, samples, tfce, pmin, cdist,
                         tstart, tstop)
        self.X = ct.X.name
        self.c0 = c0
        self.c1 = c1
        self.tail = tail
        self.tmin = tmin

        self.n = n
        self.df = df

        self.c1_mean = c1_mean
        self.c0_mean = c0_mean
        self.t = t

        self._expand_state()

    def _expand_state(self):
        _Result._expand_state(self)

        cdist = self._cdist
        t = self.t

        # difference
        diff = self.c1_mean - self.c0_mean
        if np.any(diff.x < 0):
            diff.info['cmap'] = 'xpolar'
        self.difference = diff

        # uncorrected p
        pmap = stats.ttest_p(t.x, self.df, self.tail)
        info = _cs.sig_info()
        p_uncorr = NDVar(pmap, t.dims, info=info, name='p')
        self.p_uncorrected = p_uncorr

        # composites
        diff_p_uncorr = [diff, t]
        self.difference_p_uncorrected = [diff_p_uncorr]
        self.uncorrected = [self.c1_mean, self.c0_mean, diff_p_uncorr]
        if self.samples:
            diff_p_corr = [diff, cdist.probability_map]
            self.difference_p = [diff_p_corr]
            self._default_plot_obj = [self.c1_mean, self.c0_mean, diff_p_corr]
        else:
            self._default_plot_obj = self.uncorrected

    def _name(self):
        if self.tail == 0:
            comp = "%s == %s" % (self.c1, self.c0)
        elif self.tail > 0:
            comp = "%s > %s" % (self.c1, self.c0)
        else:
            comp = "%s < %s" % (self.c1, self.c0)

        if self.Y:
            return "Related-Samples T-Test:  %s ~ %s" % (self.Y, comp)
        else:
            return "Related-Samples T-Test:  %s" % comp

    def _plot_model(self):
        return self.X

    def _plot_sub(self):
        return "(%s).isin(%s)" % (self.X, (self.c1, self.c0))

    def _repr_test_args(self):
        args = [repr(self.Y), repr(self.X), repr(self.c1), repr(self.c0),
                "%r (n=%i)" % (self.match, self.n)]
        if self.tail:
            args.append("tail=%i" % self.tail)
        return args


class _MultiEffectResult(_Result):

    def __repr__(self):
        temp = "<%s %%s>" % self.__class__.__name__

        args = [repr(self.Y), repr(self.X)]
        if self.sub is not None:
            args.append(', sub=%r' % self.sub)
        if self._cdist:
            cdist = self._cdist[0]
            args += cdist._repr_test_args(self.pmin)
            for cdist in self._cdist:
                effect_args = cdist._repr_clusters()
                args += ["%r: %s" % (cdist.name, ', '.join(effect_args))]

        out = temp % ', '.join(args)
        return out

    def _expand_state(self):
        self.effects = tuple(e.name for e in self._effects)

        # clusters
        cdists = self._cdist
        if cdists is None:
            self._kind = None
        else:
            self.tfce_maps = [cdist.tfce_map for cdist in cdists]
            self.probability_maps = [cdist.probability_map for cdist in cdists]
            self._kind = cdists[0].kind

    def _iter_cdists(self):
        for cdist in self._cdist:
            yield cdist.name.capitalize(), cdist

    @property
    def _first_cdist(self):
        if self._cdist is None:
            return None
        else:
            return self._cdist[0]

    def cluster(self, cluster_id, effect=0):
        """Retrieve a specific cluster as NDVar

        Parameters
        ----------
        cluster_id : int
            Cluster id.
        effect : int | str
            Index or name of the effect from which to retrieve a cluster
            (default is the first effect).

        Returns
        -------
        cluster : NDVar
            NDVar of the cluster, 0 outside the cluster.

        Notes
        -----
        Clusters only have stable ids for thresholded cluster distributions.
        """
        if self._cdist is None:
            err = "Method only applies to results with samples > 0"
            raise RuntimeError(err)
        elif isinstance(effect, basestring):
            effect = self.effects.index(effect)
        return self._cdist[effect].cluster(cluster_id)

    def compute_probability_map(self, effect=0, **sub):
        """Compute a probability map

        Parameters
        ----------
        effect : int | str
            Index or name of the effect from which to use the parameter map
            (default is the first effect).

        Returns
        -------
        probability : NDVar
            Map of p-values.
        """
        if self._cdist is None:
            err = "Method only applies to results with samples > 0"
            raise RuntimeError(err)
        elif isinstance(effect, basestring):
            effect = self.effects.index(effect)
        return self._cdist[effect].compute_probability_map(**sub)

    def masked_parameter_map(self, effect=0, pmin=0.05, **sub):
        """Create a copy of the parameter map masked by significance

        Parameters
        ----------
        effect : int | str
            Index or name of the effect from which to use the parameter map.
        pmin : None | scalar
            Threshold p-value for masking (default 0.05). For threshold-based
            cluster tests, pmin=None includes all clusters regardless of their
            p-value.

        Returns
        -------
        masked_map : NDVar
            NDVar with data from the original parameter map wherever p <= pmin
            and 0 everywhere else.
        """
        if self._cdist is None:
            err = "Method only applies to results with samples > 0"
            raise RuntimeError(err)
        elif isinstance(effect, basestring):
            effect = self.effects.index(effect)
        return self._cdist[effect].masked_parameter_map(pmin, **sub)

    def find_clusters(self, pmin=None, maps=False, **sub):
        """Find significant regions in a TFCE distribution

        Parameters
        ----------
        pmin : None | scalar, 1 >= p  >= 0
            Threshold p-value for clusters (for thresholded cluster tests the
            default is 1, for others 0.05).
        maps : bool
            Include in the output a map of every cluster (can be memory
            intensive if there are large statistical maps and/or many
            clusters; default False).

        Returns
        -------
        ds : Dataset
            Dataset with information about the clusters.
        """
        if self._cdist is None:
            err = ("Test results have no clustering (set samples to an int "
                   " >= 0 to find clusters")
            raise RuntimeError(err)
        dss = []
        info = {}
        for cdist in self._cdist:
            ds = cdist.clusters(pmin, maps, **sub)
            ds[:, 'effect'] = cdist.name
            if 'clusters' in ds.info:
                info['%s clusters' % cdist.name] = ds.info.pop('clusters')
            dss.append(ds)
        out = combine(dss)
        out.info.update(info)
        return out

    def find_peaks(self):
        """Find peaks in a TFCE distribution

        Returns
        -------
        ds : Dataset
            Dataset with information about the peaks.
        """
        if self._cdist is None:
            err = "Method only applies to results with samples > 0"
            raise RuntimeError(err)
        dss = []
        for cdist in self._cdist:
            ds = cdist.find_peaks()
            ds[:, 'effect'] = cdist.name
            dss.append(ds)
        return combine(dss)


class anova(_MultiEffectResult):
    """Element-wise ANOVA

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
        Threshold for forming clusters:  use an f-value equivalent to an
        uncorrected p-value.
    fmin : None | scalar
        Threshold for forming clusters.
    tfce : bool
        Use threshold-free cluster enhancement (Smith & Nichols, 2009).
        Default is False.
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

    Attributes
    ----------
    effects : tuple of str
        Names of all the effects as they occur in the ``.clusters`` Dataset.
    clusters : None | Dataset
        When performing a cluster permutation test, a Dataset of all clusters.
    f : list
        Maps of f values with probability contours.
    p : list
        Maps of p values.
    """
    _state_specific = ('X', 'pmin', '_effects', '_dfs_denom', 'f')

    def __init__(self, Y, X, sub=None, ds=None, samples=None, pmin=None,
                 fmin=None, tfce=False, tstart=None, tstop=None, match=None,
                 dist_dim=(), parc=(), dist_tstep=None, **criteria):
        sub = assub(sub, ds)
        Y = asndvar(Y, sub, ds)
        x_ = asmodel(X, sub, ds)
        if match is not None:
            match = ascategorial(match, sub, ds)

        lm = _nd_anova(x_)
        effects = lm.effects
        dfs_denom = lm.dfs_denom
        fmaps = lm.map(Y.x)

        if samples is None:
            cdists = None
        else:
            # threshold
            if sum((pmin is not None, fmin is not None, tfce)) > 1:
                msg = "Only one of pmin, fmin and tfce can be specified"
                raise ValueError(msg)
            elif pmin is not None:
                thresholds = (stats.ftest_f(pmin, e.df, df_den) for e, df_den in
                              izip(effects, dfs_denom))
            elif fmin is not None:
                thresholds = (abs(fmin) for _ in xrange(len(effects)))
            elif tfce:
                thresholds = ('tfce' for _ in xrange(len(effects)))
            else:
                thresholds = (None for _ in xrange(len(effects)))

            cdists = [_ClusterDist(Y, samples, thresh, 1, 'F', e.name, tstart,
                                   tstop, criteria, dist_dim, parc,
                                   dist_tstep)
                      for e, thresh in izip(effects, thresholds)]

            # Find clusters in the actual data
            do_permutation = 0
            for cdist, fmap in izip(cdists, fmaps):
                cdist.add_original(fmap)
                do_permutation += cdist.do_permutation

            if do_permutation:
                iterator = permute_order(len(Y), samples, unit=match)
                run_permutation_me(lm, cdists, iterator)

        # create ndvars
        dims = Y.dims[1:]
        f = []
        for e, fmap, df_den in izip(effects, fmaps, dfs_denom):
            f0, f1, f2 = stats.ftest_f((0.05, 0.01, 0.001), e.df, df_den)
            info = _cs.stat_info('f', f0, f1, f2)
            f_ = NDVar(fmap, dims, info, e.name)
            f.append(f_)

        # store attributes
        _MultiEffectResult.__init__(self, Y, match, sub, samples, tfce, pmin,
                                    cdists, tstart, tstop)
        self.X = X if isinstance(X, basestring) else x_.name
        self._effects = effects
        self._dfs_denom = dfs_denom
        self.f = f

        self._expand_state()

    def _expand_state(self):
        # backwards compatibility
        if hasattr(self, 'effects'):
            self._effects = self.effects

        _MultiEffectResult._expand_state(self)

        # backwards compatibility
        if hasattr(self, 'df_den'):
            df_den_temp = {e.name: df for e, df in self.df_den.iteritems()}
            del self.df_den
            self._dfs_denom = tuple(df_den_temp[e] for e in self.effects)

        # f-maps with clusters
        pmin = self.pmin or 0.05
        if self.samples:
            f_and_clusters = []
            for e, fmap, df_den, cdist in izip(self._effects, self.f,
                                               self._dfs_denom, self._cdist):
                # create f-map with cluster threshold
                f0 = stats.ftest_f(pmin, e.df, df_den)
                info = _cs.stat_info('f', f0)
                f_ = NDVar(fmap.x, fmap.dims, info, e.name)
                # add overlay with cluster
                if cdist.probability_map is not None:
                    f_and_clusters.append([f_, cdist.probability_map])
                else:
                    f_and_clusters.append([f_])
            self.f_probability = f_and_clusters

        # uncorrected probability
        p_uncorr = []
        for e, f, df_den in izip(self._effects, self.f, self._dfs_denom):
            info = _cs.sig_info()
            pmap = stats.ftest_p(f.x, e.df, df_den)
            p_ = NDVar(pmap, f.dims, info, e.name)
            p_uncorr.append(p_)
        self.p_uncorrected = p_uncorr

        if self.samples:
            self._default_plot_obj = f_and_clusters
        else:
            self._default_plot_obj = self.f

    def _name(self):
        if self.Y:
            return "ANOVA:  %s ~ %s" % (self.Y, self.X)
        else:
            return "ANOVA:  %s" % self.X

    def _plot_model(self):
        factors = [f.strip() for f in self.X.split('*')]
        if self.match in factors:
            factors.remove(self.match)
        return '%'.join(factors)

    def _plot_sub(self):
        return super(anova, self)._plot_sub()


def flatten(spm, all_adjacent):
    """Reshape SPM buffer array to 2-dimensional map for connectivity processing

    Parameters
    ----------
    spm : array
        N-dimensional array (with non-adjacent dimension at first position).
    all_adjacent : bool
        Whether reshape is unneccessary.

    Returns
    -------
    flat_spm : array
        The input spm reshaped if necessary, making sure that input and output
        arrays share the same underlying data buffer.
    """
    if spm.ndim == 2 or all_adjacent:
        return spm
    else:
        out = spm.reshape((spm.shape[0], -1))
        assert out.base is spm
        return out


def label_clusters(stat_map, threshold, tail, connectivity, criteria):
    """Label clusters

    Parameters
    ----------
    stat_map : array
        Statistical parameter map (non-adjacent dimension on the first
        axis).

    Returns
    -------
    cmap : np.ndarray of uint32
        Array with clusters labelled as integers.
    cluster_ids : np.ndarray of uint32
        Identifiers of the clusters that survive the minimum duration
        criterion.
    """
    all_adjacent = connectivity is None
    cmap = np.empty(stat_map.shape, np.uint32)
    bin_buff = np.empty(stat_map.shape, np.bool8)
    cmap_flat = flatten(cmap, all_adjacent)

    if tail == 0:
        int_buff = np.empty(stat_map.shape, np.uint32)
        int_buff_flat = flatten(int_buff, all_adjacent)
    else:
        int_buff = int_buff_flat = None

    struct = _make_struct(stat_map.ndim, all_adjacent)

    cids = _label_clusters(stat_map, threshold, tail, struct, all_adjacent,
                           connectivity, criteria, cmap, cmap_flat, bin_buff,
                           int_buff, int_buff_flat)
    return cmap, cids


def _label_clusters(stat_map, threshold, tail, struct, all_adjacent, conn,
                    criteria, cmap, cmap_flat, bin_buff, int_buff,
                    int_buff_flat):
    """Find clusters on a statistical parameter map

    Parameters
    ----------
    stat_map : array
        Statistical parameter map (non-adjacent dimension on the first
        axis).
    cmap : array of int
        Buffer for the cluster id map (will be modified).

    Returns
    -------
    cluster_ids : np.ndarray of uint32
        Identifiers of the clusters that survive the minimum duration
        criterion.
    """
    # compute clusters
    if tail >= 0:
        bin_map_above = np.greater(stat_map, threshold, bin_buff)
        cids = _label_clusters_binary(bin_map_above, cmap, cmap_flat, struct,
                                      all_adjacent, conn, criteria)

    if tail <= 0:
        bin_map_below = np.less(stat_map, -threshold, bin_buff)
        if tail < 0:
            cids = _label_clusters_binary(bin_map_below, cmap, cmap_flat,
                                          struct, all_adjacent, conn, criteria)
        else:
            cids_l = _label_clusters_binary(bin_map_below, int_buff,
                                            int_buff_flat, struct,
                                            all_adjacent, conn, criteria)
            x = cmap.max()
            int_buff[bin_map_below] += x
            cids_l += x
            cmap += int_buff
            cids = np.concatenate((cids, cids_l))

    return cids


def label_clusters_binary(bin_map, connectivity, criteria):
    """Label clusters in a boolean map

    Returns
    -------
    cmap : np.ndarray of uint32
        Array with clusters labelled as integers.
    cluster_ids : np.ndarray of uint32
        Sorted identifiers of the clusters that survive the selection criteria.
    """
    all_adjacent = connectivity is None
    cmap = np.empty(bin_map.shape, np.uint32)
    cmap_flat = flatten(cmap, all_adjacent)

    struct = _make_struct(bin_map.ndim, all_adjacent)

    cids = _label_clusters_binary(bin_map, cmap, cmap_flat, struct, all_adjacent,
                                  connectivity, criteria)
    return cmap, cids


def _label_clusters_binary(bin_map, cmap, cmap_flat, struct, all_adjacent,
                           conn, criteria):
    """Label clusters in a binary array

    Parameters
    ----------
    bin_map : np.ndarray
        Binary map of where the parameter map exceeds the threshold for a
        cluster (non-adjacent dimension on the first axis).
    cmap : np.ndarray
        Array in which to label the clusters.
    cmap_flat : np.ndarray
        Flat copy of cmap (ndim=2, only used when all_adjacent==False)
    struct : np.ndarray
        Struct to use for scipy.ndimage.label
    all_adjacent : bool
        Whether all dimensions have line-graph connectivity.
    conn : dict
        Connectivity (if first dimension is not a line graph).
    criteria : None | list
        Cluster size criteria, list of (axes, v) tuples. Collapse over axes
        and apply v minimum length).

    Returns
    -------
    cluster_ids : np.ndarray of uint32
        Sorted identifiers of the clusters that survive the selection criteria.
    """
    # find clusters
    n = ndimage.label(bin_map, struct, cmap)
    # n is 1 even when no cluster is found
    if n == 1:
        if cmap.max() == 0:
            return np.array((), np.uint32)
        else:
            cids = np.array((1,), np.uint32)
    elif all_adjacent:
        cids = np.arange(1, n + 1, 1, np.uint32)
    else:
        cids = merge_labels(cmap_flat, n, conn)

    # apply minimum cluster size criteria
    if criteria and cids.size:
        for axes, v in criteria:
            cids = np.setdiff1d(cids,
                                [i for i in cids if np.count_nonzero(np.equal(cmap, i).any(axes)) < v],
                                True)
            if cids.size == 0:
                break

    return cids


def tfce(stat_map, tail, connectivity):
    all_adjacent = connectivity is None
    tfce_map = np.empty(stat_map.shape)
    bin_buff = np.empty(stat_map.shape, np.bool8)
    int_buff = np.empty(stat_map.shape, np.uint32)
    int_buff_flat = flatten(int_buff, all_adjacent)

    struct = _make_struct(stat_map.ndim, all_adjacent)

    _tfce(stat_map, tail, struct, all_adjacent, connectivity, tfce_map,
          bin_buff, int_buff, int_buff_flat)
    return tfce_map


def _tfce(stat_map, tail, struct, all_adjacent, conn, out, bin_buff, int_buff,
          int_buff_flat, dh=0.1, e=0.5, h=2.0):
    "Threshold-free cluster enhancement"
    out.fill(0)

    # determine slices
    if tail == 0:
        hs = np.hstack((np.arange(-dh, stat_map.min(), -dh),
                        np.arange(dh, stat_map.max(), dh)))
    elif tail < 0:
        hs = np.arange(-dh, stat_map.min(), -dh)
    else:
        hs = np.arange(dh, stat_map.max(), dh)

    # label clusters in slices at different heights
    # fill each cluster with total section value
    # each point's value is the vertical sum
    for h_ in hs:
        if h_ > 0:
            np.greater_equal(stat_map, h_, bin_buff)
            h_factor = h_ ** h
        else:
            np.less_equal(stat_map, h_, bin_buff)
            h_factor = (-h_) ** h

        c_ids = _label_clusters_binary(bin_buff, int_buff, int_buff_flat, struct,
                                       all_adjacent, conn, None)
        for id_ in c_ids:
            np.equal(int_buff, id_, bin_buff)
            v = np.count_nonzero(bin_buff) ** e * h_factor
            out[bin_buff] += v

    return out


class StatMapProcessor(object):

    def __init__(self, tail, max_axes, parc, tstep_reshape):
        """Reduce a statistical map to the relevant maximum statistic

        Parameters
        ----------
        dims : tuple
            Dimensions of the map (without case).
        dims : tuple
            Dimensions of the map (without case).
        """
        self.tail = tail
        self.max_axes = max_axes
        self.parc = parc
        self.tstep_reshape = tstep_reshape

    def max_stat(self, stat_map):
        stat_map = stat_map.reshape(self.tstep_reshape)
        if self.tail == 0:
            v = np.abs(stat_map, stat_map).max(self.max_axes)
        elif self.tail > 0:
            v = stat_map.max(self.max_axes)
        else:
            v = -stat_map.min(self.max_axes)

        if self.parc is not None:
            v = [v[idx].max() for idx in self.parc]

        return v


class TFCEProcessor(StatMapProcessor):

    def __init__(self, tail, max_axes, parc, tstep_reshape, shape, all_adjacent,
                 connectivity):
        StatMapProcessor.__init__(self, tail, max_axes, parc, tstep_reshape)
        self.shape = shape
        self.all_adjacent = all_adjacent
        self.connectivity = connectivity
        self.struct = _make_struct(len(shape), all_adjacent)

        # Pre-allocate memory buffers used for cluster processing
        self._bin_buff = np.empty(shape, np.bool8)
        self._int_buff = np.empty(shape, np.uint32)
        self._tfce_map = np.empty(shape)

        if all_adjacent or len(shape) <= 2:
            self._int_buff_flat = self._int_buff
        else:
            self._int_buff_flat = self._int_buff.reshape((shape[0], -1))

        if tstep_reshape is None:
            self._tfce_map_stacked = self._tfce_map
        else:
            self._tfce_map_stacked = self._tfce_map.reshape(tstep_reshape)

    def max_stat(self, stat_map):
        _tfce(stat_map, self.tail, self.struct, self.all_adjacent,
              self.connectivity, self._tfce_map, self._bin_buff, self._int_buff,
              self._int_buff_flat)
        return self._tfce_map_stacked.max(self.max_axes)


class ClusterProcessor(StatMapProcessor):

    def __init__(self, tail, max_axes, parc, tstep_reshape, shape, all_adjacent,
                 connectivity, threshold, criteria):
        StatMapProcessor.__init__(self, tail, max_axes, parc, tstep_reshape)
        self.shape = shape
        self.all_adjacent = all_adjacent
        self.connectivity = connectivity
        self.struct = _make_struct(len(shape), all_adjacent)
        self.threshold = threshold
        self.criteria = criteria

        # Pre-allocate memory buffers used for cluster processing
        self._bin_buff = np.empty(shape, np.bool8)
        if parc is not None:
            self.out = np.empty(len(parc))

        self._cmap = np.empty(shape, np.uint32)
        self._cmap_flat = flatten(self._cmap, all_adjacent)

        if tail == 0:
            self._int_buff = np.empty(shape, np.uint32)
            self._int_buff_flat = flatten(self._int_buff, all_adjacent)
        else:
            self._int_buff = self._int_buff_flat = None

    def max_stat(self, stat_map, threshold=None):
        if threshold is None:
            threshold = self.threshold
        cmap = self._cmap
        cids = _label_clusters(stat_map, threshold, self.tail, self.struct,
                               self.all_adjacent, self.connectivity,
                               self.criteria, cmap, self._cmap_flat,
                               self._bin_buff, self._int_buff,
                               self._int_buff_flat)
        if self.parc is not None:
            v = self.out
            for i, idx in enumerate(self.parc):
                clusters_v = ndimage.sum(stat_map[idx], cmap[idx], cids)
                if len(clusters_v):
                    np.abs(clusters_v, clusters_v)
                    v[i] = clusters_v.max()
                else:
                    v[i] = 0
        elif len(cids):
            clusters_v = ndimage.sum(stat_map, cmap, cids)
            np.abs(clusters_v, clusters_v)
            v = clusters_v.max()
        else:
            v = 0

        return v


def get_map_processor(kind, *args):
    if kind == 'tfce':
        return TFCEProcessor(*args)
    elif kind == 'cluster':
        return ClusterProcessor(*args)
    elif kind == 'raw':
        return StatMapProcessor(*args)
    else:
        raise ValueError("kind=%s" % repr(kind))


def _make_struct(ndim, all_adjacent):
    struct = ndimage.generate_binary_structure(ndim, 1)
    if not all_adjacent:
        struct[::2] = False
    return struct


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
    def __init__(self, y, samples, threshold, tail=0, meas='?', name=None,
                 tstart=None, tstop=None, criteria={}, dist_dim=(), parc=(),
                 dist_tstep=None):
        """Accumulate information on a cluster statistic.

        Parameters
        ----------
        Y : NDVar
            Dependent variable.
        samples : int
            Number of permutations.
        threshold : None | scalar > 0 | 'tfce'
            Threshold for finding clusters. None for forming distribution of
            largest value in parameter map. 'TFCE' for threshold-free cluster
            enhancement.
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
        dist_dim : str | sequence of str
            Collect permutation extrema for all points in this dimension(s)
            instead of only collecting the overall maximum. This allows
            deriving p-values for regions of interest from the same set of
            permutations. Threshold-free distributions only.
        parc : str | sequence of str
            Collect permutation extrema for all regions of the parcellation of
            this dimension(s). For threshold-based test, the regions are
            disconnected.
        dist_tstep : None | scalar [seconds]
            Instead of collecting the distribution for the maximum across time,
            collect the maximum in several time bins. The value of tstep has to
            divide the time between tstart and tstop in even sections. TFCE
            only.
        n_workers : int
            Number of clustering workers (for threshold based clusters and
            TFCE). Negative numbers are added to the cpu-count, 0 to disable
            multiprocessing.
        """
        assert y.has_case
        if threshold is None:
            kind = 'raw'
        elif isinstance(threshold, str):
            if threshold.lower() == 'tfce':
                kind = 'tfce'
                threshold = None
            else:
                raise ValueError("Invalid value for pmin: %s" % repr(threshold))
        else:
            try:
                threshold = float(threshold)
            except:
                raise TypeError("Invalid value for pmin: %s" % repr(threshold))

            if threshold > 0:
                kind = 'cluster'
            else:
                raise ValueError("Invalid value for pmin: %s" % repr(threshold))

        # adapt arguments
        if isinstance(dist_dim, basestring):
            dist_dim = (dist_dim,)
        elif dist_dim is None:
            dist_dim = ()

        if isinstance(parc, basestring):
            parc = (parc,)
        elif parc is None:
            parc = ()

        # prepare temporal cropping
        if (tstart is None) and (tstop is None):
            self._crop_for_permutation = False
            y_perm = y
        else:
            t_ax = y.get_axis('time') - 1
            self._crop_for_permutation = True
            y_perm = y.sub(time=(tstart, tstop))
            t_slice = y.time._slice(tstart, tstop)
            self._crop_idx = (full_slice,) * t_ax + (t_slice,)
            self._uncropped_shape = y.shape[1:]

        # cluster map properties
        ndim = y_perm.ndim - 1
        shape = y_perm.shape[1:]
        stat_map_dims = y_perm.dims[1:]

        # prepare adjacency
        adjacent = [d.adjacent for d in y_perm.dims[1:]]
        all_adjacent = all(adjacent)
        if all_adjacent:
            nad_ax = 0
            connectivity = None
        else:
            if sum(adjacent) < len(adjacent) - 1:
                err = "more than one non-adjacent dimension"
                raise NotImplementedError(err)
            nad_ax = adjacent.index(False)
            # prepare flattening (cropped) maps with swapped axes
            if nad_ax:
                shape = list(shape)
                shape[0], shape[nad_ax] = shape[nad_ax], shape[0]
                shape = tuple(shape)
                stat_map_dims = list(stat_map_dims)
                stat_map_dims[0], stat_map_dims[nad_ax] = stat_map_dims[nad_ax], stat_map_dims[0]
                stat_map_dims = tuple(stat_map_dims)

            # prepare connectivity
            nad_dim = stat_map_dims[0]
            disconnect_parc = (nad_dim.name in parc)
            if disconnect_parc:
                connectivity = nad_dim.connectivity(disconnect_parc=disconnect_parc)
            else:
                connectivity = nad_dim.connectivity()

        # prepare cluster minimum size criteria
        if criteria:
            criteria_ = []
            for k, v in criteria.iteritems():
                if k == 'mintime':
                    ax = y.get_axis('time') - 1
                    v = int(ceil(v / y.time.tstep))
                else:
                    m = re.match('min(\w+)', k)
                    if m:
                        ax = y.get_axis(m.group(1)) - 1
                    else:
                        raise ValueError("Unknown argument: %s" % k)

                if nad_ax:
                    if ax == 0:
                        ax = nad_ax
                    elif ax == nad_ax:
                        ax = 0

                axes = tuple(i for i in xrange(ndim) if i != ax)
                criteria_.append((axes, v))

            if kind != 'cluster':
                # here so that invalid keywords raise explicitly
                err = ("Can not use cluster size criteria when doing "
                       "threshold free cluster evaluation")
                raise ValueError(err)
        else:
            criteria_ = None

        # prepare distribution
        samples = int(samples)
        if dist_dim or parc or dist_tstep:
            # raise for incompatible cases
            if (dist_dim or dist_tstep) and kind == 'cluster':
                err = ("The dist_dim and dist_tstep parameters only apply to "
                       "threshold-free cluster distributions.")
                raise ValueError(err)
            if parc and kind == 'tfce':
                msg = "parc does not apply to TFCE"
                raise NotImplementedError(msg)

            # check all dims are in order
            if dist_tstep and not y.has_dim('time'):
                msg = "dist_tstep specified but data has no time dimension"
                raise ValueError(msg)
            dim_names = tuple(dim.name for dim in y_perm.dims[1:])
            err = tuple(name for name in chain(dist_dim, parc) if name not in
                        dim_names)
            if err:
                if len(err) == 1:
                    msg = ("%r is contained in dist_dim or parc but is not a "
                           "valid dimension in the input ndvar" % err)
                else:
                    msg = ("%r are contained in dist_dim or parc but are not "
                           "valid dimensions in the input ndvar" % str(err))
                raise ValueError(msg)
            duplicates = set(dist_dim)
            duplicates.intersection_update(parc)
            if duplicates:
                msg = ("%s were specified as dist_dim as well as parc. Each "
                       "dimension can only be either dist_dim or parc.")
                raise ValueError(msg)

            # find parameters for aggregating dist
            dist_shape = [samples]
            dist_dims = ['case']
            tstep_reshape = []  # reshape value map for dist_tstep before .max()
            max_axes = []  # v_map.max(max_axes)
            reshaped_ax_shift = 0  # number of inserted axes after reshaping cmap
            parc_indexes = None  # (ax, parc-Factor) tuples
            for i, dim in enumerate(stat_map_dims):
                if dim.name in dist_dim:  # keep the dimension
                    length = len(dim)
                    dist_shape.append(length)
                    dist_dims.append(dim)
                    tstep_reshape.append(length)
                elif dim.name in parc:
                    if not hasattr(dim, 'parc'):
                        msg = "%r dimension has no parcellation" % dim.name
                        raise NotImplementedError(msg)
                    elif i != 0:
                        msg = "parc that is not non-adjacent axis"
                        raise NotImplementedError(msg)
                    parc_ = dim.parc
                    parc_dim = Categorial(dim.name, parc_.cells)
                    length = len(parc_dim)
                    dist_shape.append(length)
                    dist_dims.append(parc_dim)
                    tstep_reshape.append(len(dim))
                    indexes = [parc_ == cell for cell in parc_.cells]
                    parc_indexes = np.array(indexes)
                elif dim.name == 'time' and dist_tstep:
                    step = int(round(dist_tstep / dim.tstep))
                    if dim.nsamples % step != 0:
                        err = ("dist_tstep={} does not divide time into even "
                               "parts ({} samples / {}).")
                        err = err.format(dist_tstep, dim.nsamples, step)
                        raise ValueError(err)
                    n_times = int(dim.nsamples / step)

                    dist_shape.append(n_times)
                    dist_dims.append(UTS(dim.tmin, dist_tstep, n_times))
                    tstep_reshape.append(step)
                    tstep_reshape.append(n_times)
                    max_axes.append(i + reshaped_ax_shift)
                    reshaped_ax_shift += 1
                else:
                    tstep_reshape.append(len(dim))
                    max_axes.append(i + reshaped_ax_shift)

            dist_shape = tuple(dist_shape)
            dist_dims = tuple(dist_dims)
            tstep_reshape = tuple(tstep_reshape)
            max_axes = tuple(max_axes)
        else:
            dist_shape = (samples,)
            dist_dims = None
            tstep_reshape = None
            max_axes = None
            parc_indexes = None

        # arguments for the map processor
        if kind == 'raw':
            map_args = (kind, tail, max_axes, parc_indexes, tstep_reshape)
        elif kind == 'tfce':
            map_args = (kind, tail, max_axes, parc_indexes, tstep_reshape,
                        shape, all_adjacent, connectivity)
        else:
            map_args = (kind, tail, max_axes, parc_indexes, tstep_reshape,
                        shape, all_adjacent, connectivity, threshold, criteria_)

        self.kind = kind
        self.y_perm = y_perm
        self.dims = y_perm.dims
        self.shape = shape  # internal shape for maps
        self._connectivity = connectivity
        self.samples = samples
        self.dist_shape = dist_shape
        self._dist_dims = dist_dims
        self._tstep_reshape = tstep_reshape
        self._max_axes = max_axes
        self.dist = None
        self.threshold = threshold
        self.tail = tail
        self._all_adjacent = all_adjacent
        self._nad_ax = nad_ax
        self.tstart = tstart
        self.tstop = tstop
        self.dist_dim = dist_dim
        self.parc = parc
        self.dist_tstep = dist_tstep
        self.meas = meas
        self.name = name
        self._criteria = criteria_
        self.criteria = criteria
        self.map_args = map_args
        self.has_original = False
        self.do_permutation = False
        self.dt_perm = None
        self._finalized = False
        self._init_time = current_time()
        self._host = socket.gethostname()

        from .. import __version__
        self._version = __version__

    def _crop(self, im):
        "Crop an original stat_map"
        if self._crop_for_permutation:
            return im[self._crop_idx]
        else:
            return im

    def _uncrop(self, im, background=0):
        "Expand a permutation-stat_map to dimensions of the original data"
        if self._crop_for_permutation:
            im_ = np.empty(self._uncropped_shape, dtype=im.dtype)
            im_[:] = background
            im_[self._crop_idx] = im
            return im_
        else:
            return im

    def add_original(self, stat_map):
        """Add the original statistical parameter map.

        Parameters
        ----------
        stat_map : array
            Parameter map of the statistic of interest (uncropped).
        """
        if self.has_original:
            raise RuntimeError("Original pmap already added")
        logger.debug("Adding original parameter map...")

        # crop/reshape stat_map
        stat_map = self._crop(stat_map)
        if self._nad_ax:
            stat_map = stat_map.swapaxes(0, self._nad_ax)

        # process map
        if self.kind == 'tfce':
            cmap = tfce(stat_map, self.tail, self._connectivity)
            cids = None
            n_clusters = True
        elif self.kind == 'cluster':
            cmap, cids = label_clusters(stat_map, self.threshold, self.tail,
                                        self._connectivity, self._criteria)
            n_clusters = len(cids)
            # clean original cluster map
            idx = np.in1d(cmap, cids, invert=True).reshape(self.shape)
            cmap[idx] = 0
        else:
            cmap = stat_map
            cids = None
            n_clusters = True

        self._t0 = current_time()
        self._original_cluster_map = cmap
        self._cids = cids
        self.n_clusters = n_clusters
        self.has_original = True
        self.dt_original = self._t0 - self._init_time
        self._original_param_map = stat_map
        if self.samples and n_clusters:
            self._create_dist()
            self.do_permutation = True
        else:
            self.dist_array = None
            self.finalize()

    def _create_dist(self):
        "Create the distribution container"
        if MULTIPROCESSING:
            n = reduce(operator.mul, self.dist_shape)
            dist_array = RawArray('d', n)
            dist = np.frombuffer(dist_array, np.float64, n)
            dist.shape = self.dist_shape
        else:
            dist_array = None
            dist = np.zeros(self.dist_shape)

        self.dist_array = dist_array
        self.dist = dist

    def _aggregate_dist(self, **sub):
        """Aggregate permutation distribution to one value per permutation

        Parameters
        ----------
        [dimname] : index
            Limit the data for the distribution.

        Returns
        -------
        dist : array, shape = (samples,)
            Maximum value for each permutation in the given region.
        """
        dist = self.dist

        if sub:
            if self._dist_dims is None:
                raise TypeError("ClusterDist does not have parcellation")
            dist_ = NDVar(dist, self._dist_dims)
            dist_sub = dist_.sub(**sub)
            dist = dist_sub.x

        if dist.ndim > 1:
            axes = tuple(xrange(1, dist.ndim))
            dist = dist.max(axes)

        return dist

    def __repr__(self):
        items = []
        if self.has_original:
            dt = timedelta(seconds=round(self.dt_original))
            items.append("%i clusters (%s)" % (self.n_clusters, dt))

            if self.samples > 0 and self.n_clusters > 0:
                if self.dt_perm is not None:
                    dt = timedelta(seconds=round(self.dt_perm))
                    items.append("%i permutations (%s)" % (self.samples, dt))
        else:
            items.append("no data")

        return "<ClusterDist: %s>" % ', '.join(items)

    def __getstate__(self):
        if not self._finalized:
            err = ("Cannot pickle cluster distribution before all permu"
                   "tations have been added.")
            raise RuntimeError(err)
        attrs = ('name', 'meas', '_version', '_host', '_init_time',
                 # settings ...
                 'kind', 'threshold', 'tail', 'criteria', 'samples', 'tstart',
                 'tstop', 'dist_dim', 'dist_tstep',
                 # data properties ...
                 'dims', 'shape', '_all_adjacent', '_nad_ax', '_connectivity',
                 '_criteria',
                 # results ...
                 'dt_original', 'dt_perm', 'n_clusters', '_dist_dims', 'dist',
                 '_original_param_map', '_original_cluster_map', '_cids')
        state = {name: getattr(self, name) for name in attrs}
        return state

    def __setstate__(self, state):
        # backwards compatibility
        if '_connectivity_src' in state:
            state['_connectivity'] = np.hstack((state.pop('_connectivity_src'),
                                                state.pop('_connectivity_dst')))
        if 'N' in state:
            state['samples'] = state.pop('N')
        if '_version' not in state:
            state['_version'] = '< 0.11'
        if '_host' not in state:
            state['_host'] = 'unknown'
        if '_init_time' not in state:
            state['_init_time'] = None

        for k, v in state.iteritems():
            setattr(self, k, v)
        self.has_original = True
        self.finalize()

    def _repr_test_args(self, pmin):
        "Argument representation for TestResult repr"
        args = ['samples=%r' % self.samples]
        if pmin:
            args.append("pmin=%r" % pmin)
        if self.tstart:
            args.append("tstart=%r" % self.tstart)
        if self.tstop:
            args.append("tstop=%r" % self.tstop)
        if self.dist_dim:
            args.append("dist_dim=%r" % self.dist_dim)
        if self.dist_tstep:
            args.append("dist_tstep=%r" % self.dist_tstep)
        for item in self.criteria.iteritems():
            args.append("%s=%r" % item)
        return args

    def _repr_clusters(self):
        info = []
        if self.kind == 'cluster':
            if self.n_clusters == 0:
                info.append("no clusters")
            else:
                info.append("%i clusters" % self.n_clusters)

        if self.n_clusters and self.samples:
            info.append("p >= %.3f" % self.probability_map.min())

        return info

    def finalize(self):
        "Package results and delete temporary data"
        if self.dt_perm is None:
            self.dt_perm = current_time() - self._t0

        # prepare container for clusters
        dims = self.dims
        param_contours = {}
        if self.kind == 'cluster':
            if self.tail >= 0:
                param_contours[self.threshold] = (0.7, 0.7, 0)
            if self.tail <= 0:
                param_contours[-self.threshold] = (0.7, 0, 0.7)

        # original parameter-map
        param_map = self._original_param_map

        # TFCE map
        if self.kind == 'tfce':
            stat_map = self._original_cluster_map
            x = stat_map.swapaxes(0, self._nad_ax)
            tfce_map_ = NDVar(x, dims[1:], {}, self.name)
        else:
            tfce_map_ = None

        # cluster map
        if self.kind == 'cluster':
            cluster_map = self._original_cluster_map
            x = cluster_map.swapaxes(0, self._nad_ax)
            cluster_map_ = NDVar(x, dims[1:], {}, self.name)
        else:
            cluster_map_ = None

        # original parameter map
        info = _cs.stat_info(self.meas, contours=param_contours)
        if self._nad_ax:
            param_map = param_map.swapaxes(0, self._nad_ax)
        param_map_ = NDVar(param_map, dims[1:], info, self.name)

        # store attributes
        self.tfce_map = tfce_map_
        self.parameter_map = param_map_
        self.cluster_map = cluster_map_
        self._finalized = True

    def _find_peaks(self, x, out=None):
        """Find peaks (local maxima, including plateaus) in x

        Returns
        -------
        out : array (x.shape, bool)
            Boolean array which is True only on local maxima. The borders are
            treated as lower than the rest of x (i.e., local maxima can touch
            the border).
        """
        if out is None:
            out = np.empty(x.shape, np.bool8)
        out.fill(True)

        # move through each axis in both directions and discard descending
        # slope. Do most computationally intensive axis last.
        for ax in xrange(x.ndim - 1, -1, -1):
            if ax == 0 and not self._all_adjacent:
                shape = (len(x), -1)
                xsa = x.reshape(shape)
                outsa = out.reshape(shape)
                axlen = xsa.shape[1]

                conn_src = self._connectivity[:, 0]
                conn_dst = self._connectivity[:, 1]
                for i in xrange(axlen):
                    data = xsa[:, i]
                    outslice = outsa[:, i]
                    if not np.any(outslice):
                        continue

                    # find all points under a slope
                    sign = np.sign(data[conn_src] - data[conn_dst])
                    no = set(conn_src[sign < 0])
                    no.update(conn_dst[sign > 0])

                    # expand to equal points
                    border = no
                    while border:
                        # forward
                        idx = np.in1d(conn_src, border)
                        conn_dst_sub = conn_dst[idx]
                        eq = np.equal(data[conn_src[idx]], data[conn_dst_sub])
                        new = set(conn_dst_sub[eq])
                        # backward
                        idx = np.in1d(conn_dst, border)
                        conn_src_sub = conn_src[idx]
                        eq = np.equal(data[conn_src_sub], data[conn_dst[idx]])
                        new.update(conn_src_sub[eq])

                        # update
                        new.difference_update(no)
                        no.update(new)
                        border = new

                    # mark vertices or whole isoline
                    if no:
                        outslice[list(no)] = False
                    elif not np.all(outslice):
                        outslice.fill(False)
            else:
                if x.ndim == 1:
                    xsa = x[:, None]
                    outsa = out[:, None]
                else:
                    xsa = x.swapaxes(0, ax)
                    outsa = out.swapaxes(0, ax)
                axlen = len(xsa)

                kernel = np.empty(xsa.shape[1:], dtype=np.bool8)

                diff = np.diff(xsa, 1, 0)

                # forward
                kernel.fill(True)
                for i in xrange(axlen - 1):
                    kernel[diff[i] > 0] = True
                    kernel[diff[i] < 0] = False
                    nodiff = diff[i] == 0
                    kernel[nodiff] *= outsa[i + 1][nodiff]
                    outsa[i + 1] *= kernel

                # backward
                kernel.fill(True)
                for i in xrange(axlen - 2, -1, -1):
                    kernel[diff[i] < 0] = True
                    kernel[diff[i] > 0] = False
                    nodiff = diff[i] == 0
                    kernel[nodiff] *= outsa[i][nodiff]
                    outsa[i] *= kernel

        return out

    def data_for_permutation(self, raw=True):
        """Retrieve data flattened for permutation

        Parameters
        ----------
        raw : bool
            Return a RawArray and a shape tuple instead of a numpy array.
        """
        # get data in the right shape
        x = self.y_perm.x
        if self._nad_ax:
            x = x.swapaxes(1, 1 + self._nad_ax)

        if not raw:
            return x.reshape((len(x), -1))

        n = reduce(operator.mul, self.y_perm.shape)
        ra = RawArray('d', n)
        ra[:] = x.ravel()  # OPT: don't copy data
        return ra, x.shape

    def _cluster_properties(self, cluster_map, cids):
        """Create a Dataset with cluster properties

        Parameters
        ----------
        cluster_map : NDVar
            NDVar in which clusters are marked by bearing the same number.
        cids : array_like of int
            Numbers specifying the clusters (must occur in cluster_map) which
            should be analyzed.

        Returns
        -------
        cluster_properties : Dataset
            Cluster properties. Which properties are included depends on the
            dimensions.
        """
        ndim = cluster_map.ndim
        n_clusters = len(cids)

        # setup compression
        compression = []
        for ax, dim in enumerate(cluster_map.dims):
            extents = np.empty((n_clusters, len(dim)), dtype=np.bool_)
            axes = tuple(i for i in xrange(ndim) if i != ax)
            compression.append((ax, dim, axes, extents))

        # find extents for all clusters
        c_mask = np.empty(cluster_map.shape, np.bool_)
        for i, cid in enumerate(cids):
            np.equal(cluster_map, cid, c_mask)
            for ax, dim, axes, extents in compression:
                np.any(c_mask, axes, extents[i])

        # prepare Dataset
        ds = Dataset()
        ds['id'] = Var(cids)

        for ax, dim, axes, extents in compression:
            properties = dim._cluster_properties(extents)
            if properties is not None:
                ds.update(properties)

        return ds

    def cluster(self, cluster_id):
        """Retrieve a specific cluster as NDVar

        Parameters
        ----------
        cluster_id : int
            Cluster id.

        Returns
        -------
        cluster : NDVar
            NDVar of the cluster, 0 outside the cluster.

        Notes
        -----
        Clusters only have stable ids for thresholded cluster distributions.
        """
        if cluster_id not in self._cids:
            raise ValueError("No cluster with id " + repr(cluster_id))

        return self.parameter_map * (self.cluster_map == cluster_id)

    def clusters(self, pmin=None, maps=True, **sub):
        """Find significant clusters

        Parameters
        ----------
        pmin : None | scalar, 1 >= p  >= 0
            Threshold p-value for clusters (for thresholded cluster tests the
            default is 1, for others 0.05).
        maps : bool
            Include in the output a map of every cluster (can be memory
            intensive if there are large statistical maps and/or many
            clusters; default True).
        [dimname] : index
            Limit the data for the distribution.

        Returns
        -------
        ds : Dataset
            Dataset with information about the clusters.
        """
        if pmin is None:
            if self.samples > 0 and self.kind != 'cluster':
                pmin = 0.05
        elif self.samples == 0:
            msg = ("Can not determine p values in distribution without "
                   "permutations.")
            if self.kind == 'cluster':
                msg += " Find clusters with pmin=None."
            raise RuntimeError(msg)

        if sub:
            param_map = self.parameter_map.sub(**sub)
        else:
            param_map = self.parameter_map

        if self.kind == 'cluster':
            if sub:
                cluster_map = self.cluster_map.sub(**sub)
                cids = np.setdiff1d(cluster_map.x, [0])
            else:
                cluster_map = self.cluster_map
                cids = np.array(self._cids)

            if len(cids):
                # measure original clusters
                cluster_v = ndimage.sum(param_map.x, cluster_map.x, cids)

                # p-values
                if self.samples:
                    # p-values: "the proportion of random partitions that
                    # resulted in a larger test statistic than the observed
                    # one" (179)
                    dist = self._aggregate_dist(**sub)
                    n_larger = np.sum(dist > np.abs(cluster_v[:, None]), 1)
                    cluster_p = n_larger / self.samples

                    # select clusters
                    if pmin is not None:
                        idx = cluster_p <= pmin
                        cids = cids[idx]
                        cluster_p = cluster_p[idx]
                        cluster_v = cluster_v[idx]

                    # p-value corrected across parc
                    if sub:
                        dist = self._aggregate_dist()
                        n_larger = np.sum(dist > np.abs(cluster_v[:, None]), 1)
                        cluster_p_corr = n_larger / self.samples
            else:
                cluster_v = cluster_p = cluster_p_corr = []

            ds = self._cluster_properties(cluster_map, cids)
            ds['v'] = Var(cluster_v)
            if self.samples:
                ds['p'] = Var(cluster_p)
                if sub:
                    ds['p_parc'] = Var(cluster_p_corr)

            threshold = self.threshold
        else:
            p_map = self.compute_probability_map(**sub)
            bin_map = np.less_equal(p_map.x, pmin)

            # threshold for maps
            if maps:
                values = np.abs(param_map.x)[bin_map]
                if len(values):
                    threshold = values.min() / 2
                else:
                    threshold = 1.

            # find clusters (reshape to internal shape for labelling)
            if self._nad_ax:
                bin_map = bin_map.swapaxes(0, self._nad_ax)
            if sub:
                raise NotImplementedError("sub")
                # need to subset connectivity!
            c_map, cids = label_clusters_binary(bin_map, self._connectivity,
                                                None)
            if self._nad_ax:
                c_map = c_map.swapaxes(0, self._nad_ax)

            # Dataset with cluster info
            cluster_map = NDVar(c_map, p_map.dims, {}, "clusters")
            ds = self._cluster_properties(cluster_map, cids)
            ds.info['clusters'] = cluster_map
            min_pos = ndimage.minimum_position(p_map.x, c_map, cids)
            ds['p'] = Var([p_map.x[pos] for pos in min_pos])

        if 'p' in ds:
            ds['sig'] = star_factor(ds['p'])

        # expand clusters
        if maps:
            shape = (ds.n_cases,) + param_map.shape
            c_maps = np.empty(shape, dtype=param_map.x.dtype)
            c_mask = np.empty(param_map.shape, dtype=np.bool_)
            for i, cid in enumerate(cids):
                np.equal(cluster_map.x, cid, c_mask)
                np.multiply(param_map.x, c_mask, c_maps[i])

            # package ndvar
            dims = ('case',) + param_map.dims
            param_contours = {}
            if self.tail >= 0:
                param_contours[threshold] = (0.7, 0.7, 0)
            if self.tail <= 0:
                param_contours[-threshold] = (0.7, 0, 0.7)
            info = _cs.stat_info(self.meas, contours=param_contours,
                                 summary_func=np.sum)
            ds['cluster'] = NDVar(c_maps, dims, info=info)
        else:
            ds.info['clusters'] = self.cluster_map

        return ds

    def find_peaks(self):
        """Find peaks in a TFCE distribution

        Returns
        -------
        ds : Dataset
            Dataset with information about the peaks.
        """
        if self.kind == 'cluster':
            raise RuntimeError("Not a threshold-free distribution")

        param_map = self._original_param_map
        probability_map = self.probability_map.x
        if self._nad_ax:
            probability_map = probability_map.swapaxes(0, self._nad_ax)

        peaks = self._find_peaks(self._original_cluster_map)
        peak_map, peak_ids = label_clusters_binary(peaks, self._connectivity,
                                                   None)

        ds = Dataset()
        ds['id'] = Var(peak_ids)
        v = ds.add_empty_var('v')
        if self.samples:
            p = ds.add_empty_var('p')

        bin_buff = np.empty(peak_map.shape, np.bool8)
        for i, id_ in enumerate(peak_ids):
            idx = np.equal(peak_map, id_, bin_buff)
            v[i] = param_map[idx][0]
            if self.samples:
                p[i] = probability_map[idx][0]

        return ds

    def compute_probability_map(self, **sub):
        """Compute a probability map

        Parameters
        ----------
        [dimname] : index
            Limit the data for the distribution.

        Returns
        -------
        probability : NDVar
            Map of p-values.
        """
        if not self.samples:
            raise RuntimeError("Can't compute probability without permutations")

        if self.kind == 'cluster':
            cpmap = np.ones(self.shape)
            if self.n_clusters:
                cids = self._cids
                dist = self._aggregate_dist(**sub)
                cluster_map = self._original_cluster_map
                param_map = self._original_param_map

                # measure clusters
                cluster_v = ndimage.sum(param_map, cluster_map, cids)

                # p-values: "the proportion of random partitions that resulted
                # in a larger test statistic than the observed one" (179)
                n_larger = np.sum(dist > np.abs(cluster_v[:, None]), 1)
                cluster_p = n_larger / self.samples

                c_mask = np.empty(self.shape, dtype=np.bool8)
                for i, cid in enumerate(cids):
                    np.equal(cluster_map, cid, c_mask)
                    cpmap[c_mask] = cluster_p[i]
            # revert to original shape
            if self._nad_ax:
                cpmap = cpmap.swapaxes(0, self._nad_ax)

            dims = self.dims[1:]
        else:
            if self.kind == 'tfce':
                stat_map = self.tfce_map
            else:
                if self.tail == 0:
                    stat_map = self.parameter_map.abs()
                elif self.tail < 0:
                    stat_map = -self.parameter_map
                else:
                    stat_map = self.parameter_map

            dist = self._aggregate_dist(**sub)
            if sub:
                stat_map = stat_map.sub(**sub)

            idx = np.empty(stat_map.shape, dtype=np.bool8)
            cpmap = np.zeros(stat_map.shape)
            for v in dist:
                cpmap += np.greater(v, stat_map.x, idx)
            cpmap /= self.samples
            dims = stat_map.dims

        info = _cs.cluster_pmap_info()
        return NDVar(cpmap, dims, info, self.name)

    def masked_parameter_map(self, pmin=0.05, **sub):
        """Create a copy of the parameter map masked by significance

        Parameters
        ----------
        pmin : None | scalar
            Threshold p-value for masking (default 0.05). For threshold-based
            cluster tests, pmin=None includes all clusters regardless of their
            p-value.

        Returns
        -------
        masked_map : NDVar
            NDVar with data from the original parameter map wherever p <= pmin
            and 0 everywhere else.
        """
        if sub:
            param_map = self.parameter_map.sub(**sub)
        else:
            param_map = self.parameter_map.copy()

        if pmin is None:
            if self.kind != 'cluster':
                msg = "pmin can only be None for thresholded cluster tests"
                raise ValueError(msg)
            c_mask = self.cluster_map.x != 0
        else:
            probability_map = self.compute_probability_map(**sub)
            c_mask = np.less_equal(probability_map.x, pmin)
        param_map.x *= c_mask
        return param_map

    @LazyProperty
    def probability_map(self):
        if self.samples:
            return self.compute_probability_map()
        else:
            return None

    @LazyProperty
    def _default_plot_obj(self):
        if self.samples:
            return [[self.parameter_map, self.probability_map]]
        else:
            return [[self.parameter_map]]

    def info_list(self, title="Computation Info"):
        "fmtxt List with information on computation"
        l = fmtxt.List(title)
        l.add_item("Eelbrain version:  %s" % self._version)
        l.add_item("Host Computer:  %s" % self._host)
        if self._init_time is not None:
            l.add_item("Created:  %s" % datetime.fromtimestamp(self._init_time)
                       .strftime('%Y-%m-%d %H:%M'))
        l.add_item("Original time:  %s" % timedelta(seconds=round(self.dt_original)))
        l.add_item("Permutation time:  %s" % timedelta(seconds=round(self.dt_perm)))
        return l


def distribution_worker(dist_array, dist_shape, in_queue):
    "Worker that accumulates values and places them into the distribution"
    n = reduce(operator.mul, dist_shape)
    dist = np.frombuffer(dist_array, np.float64, n)
    dist.shape = dist_shape
    samples = dist_shape[0]
    # logger
    t0 = tn = current_time()
    logger.info('starting permutation')
    for i in xrange(samples):
        dist[i] = in_queue.get()
        # logger
        t = current_time()
        dt = t - tn
        if i and dt > 10:
            time_left = (samples - i) * (t - t0) / i
            td = timedelta(seconds=round(time_left))
            logger.info("max stat %i received, estimated time left: %s" % (i, td))
            tn = t
    time_taken = t - t0
    td = timedelta(seconds=round(time_taken))
    logger.info("%i permutations done in %s" % (samples, td))


def permutation_worker(in_queue, out_queue, y, shape, test_func, map_args):
    "Worker for 1 sample t-test"
    n = reduce(operator.mul, shape)
    y = np.frombuffer(y, np.float64, n).reshape((shape[0], -1))
    stat_map = np.empty(shape[1:])
    stat_map_flat = stat_map.ravel()
    map_processor = get_map_processor(*map_args)
    while True:
        perm = in_queue.get()
        if perm is None:
            break
        test_func(y, stat_map_flat, perm)
        max_v = map_processor.max_stat(stat_map)
        out_queue.put(max_v)


def run_permutation(test_func, dist, iterator):
    if MULTIPROCESSING:
        workers, out_queue = setup_workers(test_func, dist)

        for perm in iterator:
            out_queue.put(perm)

        for _ in xrange(len(workers) - 1):
            out_queue.put(None)

        for w in workers:
            w.join()
            logger.debug("worker joined")
    else:
        y = dist.data_for_permutation(False)
        map_processor = get_map_processor(*dist.map_args)
        stat_map = np.empty(dist.shape)
        stat_map_flat = stat_map.ravel()
        for i, perm in enumerate(iterator):
            test_func(y, stat_map_flat, perm)
            dist.dist[i] = map_processor.max_stat(stat_map)
    dist.finalize()


def setup_workers(test_func, dist, n_workers=None):
    "Initialize workers for permutation tests"
    if n_workers is None:
        n_workers = cpu_count()
    elif n_workers < 0:
        n_workers = max(1, cpu_count() + n_workers)
    elif not isinstance(n_workers, int):
        raise TypeError("n_workers must be int, got %s" % repr(n_workers))

    logger.debug("Setting up %i worker processes..." % n_workers)
    permutation_queue = SimpleQueue()
    dist_queue = SimpleQueue()

    # permutation workers
    y, shape = dist.data_for_permutation()
    args = (permutation_queue, dist_queue, y, shape, test_func, dist.map_args)
    workers = []
    for _ in xrange(n_workers):
        w = Process(target=permutation_worker, args=args)
        w.start()
        workers.append(w)

    # distribution worker
    args = (dist.dist_array, dist.dist_shape, dist_queue)
    w = Process(target=distribution_worker, args=args)
    w.start()
    workers.append(w)

    return workers, permutation_queue


def run_permutation_me(test, dists, iterator):
    dist = dists[0]
    if dist.kind == 'cluster':
        thresholds = tuple(d.threshold for d in dists)
    else:
        thresholds = None

    if MULTIPROCESSING:
        workers, out_queue = setup_workers_me(test, dists, thresholds)

        for perm in iterator:
            out_queue.put(perm)

        for _ in xrange(len(workers) - 1):
            out_queue.put(None)

        for w in workers:
            w.join()
            logger.debug("worker joined")
    else:
        y = dist.data_for_permutation(False)
        map_processor = get_map_processor(*dist.map_args)

        stat_maps = test.preallocate((0,) + dist.shape)
        stat_maps_iter = [stat_maps[i] for i in xrange(len(stat_maps))]
        if thresholds:
            stat_maps_iter = zip(stat_maps_iter, thresholds, dists)
        else:
            stat_maps_iter = zip(stat_maps_iter, dists)

        for i, perm in enumerate(iterator):
            test.map(y, perm)
            if thresholds:
                for m, t, d in stat_maps_iter:
                    if d.do_permutation:
                        d.dist[i] = map_processor.max_stat(m, t)
            else:
                for m, d in stat_maps_iter:
                    if d.do_permutation:
                        d.dist[i] = map_processor.max_stat(m)

    for d in dists:
        if d.do_permutation:
            d.finalize()


def setup_workers_me(test_func, dists, thresholds, n_workers=None):
    "Initialize workers for permutation tests"
    if n_workers is None:
        n_workers = cpu_count()
    elif n_workers < 0:
        n_workers = max(1, cpu_count() + n_workers)
    elif not isinstance(n_workers, int):
        raise TypeError("n_workers must be int, got %s" % repr(n_workers))

    logger.debug("Setting up %i worker processes..." % n_workers)
    permutation_queue = SimpleQueue()
    dist_queue = SimpleQueue()

    # permutation workers
    dist = dists[0]
    y, shape = dist.data_for_permutation()
    args = (permutation_queue, dist_queue, y, shape, test_func, dist.map_args,
            thresholds)
    workers = []
    for _ in xrange(n_workers):
        w = Process(target=permutation_worker_me, args=args)
        w.start()
        workers.append(w)

    # distribution worker
    args = ([d.dist_array for d in dists], dist.dist_shape, dist_queue)
    w = Process(target=distribution_worker_me, args=args)
    w.start()
    workers.append(w)

    return workers, permutation_queue


def permutation_worker_me(in_queue, out_queue, y, shape, test, map_args,
                          thresholds):
    n = reduce(operator.mul, shape)
    y = np.frombuffer(y, np.float64, n).reshape((shape[0], -1))
    stat_maps = test.preallocate(shape)
    iterator = [stat_maps[i] for i in xrange(len(stat_maps))]
    if thresholds:
        iterator = zip(iterator, thresholds)
    map_processor = get_map_processor(*map_args)
    while True:
        perm = in_queue.get()
        if perm is None:
            break
        test.map(y, perm)

        if thresholds:
            max_v = [map_processor.max_stat(m, t) for m, t in iterator]
        else:
            max_v = [map_processor.max_stat(m) for m in iterator]
        out_queue.put(max_v)


def distribution_worker_me(dist_arrays, dist_shape, in_queue):
    "Worker that accumulates values and places them into the distribution"
    n = reduce(operator.mul, dist_shape)
    dists = [d if d is None else np.frombuffer(d, np.float64, n).reshape(dist_shape)
             for d in dist_arrays]
    samples = dist_shape[0]
    # logger
    t0 = tn = current_time()
    logger.info('starting permutation')
    for i in xrange(samples):
        for dist, v in izip(dists, in_queue.get()):
            if dist is not None:
                dist[i] = v
        # logger
        t = current_time()
        dt = t - tn
        if i and dt > 10:
            time_left = (samples - i) * (t - t0) / i
            td = timedelta(seconds=round(time_left))
            logger.info("max stat %i received, estimated time left: %s" % (i, td))
            tn = t
    time_taken = t - t0
    td = timedelta(seconds=round(time_taken))
    logger.info("%i permutations done in %s" % (samples, td))
