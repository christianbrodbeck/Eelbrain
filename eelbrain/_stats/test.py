"""
statistical tests for data objects

"""
from __future__ import division

import itertools
from itertools import izip

import numpy as np
import scipy.stats

from .. import fmtxt
from .._data_obj import (ascategorial, asfactor, assub, asvar, cellname,
                         Celltable, Factor, isvar)
from .permutation import resample


__test__ = False


def lilliefors(data, formatted=False, **kwargs):
    """Lilliefors test for normal distribution.

    The Lilliefors test is an adaptation of the Kolmogorov-Smirnov test. It
    is used to test the null hypothesis that data come from a normally
    distributed population, when the null hypothesis does not specify which
    normal distribution, i.e. does not specify the expected value and variance.

    Parameters
    ----------
    data : array_like

    formatted : bool
        Return a single string with the results instead of the numbers.
    kwargs :
        All keyword arguments are forwarded to :func:`scipy.stats.kstest`.

    Returns
    -------
    D : float
        The D-value of the Kolmogorov-Smirnov Test
    p_estimate : float
        The approximate p value according to Dallal and Wilkinson (1986).
        Requires minimal sample size of 5. p is reasonably accurate only when
        it is <= .1 (cf. Dallal and Wilkens).

    Notes
    -----
    Uses the scipy.stats.kstest implementation of the Kolmogorov-Smirnov test.

    References
    ----------
    Dallal, G. E. and Wilkinson, L. (1986). An Analytic Approximation to the
            Distribution of Lilliefors's Test Statistic for Normality. The
            American Statistician, 40(4), 294--296.
    Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov Test for Normality
            with Mean and Variance Unknown. Journal of the American
            Statistical Association, 62(318), 399--402.
    """
    # p values agree with R lillie.test (nortest package) on low values of p.
    # lillie.test adjusts something at p>.1
    # http://pbil.univ-lyon1.fr/library/nortest/R/nortest
    data = np.asarray(data)
    N = len(data)  # data.shape[-1] #axis]
    assert N >= 5, "sample size must be greater than 4"
    # perform Kolmogorov-Smirnov with estimated mean and std
    m = np.mean(data)  # , axis=axis)
    s = np.std(data, ddof=1)  # , axis=axis)
    D, ks_p = scipy.stats.kstest(data, 'norm', args=(m, s), **kwargs)
    # approximate p (Dallal)
    if N > 100:
        D *= (N / 100) ** .49
        N = 100
    p_estimate = np.exp(-7.01256 * D ** 2 * (N + 2.78019)             \
                     + 2.99587 * D * (N + 2.78019) ** .5 - .122119    \
                     + .974598 / (N ** .5) + 1.67997 / N)
    # approximate P (Molin & Abdi)
    L = D  # ???
    b2 = 0.08861783849346
    b1 = 1.30748185078790
    b0 = 0.37872256037043
    A = (-(b1 + N) + np.sqrt((b1 + N) ** 2 - 4 * b2 * (b0 - L ** -2))) / 2 * b2
    Pr = -.37782822932809         \
         + 1.67819837908004 * A     \
         - 3.02959249450445 * A ** 2  \
         + 2.80015798142101 * A ** 3  \
         - 1.39874347510845 * A ** 4  \
         + 0.40466213484419 * A ** 5  \
         - 0.06353440854207 * A ** 6  \
         + 0.00287462087623 * A ** 7  \
         + 0.00069650013110 * A ** 8  \
         - 0.00011872227037 * A ** 9  \
         + 0.00000575586834 * A ** 10
    #
    if formatted:
        txt = "D={0:.4f}, Dallal p={1:.4f}, Molin&Abdi p={2:.4f}"
        return txt.format(D, p_estimate, Pr)
    else:
        return D, p_estimate



def _hochberg_threshold(N, alpha=.05):
    j = np.arange(N)
    threshold = alpha / (N - j)
    return threshold



def mcp_adjust(ps, method='Hochberg'):
    """Adjust p-values for multiple comparison

    Parameters
    ----------
    ps : sequence of scalar
        P-values.
    method : 'hochberg' | 'bonferroni' | 'holm'
        Correction method. Default is 'hochberg'.

    Returns
    -------
    adjusted_ps : list of scalar
        Adjusted p-values.
    """
    n = len(ps)
    if n <= 1:
        return ps
    method_ = method.lower()
    if method_ == 'bonferroni':
        ps_adjusted = [p * n for p in ps]
    elif method_ in ('hochberg', 'holm'):
        ascsort = np.argsort(ps)
        ps_asc = np.array(ps)[ascsort]
        iout_asc = np.arange(n)[ascsort]
        ps_adjusted = [-1] * n
        p_buffer = 1
        if method_ == 'holm':
            for i in range(n):
                p = ps_asc[i]
                p_adj = (n - i) * p
                p_buffer = max(p_buffer, p_adj)
                ps_adjusted[iout_asc[i]] = p_buffer
        elif method_ == 'hochberg':
            for i in range(1, n + 1):
                p = ps_asc[-i]
                p_adj = (i) * p
                p_buffer = min(p_adj, p_buffer)
                ps_adjusted[iout_asc[-i]] = p_buffer
    else:
        msg = ('%r is not a valid argument for multiple comparison correction '
               'method' % method)
        raise ValueError(msg)

    return ps_adjusted



def _get_correction_caption(corr, n):
    if corr == 'Hochberg':
        return "(* Corrected after Hochberg, 1988)"
    elif corr == 'Bonferroni':
        return "(* Bonferroni corrected for %i tests)" % n
    elif corr == 'Holm':
        return "(* Corrected after Holm)"
    else:
        return "(* Uncorrected)"


def star(p_list, out=str, levels=True, trend=False, eq_strlen=False):
    """

    out=str: convert n stars into string containing '**'
       =int: leave n stars as integer

    levels: {p: string, ...} dictionary. Default (levels=True) creates the
            levels {.05 : '*',    . trend=True adds {.1: "'"}.
                    .01 : '**',
                    .001: '***'}

    eq_strlen: ("equalize string lengths") when strings are returned make sure
               they all have the same length (False by default).

    """
    # set default levels
    if levels is True:
        levels = {.05 : '*',
                  .01 : '**',
                  .001: '***'}
        if trend is True:
            levels[.1] = "`"  # "`"
        elif fmtxt.isstr(trend):
            levels[.1] = trend
    elif trend:
        raise AssertionError("'trend' kwarg only meaningful when levels is True")

    a_levels = sorted(levels.keys(), reverse=True)
    symbols = [''] + [levels[p] for p in a_levels]

    # allow input (p_list) to contain single p-value
    if np.iterable(p_list):
        int_out = False
    else:
        int_out = True
        p_list = [p_list]

    nstars = np.zeros(len(p_list), dtype=int)
    p_list = np.asarray(p_list)
    for a in a_levels:
        nstars += (p_list <= a)

    # out
    if out == str:
        if eq_strlen:
            maxlen = max([len(s) for s in symbols])
            out = [(symbols[n]).ljust(maxlen) for n in nstars]
        else:
            out = [symbols[n] for n in nstars]
    else:
        out = nstars
    # out format to in format
    if int_out:
        return out[0]
    else:
        return out


def star_factor(p, levels={.1: '`', .05 : '*', .01 : '**', .001: '***'}):
    """Create a factor with stars for a sequence of p-values

    Parameters
    ----------
    p : sequence of scalar
        Sequence of p-values.
    levels : dict {scalar: str}
        {value: star-mark} dictionary.

    Returns
    -------
    stars : Factor
        Factor with the appropriate star marking for each item in p.
    """
    sorted_levels = sorted(levels, reverse=True)
    star_labels = {i: levels[v] for i, v in enumerate(sorted_levels, 1)}
    star_labels[0] = ''
    level_values = np.reshape(sorted_levels, (-1, 1))
    stars = Factor(np.sum(p < level_values, 0), labels=star_labels)
    return stars


def ttest(Y, X=None, against=0, match=None, sub=None, corr='Hochberg',
           title='{desc}', ds=None):
    """T tests for one or more samples.

    parameters
    ----------
    Y : Var
        Dependent variable
    X : None | categorial
        Perform tests separately for all categories in X.
    against : scalar | str | tuple
        Baseline against which to test (scalar or category in X).
    match : None | Factor
        Repeated measures factor.
    sub : index
        Only use part of the data.
    corr : None | 'hochberg' | 'bonferroni' | 'holm'
        Method for multiple comparison correction (default 'hochberg').
    title : str
        Title for the table.
    ds : None | Dataset
        If a Dataset is given, all data-objects can be specified as names of
        Dataset variables

    Returns
    -------
    table : FMText Table
        Table with results.
    """
    ct = Celltable(Y, X, match, sub, ds=ds, coercion=asvar)

    par = True
    if par:
        infl = '' if X is None else 's'
        if ct.Y.name is None:
            title_desc = "T-test%s against %s" % (infl, cellname(against))
        else:
            title_desc = "T-test%s of %s against %s" % (infl, ct.Y.name, cellname(against))
        statistic_name = 't'
    else:
        raise NotImplementedError

    names = []
    diffs = []
    ts = []
    dfs = []
    ps = []

    if isinstance(against, (str, tuple)):
        if against not in ct.cells:
            x_repr = 'X' if ct.X.name is None else repr(ct.X.name)
            raise ValueError("agains=%r: %r is not a cell in %s"
                             % (against, against, x_repr))
        k = len(ct.cells) - 1
        baseline = ct.data[against]

        for cell in ct.cells:
            if cell == against:
                continue
            names.append(cell)
            if match is not None and ct.within[cell, against]:
                diffs.append(ct.data[cell].mean() - baseline.mean())
                t, p = scipy.stats.ttest_rel(ct.data[cell], baseline)
                df = len(baseline) - 1
            else:
                data = ct.data[cell]
                diffs.append(data.mean() - baseline.mean())
                t, p = scipy.stats.ttest_ind(data, baseline)
                df = len(baseline) + len(data) - 2
            ts.append(t)
            dfs.append(df)
            ps.append(p)

    elif np.isscalar(against):
        k = len(ct.cells)

        for cell in ct.cells:
            label = cellname(cell)
            data = ct.data[cell].x
            t, p = scipy.stats.ttest_1samp(data, against)
            df = len(data) - 1
            names.append(label)
            diffs.append(data.mean() - against)
            ts.append(t)
            dfs.append(df)
            ps.append(p)
    else:
        raise TypeError("against=%s" % repr(against))

    if k <= 1:
        corr = None

    if corr:
        ps_adjusted = mcp_adjust(ps, corr)
    else:
        ps_adjusted = ps
    stars = star(ps_adjusted, out=str)
    if len(set(dfs)) == 1:
        df_in_header = True
    else:
        df_in_header = False

    table = fmtxt.Table('l' + 'r' * (4 - df_in_header + bool(corr)))
    table.title(title.format(desc=title_desc))
    if corr:
        table.caption(_get_correction_caption(corr, k))

    # header
    table.cell("Effect")
    table.cell("Difference")
    if df_in_header:
        table.cell(fmtxt.symbol(statistic_name, dfs[0]))
    else:
        table.cell(statistic_name, 'math')
        table.cell('df', 'math')
    table.cell('p', 'math')
    if corr:
        table.cell(fmtxt.symbol('p', corr))
    table.midrule()

    # body
    for name, diff, t, mark, df, p, p_adj in izip(names, diffs, ts, stars, dfs, ps, ps_adjusted):
        table.cell(name)
        table.cell(diff)
        table.cell(fmtxt.stat(t, stars=mark, of=3))
        if not df_in_header:
            table.cell(df)

        table.cell(fmtxt.p(p))
        if corr:
            table.cell(fmtxt.p(p_adj))
    return table



def pairwise(Y, X, match=None, sub=None, ds=None,  # data in
             par=True, corr='Hochberg', trend=True,  # stats
             title='{desc}', mirror=False,  # layout
             ):
    """Pairwise comparison table.

    Parameters
    ----------
    Y : Var
        Dependent measure.
    X : categorial
        Categories to compare.
    match : None | Factor
        Repeated measures factor.
    sub : None | index-array
        Perform tests with a subset of the data.
    ds : None | Dataset
        If a Dataset is given, all data-objects can be specified as names of
        Dataset variables

    Returns
    -------
    table : FMText Table
        Table with results.
    """
    ct = Celltable(Y, X, match=match, sub=sub, ds=ds, coercion=asvar)
    test = _pairwise(ct.get_data(), within=ct.all_within, parametric=par,
                     corr=corr, trend=trend)

    # extract test results
    k = len(ct)
    indexes = test['pw_indexes']
    statistic = test['statistic']
    _K = test[statistic]
    _P = test['p']
    if corr:
        _Pc = mcp_adjust(_P, corr)
    _df = test['df']
    _NStars = test['stars']
    symbols = test['symbols']

    # create TABLE
    table = fmtxt.Table('l' + 'l' * (k - 1 + mirror))
    title_desc = "Pairwise {0}".format(test['test'])
    table.title(title.format(desc=title_desc))
    table.caption(test['caption'])

    # headings
    table.cell()
    cellnames = ct.cellnames()
    for name in cellnames[1 - mirror:]:
        table.cell(name)
    table.midrule()

    # tex_df = fmtxt.Element(df, "_", digits=0)
    if corr and not mirror:
        subrows = range(3)
    else:
        subrows = range(2)

    for row in range(0, k - 1 + mirror):
        for subrow in subrows:  # contains t/p
            # names column
            if subrow is 0:
                table.cell(cellnames[row])
            else:
                table.cell()
            # rows
            for col in range(1 - mirror, k):
                if row == col:
                    table.cell()
                elif col > row:
                    index = indexes[(row, col)]
                    if subrow is 0:
                        table.cell(fmtxt.eq(statistic, _K[index], _df[index],
                                            stars=symbols[index], of=3 + trend))
                    elif subrow is 1:
                        table.cell(fmtxt.peq(_P[index]))
                    elif subrow is 2:
                        table.cell(fmtxt.peq(_Pc[index], 'c'))
                elif mirror and corr and subrow == 0:
                    index = indexes[(col, row)]
                    table.cell(fmtxt.P(_Pc[index]))
                else:
                    table.cell()
    return table


def _pairwise(data, within=True, parametric=True, corr='Hochberg',
              levels=True, trend=True):
    """
    Parameters
    ----------
    data
        list of groups/treatments
    corr : 'Hochberg' | 'Holm' | 'Bonferroni'
        MCP.

    Returns
    -------
    results : dict
        dictionary with results:
        'test': test name
        'caption': information about correction
        'statistic': abbreviation used for the staistic (e.g. 'Q')
        statistic: list of values
        'df': df
        'p': list of corresponding pa values
        'stars': list of n stars (ints)
        'pw_indexes': dict linking table index (i,j) to the list index for p etc.
    """
    # find test
    k = len(data)
    if k < 3:  # need no correction for single test
        corr = None
    if parametric:
        test_name = "t-Tests ({0} samples)"
        statistic = "t"
        if within:
            test_func = scipy.stats.ttest_rel
            test_name = test_name.format('paired')
        else:
            test_func = scipy.stats.ttest_ind
            test_name = test_name.format('independent')
    elif within:
        test_name = "Wilcoxon Signed-Rank Test"
        test_func = scipy.stats.wilcoxon
        statistic = "z"
    else:
        test_name = "Mann-Whitney U Test"
        raise NotImplementedError("mannwhitneyu returns one-sided p")
        test_func = scipy.stats.mannwhitneyu
        statistic = "u"

    # perform test
    _K = []  # kennwerte
    _P = []
    _df = []
    i = 0
    indexes = {}
    for x in range(k):
        for y in range(x + 1, k):
            Y1, Y2 = data[x], data[y]
            t, p = test_func(Y1, Y2)
            _K.append(t)
            if within:
                _df.append(len(Y1) - 1)
            else:
                _df.append(len(Y1) + len(Y2) - 2)

            _P.append(p)
            indexes[(x, y)] = indexes[(y, x)] = i
            i += 1
    # add stars
    if corr:
        p_adjusted = mcp_adjust(_P, corr)
    else:
        p_adjusted = _P
    _NStars = star(p_adjusted, int, levels, trend)
    _str_Stars = star(p_adjusted, str, levels, trend)
    caption = _get_correction_caption(corr, len(_P))
    # prepare output
    out = {'test': test_name,
           'caption': caption,
           'statistic': statistic,
           statistic: _K,
           'df': _df,
           'p': _P,
           'stars': _NStars,
           'symbols': _str_Stars,
           'pw_indexes': indexes}
    return out



def correlations(Y, Xs, cat=None, sub=None, ds=None, levels=[.05, .01, .001],
                 diff=None, pmax=None, nan=True):  # , match=None):
    """Correlation with one or more predictors.

    Parameters
    ----------
    Y : Var
        First variable
    X : Var | list of Var
        second variable (or list of variables).
    cat : categorial
        Show correlations separately for different groups in the
        data. Can be a ``Factor`` (the correlation for each level is shown
        separately) or an array of ``bool`` values (e.g. from a comparison like
        ``Stim==1``)
    levels : list of float
        Significance levels to mark.
    diff :
        (Factor, cat_1, cat_2)
    sub :
        Use only a subset of the data
    pmax : float | None
        Don't show correlations with p > pmax
    nan : bool
        Display correlation which yield NAN;

    Returns
    -------
    table : FMText Table
        Table with correlations.
    """
    sub = assub(sub, ds)
    Y = asvar(Y, sub, ds)
    if isvar(Xs) or isinstance(Xs, basestring):  # FIXME: better way to specify Xs
        Xs = (Xs,)
    Xs = [asvar(X, sub, ds) for X in Xs]
    if cat is not None:
        cat = ascategorial(cat, sub, ds)

    levels = np.array(levels)

    if diff is not None:
        raise NotImplementedError

    if cat is None:
        table = fmtxt.Table('l' * 4)
        table.cells('Variable', 'r', 'p', 'n')
    else:
        table = fmtxt.Table('l' * 5)
        table.cells('Variable', 'Category', 'r', 'p', 'n')

    table.midrule()
    table.title("Correlations with %s" % (Y.name))

    table._my_nan_count = 0

    for X in Xs:
        if cat is None:
            _corr_to_table(table, Y, X, cat, levels, pmax=pmax, nan=nan)
        else:
            printXname = True
            for cell in cat.cells:
                tlen = len(table)
                sub = (cat == cell)
                _corr_to_table(table, Y, X, sub, levels, pmax=pmax, nan=nan,
                               printXname=printXname, label=cellname(cell))

                if len(table) > tlen:
                    printXname = False

    # last row
    if pmax is None:
        p_text = ''
    else:
        p_text = 'all other p>{p}'.format(p=pmax)
    if nan is False and table._my_nan_count > 0:
        nan_text = '%s NANs' % table._my_nan_count
    else:
        nan_text = ''
    if p_text or nan_text:
        if p_text and nan_text:
            text = ', '.join([p_text, nan_text])
        else:
            text = ''.join([p_text, nan_text])
        table.cell("(%s)" % text)
    return table



def _corr(Y, X, sub=None):
    """
    index has to be bool array; returns r, p, n

    """
    if sub is not None:
        Y = Y[sub]
        X = X[sub]
    n = len(Y)
    assert n == len(X)
    df = n - 2
    r = np.corrcoef(Y.x, X.x)[0, 1]
    t = r / np.sqrt((1 - r ** 2) / df)
    p = scipy.stats.t.sf(np.abs(t), df) * 2
    return r, p, n



def _corr_to_table(table, Y, X, categories, levels, printXname=True, label=False,
                   pmax=None, nan=True):
    r, p, n = _corr(X, Y, categories)
    if (pmax is None) or (p <= pmax):
        if nan or (not np.isnan(r)):
            nstars = np.sum(p <= levels)
            if printXname:
                table.cell(X.name)
            else:
                table.cell()
            if label:
                table.cell(label)
            table.cell(fmtxt.stat(r, '%.3f', nstars, len(levels), drop0=True))
            table.cell(fmtxt.P(p))
            table.cell(n)
        else:
            table._my_nan_count += 1



class bootstrap_pairwise(object):
    def __init__(self, Y, X, match=None, sub=None,
                 samples=1000, replacement=True,
                 title="Bootstrapped Pairwise Tests", ds=None):
        sub = assub(sub, ds)
        Y = asvar(Y, sub, ds)
        X = asfactor(X, sub, ds)
        assert len(Y) == len(X), "data length mismatch"
        if match is not None:
            match = ascategorial(match, sub, ds)
            assert len(match) == len(Y), "data length mismatch"

        # prepare data container
        resampled = np.empty((samples + 1, len(Y)))  # sample X subject within category
        resampled[0] = Y.x
        # fill resampled
        for i, Y_ in enumerate(resample(Y, samples, replacement, match), 1):
            resampled[i] = Y_.x
        self.resampled = resampled

        cells = X.cells
        n_groups = len(cells)

        if match is not None:
            # if there are several values per X%match cell, take the average
            # T: indexes to transform Y.x to [X%match, value]-array
            match_cell_ids = match.cells
            group_size = len(match_cell_ids)
            T = None; i = 0
            for X_cell in cells:
                for match_cell in match_cell_ids:
                    source_indexes = np.where((X == X_cell) * (match == match_cell))[0]
                    if T is None:
                        n_cells = n_groups * group_size
                        T = np.empty((n_cells, len(source_indexes)), dtype=int)
                    T[i, :] = source_indexes
                    i += 1

            if T.shape[1] == 1:
                T = T[:, 0]
                ordered = resampled[:, T]
            else:
                ordered = resampled[:, T].mean(axis=2)
            self.ordered = ordered

            # t-tests
            n_comparisons = sum(range(n_groups))
            t = np.empty((samples + 1, n_comparisons))
            comp_names = []
            one_group = np.arange(group_size)
            groups = [one_group + i * group_size for i in range(n_groups)]
            for i, (g1, g2) in enumerate(itertools.combinations(range(n_groups), 2)):
                group_1 = groups[g1]
                group_2 = groups[g2]
                diffs = ordered[:, group_1] - ordered[:, group_2]
                t[:, i] = np.mean(diffs, axis=1) * np.sqrt(group_size) / np.std(diffs, axis=1, ddof=1)
                comp_names.append(' - '.join((cells[g1], cells[g2])))

            self.diffs = diffs
            self.t_resampled = np.max(np.abs(t[1:]), axis=1)
            self.t = t = t[0]
        else:
            raise NotImplementedError

        self._Y = Y
        self._X = X
        self._group_names = cells
        self._group_data = np.array([ordered[0, g] for g in groups])
        self._group_size = group_size
        self._df = group_size - 1
        self._match = match
        self._n_samples = samples
        self._replacement = replacement
        self._comp_names = comp_names
        self._p_parametric = self.test_param(t)
        self._p_boot = self.test_boot(t)
        self.title = title

    def __repr__(self):
        out = ['bootstrap_pairwise(', self._Y.name, self._X.name]
        if self._match:
            out.append('match=%s ' % self._match.name)
        out.append('saples=%i ' % self._n_samples)
        out.append('replacement=%s)' % self._replacement)
        return ''.join(out)

    def __str__(self):
        return str(self.table())

    def table(self):
        table = fmtxt.Table('lrrrr')
        table.title(self.title)
        table.caption("Results based on %i samples" % self._n_samples)
        table.cell('Comparison')
        table.cell(fmtxt.symbol('t', self._df))
        table.cell(fmtxt.symbol('p', 'param'))
        table.cell(fmtxt.symbol('p', 'corr'))
        table.cell(fmtxt.symbol('p', 'boot'))
        table.midrule()

        p_corr = mcp_adjust(self._p_parametric)
        stars_parametric = star(p_corr)
        stars_boot = star(self._p_boot)

        for name, t, p1, pc, s1, p2, s2 in izip(self._comp_names, self.t,
                                                self._p_parametric, p_corr,
                                                stars_parametric,
                                                self._p_boot, stars_boot):
            table.cell(name)
            table.cell(t, fmt='%.2f')
            table.cell(fmtxt.p(p1))
            table.cell(fmtxt.p(pc, stars=s1))
            table.cell(fmtxt.p(p2, stars=s2))
        return table

    def plot_t_dist(self):
        """
        After:
        http://stackoverflow.com/questions/4150171/how-to-create-a-density-plot-in-matplotlib
        """
        from matplotlib import pyplot

        t = self.t_resampled
        density = scipy.stats.gaussian_kde(t)
#        density.covariance_factor = lambda : .25
#        density._compute_covariance()
        xs = np.linspace(0, max(t), 200)
        pyplot.plot(xs, density(xs))

    def plot_dv_dist(self):
        from matplotlib import pyplot

        xs = np.linspace(np.min(self._group_data), np.max(self._group_data), 200)
        for i, name in enumerate(self._group_names):
            density = scipy.stats.gaussian_kde(self._group_data[i])
            pyplot.plot(xs, density(xs), label=name)
        pyplot.legend()

    def test_param(self, t):
        return scipy.stats.t.sf(np.abs(t), self._group_size - 1) * 2

    def test_boot(self, t):
        "t: scalar or array; returns p for each t"
        test = self.t_resampled[:, None] > np.abs(t)
        return np.sum(test, axis=0) / self.t_resampled.shape[0]
