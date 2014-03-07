'''


References
----------

Fox, J. (2008). Applied regression analysis and generalized linear models.
    2nd ed. Sage, Los Angeles.

Hopkins, K. D. (1976). A Simplified Method for Determining Expected Mean
    Squares and Error Terms in the Analysis of Variance. Journal of
    Experimental Education, 45(2), 13--18.



Created on Oct 17, 2010

@author: christian
'''
from __future__ import division

from itertools import izip
import logging, os

import numpy as np
from numpy import dot
from scipy.linalg import lstsq
import scipy.stats

from ... import fmtxt
from ...utils import LazyProperty
from ...utils.print_funcs import strdict
from ..data_obj import (isvar, asvar, assub, isbalanced, isnestedin, hasrandom,
                        find_factors, Model, asmodel)
from .stats import ftest_p
from . import test


_max_array_size = 26  # constant for max array size in LMFitter

# Method to use for least squares estimation:
# (0) Use scipy.linalg.lstsq
# (1) Use lstsq after Fox (2008) with caching of the model transformation
_lmf_lsq = 1  # for the LMFitter class
_lm_lsq = 0  # for the LM class



class hopkins_ems(dict):
    """
    A dictionary supplying for each of a model's effects the components of
    the F-test denominator (error term).

    "The E(MS) for any source of variation for any ANOVA model, in addition
    to the specified effect (main effect or interaction), includes the
    interaction of this effect with any random factor or combinations of
    random factors, plus any random effect nested within the specified effect,
    plus any random effect that either crosses or is nested within each
    ingredient and the specified effect." (Hopkins, 1976: 18)

    """
    def __init__(self, X):
        """
        Components of the F-test denominator according to Hopkins (1976)

        Parameters
        ----------
        X : Model
            ANOVA model. Needs to balanced and completely specified.
        """
        super(hopkins_ems, self).__init__()

        if X.df_error > 0:
            err = "Hopkins E(MS) estimate requires a fully specified model"
            raise ValueError(err)
        elif not any(f.random for f in find_factors(X)):
            err = ("Need at least one random effect in fully specified model "
                   "(got %s)" % X.name)
            raise ValueError(err)
        elif not isbalanced(X):
            logging.warn('X is not balanced')

        for e in X.effects:
            self[e] = _find_hopkins_ems(e, X)

    def __repr__(self):
        items = {}
        for k, v in self.iteritems():
            kstr = ' %s' % k.name
            vstr = '(%s)' % ''.join(e.name + ', ' for e in v)
            items[kstr] = vstr
        return strdict(items, fmt='%s')


def _hopkins_test(e, e2):
    """
    Tests whether e2 is in the E(MS) of e.

    e : effect
        effect whose E(MS) is being constructed
    e2 : effect
        Model effect which is tested for inclusion in E(MS) of e

    """
    if e is e2:
        return False
    else:
        e_factors = find_factors(e)
        e2_factors = find_factors(e2)

        a = np.all([(f in e_factors or f.random) for f in e2_factors])
        b = np.all([(f in e2_factors or isnestedin(e2, f)) for f in e_factors])

        return a and b

def _find_hopkins_ems(e, X):
    return tuple(e2 for e2 in X.effects if _hopkins_test(e, e2))



def is_higher_order(e1, e0):
    """
    Returns True if e1 is a higher order term of e0 (i.e., all factors in e0
    are contained in e1).

    e1, e0 : effects
        The effects to compare.

    """
    f1s = find_factors(e1)
    return all(f in f1s for f in find_factors(e0))



class LM(object):
    """
    Fit a linear model to a dependent variable


    Attributes
    ----------

    F, p : scalar
        Test of the null-hypothesis that the model does not explain a
        significant amount of the variance in the dependent variable.

    """
    def __init__(self, Y, X, sub=None, ds=None):
        """
        Fit the model X to the dependent variable Y.

        Parameters
        ----------
        Y : Var
            Dependent variable.
        X : Model
            Model.
        sub : None | index
            Only use part of the data
        """
        # prepare input
        sub = assub(sub, ds)
        Y = asvar(Y, sub, ds)
        X = asmodel(X, sub, ds)

        assert len(Y) == len(X)
        assert X.df_error > 0

        # fit
        if _lm_lsq == 0:  # use scipy (faster)
            beta, SS_res, _, _ = lstsq(X.full, Y.x)
        elif _lm_lsq == 1:  # Fox
            # estimate least squares approximation
            beta = X.fit(Y)
            # estimate
            values = beta * X.full
            Y_est = values.sum(1)
            self._residuals = residuals = Y.x - Y_est
            SS_res = np.sum(residuals ** 2)
            if not Y.mean() == Y_est.mean():
                msg = ("Y.mean() != Y_est.mean() (%s vs "
                       "%s)" % (Y.mean(), Y_est.mean()))
                logging.warning(msg)
        else:
            raise ValueError

        # SS total
        SS_total = self.SS_total = np.sum((Y.x - Y.mean()) ** 2)
        df_total = self.df_total = X.df_total
        self.MS_total = SS_total / df_total

        # SS residuals
        self.SS_res = SS_res
        df_res = self.df_res = X.df_error
        self.MS_res = SS_res / df_res

        # SS explained
#        SS_model = self.SS_model = np.sum((Y_est - Y.mean())**2)
        SS_model = self.SS = self.SS_model = SS_total - SS_res
        df_model = self.df = self.df_model = X.df
        self.MS_model = self.MS = SS_model / df_model

        # store stuff
        self.Y = Y
        self.X = X
        self.sub = sub
        self.beta = beta

    def __repr__(self):
        # repr kwargs
        kwargs = []
        if self.sub:
            kwargs.append(('sub', getattr(self.sub, 'name', '<...>')))

        fmt = {'Y': getattr(self.Y, 'name', '<Y>'),
               'X': repr(self.X)}
        if kwargs:
            fmt['kw'] = ', '.join([''] + map('='.join, kwargs))
        else:
            fmt['kw'] = ''
        return "LM({Y}, {X}{kw})".format(**fmt)

    def anova(self, title='ANOVA', empty=True, ems=False):
        """
        returns an ANOVA table for the linear model

        """
        X = self.X
        values = self.beta * self.X.full

        if X.df_error == 0:
            e_ms = hopkins_ems(X)
        elif hasrandom(X):
            err = ("Models containing random effects need to be fully "
                   "specified.")
            raise NotImplementedError(err)
        else:
            e_ms = False

        # table head
        table = fmtxt.Table('l' + 'r' * (5 + ems))
        if title:
            table.title(title)

        if not isbalanced(X):
            table.caption("Warning: Model is unbalanced, use anova class")

        table.cell()
        headers = ["SS", "df", "MS"]
        if ems:
            headers += ["E(MS)"]
        headers += ["F", "p"]
        for hd in headers:
            table.cell(hd, r"\textbf", just='c')
        table.midrule()

        # MS for factors (Needed for models involving random effects)
        MSs = {}
        SSs = {}
        for e in X.effects:
            idx = X.full_index[e]
            SSs[e] = SS = np.sum(values[:, idx].sum(1) ** 2)
            MSs[e] = (SS / e.df)

        # table body
        results = {}
        for e in X.effects:
            MS = MSs[e]
            if e_ms:
                e_EMS = e_ms[e]
                df_d = sum(c.df for c in e_EMS)
                MS_d = sum(MSs[c] for c in e_EMS)
                e_ms_name = ' + '.join(repr(c) for c in e_EMS)
            else:
                df_d = self.df_res
                MS_d = self.MS_res
                e_ms_name = "Res"

            # F-test
            if MS_d != False:
                F = MS / MS_d
                p = 1 - scipy.stats.distributions.f.cdf(F, e.df, df_d)
                stars = test.star(p)
                tex_stars = fmtxt.Stars(stars)
                F_tex = [F, tex_stars]
            else:
                F_tex = None
                p = None
            # add to table
            if e_ms_name or empty:
                table.cell(e.name)
                table.cell(SSs[e])
                table.cell(e.df, fmt='%i')
                table.cell(MS)
                if ems:
                    table.cell(e_ms_name)
                table.cell(F_tex, mat=True)
                table.cell(fmtxt.p(p))
            # store results
            results[e.name] = {'SS': SS, 'df': e.df, 'MS': MS, 'E(MS)': e_ms_name,
                             'F': F, 'p':p}

        # Residuals
        if self.df_res > 0:
            table.cell("Residuals")
            table.cell(self.SS_res)
            table.cell(self.df_res, fmt='%i')
            table.cell(self.MS_res)

        return table

    @LazyProperty
    def F(self):
        return self.MS_model / self.MS_res

    @LazyProperty
    def p(self):
        return scipy.stats.distributions.f.sf(self.F, self.df_model, self.df_res)

    @LazyProperty
    def regression_table(self):
        """
        Not fully implemented!

        A table containing slope coefficients for all effects.

        """
        # prepare table
        table = fmtxt.Table('l' * 4)
        df = self.X.df_error
        table.cell()
        table.cell('\\beta', mat=True)
        table.cell('T_{%i}' % df, mat=True)
        table.cell('p', mat=True)
        table.midrule()
        #
        q = 1  # track start location of effect in Model.full
        ne = len(self.X.effects)
        for ie, e in enumerate(self.X.effects):
            table.cell(e.name + ':')
            table.endline()
            for i, name in enumerate(e.beta_labels):  # Fox pp. 106 ff.
                beta = self.beta[q + i]
#                    Evar_pt_est = self.SS_res / df
                # SEB
#                    SS = (self.values[q+i])**2
                T = 0
                p = 0
                # todo: T/p
                table.cell(name)
                table.cell(beta)
                table.cell(T)
                table.cell(fmtxt.p(p))
            q += e.df
            if ie < ne - 1:
                table.empty_row()
        return table

    @LazyProperty
    def residuals(self):
        values = self.beta * self.X.full
        Y_est = values.sum(1)
        return self.Y.x - Y_est



class LMFitter(object):
    """
    Object for efficiently fitting a model to multiple dependent variables.

    Notes
    -----
    Currently only implemented for balanced models.
    E(MS) for F statistic after Hopkins (1976)

    """
    def __init__(self, x, y_shape=None):
        """
        Object for efficiently fitting a model to multiple dependent variables.

        Parameters
        ----------
        x : Model
            Model which will be fitted to the data.
        y_shape : None | tuple
            Data shape (if known) will allow preallocation of containers for
            intermediate results.
        """
        # prepare input
        x = asmodel(x)
        if not isbalanced(x):
            raise NotImplementedError("Unbalanced models")
        self._x_full = x.full

        full_model = (x.df_error == 0)
        if full_model:
            E_MS = hopkins_ems(x)
            df_den = {e: sum(e_.df for e_ in E_MS[e]) for e in x.effects}
            effects = tuple(e for e in x.effects if df_den[e])
        elif hasrandom(x):
            err = ("Models containing random effects need to be fully "
                   "specified.")
            raise NotImplementedError(err)
        else:
            E_MS = None
            effects = x.effects
            df_den = {e: x.df_error for e in effects}

        # pre-compute dfs
        dfs_nom = [e.df for e in effects]
        dfs_denom = [df_den[e] for e in effects]

        # determine how many tests can be done in one call
        self._max_n_tests = int(2 ** _max_array_size // x.df ** 2)

        if _lmf_lsq == 0:
            pass
        elif _lmf_lsq == 1:
            self._Xsinv = x.Xsinv
        else:
            raise ValueError('version')

        # store public attributes
        self.x = x
        self.n_obs = len(x)
        self.full_model = full_model
        self.effects = effects
        self.n_effects = len(effects)
        self.df_den = df_den
        self.dfs_nom = dfs_nom
        self.dfs_denom = dfs_denom
        self.E_MS = E_MS
        self._flat_f_maps = None

        # preallocate large arrays
        self._preallocate_internals(y_shape)

    def _preallocate_internals(self, y_shape):
        self.y_shape = y_shape
        if y_shape is None:
            self._buffer_obt = None
            self._buffer_ot = None
            self._buffer_bt = None
            self._buffer_t = None
        else:
            n_obs, n_betas = self._x_full.shape
            n_tests = min(self._max_n_tests, np.product(y_shape[1:]))
            self._buffer_obt = np.empty((n_obs, n_betas, n_tests))
            self._buffer_ot = np.empty((n_obs, n_tests))
            self._buffer_bt = np.empty((n_betas, n_tests))
            self._buffer_t = np.empty(n_tests)

    def __repr__(self):
        return 'LMFitter((%s))' % self.x.name

    def map(self, Y, out=None):
        """
        Fits the model to multiple dependent variables and returns arrays of
        F-values and optionally p-values.

        Parameters
        ----------
        Y : np.array
            Assumes that the first dimension of Y provides cases.
            Other than that, shape is free to vary and output shape will match
            input shape.
        out : list of array
            List of arrays in which to place the resulting (ravelled) f-maps.
            Can only be used in conjunction with ``p=False``.

        Returns
        -------
        f_maps : list
            A list with maps of F values (order corresponding to self.effects).
        """
        n_obs = self.n_obs

        original_shape = Y.shape
        if original_shape[0] != n_obs:
            raise ValueError("first dimension of Y must contain cases")
        if len(original_shape) > 2:
            Y = Y.reshape((n_obs, -1))
        out_shape = original_shape[1:]
        n_tests = Y.shape[1]

        # find result container
        if out is None:
            if self._flat_f_maps is None:
                f_maps = [np.empty(out_shape) for _ in xrange(self.n_effects)]
                flat_maps = tuple(f_map.ravel() for f_map in f_maps)
            else:
                f_maps = None
                flat_maps = self._flat_f_maps
        else:
            f_maps = None
            flat_maps = out

        # Split Y that are too long
        if n_tests > self._max_n_tests:
            splits = xrange(0, n_tests, self._max_n_tests)

            msg = ("LMFitter: Y.shape=%s; splitting Y at %s" %
                   (Y.shape, list(splits)))
            logging.debug(msg)

            # compute f-maps
            for s in splits:
                s1 = s + self._max_n_tests
                y_sub = Y[:, s:s1]
                out_ = tuple(f_map[s:s1] for f_map in flat_maps)
                self.map(y_sub, out_)

            return f_maps

        # do the actual estimation
        x = self.x
        x_full = self._x_full
        full_model = self.full_model

        # pre-allocated memory
        _buffer_obt = self._buffer_obt
        _buffer_ot = self._buffer_ot
        _buffer_bt = self._buffer_bt
        _buffer_t = self._buffer_t
        if _buffer_obt is not None and n_tests != _buffer_obt.shape[2]:
            _buffer_obt = _buffer_obt[:, :, :n_tests]
            _buffer_ot = _buffer_ot[:, :n_tests]
            _buffer_t = _buffer_t[:n_tests]
            _buffer_bt = None  # needs to be C-contiguous

        # beta: coefficient X test
        if _lmf_lsq == 0:
            beta, SS_res, _, _ = lstsq(x_full, Y)
        elif _lmf_lsq == 1:
            beta = dot(self._Xsinv, Y, _buffer_bt)

        # values: observation x coefficient x test
        values = np.multiply(beta[None, :, :], x_full[:, :, None], _buffer_obt)

        # MS of the residuals
        if not full_model:
            df_res = x.df_error
            if _lmf_lsq == 1:
                Yp = values.sum(1)  # case x test
                SS_res = ((Y - Yp) ** 2).sum(0)
            MS_d = SS_res / df_res

        # collect MS of effects
        MSs = {}
        for e in x.effects:
            index = x.beta_index[e]
            # observation x test
            Yp = np.sum(values[:, index, :], 1, out=_buffer_ot)
            Sq = np.power(Yp, 2, Yp)
            # test
            SS = np.sum(Sq, 0, out=_buffer_t)
            MS = SS / e.df
            MSs[e] = MS

        # F Tests
        # n = numerator, d = denominator
        for e_n, f_map in izip(self.effects, flat_maps):
            if full_model:
                E_MS_cmp = self.E_MS[e_n]
                MS_d = _buffer_t  # re-use memory
                MS_d.fill(0)
                for e_d in E_MS_cmp:
                    MS_d += MSs[e_d]

            MS_n = MSs[e_n]
            np.divide(MS_n, MS_d, f_map)

        return f_maps

    def p_maps(self, f_maps):
        """Convert F-maps for uncorrected p-maps

        Parameters
        ----------
        f_maps : list
            List of f_maps in the same order as self.effects, as returned by
            self.map().

        Returns
        -------
        p_maps : list, optional
            A list with maps of uncorrected p values (order corresponding to
            self.effects).
        """
        p_maps = map(ftest_p, f_maps, self.dfs_nom, self.dfs_denom)
        return p_maps

    def preallocate(self, y_shape):
        """Pre-allocate an output array container.

        Returns
        -------
        f_maps : array
            Properly shaped output array. Every time .map() is called, the
            content of this array will change (and map() will not return
            anything)
        """
        if y_shape is None:
            err = "Can only preallocate output of LMFitter with known y_shape"
            raise RuntimeError(err)
        self._preallocate_internals(y_shape)
        out_shape = y_shape[1:]
        f_maps = [np.empty(out_shape) for _ in xrange(self.n_effects)]
        self._flat_f_maps = tuple(f_map.ravel() for f_map in f_maps)
        return f_maps


class incremental_F_test:
    """
    Attributes
    ----------

    lm1 : Model
        The extended model
    lm0 : Model
        The control model
    SS : scalar
        the difference in the SS explained by the two models
    df : int
        The difference in df between the two models
    MS : scalar
        The MS of the difference
    F, p : scalar
        F and p valuer of the comparison

    """
    def __init__(self, lm1, lm0, MS_e=None, df_e=None, name=None):
        """
        tests the null hypothesis that the model of lm1 does not explain more
        variance than that of lm0 (with model_1 == model_0 + q factors,  q > 0).
        If lm1 is None it is assumed that lm1 is the full model with 0 residuals.

        lm1, lm0 : Model
            The two models to compare.
        MS_e, df_e : scalar | None
            Parameters for random effects models: the Expected value of MS;
            if None, the error MS of lm1 is used (valid for fixed effects
            models).


        (Fox 2008, p. 109 f.)

        """
        if lm1 is None:
            lm1_SS_res = 0
            lm1_MS_res = 0
            lm1_df_res = 0
        else:
            lm1_SS_res = lm1.SS_res
            lm1_MS_res = lm1.MS_res
            lm1_df_res = lm1.df_res

        if MS_e is None:
            assert df_e is None
            MS_e = lm1_MS_res
            df_e = lm1_df_res
        else:
            assert df_e is not None

        SS_diff = lm0.SS_res - lm1_SS_res
        df_diff = lm0.df_res - lm1_df_res
        MS_diff = SS_diff / df_diff

        if df_e > 0:
            F = MS_diff / MS_e
            p = ftest_p(F, df_diff, df_e)
        else:
            F = None
            p = None

        self.lm0 = lm0
        self.lm1 = lm1
        self.SS = SS_diff
        self.df = df_diff
        self.MS = MS_diff
        self.F = F
        self.p = p
        self.name = name

    def __repr__(self):
        name = ' %r' % self.name if self.name else ''
        return "<incremental_F_test%s: F=%.2f, p=%.3f>" % (name, self.F, self.p)




def comparelm(lm1, lm2):
    """
    Fox (p. 109)

    """
    if lm2.df_res > lm1.df_res:
        mtemp = lm1
        lm1 = lm2
        lm2 = mtemp
    else:
        assert lm1.df_res != lm2.df_res
    SS_diff = lm1.SS_res - lm2.SS_res
    df_diff = lm1.df_res - lm2.df_res
    MS_diff = SS_diff / df_diff
    F = MS_diff / lm2.MS_res
    p = ftest_p(F, df_diff, lm2.df_res)
    stars = test.star(p).replace(' ', '')
    difftxt = "Residual SS reduction: {SS}, df difference: {df}, " + \
              "F = {F:.3f}{s}, p = {p:.4f}"
    return difftxt.format(SS=SS_diff, df=df_diff, F=F, s=stars, p=p)




class anova(object):
    """
    Fits a univariate ANOVA model. The objects' string representation is the
    anova table, so the model can be created and examined inone command::

        >>> print anova(Y, X)

    For other uses, properties of the fit can be accessed with methods and
    attributes.


    """
    def __init__(self, Y, X, sub=None,
                 title=None, empty=True, ems=None,
                 showall=False, ds=None):
        """
        Fits a univariate ANOVA model.

        Mixed effects models require full model specification so that E(MS)
        can be estimated according to Hopkins (1976)


        Parameters
        ----------
        Y : Var
            dependent variable
        X : Model
            Model to fit to Y
        empty : bool
            include rows without F-Tests (True/False)
        ems : bool | None
            display source of E(MS) for F-Tests (True/False; None = use default)
        lsq : int
            least square fitter to use;
            0 -> scipy.linalg.lstsq
            1 -> after Fox
        showall : bool
            show SS, df and MS for effects without F test
        """
#  TODO:
#         - sort model
#          - reuse lms which are used repeatedly
#          - provide threshold for including interaction effects when testing lower
#            level effects
#
#        Problem with unbalanced models
#        ------------------------------
#          - The SS of Effects which do not include the between-subject factor are
#            higher than in SPSS
#          - The SS of effects which include the between-subject factor agree with
#            SPSS

        # prepare kwargs
        Y = asvar(Y, sub=sub, ds=ds)
        X = asmodel(X, sub=sub, ds=ds)

        if len(Y) != len(X):
            raise ValueError("Y and X must describe same number of cases")

        # save args
        self.Y = Y
        self.X = X
        self.title = title
        self.show_ems = ems
        self._log = []

        # decide which E(MS) model to use
        if X.df_error == 0:
            rfx = 1
            fx_desc = 'Mixed'
        elif X.df_error > 0:
            if hasrandom(X):
                err = ("Models containing random effects need to be fully "
                       "specified.")
                raise NotImplementedError(err)
            rfx = 0
            fx_desc = 'Fixed'
        else:
            raise ValueError("Model Overdetermined")
        self._log.append("Using %s effects model" % fx_desc)

        # list of (name, SS, df, MS, F, p)
        self.f_tests = []
        self.names = []


        if len(X.effects) == 1:
            self._log.append("single factor model")
            lm1 = LM(Y, X)
            self.f_tests.append(lm1)
            self.names.append(X.name)
            self.residuals = lm1.SS_res, lm1.df_res, lm1.MS_res
        else:
            if rfx:
                pass  # <- Hopkins
            else:
                full_lm = LM(Y, X)
                SS_e = full_lm.SS_res
                MS_e = full_lm.MS_res
                df_e = full_lm.df_res


            for e_test in X.effects:
                skip = False
                name = e_test.name

                # find model 0
                effects = []
                excluded_e = []
                for e in X.effects:
                    # determine whether e_test
                    if e is e_test:
                        pass
                    else:
                        if is_higher_order(e, e_test):
                            excluded_e.append(e)
                        else:
                            effects.append(e)

                model0 = Model(*effects)
                if e_test.df > model0.df_error:
                    skip = "overspecified"
                else:
                    lm0 = LM(Y, model0)

                    # find model 1
                    effects.append(e_test)
                    model1 = Model(*effects)
                    if model1.df_error > 0:
                        lm1 = LM(Y, model1)
                    else:
                        lm1 = None

                    if rfx:
                        # find E(MS)
                        EMS_effects = _find_hopkins_ems(e_test, X)

                        if len(EMS_effects) > 0:
                            lm_EMS = LM(Y, Model(*EMS_effects))
                            MS_e = lm_EMS.MS_model
                            df_e = lm_EMS.df_model
                        else:
                            if lm1 is None:
                                SS = lm0.SS_res
                                df = lm0.df_res
                            else:
                                SS = lm0.SS_res - lm1.SS_res
                                df = lm0.df_res - lm1.df_res
                            MS = SS / df
                            skip = ("no Hopkins E(MS); SS=%.2f, df=%i, "
                                    "MS=%.2f" % (SS, df, MS))

                if skip:
                    self._log.append("SKIPPING: %s (%s)" % (e_test.name, skip))
                else:
                    res = incremental_F_test(lm1, lm0, MS_e=MS_e, df_e=df_e, name=name)
                    self.f_tests.append(res)
                    self.names.append(name)
            if not rfx:
                self.residuals = SS_e, df_e, MS_e

    def __repr__(self):
        return "anova(%s, %s)" % (self.Y.name, self.X.name)

    def __str__(self):
        return str(self.table())

    def print_log(self):
        out = self._log[:]
        print os.linesep.join(out)

    def table(self):
        """Create an ANOVA table

        Returns
        -------
        anova_table : eelbrain.fmtxt.Table
            Anova table.
        """
        # table head
        table = fmtxt.Table('l' + 'r' * 5)
        if self.title:
            table.title(self.title)
        table.cell()
        headers = ["SS", "df", "MS"]
        headers += ["F", "p"]
        for hd in headers:
            table.cell(hd, r"\textbf", just='c')
        table.midrule()

        # table body
        for name, f_test in izip(self.names, self.f_tests):
            table.cell(name)
            table.cell(fmtxt.stat(f_test.SS))
            table.cell(fmtxt.stat(f_test.df, fmt='%i'))
            table.cell(fmtxt.stat(f_test.MS))
            if f_test.F:
                stars = test.star(f_test.p)
                table.cell(fmtxt.stat(f_test.F, stars=stars))
                table.cell(fmtxt.p(f_test.p))
            else:
                table.cell()
                table.cell()

        # residuals
        if self.X.df_error > 0:
            table.empty_row()
            table.cell("Residuals")
            SS, df, MS = self.residuals
            table.cell(SS)
            table.cell(df, fmt='%i')
            table.cell(MS)
            table.endline()

        # total
        table.midrule()
        table.cell("Total")
        SS = np.sum((self.Y.x - self.Y.mean()) ** 2)
        table.cell(fmtxt.stat(SS))
        table.cell(fmtxt.stat(len(self.Y) - 1, fmt='%i'))
        return table




def ancova(Y, factorial_model, covariate, interaction=None, sub=None, v=True,
           empty=True, ems=None):
    """
    OBSOLETE

    args
    ----

    Y: dependent variable
    factorial model:
    covariate:


    kwargs
    ------

    interaction: term from the factorial model to check for interaction with
                 the covariate
    v=True: display more information
    **anova_kwargs: ems, empty


    Based on:
        `Exercise to STATISTICS: AN INTRODUCTION USING R
        <http://www.bio.ic.ac.uk/research/crawley/statistics/exercises/R6Ancova.pdf>`_

    """
    assert isvar(covariate)
    anova_kwargs = {'empty': empty, 'ems': ems}
    if sub != None:
        Y = Y[sub]
        factorial_model = factorial_model[sub]
        covariate = covariate[sub]
        if interaction != None:
            interaction = interaction[sub]
    # if interaction: assert type(interaction) in [Factor]
    factorial_model = asmodel(factorial_model)
    a1 = LM(Y, factorial_model)
    if v:
        print a1.table(title="MODEL 1", **anova_kwargs)
        print '\n'
    a2 = LM(Y, factorial_model + covariate)
    if v:
        print a2.table(title="MODEL 2: Main Effect Covariate", **anova_kwargs)
        print '\n'
    print 'Model with "%s" Covariate > without Covariate' % covariate.name
    print comparelm(a1, a2)

    if interaction:
        logging.debug("%s / %s" % (covariate.name, interaction.name))
        logging.debug("%s" % (covariate.__div__))
        i_effect = covariate.__div__(interaction)
#        i_effect = covariate / interaction
        a3 = LM(Y, factorial_model + i_effect)
        if v:
            print '\n'
            print a3.table(title="MODEL 3: Interaction")
        # compare
        print '\n"%s"x"%s" Interaction > No Covariate:' % (covariate.name, interaction.name)
        print comparelm(a1, a3)
        print '\n"%s"x"%s" Interaction > Main Effect:' % (covariate.name, interaction.name)
        print comparelm(a2, a3)


def compare(Y, first_model, test_effect, sub=None):
    """
    OBSOLETE

    Tests for a significant influence of test_effect by comparing whether
    (first_model + test_effect) explains significantly more variance than
    first_model alone.

    """
    a1 = LM(Y, first_model, sub=sub)
    a2 = LM(Y, first_model + test_effect, sub=sub)
    print
    print a1.table(title='MODEL 1:')
    print '\n'
    print a2.table(title='MODEL 2:')
    # compare
    SS_diff = a1.SS_res - a2.SS_res
    df_diff = test_effect.df
    MS_diff = SS_diff / df_diff
    # if not round(SS_diff, 6) == round(SS_cov_1 - SS_cov_2, 6):
    #    txt = "\nWARNING: SS_diff: {0} a1.SS_res - a2.SS_res: {1}"
    #    print txt.format(SS_diff, a1.SS_res - a2.SS_res)
    F = MS_diff / a2.MS_res
    p = 1 - scipy.stats.distributions.f.cdf(F, df_diff, a2.df_res)
    stars = test.star(p).replace(' ', '')
    difftxt = "Residual SS reduction: {SS}, df difference: {df}, " + \
              "F = {F}{s}, p = {p}"
    print '\n' + difftxt.format(SS=SS_diff, df=df_diff, F=F, s=stars, p=p)
