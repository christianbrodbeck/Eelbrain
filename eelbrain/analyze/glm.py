'''
Created on Oct 17, 2010

@author: christian
'''
from __future__ import division

import logging, os
from copy import deepcopy

import numpy as np
import scipy as sp

import eelbrain.fmtxt as textab

import test
from eelbrain.vessels.data import model, var, ismodel, asmodel, isvar, asvar



defaults = dict(show_ems=False, #True or False; show E(MS) in Anova tables
                p_fmt='%.4f',
                )

_max_array_size = 25 # constant for max array size in lm_fitter



def _leastsq(Y, X):
    Y = np.matrix(Y, copy=False).T
    X = np.matrix(X, copy=False)
    B = (X.T * X).I * X.T * Y
    return np.ravel(B)
    
def _leastsq_2(Y, X):
    # same calculations
    Xsinv = np.dot(np.matrix(np.dot(X.T, X)).I.A,
                   X.T)
    beta = np.dot(Xsinv, Y)
    return beta


def _hopkins_ems(regression_model, v=False):
    """
    Returns a table that can be used
    
    v=True prints E(MS) components (False by default)
    
    """
    assert ismodel(regression_model)
    # check that no var is in model
    if any([type(f) == var for f in regression_model.factors]):
        return [None] * len(regression_model.effects)
    # E(MS) table (after Hopkins, 1976)
    E_MS_table = []
    for e in regression_model.effects:
        E_MS_row = []
        for e2 in regression_model.effects:
            if np.all([(f in e or f.random) for f in e2.factors]) \
               and np.all([(f in e2 or e2.nestedin(f)) for f in e.factors]):
                E_MS_row.append(True)
            else:
                E_MS_row.append(False)
        E_MS_table.append(E_MS_row)
    E_MS = np.array(E_MS_table, dtype = bool)
    
    if v:
        print "E(MS) component table:\n", E_MS
    
    # read MS denominator for F tests from table
    MS_denominators = []
    for i,f in enumerate(regression_model.effects):
        e_ms_den = deepcopy(E_MS[i])
        e_ms_den[i] = False
        match = np.all(E_MS == e_ms_den, axis=1)
        if v:
            print f.name, ':', match
        if match.sum() == 1:
            match_i = np.where(match)[0][0]
            MS_denominators.append(match_i)
        elif match.sum() == 0:
            MS_denominators.append(None)
        else:
            raise NotImplementedError("too many matches")

    return MS_denominators


def _is_higher_order(e1, e0):
    "returns True if e1 is a higher order term of e0"
    # == all factors in e0 are contained in e1
    f1s = e1.factors
    for f0 in e0.factors:
        # if f0 not in f1s
        if any(f0 is f1 for f1 in f1s):
            continue
        else:
            return False
    return True



#def isbalanced(X):
#    """
#    tests whether a model is balanced
#    
#    """
#    # TODO
##    return False
#    return True


#def fit(Y, X, lsq=1):
#    """
#    fits the model X to Y, returns
#    
#    {
#     'total': {'SS':xx, 'df':xx, 'MS':xx},
#     'model': {...},
#     'residuals': {...},
#     'beta': beta weights,
#     'values': beta * X
#     }
#    
#    
#    lsq: least square estimator
#         1: after Fox
#         else: numpy 
#    
#    """
#    out = {}
#    Y = asvar(Y)
#    X = asmodel(X)
#    if lsq == 1:
#        beta = _leastsq(Y.x, X.full)
#    else:
#        beta = _leastsq_np(Y.x, X.full)
#    out['beta'] = beta
#    mu = Y.x.mean()
#    SS = np.sum((Y.x - mu)**2)
#    df = Y.N - 1
#    MS = SS / df
#    out['total'] = dict(SS=SS, df=df, MS=MS)
#    out['values'] = values = beta * X.full
#    Y_est = values.sum(1)
#    if not mu == Y_est.mean():
#        logging.debug("mu=%s != Y_est.mean()=%s"%(mu, Y_est.mean()))
#    SSm = np.sum((Y_est - mu)**2)
#    dfm = X.df
#    MSm = SSm / dfm
#    out['model'] = dict(SS=SSm, df=dfm, MS=MSm)
#    residuals = Y.x - Y_est
#    SS_res = np.sum(residuals**2)
#    df_res = X.df_error
#    MS_res = SS_res / df_res
#    out['residuals'] = dict(SS=SS_res, df=df_res, MS=MS_res)
#    return out


class lm:
    """
    object for fitting a linear model to one set of data, implementing lazy
    creation of results
    
    """
    def __init__(self, Y, X, sub=None, v=False, lsq=0, title=None):
        """
        Y: 
            dependent variable
        X: 
            model
        v: 
            print some intermediate results for inspection
        
        lsq : int
            0 = numpy lsq
            1 = my least sq (Fox)
        
        performs an anova/ancova based on residuals. Fixed effects only
        
        """
        self.title = title
        self.results = {} # will store results
        # prepare input
        Y = asvar(Y)
        X = asmodel(X)#.sorted()
        if sub is not None:
            Y = Y[sub]
            X = X[sub]
        
        assert Y.N == X.N
        assert X.df_error > 0

        # fit
        if lsq == 1:
            # estimate least squares approximation
            beta = _leastsq(Y.x, X.full)
            # estimate
            values = self.values = beta * X.full
            Y_est = values.sum(1)
            self._residuals = residuals = Y.x - Y_est
            SS_res = np.sum(residuals**2)
            if not Y.mu == Y_est.mean():
                logging.warning("Y.mu=%s != Y_est.mean()=%s"%(Y.mu, Y_est.mean()))
        else:
            # use numpy
            beta, SS_res, rank, s = np.linalg.lstsq(X.full, Y.x)
            if len(SS_res) == 1:
                SS_res = SS_res[0]
            else:
                SS_res = 0
        
        # SS total
        SS_total = self.SS_total = np.sum((Y.x - Y.mu)**2)
        df_total = self.df_total = X.df_total
        MS_total = self.MS_total = SS_total / df_total
        # SS residuals
        self.SS_res = SS_res
        df_res = self.df_res = X.df_error
        MS_res = self.MS_res = SS_res / df_res
        # SS explained
#        SS_model = self.SS_model = np.sum((Y_est - Y.mu)**2)
        SS_model = self.SS_model = SS_total - SS_res
        df_model = self.df_model = X.df
        MS_model = self.MS_model = SS_model / df_model

        # store stuff
        self.Y = Y
        self.X = X
        self.beta = beta
    def F_test(self):
        """
        Tests the null-hypothesis that the model does not explain a significant
        amount of the variance in the dependent variable. Returns F, p.
        
        """
        F = self.MS_model / self.MS_res
        p = sp.stats.distributions.f.sf(F, self.df_model, self.df_res)
        return F, p
    @property
    def residuals(self):
        if not hasattr(self, '_residuals'):
            values = self.beta * self.X.full
            Y_est = values.sum(1)
            self._residuals = self.Y.x - Y_est
        return self._residuals

    def __repr__(self):
        txt = "lm({Y} ~ {model})"
        if hasattr(self.Y, 'name'):
            yname = self.Y.name
        else:
            yname = 'Y'
        return txt.format(Y=yname, model=self.X.model_eq)
# TODO:    def assumptions(self):
#        """
#        Tests the assumptions required for the validity of inferential 
#        statistics
#        
#        """
#        pass


class _old_lm_(lm):
    def anova(self, title=None, empty=True, ems=None):
        """
        returns an ANOVA table for the linear model
         
        """
        if ems is None:
            ems = defaults['show_ems']
        Y = self.Y
        X = self.X
        values = self.values
        # method
        #if X.df_error == 0:
        #    hopkins = True
        e_ms = _hopkins_ems(X)
        #else:
        #    hopkins = False

        # table head
        table = textab.Table('l'+'r'*(5+ems))
        if title:
            table.title(title)
        elif self.title:
            table.title(self.title)
#        for msg in X.check():
#            table.caption('! '+msg)
        table.cell()
        headers = ["SS", "df", "MS"]
        if ems: headers += ["E(MS)"]
        headers += ["F", "p"]
        for hd in headers:
            table.cell(hd, r"\textbf", just='c')
        table.midrule()
        
        if isbalanced(X):
            # MS for factors (Needed for models involving random effects)  
            self.MS = []
            for i, name, index, df in X.iter_effects():
                SS = np.sum(values[:,index].sum(1)**2)
                self.MS.append(SS / df)
        else:
            raise NotImplementedError()
            tests = {}
            for e in X.effects: # effect to test
                m0effects = []
                for e0 in X.effects: # effect in model0
                    if e0 is e:
                        pass
                    elif all([f in e0.factors for f in e.factors]):
                        pass
                    else:
                        m0effects.append(e0)
                model0 = model(m0effects)
                model1 = model0 + e
                SS, df, MS, F, p = incremental_F_test(Y, model1, model0)
                tests[e.name] = dict(SS=SS, df=df, MS=MS, F=F, p=p)
        
        # table body
        self.results = {}
        for i, name, index, df in X.iter_effects():
            SS = np.sum(values[:,index].sum(1)**2)
            #if v: print name, index, SS
            MS = SS / df
            #self.results[name] = {'SS':SS, 'df':df, 'MS':MS}
            if e_ms[i] != None: #hopkins and 
                e_ms_i = e_ms[i]
                MS_d = self.MS[e_ms_i]
                df_d = X.effects[e_ms_i].df
                e_ms_name = X.effects[e_ms_i].name
            elif self.df_res > 0:
                df_d = self.df_res
                MS_d = self.MS_res
                e_ms_name = "Res"
            else:
                MS_d = False
                e_ms_name = None
            # F-test
            if MS_d != False:
                F = MS / MS_d
                p = 1 - sp.stats.distributions.f.cdf(F, df, df_d)
                stars = test.star(p)
                tex_stars = textab.Stars(stars)
                F_tex = [F, tex_stars]
            else:
                F_tex = None
                p = None
            # add to table
            if e_ms_name or empty:
                table.cell(name)
                table.cell(SS)
                table.cell(df, fmt='%i')
                table.cell(MS)
                if ems:
                    table.cell(e_ms_name)
                table.cell(F_tex, mat=True)
                table.cell(p, fmt=defaults['p_fmt'], drop0=True)
            # store results
            self.results[name] = {'SS':SS,
                                  'df':df,
                                  'MS':MS,
                                  'E(MS)':e_ms_name,
                                  'F':F,
                                  'p':p}
            #self.indexes[name] = index # for self.Ysub()
        # table end
        if self.df_res > 0:
            table.cell("Residuals")
            table.cell(self.SS_res)
            table.cell(self.df_res, fmt='%i')
            table.cell(self.MS_res)
        return table
    def reg(self):
        """
        returns a table containing slope coefficitions for covariates. 
        (Slope coefficients are shown for all factors with attribute 
        showreg==True)
        
        
        """
        # prepare table
        table = textab.Table('l'*4)
        df = self.X.df_error
        table.cell()
        table.cell('\\beta', mat=True)
        table.cell('T_{%i}'%df, mat=True)
        table.cell('p', mat=True)
        table.midrule()
        #
        q = 1 # track start location of effect in model.full
        for e in self.X.effects:
            if True:#e.showreg:
                table.cell(e.name+':')
                table.newrow()
                for i, name in e.iter_beta(): # Fox pp. 106 ff.
                    beta = self.beta[q+i]
#                    Evar_pt_est = self.SS_res / df
                    #SEB 
#                    SS = (self.values[q+i])**2
                    T = 0
                    p =  0
                    table.cell(name)
                    table.cell(beta)
                    table.cell(T)
                    table.cell(p, fmt=defaults['p_fmt'])
            q += e.df
        return table
    ## older functions
#    def plot(self, sub=None):
#        return plot(self, sub=sub)
    def Ysub(self, effects):
        "return Y after subtracting the SS explained by the given effects"
        # FIXME: Ysub
        effects = asmodel(effects)
        Yout = deepcopy(self.Y.x)
        for e in effects.effects:
            Yout -= self.values[:,self.indexes[e.name]].sum(1)
        return Yout

# MARK: Multitest object


class lm_fitter(object):
    """
    ANOVA model class. E(MS) for F statistic after Hopkins (1976)
    
    Each method takes the kwarg v ("verbose", default ``False``); displays more
    information if True.
    
    use .aov method for single anova, .map method for list.
    
    Reference
    ---------
    Hopkins, K. D. (1976). A Simplified Method for Determining Expected Mean 
        Squares and Error Terms in the Analysis of Variance. Journal of 
        Experimental Education, 45(2), 13--18.

    """
    def __init__(self, X, title=False):
        self.title = title
        self.results = {} # will store results
        # prepare input
        X = asmodel(X)
        self.X = X
        # X inverse
        X_ = X.full
        self.Xinv = np.matrix(X_).I.A # alternative still used for map
        self.Xsinv = np.dot(np.matrix(np.dot(X_.T, X_)).I.A,
                            X_.T)
        # E MS
        self.E_ms = _hopkins_ems(X)
        self.df_res = X.df_error
    
    def map(self, Y, v=False, sender=None, new=False):
        """
        Returns results for multiple sets of dependents. Returns a list with 
        (name, F-field, P-field) tuples for all effects that can be estimated 
        with the current method.
        
        Y: np.array
            Assumes that the last dimension of Y provides cases. 
            Other than that, shape is free to vary and output shape will match 
            input shape.
        
        
        """
        original_shape = Y.shape
        assert original_shape[-1] == self.X.N, "last dimension must contain cases"
        Y = Y.reshape((-1, self.X.N))
        if v:
            print Y.shape, self.Xinv.shape
        # Split Y that are too long
        df = self.Xinv.shape[0]
        if np.log2(Y.shape[0] * df**2) > _max_array_size:
            max_len = int(2**_max_array_size // df**2)
            n_parts = Y.shape[0] // max_len + 1
            logging.debug(" S.lm_fitter: Splitting Y in {0} parts".format(n_parts))
            Y_list = [Y[i*max_len:(i+1)*max_len,:] for i in range(n_parts)]
            out_maps = [self.map(Yi) for Yi in Y_list]
            out_map = []
            for i in range(len(out_maps[0])):
                e = out_maps[0][i][0]
                if v:
                    print str([m[i][1].shape for m in out_maps])
                F = np.hstack([m[i][1] for m in out_maps]).reshape(original_shape[:-1])
                P = np.hstack([m[i][2] for m in out_maps]).reshape(original_shape[:-1])
                out_map.append((e, F, P))
            return out_map
        # do the actual estimation
        else:
            if new:
                raise NotImplementedError
            else:
                params = (Y[:,None,:] * self.Xinv).sum(2) # c * effect-code
                values = params[:,None,:] * self.X.full # c x param x effect-code
                # MS res
                if self.df_res > 0:
                    Yp = values.sum(2)
                    SS_res = ((Y - Yp)**2).sum(1)
                    MS_res = SS_res / self.df_res
                # collect SS, df, MS
                i = 0
                e_list = [] #<- (ss, df, ms)
                for i, name, index, df in self.X.iter_effects():
                    Yp = values[:,:,index].sum(2)
                    SS = (Yp**2).sum(1)
                    MS = SS / df
                    e_list.append([name, df, MS])
                # F Tests
                out_map = [] #<- (name, F, P)
                for i, e in enumerate(e_list):
                    df_n = e[1]
                    MS_n = e[2]
                    E_ms = self.E_ms[i]
                    if E_ms != None:
                        df_d = e_list[E_ms][1]
                        MS_d = e_list[E_ms][2]
                    elif self.df_res > 0:
                        df_d = self.df_res
                        MS_d = MS_res
                    else:
                        df_d = 0
                    #
                    if df_d > 0:                        
                        F = MS_n / MS_d
                        P = 1 - sp.stats.distributions.f.cdf(F, df_n, df_d)
                        out_map.append((e[0], F.reshape(original_shape[:-1]), 
                                        P.reshape(original_shape[:-1])))
                # append RESIDUALS to output
                #if self.df_res > 0:
                #    out_map.append
                return out_map
    
    def __repr__(self):
        txt = ''.join(['lm_fitter(', self.X.__repr__(), ')'])
        return txt
#    def __str__(self):
#        eq = self.model.__repr__()
#        space = max([len(e.name) for e in self.model.effects]) + 1
#        E_MS = ['  '.join([e.name.rjust(space)] + [str(int(i)) for i in line]) \
#                for e, line in zip(self.model.effects, self.E_MS_table)]
#        return '\n'.join([eq, str(self.MS_denominators), "E(MS):"] + E_MS)


'''   
    # deriving params for 1 set of measurements
    def fit(self, Y, v=False):
        """
        Returns a list with the results of fitting Y. Use aov(Y) method to get
        reults in the form of a table.
        
        Output list has [ss, df, ms, F, p] list for each effect (in the order of 
        self.model.effects).
        
        
        """
        Y = np.ravel(Y)
        self.Y = Y 
        
        #params = (Y * self.Xinv).sum(1)
        params = np.dot(self.Xsinv, Y)
        self.params = params
        if v:
            print params
            print (params * self.model.full).sum(1)
        values = params * self.model.full
        
        # collect SS, df, MS
        i = 0
        e_list = [] # (ss, df, ms)
        for e in self.model.effects:
            Yp = values[:,i:i+e.df].sum(1)
            SS = (Yp**2).sum(0)
            MS = SS / e.df
            e_list.append([SS, e.df, MS])
            i += e.df
        
        # residuals
        if self.model.df_error > 0: 
            SS = np.sum((Y - values.sum(1))**2)
            df = self.model.df_error
            MS = SS / df
            self.res = {'SS':SS, 'df':df, 'MS':MS}
        else:
            self.res = None
        
        # F Tests
        for i, e in enumerate(e_list):
            denominator = self.MS_denominators[i]
            if denominator or self.res:
                df_n = e[1]
                MS_n = e[2]
                if denominator:
                    df_d = e_list[denominator][1]
                    MS_d = e_list[denominator][2]
                elif self.res:
                    print "WARNING: Residuals --> test # %s"%i
                    df_d = self.res['df']
                    MS_d = self.res['MS']
                else:
                    raise ValueError("WTF Error")
                F = MS_n / MS_d
                p = 1 - sp.stats.distributions.f.cdf(F, df_n, df_d)
                e += [F, p]
            else:
                e += [None, None]
        return e_list
    # casting params in nice table for one set of measurements
    def aov(self, Y, spss=False, **kwargs):
        "returns a textab.table with the results of fitting Y"
        if spss:
            self.table(Y)
        table = textab.Table('lrrrrr')
        table.cell()
        for hd in ["SS", "df", "MS", "F", "p"]:
            table.cell(hd, "textbf", just='c')
        table.midrule()
        for e,line in zip(self.model.effects, self.fit(Y)):
            SS, df, MS, F, p = line
            table.cell(e.name)
            table.cell(SS)
            table.cell(df, digits=0)
            table.cell(MS)
            if p:
                stars = test.star(p)
                tex_stars = textab.Element(stars, "^", )
                table.cell([F, tex_stars], mat=True)
            else:
                table.cell(F)
            table.cell(p, drop0=True)
        if self.res:
            table.cell("Residuals")
            table.cell(self.res['SS'])
            table.cell(self.res['df'], digits=0)
            table.cell(self.res['MS'])
        return table
    def table(self, Y, within=None):
        "convenience method for transferring Y with factors to SPSS"
        data = np.ravel(Y)[:,None].astype('|S8')
        names = ['Y']
        # collect factors
        for e in self.model.effects:
            for f in e.factors:
                if f.name not in names:
                    names.append(f.name)
                    data = np.hstack((data, f.x[:,None].astype(int)))
        # print
        print '\t'.join(names)
        for line in data:
            print '\t'.join(line)
    def sorted(self, factors):
        pass
'''

#def incremental_F_test(lm1, lm0, lmEMS=None):
#    """
#    IMPLEMENTATION FOR model OBJECTS
#
#    tests the hypothesis that model1 does not explain more variance than 
#    model0. 
#    (model1 is the model0 + q factors)
#    
#    EMS: model for the Expected value of MS; if == None, the error MS of model1
#         is used (valid as long as the model consists of fixed effects only)
#    
#    SS, df, MS, F, p = incremental_F_test(Y, model1, model0)
#    
#    """
#    n = lm1.X.N
#    RegSS1 = lm1.SS_model
#    k = lm1.df_model
#    RegSS0 = lm0.SS_model
#    
#    if lmEMS == None:
#        MS_e = lm1.MS_res
#    else:
#        MS_e = lmEMS.MS_model
#    
#    df = lm1.df_model - lm0.df_model
#    SS = RegSS1 - RegSS0
#    MS = SS / df
#    
#    F = MS / MS_e
#    p = 1 - sp.stats.distributions.f.cdf(F, df, n-k-1)
#    return SS, df, MS, F, p


def incremental_F_test(lm1, lm0, MS_e=None, df_e=None):
    """
    tests the null hypothesis that the model of lm1 does not explain more 
    variance than that of lm0 (with model_1 == model_0 + q factors,  q > 0).
    If lm1 is None it is assumed that lm1 is the full model with 0 residuals.
    
    MS_e, f_e:  model for the Expected value of MS; if == None, the error MS of 
           model1 is used (valid for fixed effects models)
    
    
    SS_diff, df_diff, MS_diff, F, p = incremental_F_test(lm1, lm0, lmEMS)
    
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
        p = sp.stats.distributions.f.sf(F, df_diff, df_e)
    else:
        F = None
        p = None
    
    return SS_diff, df_diff, MS_diff, F, p

    
    

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
    p = 1 - sp.stats.distributions.f.cdf(F, df_diff, lm2.df_res)
    stars = test.star(p).replace(' ', '')
    difftxt = "Residual SS reduction: {SS}, df difference: {df}, " + \
              "F = {F:.3f}{s}, p = {p:.4f}"
    return difftxt.format(SS=SS_diff, df=df_diff, F=F, s=stars, p=p)




# MARK: Convenience functions

class anova(object):
    def __init__(self, Y, X, sub=None, 
                 title=None, empty=True, ems=None, lsq=0,
                 showall=False):
        """
        Returns an ANOVA table for the linear model. Mixed effects models require 
        full model specification so that E(MS) can be estimated according to 
        Hopkins (1976)
        
        Random effects: If the model is fully specified, a Hopkins E(MS) table 
        is used to determine error terms in the mixed effects model. Otherwise,
        random factors are treated as fixed factors.
        
        kwargs
        ------
        empty:  include rows without F-Tests (True/False)
        ems:  display source of E(MS) for F-Tests (True/False; None = use default)
        lsq:  least square fitter
                = 0 -> numpy.linalg.lstsq 
                = 1 -> after Fox
        showall: show SS, df and MS for effects without F test
        
        
        TODO
        ----
          - sort model
          - reuse lms which are used repeatedly
          - provide threshold for including interaction effects when testing lower 
            level effects
        
        
        Problem with unbalanced models
        ------------------------------
          - The SS of Effects which do not include the between-subject factor are 
            higher than in SPSS
          - The SS of effects which include the between-subject factor agree with 
            SPSS
        
        """
        # prepare kwargs
        Y = asvar(Y)
        X = asmodel(X)
        
        if sub is not None:
            Y = Y[sub]
            X = X[sub]
        
        assert Y.N == X.N

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
            rfx = 0
            fx_desc = 'Fixed'
        else:
            raise ValueError("Model Overdetermined")
        self._log.append("%s effects model"%fx_desc)
        
        if lsq == 1:
            self._log.append("(my lsq)")
        elif lsq ==0:
            self._log.append("\n (np lsq)")
        
    
        # create testing table:  
        # list of (effect, lm, lm_comp, lm_EMS)
        test_table = []
        #    
        # list of (name, SS, df, MS, F, p)
        results_table = []
        
        
        if len(X.effects) == 1:
            self._log.append("single factor model")
            lm0 = lm(Y, X, lsq=lsq)
            SS = lm0.SS_model
            df = lm0.df_model
            MS = lm0.MS_model
            F, p = lm0.F_test()
            results_table.append((X.name, SS, df, MS, F, p))
            results_table.append(("Residuals", lm0.SS_res, lm0.df_res, 
                                  lm0.MS_res, None, None))
        else:
            if not rfx:
                full_lm = lm(Y, X, lsq=lsq)
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
                        if _is_higher_order(e, e_test):
                            excluded_e.append(e)
                        else:
                            effects.append(e)
        
                model0 = model(*effects)
                if e_test.df > model0.df_error:
                    skip = "overspecified"
                else:
                    lm0 = lm(Y, model0, lsq=lsq)
                    
                    # find model 1
                    effects.append(e_test)
                    model1 = model(*effects)
                    if model1.df_error > 0:
                        lm1 = lm(Y, model1, lsq=lsq)
                    else:
                        lm1 = None
                    
                    if rfx:
                        # find E(MS)
                        EMS_effects = []
                        for e in X.effects:
                            if e is e_test:
                                pass
                            elif all([(f in e_test or f.random) for f in e.factors]):
                                if all([(f in e or e.nestedin(f)) for f in e_test.factors]):
                                    EMS_effects.append(e)
                        if len(EMS_effects) > 0:
                            lm_EMS = lm(Y, model(*EMS_effects), lsq=lsq)
                            MS_e = lm_EMS.MS_model
                            df_e = lm_EMS.df_model
                        else:
                            if showall:
                                if lm1 is None:
                                    SS = lm0.SS_res
                                    df = lm0.df_res                                
                                else:
                                    SS = lm0.SS_res - lm1.SS_res
                                    df = lm0.df_res - lm1.df_res
                                MS = SS / df
                                results_table.append((name, SS, df, MS, None, None))
                            skip = "no Hopkins E(MS)"
                    
                
                if skip:
                    self._log.append("SKIPPING: %s (%s)"%(e_test.name, skip))
                else:
                    test_table.append((e_test, lm1, lm0, MS_e, df_e))
                    SS, df, MS, F, p = incremental_F_test(lm1, lm0, MS_e=MS_e, df_e=df_e)
                    results_table.append((name, SS, df, MS, F, p))
            if not rfx:
                results_table.append(("Residuals", SS_e, df_e, MS_e, None, None))
        self._test_table = test_table
        self._results_table = results_table
    def __repr__(self):
        return "anova(%s, %s)" % (self.Y.name, self.X.name)
    def __str__(self):
        return str(self.anova())
    def log(self):
        print os.linesep.join(self._log)
    def anova(self):
        "Return ANOVA table"
        if self.show_ems is None:
            ems = defaults['show_ems']
        else:
            ems = self.show_ems
        
        # table head
        table = textab.Table('l'+'r'*(5+ems))
        if self.title:
            table.title(self.title)
        table.cell()
        headers = ["SS", "df", "MS"]
#        if ems: headers += ["E(MS)"]
        headers += ["F", "p"]
        for hd in headers:
            table.cell(hd, r"\textbf", just='c')
        table.midrule()
    
        # table body
        for name, SS, df, MS, F, p in self._results_table:
            table.cell(name)
            table.cell(textab.stat(SS))
            table.cell(textab.stat(df, fmt='%i'))
            table.cell(textab.stat(MS))
            if F:
                stars = test.star(p)
                table.cell(textab.stat(F, stars=stars))
                table.cell(textab.p(p))
            else:
                table.cell()
                table.cell()
        
        # total
        table.midrule()
        table.cell("Total")
        table.cell(textab.stat(self.Y.SS))
        table.cell(textab.stat(self.Y.N-1, fmt='%i'))
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
    
    
    Based on
    --------
    Exercise to STATISTICS: AN INTRODUCTION USING R
    http://www.bio.ic.ac.uk/research/crawley/statistics/exercises/R6Ancova.pdf
    """
    assert isvar(covariate)
    anova_kwargs = {'empty': empty, 'ems': ems}
    if sub != None:
        Y = Y[sub]
        factorial_model = factorial_model[sub]
        covariate = covariate[sub]
        if interaction != None:
            interaction = interaction[sub]
    #if interaction: assert type(interaction) in [factor]
    factorial_model = asmodel(factorial_model)
    a1 = lm(Y, factorial_model)
    if v: 
        print a1.anova(title="MODEL 1", **anova_kwargs)
        print '\n'
    a2 = lm(Y, factorial_model + covariate)
    if v:
        print a2.anova(title="MODEL 2: Main Effect Covariate", **anova_kwargs)
        print '\n'
    print 'Model with "%s" Covariate > without Covariate' % covariate.name
    print comparelm(a1, a2)
    
    if interaction:
        logging.debug("%s / %s"%(covariate.name, interaction.name))
        logging.debug("%s"%(covariate.__div__))
        i_effect = covariate.__div__(interaction)
#        i_effect = covariate / interaction
        a3 = lm(Y, factorial_model + i_effect)        
        if v:
            print '\n'
            print a3.anova(title="MODEL 3: Interaction")
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
    a1 = lm(Y, first_model, sub=sub)
    a2 = lm(Y, first_model + test_effect, sub=sub)
    print 
    print a1.anova(title='MODEL 1:')
    print '\n'
    print a2.anova(title='MODEL 2:')
    # compare
    SS_diff = a1.SS_res - a2.SS_res
    df_diff = test_effect.df
    MS_diff = SS_diff / df_diff
    #if not round(SS_diff, 6) == round(SS_cov_1 - SS_cov_2, 6):
    #    txt = "\nWARNING: SS_diff: {0} a1.SS_res - a2.SS_res: {1}"
    #    print txt.format(SS_diff, a1.SS_res - a2.SS_res)
    F = MS_diff / a2.MS_res
    p = 1 - sp.stats.distributions.f.cdf(F, df_diff, a2.df_res)
    stars = test.star(p).replace(' ', '')
    difftxt = "Residual SS reduction: {SS}, df difference: {df}, " + \
              "F = {F}{s}, p = {p}"
    print '\n'+difftxt.format(SS=SS_diff, df=df_diff, F=F, s=stars, p=p)



