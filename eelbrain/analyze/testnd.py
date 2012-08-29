'''
tests for varnd objects


Created on Feb 22, 2012

@author: christian
'''

import numpy as np
import scipy.stats
import scipy.ndimage
import mne

from eelbrain import fmtxt
from eelbrain import vessels as _vsl
from eelbrain.vessels.data import ndvar, asmodel

import glm as _glm
from test import _resample


__hide__ = ['test_result', 'test',
            'test_result_subdata_helper']


class test_result(_vsl.data.dataset):
    """
    Subclass of dataset that holds results of a statistical test. Its special 
    property is that all entries describe the same dimensional space. That 
    property makes a .subdata() method possible.
    
    """
    def subdata(self, **kwargs):
        "see ndvar.subdata() documentation"
        # create a subclass that inherits test-specific properties while
        # bypassing the test's __init__ method:
        class test_result_subdata(test_result_subdata_helper, self.__class__):
            pass
        
        out = test_result_subdata(self.name, self.info)
        for k,v in self.iteritems():
            out[k] = v.subdata(**kwargs)
        
        return out
    
    @property
    def _default_plot_obj(self):
        return self.all


class test_result_subdata_helper(test_result):
    def __init__(self, name, info):
        super(test_result, self).__init__(name=name, info=info)



class ttest(test_result):
    """
    **Attributes:**
    
    all :
        c1, c0, [c0 - c1, P]
    diff :
        [c0 - c1, P]
    
    """
    def __init__(self, Y='MEG', X=None, c1=None, c0=0, 
                 match=None, sub=None, ds=None, 
                 contours={.05: (.8, .2, .0),  .01: (1., .6, .0),  .001: (1., 1., .0)}):
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
        
        # create dataset
        super(ttest, self).__init__(T, P, name=test_name)
        self['c1_m'] = c1_mean
        if c0_mean: 
            self['c0_m'] = c0_mean
        if diff:
            self['diff'] = diff
        
    @property
    def all(self):
        if 'c0_m' in self:
            return [self['c1_m'], self['c0_m']] + self.diff
        elif 'diff' in self:
            return [self['c1_m']] + self.diff
        else:
            return self.diff
    
    @property
    def diff(self):
        if 'diff' in self:
            layers = [self['diff']]
        else:
            layers = [self['c1_m']]
          
        layers.append(self['p'])
        return [layers]






class f_oneway(test_result):
    def __init__(self, Y='MEG', X='condition', sub=None, dataset=None,
                 p=.05, contours={.01: '.5', .001: '0'}):
        """
        uses scipy.stats.f_oneway
        
        """
        if isinstance(Y, basestring):
            Y = dataset[Y]
        if isinstance(X, basestring):
            X = dataset[X]
        
        if sub is not None:
            Y = Y[sub]
            X = X[sub]
        
        Ys = [Y[X==c] for c in X.cells]
        Ys = [y.x.reshape((y.x.shape[0], -1)) for y in Ys]
        N = Ys[0].shape[1]
        
        Ps = []
        for i in xrange(N):
            groups = (y[:,i] for y in Ys)
            F, p = scipy.stats.f_oneway(*groups)
            Ps.append(p)
        test_name = 'One-way ANOVA'
        
        dims = Y.dims[1:]
        Ps = np.reshape(Ps, tuple(len(dim) for dim in dims))
        
        properties = Y.properties.copy()
        properties['colorspace'] = _vsl.colorspaces.get_sig(p=p, contours=contours)
        properties['test'] = test_name
        p = ndvar(Ps, dims, properties=properties, name=X.name)

        # create dataset
        super(f_oneway, self).__init__(name="anova")
        self['p'] = p
    
    @property
    def all(self):
        return self['p']



class anova(test_result):
    """
    
    """
    def __init__(self, Y='MEG', X='condition', sub=None, ds=None, info={},
                 p=.05, contours={.01: '.5', .001: '0'}):
        if isinstance(Y, basestring):
            Y = ds[Y]
        if isinstance(X, basestring):
            X = ds[X]
        if sub is not None:
            if isinstance(sub, basestring):
                sub = ds[sub]
            Y = Y[sub]
            X = X[sub]
        
        fitter = _glm.lm_fitter(X)
        
        info['effect_names'] = effect_names = []
        super(anova, self).__init__(name="anova", info=info)
        properties = Y.properties.copy()
        properties['colorspace'] = _vsl.colorspaces.get_sig(p=p, contours=contours)
        kwargs = dict(dims = Y.dims[1:],
                      properties = properties)
        
        for e, _, Ps in fitter.map(Y.x):
            name = e.name
            effect_names.append(name)
            P = ndvar(Ps, name=name, **kwargs)
            self[name + '_p'] = P
    
    @property
    def all(self):
        epochs = []
        for name in self.info['effect_names']:
            epochs.append(self[name+'_p'])
        
        return epochs


class cluster_anova(test_result):
    def __init__(self, Y, X, t=.1, sub=None, samples=1000, replacement=False,
                 pmax=1):
        """
        t : scalar
            Threshold: uncorrected p-value to use as threshold for finding 
            clusters
        pmax : scalar <= 1
            Maximum cluster p-values to keep cluster.
        
        FIXME: connectivity for >2 dimensional data
        """
        if sub is not None:
            Y = Y[sub]
            X = X[sub]
        
        X = self.X = asmodel(X)
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
        dists = {e: np.empty(samples) for e in tF}
        for i, Yrs in _resample(Y, replacement=replacement, samples=samples):
            for e, F in lm.map(Yrs.x, p=False):
                clusters, n = scipy.ndimage.label(F > tF[e])
                if n:
                    clusters_v = scipy.ndimage.measurements.sum(F, clusters, xrange(1, n+1))
                    dists[e][i] = max(clusters_v)
                else:
                    dists[e][i] = 0
        
        # 
        test0 = lm.map(Y.x, p=False)
        self.clusters = {}
        self.F_maps = {}
        dims = Y.dims[1:]
        for e, F in test0:
            dist = dists[e]
            clusters, n = scipy.ndimage.label(F > tF[e])#, structure, output)
            clusters_v = scipy.ndimage.measurements.sum(F, clusters, xrange(1,n+1))
            clist = self.clusters[e] = []
            for i in xrange(n):
                v = clusters_v[i]
                p = 1 - scipy.stats.percentileofscore(dist, v, 'mean') / 100
                if p <= pmax:
                    im = (clusters == i+1)
                    name = '%s (p=%s)' % (e.name, p)
                    properties = {'p': p}
                    ndv = ndvar(im, dims=dims, name=name, properties=properties)
                    clist.append(ndv)
            
            props = {'tF': tF[e], 'unit': 'F'}
            self.F_maps[e] = ndvar(F, dims=dims, name=e.name, properties=props)
        
        super(cluster_anova, self).__init__(name="ANOVA Permutation Cluster Test")#, info=info)
        self.tF = tF
    
    @property
    def all(self):
        epochs = []
        for e in self.X.effects:
            if e in self.F_maps:
                epochs.append([self.F_maps[e]] + self.clusters[e])
        
        return epochs

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
                    if p > pmax:
                        break
                    table.cell(i)
                    table.cell(p)
        return table




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
    k = len(ndvars) # number of levels
    if k == 0:
        raise ValueError("no segments provided")
    
    # perform test
    if parametric: # simple tests
        if k==1:
            statistic = 't'
            T, P = scipy.stats.ttest_1samp(*data, popmean=0, axis=0)
            test_name = '1-Sample $t$-Test'
        elif k==2:
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

    else: # non-parametric:
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
                index = tuple( [slice(int(i),int(i+1)) for i in indexes] + [slice(0, None)] )
                testArgs = tuple( [ d[index].ravel() for d in data] )
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
