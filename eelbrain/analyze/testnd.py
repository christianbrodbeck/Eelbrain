'''
tests for varnd objects


Created on Feb 22, 2012

@author: christian
'''

import numpy as np
import scipy.stats
#import mne

from eelbrain import vessels as _vsl

import glm as _glm


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
        c1, c2, [c2 - c1, P]
    diff :
        [c2 - c1, P]
    
    """
    def __init__(self, Y='MEG', X=None, c1=None, c2=0, 
                 match=None, sub=None, ds=None, contours=None):
        """
        c1 and c2 : ndvars (or dataset with default_DV)
            segments between which to perform the test
        
        """
        if not contours:
            contours = { .05: (.8, .2, .0),  .01: (1., .6, .0),  .001: (1., 1., .0),
                        -.05: (0., .2, 1.), -.01: (.4, .8, 1.), -.001: (.5, 1., 1.),
                        }
        
        if X is None:
            pass
        elif c1 is None:
            v = X.values()
            if len(v) == 2:
                c1, c2 = v
            else:
                err = "If X has more than 2 categories, 2 must be chosen"
                raise ValueError(err)
        
        ct = _vsl.structure.celltable(Y, X, match, sub, ds=ds)
        
        if isinstance(c2, basestring):
            c1_mean = ct.data[c1].summary(name=c1)
            c2_mean = ct.data[c2].summary(name=c2)
            diff = c1_mean - c2_mean
            if match:
                if not ct.within[(c1, c2)]:
                    err = ("match kwarg: Conditions have different values on"
                           " <%r>" % match.name)
                    raise ValueError(err)
                T, P = scipy.stats.ttest_rel(ct.data[c1].x, ct.data[c2].x, axis=0)
                test_name = 'Related Samples t-Test'
            else:
                T, P = scipy.stats.ttest_ind(ct.data[c1].x, ct.data[c2].x, axis=0)
                test_name = 'Independent Samples t-Test'
        elif np.isscalar(c2):
            c1_data = ct.data[str(Y.name)]
            c1_mean = c1_data.summary()
            c2_mean = None
            T, P = scipy.stats.ttest_1samp(c1_data.x, popmean=c2, axis=0)
            test_name = '1-Sample t-Test'
            if c2:
                diff = c1_mean - c2
            else:
                diff = None
        else:
            raise ValueError('invalid c2: %r' % c2)
        
        # fix dimensionality
        T = T[None]
        P = P[None]
        
#        direction = np.sign(diff.x)
#        P = P * direction# + 1 # (1 - P)
#        for k in contours.copy():
#            contours[k+1] = contours.pop(k)
        
        dims = ct.Y.dims
        properties = ct.Y.properties.copy()
        
        properties['colorspace'] = _vsl.colorspaces.Colorspace(contours=contours)
        P = _vsl.data.ndvar(dims, P, properties=properties, name='p', info=test_name)
        
        properties['colorspace'] = _vsl.colorspaces.get_default()
        T = _vsl.data.ndvar(dims, T, properties=properties, name='T', info=test_name)
        
        # add Y.name to dataset name
        Yname = getattr(Y, 'name', None)
        if Yname: 
            test_name = ' of '.join((test_name, Yname))
        
        # create dataset
        super(ttest, self).__init__(T, P, name=test_name)
        self['c1_m'] = c1_mean
        if c2_mean: 
            self['c2_m'] = c2_mean
        if diff:
            self['diff'] = diff
        
    @property
    def all(self):
        if 'c2_m' in self:
            return [self['c1_m'], self['c2_m']] + self.diff
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
    """
    Attributes:
    
    p:
        p-map
    
    """
    def __init__(self, Y='MEG', X='condition', dataset=None):
        """
        uses scipy.stats.f_oneway
        
        """
        if isinstance(Y, basestring):
            Y = dataset[Y]
        if isinstance(X, basestring):
            X = dataset[X]
        
        Ys = [Y[X==c] for c in X.cells.values()]
        Ys = [y.x.reshape((y.x.shape[0], -1)) for y in Ys]
        N = Ys[0].shape[1]
        
        Ps = []
        for i in xrange(N):
            groups = (y[:,i] for y in Ys)
            F, p = scipy.stats.f_oneway(*groups)
            Ps.append(p)
        test_name = 'One-way ANOVA'
        
        dims = Y.dims
        Ps = np.reshape(Ps, (1,) + tuple(len(dim) for dim in dims))
        
        properties = Y.properties.copy()
        properties['colorspace'] = _vsl.colorspaces.get_sig()
        p = _vsl.data.ndvar(dims, Ps, properties=properties, name=X.name, info=test_name)

        # create dataset
        super(f_oneway, self).__init__(name="anova")
        self['p'] = p
    
    @property
    def all(self):
        return self['p']



class anova(test_result):
    """
    Attributes:
    
    allps:
        p-map for each effect
    
    """
    def __init__(self, Y='MEG', X='condition', sub=None, ds=None, info={}, v=False):
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
        properties['colorspace'] = _vsl.colorspaces.get_sig()
        # Y.data:  epoch X [time X sensor X freq]
        for name, Fs, Ps in fitter.map(Y.x.T, v=v):
            effect_names.append(name)
            P = _vsl.data.ndvar(Y.dims, Ps.T[None], properties=properties, name=name, info=name)
            self[name+'_p'] = P
    
    @property
    def all(self):
        epochs = []
        for name in self.info['effect_names']:
            epochs.append(self[name+'_p'])
        
        return epochs






def test(ndvars, parametric=True, match=None, func=None, attr='data',
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
