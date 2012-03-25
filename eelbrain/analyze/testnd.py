'''

desired qualities:

 * .subdata method
 * contain overlay specification (data & contours (colorspace))

 * when subdata(timewindow=(.1,.3)), the test should be newly executed for the
   new data

-> ??? do make new subclass to ensure that 


Created on Feb 22, 2012

@author: christian
'''

import numpy as np
import scipy.stats
#import mne

from eelbrain import vessels as _vsl



class TestResults(_vsl.data.dataset):
    def __repr__(self):
        tmp = "<TestResult: %s>"
        return tmp % self.__class__.__name__


class ttest(_vsl.data.dataset):
    def __init__(self, dataset, Y='MEG', X='condition', c1='c1', c2=0, match=None, contours=None):
        """
        c1 and c2 : ndvars (or dataset with default_DV)
            segments between which to perform the test
        
        """
        if isinstance(Y, basestring):
            Y = dataset[Y]
        if isinstance(X, basestring):
            X = dataset[X]
        if not contours:
            contours = { .05: (.8, .2, .0),  .01: (1., .6, .0),  .001: (1., 1., .0),
                        -.05: (0., .2, 1.), -.01: (.4, .8, 1.), -.001: (.5, 1., 1.),
                        }
        
        c1DV = Y[X==c1]
        c1_mean = c1DV.get_summary(name=c1)
        if isinstance(c2, basestring):
            c2DV = Y[X==c2]
            c2DV.name = c2
            c2_mean = c2DV.get_summary(name=c2)
            data = [c1_mean, c2_mean]
            diff = c1_mean - c2_mean
            if match:
                match1 = match[X==c1]
                match2 = match[X==c2]
                index = match2.get_index_to_match(match1)
                data2 = c2DV.data[index]            
                T, P = scipy.stats.ttest_rel(c1DV.data, data2, axis=0)
                test_name = 'Related Samples $t$-Test'
            else:
                T, P = scipy.stats.ttest_ind(c1DV.data, c2DV.data, axis=0)
                test_name = 'Independent Samples $t$-Test'
        elif np.isscalar(c2):
            data = [c1_mean]
            T, P = scipy.stats.ttest_1samp(c1DV.data, popmean=c2, axis=0)
            test_name = '1-Sample $t$-Test'
            if c2:
                diff = c1_mean - c2
            else:
                diff = None
        else:
            raise ValueError('invalid c2: %r' % c2)
        
        T = T[None]
        P = P[None]
        
#        direction = np.sign(diff.data)
#        P = P * direction# + 1 # (1 - P)
#        for k in contours.copy():
#            contours[k+1] = contours.pop(k)
        
        dims = c1DV.dims
        properties = c1DV.properties.copy()
        
        properties['colorspace'] = _vsl.colorspaces.Colorspace(contours=contours)
        P = _vsl.data.ndvar(dims, P, properties=properties, name='p', info=test_name)
        
        properties['colorspace'] = _vsl.colorspaces.get_default()
        T = _vsl.data.ndvar(dims, T, properties=properties, name='T', info=test_name)
        
        self.data = data
        if diff is None:
            self.diff = self.all = [data + [P]]
            items = data + [T, P]
        else:
            self.diff = [[diff, P]]
            self.all = data + self.diff
            items = data + [diff, T, P]
        
        _vsl.data.dataset.__init__(self, name=test_name, *items)
        




class old_ttest(TestResults):
    def __init__(self, c1, c2=0, match=None, contours=None):
        """
        c1 and c2 : ndvars (or dataset with default_DV)
            segments between which to perform the test
        
        """
        if not contours:
            contours = {.05: '.5', .01: '.75', .001:'1.'}
        
        c1DV = c1[c1.default_DV]
        c1_mean = c1DV.as_epoch()
        if np.isscalar(c2):
            data = [c1_mean]
            T, P = scipy.stats.ttest_1samp(c1DV.data, popmean=c2, axis=0)
            test_name = '1-Sample $t$-Test'
            if c2:
                diff = c1_mean - c2
            else:
                diff = c1_mean.copy()
        else:
            c2DV = c2[c2.default_DV]
            c2_mean = c2DV.as_epoch()
            data = [c1_mean, c2_mean]
            diff = c1_mean - c2_mean
            if match:
                match1 = c1[match]
                match2 = c2[match]
                index = match2.get_index_to_match(match1)
                data2 = c2DV.data[index]            
                T, P = scipy.stats.ttest_rel(c1DV.data, data2, axis=0)
                test_name = 'Related Samples $t$-Test'
            else:
                T, P = scipy.stats.ttest_ind(c1DV.data, c2DV.data, axis=0)
                test_name = 'Independent Samples $t$-Test'
            
#        direction = np.sign(( - c2_mean).data)
#        Pdir = (1 - P) * direction
        
        dims = c1DV.dims
        properties = c1DV.properties.copy()
        properties['colorspace'] = _vsl.colorspaces.Colorspace(contours=contours)
        P = _vsl.data.epoch(dims, P, properties=properties, name='p', info=test_name)
#        Tepoch = _vsl.data.epoch(dims, T, properties=None, name="???", variables={}, info="")
        
        diff.overlay = P
        
        self.data = data
        self.diff = diff
        if np.isscalar(c2) and c2==0:
            self.all = [diff]
        else:
            self.all = data + [diff]



def test(ndvars, parametric=True, match=None, func=None, attr='data',
         name="{test}"):
    """
    use func (func) or attr (str) to customize data
    (func=abs for )
    
    """
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
