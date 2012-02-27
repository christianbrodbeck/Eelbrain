'''
Created on Apr 17, 2011

statistical functions that could not be in module :mod:`psystats.test` because
of a circular import 

@author: christianmbrodbeck
'''

import numpy as np
import scipy.stats



def CI(x, p=.95):
    """
    :returns: list with the endpoints of the confidence interval based on the
        inverse t-test 
        (`<http://en.wikipedia.org/wiki/Confidence_interval#Statistical_hypothesis_testing>`_). 
    
    :arg array x: data
    :arg float p: p value for confidence interval
    
    """
    M = np.mean(x)
    c = CIhw(x, p)
    return [M-c, M+c]


def CIhw(x, p=.95):
    """
    :returns: half-width of the confidence interval based on the inverse t-test 
        (`<http://en.wikipedia.org/wiki/Confidence_interval#Statistical_hypothesis_testing>`_). 
    
    :arg array x: data
    :arg float p: p value for confidence interval
    
    """
    N = len(x)
    t = scipy.stats.t.isf((1 - p) / 2, N - 1)
    c = (np.std(x, ddof=1) * t) / np.sqrt(N)
    return c

