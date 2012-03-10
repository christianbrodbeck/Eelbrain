'''
Created on Dec 2, 2010

@author: christian
'''

import numpy as np



def RMSSD(Y):
    """
    root mean square of successive differences. Used for heart rate variance
    analysis.
    
    """
    assert np.ndim(Y) == 1
    N = len(Y)
    assert N > 1
    dY = np.diff(Y)
    X = N/(N-1) * np.sum(dY**2)
    return np.sqrt(X)


def RMS(Y):
    """
    Root mean square. 
    
    asummes that Y is time x electrode (x epoch) array.
    
    Used as 'Global Field Power' (Murray et al., 2008).
    
    Murray, M. M., Brunet, D., and Michel, C. M. (2008). Topographic ERP 
            analyses: a step-by-step tutorial review. Brain Topogr, 20(4), 
            249-64.
    """
    # avg reference
    Yav = Y - Y.mean(1)[:,None,...]
    # root mean square
    rms = np.sqrt(np.mean(Yav**2, 1)[:,None,...])
    return rms