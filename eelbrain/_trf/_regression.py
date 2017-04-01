# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from inspect import getargspec
from math import sqrt

import numpy as np
from sklearn.linear_model import Ridge

from .shared import RevCorrData


class RegressionResult(object):
    def __init__(self, h, r, y, x, tstart, tstop, alpha):
        self.h = h
        self.r = r
        self.y = y
        self.x = x
        self.tstart = tstart
        self.tstop = tstop
        self.alpha = alpha

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in
                getargspec(self.__init__).args[1:]}

    def __setstate__(self, state):
        self.__init__(**state)

    def __repr__(self):
        if self.x is None or isinstance(self.x, basestring):
            x = self.x
        else:
            x = ' + '.join(map(str, self.x))
        items = ['regression %s ~ %s' % (self.y, x),
                 '%g - %g' % (self.tstart, self.tstop),
                 'alpha=%s' % (self.alpha,)]
        return '<%s>' % ', '.join(items)


def regression(y, x, tstart, tstop, alpha=0.1):
    """Reverse correlation using ridge regression
    
    Parameters
    ----------
    y : NDVar
        Signal to predict.
    x : NDVar | sequence of NDVar
        Signal to use to predict ``y``. Can be sequence of NDVars to include
        multiple predictors. Time dimension must correspond to ``y``.
    tstart : float
        Start of the TRF in seconds.
    tstop : float
        Stop of the TRF in seconds.
    alpha : scalar
        Regularization parameter (default ``0.1``).
    """
    data = RevCorrData(y, x, 'l2', False)

    # prepare estimator
    tstep = data.time.tstep
    h_n_samples = int(round((tstop - tstart) / tstep))
    imin = int(round(tstart / tstep))

    # S:  y-time by (x inside h-time)
    n_y, n_times = data.y.shape
    n_x = len(data.x)
    istop = imin + h_n_samples
    s = np.zeros((n_times, h_n_samples * n_x))
    si = 0
    for shift in xrange(imin, istop):
        si_t = slice(max(0, shift), min(n_times, n_times + shift))
        xi_t = slice(max(0, -shift), min(n_times, n_times - shift))
        for xi in xrange(n_x):
            s[si_t, si] = data.x[xi, xi_t]
            si += 1

    # regression
    y = data.y.T
    ridge = Ridge(alpha)
    ridge.fit(s, y)
    h_x = ridge.coef_.reshape((n_y, h_n_samples, n_x)).swapaxes(1, 2)
    # r
    y_pred = ridge.predict(s)
    r_x = np.empty(n_y)
    for i in xrange(n_y):
        r_x[i] = np.corrcoef(y[:, i], y_pred[:, i])[0, 1]

        # r_x[i] = sqrt(ridge.score(s, y[:, i]))   returns scalar

    # package output
    h = data.package_kernel(h_x, tstart)
    r = data.package_statistic(r_x, 'r', 'correlation')
    return RegressionResult(h, r, data.y_name, data.x_name, tstart, tstop, alpha)
