'''
Created on Sep 13, 2013

@author: Christian M Brodbeck
'''
import numpy as np

from ..data_obj import isvar, isndvar


def resample(Y, samples=10000, replacement=False, unit=None):
    """
    Generator function to resample a dependent variable (Y) multiple times

    Parameters
    ----------
    Y : Var | NDVar
        Variable which is to be resampled.
    samples : int
        number of samples to yield.
    replacement : bool
        whether random samples should be drawn with replacement or without.
    unit : categorial
        Factor specifying unit of measurement (e.g. subject). If unit is
        specified, resampling proceeds by first resampling the categories of
        unit (with or without replacement) and then shuffling the values
        within units (no replacement).

    Returns
    -------
    Iterator over Y_resampled. The same copy of ``Y`` is yielded in each
    iteration with different data.
    """
    samples = int(samples)
    if isvar(Y):
        pass
    elif isndvar(Y):
        if not Y.has_case:
            raise ValueError("Need NDVar with cases")
    else:
        raise TypeError("need Var or NDVar")

    Yout = Y.copy('{name}_resampled')

    if unit is None:
        if replacement:
            N = len(Y)
            for _ in xrange(samples):
                index = np.random.randint(N, N)
                Yout.x = Y.x[index]
                yield Yout
        else:  # OK
            for _ in xrange(samples):
                np.random.shuffle(Yout.x)
                yield Yout
    else:
        if replacement:
            raise NotImplementedError("replacement and units")
        else:
            idx_orig = np.arange(len(Y))
            idx_perm = np.arange(len(Y))
            unit_idxs = [np.nonzero(unit == cell)[0] for cell in unit.cells]
            for _ in xrange(samples):
                for idx_ in unit_idxs:
                    v = idx_orig[idx_]
                    np.random.shuffle(v)
                    idx_perm[idx_] = v
                Yout.x[idx_perm] = Y.x
                yield Yout
