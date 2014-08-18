'''
Created on Sep 13, 2013

@author: Christian M Brodbeck
'''
import random

import numpy as np

from .._data_obj import isvar, isndvar


def _resample_params(N, samples):
    """Decide whether to do permutations or random resampling

    Parameters
    ----------
    N : int
        Number of observations.
    samples : int
        ``samples`` parameter (number of resampling iterations, or < 0 to
        sample all permutations).

    Returns
    -------
    n_samples : int
        Adapted number of resamplings that will be done.
    samples : int
        Samples parameter for the resample function (-1 to do all permutations,
        otherwise same as n_samples).
    """
    n_perm = 2 ** N
    if n_perm - 1 <= samples:
        samples = -1

    if samples < 0:
        n_samples = n_perm - 1
    else:
        n_samples = samples

    return n_samples, samples


def resample(Y, samples=10000, replacement=False, unit=None, sign_flip=False,
             seed=0):
    """
    Generator function to resample a dependent variable (Y) multiple times

    Parameters
    ----------
    Y : Var | NDVar
        Variable which is to be resampled.
    samples : int
        Number of samples to yield. If < 0, all possible permutations are
        performed.
    replacement : bool
        whether random samples should be drawn with replacement or without.
    unit : categorial
        Factor specifying unit of measurement (e.g. subject). If unit is
        specified, resampling proceeds by first resampling the categories of
        unit (with or without replacement) and then shuffling the values
        within units (no replacement).
    sign_flip : bool
        Instead of shuffling the observations in the data (default), randomly
        do or do not flip the sign of each observation. If the number of
        possible permutations is smaller than ``samples``, ``resample(...,
        sign_flip=True)`` iterates over all possible permutations.
    seed : None | int
        Seed the random state of the relevant randomization module
        (:mod:`random` or :mod:`numpy.random`) to make replication possible.
        None to skip seeding (default 0).

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
        raise TypeError("Need Var or NDVar")

    Yout = Y.copy('{name}_resampled')

    if samples < 0 and not sign_flip:
        err = "Complete permutation for resampling through reordering"
        raise NotImplementedError(err)

    if seed is not None:
        if sign_flip:
            random.seed(seed)
        else:
            np.random.seed(seed)

    if sign_flip:
        if replacement is not False:
            err = ("`replacement` parameter does not apply when sign_flip is "
                   "True")
            raise ValueError(err)
        elif unit is not None:
            err = "`unit` parameter does not apply when sign_flip is True"
            raise ValueError(err)

        # determine possible number of permutations
        N = len(Y)
        n_perm = 2 ** N
        if samples < 0:
            # do all permutations
            sample_sequences = xrange(1, n_perm)
        else:
            # random resampling
            sample_sequences = random.sample(xrange(1, n_perm), samples)

        sign_shape = (N,) + (1,) * (Y.ndim - 1)
        mult = 2 ** np.arange(N, dtype=np.uint32).reshape(sign_shape)
        buffer_ = np.empty(sign_shape, dtype=np.uint32)
        for i in sample_sequences:
            np.floor_divide(i, mult, buffer_, dtype=np.uint32)
            buffer_ %= 2
            sign = np.where(buffer_, -1, 1)
            np.multiply(Y.x, sign, Yout.x)
            yield Yout
    elif unit is None:
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
            raise NotImplementedError("Replacement and units")
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
