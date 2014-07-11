'''
Created on Sep 13, 2013

@author: Christian M Brodbeck
'''
import logging

from nose.tools import eq_, ok_
import numpy as np

from eelbrain import Factor, Var
from eelbrain._stats.permutation import resample


def test_permutation():
    """Test permutation"""
    v = Var(np.arange(6))
    res = np.empty((5, 6))
    for i, y in enumerate(resample(v, samples=5)):
        res[i] = y.x
    logging.info('Standard Permutation:\n%s' % res)

    # with unit
    s = Factor('abc', tile=2)
    for i, y in enumerate(resample(v, samples=5, unit=s)):
        res[i] = y.x
    logging.info('Permutation with Unit:\n%s' % res)

    # check we have only appropriate cells
    cols = [np.unique(res[:, i]) for i in xrange(res.shape[1])]
    for i in xrange(3):
        eq_(len(np.setdiff1d(cols[i], [i, i + 3])), 0)
    for i in xrange(3, 6):
        eq_(len(np.setdiff1d(cols[i], [i, i - 3])), 0)

    # check we have some variability
    eq_(max(map(len, cols)), 2)

    # with sign flip
    v = Var(np.arange(1, 7))
    res = np.empty((2 ** 6 - 1, 6))
    for i, y in enumerate(resample(v, samples=-1, sign_flip=True)):
        res[i] = y.x
    logging.info('Permutation with sign_flip:\n%s' % res)
    ok_(np.all(res.min(1) < 0), "Not all permutations have a sign flip")
