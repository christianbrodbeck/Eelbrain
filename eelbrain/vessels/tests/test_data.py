'''
Created on Dec 2, 2012

@author: christian
'''
import numpy as np
from numpy.testing import assert_array_equal

from ..data import align, align1
from .. import datasets


def test_print():
    "Run the string representation methods"
    ds = datasets.get_basic()
    print ds
    print repr(ds)
    A = ds['A']
    print A
    print repr(A)
    Y = ds['Y']
    print Y
    print repr(Y)
    Ynd = ds['Ynd']
    print Ynd
    print repr(Ynd)


def test_align():
    "Testing align() and align1() functions"
    ds = datasets.get_basic()
    ds.index()
    idx = np.arange(0, ds.n_cases, 4)
    ds_sub = ds.subset(np.arange(0, ds.n_cases, 2))
    dsa = align1(ds_sub, idx)
    assert_array_equal(dsa['index'].x, idx, "align1() failure")

    dsa1, dsa2 = align(dsa, ds_sub)
    assert_array_equal(dsa1['index'].x, dsa2['index'].x, "align() failed")
