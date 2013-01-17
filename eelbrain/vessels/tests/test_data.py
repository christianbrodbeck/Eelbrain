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
