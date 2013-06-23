'''
Created on Dec 2, 2012

@author: christian
'''
import os
import cPickle as pickle
import shutil
import tempfile

from nose.tools import assert_equal, assert_true, eq_, ok_
import numpy as np
from numpy.testing import assert_array_equal

from eelbrain.vessels import datasets
from eelbrain.vessels.data import (var, factor, align, align1, combine,
                                   isdatalist, isndvar, isuv)


def assert_dataset_equal(ds1, ds2):
    assert_equal(ds1.keys(), ds2.keys(), "Datasets unequal: different keys")
    for k in ds1.keys():
        assert_dataobj_equal(ds1[k], ds2[k])
    assert_equal(ds1.info.keys(), ds2.info.keys())


def assert_dataobj_equal(d1, d2):
    assert_equal(d1.name, d2.name, "Names unequal")
    assert_equal(len(d1), len(d2), "Unequal length")
    if isuv(d1):
        assert_true(np.all(d1 == d2), "Values unequal")
    elif isndvar(d1):
        assert_true(np.all(d1.x == d2.x), "Values unequal")
    elif isdatalist(d1):
        for i in xrange(len(d1)):
            assert_equal(d1[i], d2[i], "Values unequal")


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


def test_combine():
    "Test combine()"
    ds1 = datasets.get_basic()
    ds2 = datasets.get_basic()
    ds = combine((ds1, ds2))
    assert_array_equal(ds2['Y'].x, ds['Y'].x[ds1.n_cases:], "Basic combine")
    del ds1['Y']
    del ds2['Cat']
    ds = combine((ds1, ds2))
    assert_array_equal(ds2['Y'].x, ds['Y'].x[ds1.n_cases:], "Combine with "
                       "missing var")
    assert_true(np.all(ds1['Cat'] == ds['Cat'][:ds1.n_cases]), "Combine with "
                "missing factor")


def test_ndvar_op():
    "Test ndvar operations"
    ds = datasets.get_basic()
    Ynd = ds['Ynd']
    Ynd_bl = Ynd - Ynd.summary(time=(None, 0))

    # assert that the baseline is 0
    bl = Ynd_bl.summary('case', time=(None, 0))
    ok_(np.abs(bl) < 1e-10, "Baseline correction")


def test_pickle_io():
    "Test io by pickling"
    ds = datasets.get_basic()
    ds.info['info'] = "Some very useful information about the dataset"
    tempdir = tempfile.mkdtemp()
    try:
        dest = os.path.join(tempdir, 'test.pickled')
        with open(dest, 'w') as fid:
            pickle.dump(ds, fid, protocol=pickle.HIGHEST_PROTOCOL)
        with open(dest) as fid:
            ds2 = pickle.load(fid)
    finally:
        shutil.rmtree(tempdir)

    assert_dataset_equal(ds, ds2)


def test_var():
    "Test var objects"
    base = factor('aabbcde')
    Y = var.from_dict(base, {'a': 5, 'e': 8}, default=0)
    assert_array_equal(Y.x, [5, 5, 0, 0, 0, 0, 8])

