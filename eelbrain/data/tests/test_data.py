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
from numpy.testing import assert_array_equal, assert_array_almost_equal

from eelbrain.data import datasets, Var, Factor, Dataset, Celltable, load
from eelbrain.data.data_obj import (isdatalist, isndvar, isuv, isvar, align,
                                    align1, combine)


def assert_dataset_equal(ds1, ds2, msg="Datasets unequal", decimal=None):
    """
    Raise an assertion if two Datasets are not equal up to desired precision.

    Parameters
    ----------
    ds1, ds2 : Dataset
        Datasets to compare.
    msg : str
        Prefix of the error message to be printed in case of failure.
    decimal : None | int
        Desired precision (default is exact match).
    """
    assert_equal(ds1.keys(), ds2.keys(), "%s: different keys" % msg)
    for k in ds1.keys():
        assert_dataobj_equal(ds1[k], ds2[k], msg=msg, decimal=decimal)
    assert_equal(ds1.info.keys(), ds2.info.keys(), "%s: keys in info" % msg)


def assert_dataobj_equal(d1, d2, msg="Data-objects unequal", decimal=None):
    """
    Raise an assertion if two data-objects are not equal up to desired
    precision.

    Parameters
    ----------
    ds1, ds2 : data-objects
        Data-objects to compare.
    msg : str
        Prefix of the error message to be printed in case of failure.
    decimal : None | int
        Desired precision (default is exact match).
    """
    msg = "%s:" % msg
    assert_equal(d1.name, d2.name, "%s unequal names (%r vs %r"
                 ")" % (msg, d1.name, d2.name))
    msg += ' %r have' % d1.name
    assert_equal(len(d1), len(d2), "%s unequal length" % msg)
    if isvar(d1) and decimal:
        assert_array_almost_equal(d1.x, d2.x, decimal)
    elif isuv(d1):
        assert_true(np.all(d1 == d2), "%s unequal values: %r vs "
                    "%r" % (msg, d1, d2))
    elif isndvar(d1):
        assert_true(np.all(d1.x == d2.x), "%s unequal values" % msg)
    elif isdatalist(d1):
        for i in xrange(len(d1)):
            assert_equal(d1[i], d2[i], "%s unequal values" % msg)


def test_print():
    "Run the string representation methods"
    ds = datasets.get_rand()
    print ds
    print repr(ds)
    A = ds['A']
    print A
    print repr(A)
    Y = ds['Y']
    print Y
    print repr(Y)
    Ynd = ds['uts']
    print Ynd
    print repr(Ynd)


def test_align():
    "Testing align() and align1() functions"
    ds = datasets.get_rand()
    ds.index()
    idx = np.arange(0, ds.n_cases, 4)
    ds_sub = ds.subset(np.arange(0, ds.n_cases, 2))
    dsa = align1(ds_sub, idx)
    assert_array_equal(dsa['index'].x, idx, "align1() failure")

    dsa1, dsa2 = align(dsa, ds_sub)
    assert_array_equal(dsa1['index'].x, dsa2['index'].x, "align() failed")


def test_celltable():
    "Test the Celltable class."
    ds = datasets.get_rand()
    ds['cat'] = Factor('abcd', rep=15)

    ct = Celltable('Y', 'A', ds=ds)
    eq_(ct.n_cases, 60)
    eq_(ct.n_cells, 2)

    ct = Celltable('Y', 'A', match='rm', ds=ds)
    eq_(ct.n_cases, 30)
    eq_(ct.n_cells, 2)

    ct = Celltable('Y', 'cat', cat=('c', 'b'), ds=ds)
    eq_(ct.n_cases, 30)
    eq_(ct.X[0], 'c')
    eq_(ct.X[-1], 'b')

    ct = Celltable('Y', 'A', match='rm', ds=ds)
    eq_(ct.n_cases, 30)
    assert np.all(ct.groups['a0'] == ct.groups['a1'])

    ct = Celltable('Y', 'cat', match='rm', cat=('c', 'b'), ds=ds)
    eq_(ct.n_cases, 30)
    eq_(ct.X[0], 'c')
    eq_(ct.X[-1], 'b')

    # test sub
    ds_sub = ds.subset("A == 'a0'")
    ct_sub = Celltable('Y', 'B', ds=ds_sub)
    ct = Celltable('Y', 'B', sub="A == 'a0'", ds=ds)
    assert_dataobj_equal(ct_sub.Y, ct.Y)

    # test sub with rm
    ct_sub = Celltable('Y', 'B', match='rm', ds=ds_sub)
    ct = Celltable('Y', 'B', match='rm', sub="A == 'a0'", ds=ds)
    assert_dataobj_equal(ct_sub.Y, ct.Y)

    # test rm sorting
    ds = Dataset()
    ds['rm'] = Factor('abc', rep=4)
    ds['Y'] = Var(np.arange(3.).repeat(4))
    ds['X'] = Factor('ab', rep=2, tile=3)
    idx = np.arange(12)
    np.random.shuffle(idx)
    ds = ds[idx]
    ct = Celltable('Y', 'X', 'rm', ds=ds)
    assert_array_equal(ct.match, Factor('abc', tile=2))
    assert_array_equal(ct.Y, np.tile(np.arange(3.), 2))
    assert_array_equal(ct.X, Factor('ab', rep=3))


def test_combine():
    "Test combine()"
    ds1 = datasets.get_rand()
    ds2 = datasets.get_rand()
    ds = combine((ds1, ds2))
    assert_array_equal(ds2['Y'].x, ds['Y'].x[ds1.n_cases:], "Basic combine")
    del ds1['Y']
    del ds2['YCat']
    ds = combine((ds1, ds2))
    assert_array_equal(ds2['Y'].x, ds['Y'].x[ds1.n_cases:], "Combine with "
                       "missing Var")
    assert_true(np.all(ds1['YCat'] == ds['YCat'][:ds1.n_cases]), "Combine "
                "with missing Factor")

    # combine NDVar with unequel dimensions
    ds = datasets.get_rand(utsnd=True)
    y = ds['utsnd']
    y1 = y.subdata(sensor=['0', '1', '2', '3'])
    y2 = y.subdata(sensor=['1', '2', '3', '4'])
    ds1 = Dataset(y1)
    ds2 = Dataset(y2)
    dsc = combine((ds1, ds2))
    y = dsc['utsnd']
    assert_equal(y.sensor.names, ['1', '2', '3'], "Sensor dimension "
                 "intersection failed.")
    dims = ('case', 'sensor', 'time')
    ref = np.concatenate((y1.get_data(dims)[:, 1:], y2.get_data(dims)[:, :3]))
    assert_array_equal(y.get_data(dims), ref, "combine utsnd")


def test_dataset_sorting():
    "Test Dataset sorting methods"
    test_array = np.arange(10)
    ds = Dataset()
    ds['v'] = Var(test_array)
    ds['f'] = Factor(test_array)

    # shuffle the Dataset
    rand_idx = test_array.copy()
    np.random.shuffle(rand_idx)
    ds_shuffled = ds[rand_idx]

    # ascending, Var, copy
    dsa = ds_shuffled.sorted('v')
    assert_dataset_equal(dsa, ds, "Copy sorted by Var, ascending")

    # descending, Factor, in-place
    ds_shuffled.sort('f', descending=True)
    assert_dataset_equal(ds_shuffled, ds[::-1], "In-place sorted by Factor, "
                         "descending")


def test_dataset_indexing():
    """Test Dataset indexing"""
    ds = datasets.get_uv()
    ds['C', :] = 'c'
    ok_(np.all(ds.eval("C == 'c'")))


def test_ndvar_op():
    "Test NDVar operations"
    ds = datasets.get_rand(utsnd=True)
    Ynd = ds['utsnd']
    Ynd_bl = Ynd - Ynd.summary(time=(None, 0))

    # assert that the baseline is 0
    bl = Ynd_bl.summary('case', 'sensor', time=(None, 0))
    ok_(np.abs(bl) < 1e-10, "Baseline correction")


def test_io_pickle():
    "Test io by pickling"
    ds = datasets.get_rand()
    ds.info['info'] = "Some very useful information about the Dataset"
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


def test_io_txt():
    "Test Dataset io as text"
    ds = datasets.get_uv()

    # Var that has integer values as float
    ds['intflt'] = ds.eval('intvar * 1.')
    ds['intflt'].name = 'intflt'

    # io test
    tempdir = tempfile.mkdtemp()
    try:
        dest = os.path.join(tempdir, 'test.txt')
        ds.save_txt(dest)
        ds2 = load.tsv(dest)
    finally:
        shutil.rmtree(tempdir)

    assert_dataset_equal(ds, ds2, decimal=6)


def test_var():
    "Test Var objects"
    base = Factor('aabbcde')
    Y = Var.from_dict(base, {'a': 5, 'e': 8}, default=0)
    assert_array_equal(Y.x, [5, 5, 0, 0, 0, 0, 8])

