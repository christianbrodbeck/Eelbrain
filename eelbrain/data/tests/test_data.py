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

from eelbrain.data import datasets, var, factor, dataset
from eelbrain.data.data_obj import (isdatalist, isndvar, isuv, align, align1,
                                    combine)


def assert_dataset_equal(ds1, ds2, msg="Datasets unequal"):
    assert_equal(ds1.keys(), ds2.keys(), "%s: different keys" % msg)
    for k in ds1.keys():
        assert_dataobj_equal(ds1[k], ds2[k], msg=msg)
    assert_equal(ds1.info.keys(), ds2.info.keys(), "%s: keys in info" % msg)


def assert_dataobj_equal(d1, d2, msg="Data-objects unequal"):
    msg = "%s:" % msg
    assert_equal(d1.name, d2.name, "%s unequal names (%r vs %r"
                 ")" % (msg, d1.name, d2.name))
    msg += ' %r have' % d1.name
    assert_equal(len(d1), len(d2), "%s unequal length" % msg)
    if isuv(d1):
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
                       "missing var")
    assert_true(np.all(ds1['YCat'] == ds['YCat'][:ds1.n_cases]), "Combine "
                "with missing factor")

    # combine ndvar with unequel dimensions
    ds = datasets.get_rand(utsnd=True)
    y = ds['utsnd']
    y1 = y.subdata(sensor=['0', '1', '2', '3'])
    y2 = y.subdata(sensor=['1', '2', '3', '4'])
    ds1 = dataset(y1)
    ds2 = dataset(y2)
    dsc = combine((ds1, ds2))
    y = dsc['utsnd']
    assert_equal(y.sensor.names, ['1', '2', '3'], "Sensor dimension "
                 "intersection failed.")
    dims = ('case', 'sensor', 'time')
    ref = np.concatenate((y1.get_data(dims)[:, 1:], y2.get_data(dims)[:, :3]))
    assert_array_equal(y.get_data(dims), ref, "combine utsnd")


def test_dataset_sorting():
    "Test dataset sorting methods"
    test_array = np.arange(10)
    ds = dataset()
    ds['v'] = var(test_array)
    ds['f'] = factor(test_array)

    # shuffle the dataset
    rand_idx = test_array.copy()
    np.random.shuffle(rand_idx)
    ds_shuffled = ds[rand_idx]

    # ascending, var, copy
    dsa = ds_shuffled.sorted('v')
    assert_dataset_equal(dsa, ds, "Copy sorted by var, ascending")

    # descending, factor, in-place
    ds_shuffled.sort('f', descending=True)
    assert_dataset_equal(ds_shuffled, ds[::-1], "In-place sorted by factor, "
                         "descending")


def test_ndvar_op():
    "Test ndvar operations"
    ds = datasets.get_rand(utsnd=True)
    Ynd = ds['utsnd']
    Ynd_bl = Ynd - Ynd.summary(time=(None, 0))

    # assert that the baseline is 0
    bl = Ynd_bl.summary('case', 'sensor', time=(None, 0))
    ok_(np.abs(bl) < 1e-10, "Baseline correction")


def test_pickle_io():
    "Test io by pickling"
    ds = datasets.get_rand()
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

