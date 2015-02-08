# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import izip, product
import os
import cPickle as pickle
import shutil
from string import ascii_lowercase
import tempfile

import mne
from nose.tools import (assert_equal, assert_almost_equal, assert_is_instance,
                        assert_true, eq_, ok_, assert_raises)
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from eelbrain import (datasets, load, Var, Factor, NDVar, Dataset, Celltable,
                      align, align1, combine)
from eelbrain._data_obj import (asvar, isdatalist, isndvar, isvar, isuv,
                                Categorial, SourceSpace, UTS)
from eelbrain._stats.stats import rms


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
    assert_equal(ds1.keys(), ds2.keys(), "%s: different keys (%s vs %s)" %
                 (msg, ds1.keys(), ds2.keys()))
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
    if not hasattr(d1, '_stype_'):
        raise TypeError("d1 is not a data-object but %s" % repr(d1))
    elif not hasattr(d2, '_stype_'):
        raise TypeError("d2 is not a data-object but %s" % repr(d2))
    else:
        eq_(d1._stype_, d2._stype_)
    msg = "%s:" % msg
    assert_equal(d1.name, d2.name, "%s unequal names (%r vs %r"
                 ")" % (msg, d1.name, d2.name))
    msg += 'Two objects named %r have' % d1.name
    len1 = len(d1)
    len2 = len(d2)
    assert_equal(len1, len2, "%s unequal length: %i/%i" % (msg, len1, len2))
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


def assert_source_space_equal(src1, src2, msg="SourceSpace Dimension objects "
                              "unequal"):
    """
    Raise an assertion if two SourceSpace objects are not equal up to desired
    precision.

    Parameters
    ----------
    src1, src2 : SourceSpace objects
        SourceSpace objects to compare.
    msg : str
        Prefix of the error message to be printed in case of failure.
    """
    msg = "%s:" % msg
    assert_array_equal(src1.vertno[0], src2.vertno[0], "%s unequal lh vertno "
                       "(%r vs %r)" % (msg, src1.vertno[0], src2.vertno[0]))
    assert_array_equal(src1.vertno[1], src2.vertno[1], "%s unequal rh vertno "
                       "(%r vs %r)" % (msg, src1.vertno[1], src2.vertno[1]))
    assert_equal(src1.subject, src2.subject, "%s unequal subject (%r vs %r"
                 ")" % (msg, src1.subject, src2.subject))
    assert_equal(src1.src, src2.src, "%s unequal names (%r vs %r"
                 ")" % (msg, src1.src, src2.src))
    assert_equal(src1.subjects_dir, src2.subjects_dir, "%s unequal names (%r "
                 "vs %r)" % (msg, src1.subjects_dir, src2.subjects_dir))


def test_print():
    "Run the string representation methods"
    ds = datasets.get_uts()
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


def test_aggregate():
    "Test aggregation methods"
    ds = datasets.get_uts()

    # don't handle inconsistencies silently
    assert_raises(ValueError, ds.aggregate, 'A%B')

    dsa = ds.aggregate('A%B', drop_bad=True)
    assert_array_equal(dsa['n'], [15, 15, 15, 15])
    idx1 = ds.eval("logical_and(A=='a0', B=='b0')")
    assert_equal(dsa['Y', 0], ds['Y', idx1].mean())

    # unequal cell counts
    ds = ds[:-3]
    dsa = ds.aggregate('A%B', drop_bad=True)
    assert_array_equal(dsa['n'], [15, 15, 15, 12])
    idx1 = ds.eval("logical_and(A=='a0', B=='b0')")
    assert_equal(dsa['Y', 0], ds['Y', idx1].mean())

    dsa = ds.aggregate('A%B', drop_bad=True, equal_count=True)
    assert_array_equal(dsa['n'], [12, 12, 12, 12])
    idx1_12 = np.logical_and(idx1, idx1.cumsum() <= 12)
    assert_equal(dsa['Y', 0], ds['Y', idx1_12].mean())


def test_align():
    "Testing align() and align1() functions"
    ds = datasets.get_uv()
    ds.index()
    idx4 = np.arange(0, ds.n_cases, 4)
    idx4i = idx4[::-1]
    ds2 = ds.sub(np.arange(0, ds.n_cases, 2))

    # align1: align Dataset to index
    dsa = align1(ds2, idx4)
    assert_array_equal(dsa['index'], idx4, "align1() failure")
    dsa = align1(ds2, idx4i)
    assert_array_equal(dsa['index'], idx4i, "align1() failure")
    # d_idx as Var
    dsa = align1(ds2[::2], idx4, idx4i)
    assert_array_equal(dsa['index'], idx4i, "align1() failure")
    assert_raises(ValueError, align1, ds2, idx4, idx4i)

    # Factor index
    assert_raises(ValueError, align1, ds, ds['rm', ::-1], 'rm')
    fds = ds[:20]
    dsa = align1(fds, fds['rm', ::-1], 'rm')
    assert_array_equal(dsa['index'], np.arange(19, -1, -1), "align1 Factor")

    # align two datasets
    dsa1, dsa2 = align(ds, ds2)
    assert_array_equal(dsa1['index'], dsa2['index'], "align() failure")
    dsa1, dsa2 = align(ds, ds2[::-1])
    assert_array_equal(dsa1['index'], dsa2['index'], "align() failure")


def test_celltable():
    "Test the Celltable class."
    ds = datasets.get_uts()
    ds['cat'] = Factor('abcd', repeat=15)

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

    # coercion of numerical X
    X = ds.eval("A == 'a0'")
    ct = Celltable('Y', X, cat=(None, None), ds=ds)
    assert_equal(('False', 'True'), ct.cat)
    assert_array_equal(ct.data['True'], ds['Y', X])

    ct = Celltable('Y', X, cat=(True, False), ds=ds)
    assert_equal(('True', 'False'), ct.cat)
    assert_array_equal(ct.data['True'], ds['Y', X])

    # test coercion of Y
    ct = Celltable(ds['Y'].x, 'A', ds=ds)
    assert_is_instance(ct.Y, np.ndarray)
    ct = Celltable(ds['Y'].x, 'A', ds=ds, coercion=asvar)
    assert_is_instance(ct.Y, Var)

    # test sub
    ds_sub = ds.sub("A == 'a0'")
    ct_sub = Celltable('Y', 'B', ds=ds_sub)
    ct = Celltable('Y', 'B', sub="A == 'a0'", ds=ds)
    assert_dataobj_equal(ct_sub.Y, ct.Y)

    # test sub with rm
    ct_sub = Celltable('Y', 'B', match='rm', ds=ds_sub)
    ct = Celltable('Y', 'B', match='rm', sub="A == 'a0'", ds=ds)
    assert_dataobj_equal(ct_sub.Y, ct.Y)

    # Interaction match
    ct = Celltable('Y', 'A', match='B % rm', ds=ds)
    ok_(ct.all_within)
    assert_dataobj_equal(combine((ct.data['a0'], ct.data['a1'])), ds['Y'])

    # test rm sorting
    ds = Dataset()
    ds['rm'] = Factor('abc', repeat=4)
    ds['Y'] = Var(np.arange(3.).repeat(4))
    ds['X'] = Factor('ab', repeat=2, tile=3)
    idx = np.arange(12)
    np.random.shuffle(idx)
    ds = ds[idx]
    ct = Celltable('Y', 'X', 'rm', ds=ds)
    assert_array_equal(ct.match, Factor('abc', tile=2))
    assert_array_equal(ct.Y, np.tile(np.arange(3.), 2))
    assert_array_equal(ct.X, Factor('ab', repeat=3))


def test_combine():
    "Test combine()"
    ds1 = datasets.get_uts()
    ds2 = datasets.get_uts()
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
    ds = datasets.get_uts(utsnd=True)
    y = ds['utsnd']
    y1 = y.sub(sensor=['0', '1', '2', '3'])
    y2 = y.sub(sensor=['1', '2', '3', '4'])
    ds1 = Dataset((y1,))
    ds2 = Dataset((y2,))
    dsc = combine((ds1, ds2))
    y = dsc['utsnd']
    assert_equal(y.sensor.names, ['1', '2', '3'], "Sensor dimension "
                 "intersection failed.")
    dims = ('case', 'sensor', 'time')
    ref = np.concatenate((y1.get_data(dims)[:, 1:], y2.get_data(dims)[:, :3]))
    assert_array_equal(y.get_data(dims), ref, "combine utsnd")


def test_dataset_combining():
    "Test Dataset combination methods"
    ds = datasets.get_uv()
    del ds['fltvar'], ds['intvar'], ds['A']

    ds2 = datasets.get_uv()
    del ds2['fltvar'], ds2['intvar']
    ds.update(ds2)
    assert_array_equal(ds['A'], ds2['A'])

    ds2 = datasets.get_uv()
    del ds2['fltvar'], ds2['intvar']
    ds2['B'][5] = 'something_else'
    del ds['A']
    assert_raises(ValueError, ds.update, ds2)


def test_dataset_indexing():
    """Test Dataset indexing"""
    ds = datasets.get_uv()

    # indexing values
    assert_equal(ds['A', 1], ds['A'][1])
    assert_equal(ds[1, 'A'], ds['A'][1])

    # indexing variables
    assert_dataobj_equal(ds[:, 'A'], ds['A'])
    assert_dataobj_equal(ds['A', :], ds['A'])
    assert_dataobj_equal(ds[:10, 'A'], ds['A'][:10])
    assert_dataobj_equal(ds['A', :10], ds['A'][:10])

    # new Dataset through indexing
    ds2 = Dataset()
    ds2['A'] = ds['A']
    assert_dataset_equal(ds[('A',)], ds2)
    ds2['B'] = ds['B']
    assert_dataset_equal(ds['A', 'B'], ds2)
    assert_dataset_equal(ds[('A', 'B'), :10], ds2[:10])
    assert_dataset_equal(ds[:10, ('A', 'B')], ds2[:10])

    # assigning value
    ds[2, 'A'] = 'hello'
    assert_equal(ds[2, 'A'], 'hello')
    ds['A', 2] = 'not_hello'
    assert_equal(ds[2, 'A'], 'not_hello')

    # assigning new factor
    ds['C', :] = 'c'
    ok_(np.all(ds.eval("C == 'c'")))

    # assigning new Var
    ds['D1', :] = 5.
    ds[:, 'D2'] = 5.
    assert_array_equal(ds['D1'], 5)
    assert_array_equal(ds['D2'], 5)

    # test illegal names
    f = Factor('aaabbb')
    assert_raises(ValueError, ds.__setitem__, '%dsa', f)
    assert_raises(ValueError, ds.__setitem__, '432', f)
    assert_raises(ValueError, ds.__setitem__, ('%dsa', slice(None)), 'value')
    assert_raises(ValueError, ds.__setitem__, (slice(None), '%dsa'), 'value')
    assert_raises(ValueError, ds.__setitem__, ('432', slice(None)), 4.)
    assert_raises(ValueError, ds.__setitem__, (slice(None), '432'), 4.)


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


def test_dim_categorial():
    "Test Categorial Dimension"
    values = ['a', 'b', 'c', 'abc']
    name = 'cat'
    dim = Categorial(name, values)

    # basic properties
    print dim
    assert_equal(len(dim), len(values))

    # persistence
    s = pickle.dumps(dim, pickle.HIGHEST_PROTOCOL)
    dim_ = pickle.loads(s)
    assert_equal(dim_, dim)

    # indexing
    sub_values = values[:2]
    idx = dim.dimindex(sub_values)
    assert_array_equal(dim.dimindex(tuple(sub_values)), idx)
    assert_equal(dim[idx], Categorial(name, sub_values))
    assert_equal(dim.dimindex('a'), values.index('a'))
    assert_equal(dim.dimindex('abc'), values.index('abc'))

    # intersection
    dim2 = Categorial(name, ['c', 'b', 'e'])
    dim_i = dim.intersect(dim2)
    assert_equal(dim_i, Categorial(name, ['b', 'c']))

    # unicode
    dimu = Categorial(name, [u'c', 'b', 'e'])
    assert_equal(dimu.values.dtype.kind, 'U')
    assert_equal(dim2.values.dtype.kind, 'S')
    assert_equal(dimu, dim2)


def test_dim_uts():
    "Test UTS Dimension"
    uts = UTS(-0.1, 0.005, 301)

    # make sure indexing rounds correctly for floats
    for i, s in enumerate(np.arange(0, 1.4, 0.05)):
        idx = uts.dimindex((-0.1 + s, s))
        assert_equal(idx.start, 10 * i)
        assert_equal(idx.stop, 20 + 10 * i)

    # intersection
    uts1 = UTS(-0.1, 0.01, 50)
    uts2 = UTS(0, 0.01, 20)
    intersection = uts1.intersect(uts2)
    assert_equal(intersection, uts2)
    idx = uts1.dimindex((0, 0.2))
    assert_equal(uts1[idx], uts2)


def test_effect():
    "Test _Effect class"
    # .enumerate_cells()
    f1 = Factor('aabbccaabbcc')
    f2 = Factor('abababababab')
    i = f1 % f2

    n1 = np.concatenate((np.tile([0, 1], 3), np.tile([2, 3], 3)))
    assert_array_equal(f1.enumerate_cells(), n1)
    assert_array_equal(f2.enumerate_cells(), np.arange(6).repeat(2))
    assert_array_equal(i.enumerate_cells(), np.arange(2).repeat(6))


def test_factor():
    "Test basic Factor functionality"
    f = Factor('aabbcc')

    # removing a cell
    assert_equal(f.cells, ('a', 'b', 'c'))
    f[f == 'c'] = 'a'
    assert_equal(f.cells, ('a', 'b'))

    # label length
    lens = [2, 5, 32, 2, 32, 524]
    f = Factor(['a' * l for l in lens])
    assert_array_equal(f.label_length(), lens)


def test_factor_relabel():
    "Test Factor.relabel() method"
    f = Factor('aaabbbccc')
    f.relabel({'a': 'd'})
    assert_array_equal(f, Factor('dddbbbccc'))
    f.relabel({'d': 'c', 'c': 'd'})
    assert_array_equal(f, Factor('cccbbbddd'))
    f.relabel({'d': 'c'})
    assert_array_equal(f, Factor('cccbbbccc'))
    assert_raises(KeyError, f.relabel, {'a':'c'})


def test_interaction():
    "Test Interaction"
    ds = datasets.get_uv()
    A = ds['A']
    B = ds['B']
    i = A % B
    # eq for sequence
    assert_array_equal(i == A % B, True)
    assert_array_equal(i == B % A, False)
    assert_array_equal(i == A, False)
    assert_array_equal(i == ds['fltvar'], False)
    assert_array_equal(ds.eval("A%B") == Factor(ds['A']) % B, True)
    # eq for element
    for a, b in product(A.cells, B.cells):
        assert_array_equal(i == (a, b), np.logical_and(A == a, B == b))


def test_isin():
    "Test .isin() methods"
    values = np.array([  6, -6, 6, -2, -1, 0, -10, -5, -10, -6])
    v = values[0]
    v2 = values[:2]
    labels = {i: c for i, c in enumerate(ascii_lowercase, -10)}
    vl = labels[v]
    v2l = [labels[v_] for v_ in v2]

    target = np.logical_or(values == v2[0], values == v2[1])
    inv_target = np.invert(target)
    index_target = np.flatnonzero(values == v)
    empty = np.array([])

    var = Var(values)
    assert_array_equal(var.index(v), index_target)
    assert_array_equal(var.isin(v2), target)
    assert_array_equal(var.isany(*v2), target)
    assert_array_equal(var.isnot(*v2), inv_target)
    assert_array_equal(var.isnotin(v2), inv_target)

    var0 = Var([])
    assert_array_equal(var0.isin(v2), empty)
    assert_array_equal(var0.isany(*v2), empty)
    assert_array_equal(var0.isnot(*v2), empty)
    assert_array_equal(var0.isnotin(v2), empty)

    f = Factor(values, labels=labels)
    assert_array_equal(f.index(vl), index_target)
    assert_array_equal(f.isin(v2l), target)
    assert_array_equal(f.isany(*v2l), target)
    assert_array_equal(f.isnot(*v2l), inv_target)
    assert_array_equal(f.isnotin(v2l), inv_target)

    f0 = Factor([])
    assert_array_equal(f0.isin(v2l), empty)
    assert_array_equal(f0.isany(*v2l), empty)
    assert_array_equal(f0.isnot(*v2l), empty)
    assert_array_equal(f0.isnotin(v2l), empty)


def test_ndvar():
    "Test the NDVar class"
    ds = datasets.get_uts(utsnd=True)
    x = ds['utsnd']

    # slicing
    assert_raises(KeyError, x.sub, sensor='5')
    assert_equal(x.sub(sensor='4').ndim, 2)
    assert_equal(x.sub(sensor=['4']).ndim, 3)
    assert_equal(x.sub(case=1, sensor='4').ndim, 1)

    # baseline correction
    x_bl = x - x.summary(time=(None, 0))
    # assert that the baseline is 0
    bl = x_bl.summary('case', 'sensor', time=(None, 0))
    ok_(abs(bl) < 1e-10, "Baseline correction")

    # NDVar as index
    sens_mean = x.mean(('case', 'time'))
    idx = sens_mean > 0
    pos = sens_mean[idx]
    assert_array_equal(pos.x > 0, True)


def test_ndvar_binning():
    "Test NDVar.bin()"
    x = np.arange(10)
    time = UTS(-0.1, 0.1, 10)
    x_dst = x.reshape((5, 2)).mean(1)
    time_dst = np.arange(0., 0.9, 0.2)

    # 1-d
    ndvar = NDVar(x, (time,))
    b = ndvar.bin(0.2)
    assert_array_equal(b.x, x_dst, "Binned data")
    assert_array_equal(b.time.x, time_dst, "Bin times")

    # 2-d
    ndvar = NDVar(np.vstack((x, x, x)), ('case', time))
    b = ndvar.bin(0.2)
    assert_array_equal(b.x, np.vstack((x_dst, x_dst, x_dst)), "Binned data")
    assert_array_equal(b.time.x, time_dst, "Bin times")

    # time:
    x = np.ones((5, 70))
    ndvar = NDVar(x, ('case', UTS(0.45000000000000007, 0.005, 70)))
    binned_ndvar = ndvar.bin(0.05)
    assert_array_equal(binned_ndvar.x, 1.)
    eq_(binned_ndvar.shape, (5, 7))


def test_ndvar_summary_methods():
    "Test NDVar methods for summarizing data over axes"
    ds = datasets.get_uts(utsnd=True)
    x = ds['utsnd']

    dim = 'sensor'
    axis = x.get_axis(dim)
    dims = ('case', 'sensor')
    axes = tuple(x.get_axis(d) for d in dims)
    idx = x > 0
    x0 = x[0]
    idx0 = idx[0]
    xsub = x.sub(time=(0, 0.5))
    idxsub = xsub > 0
    idx1d = x.mean(('case', 'time')) > 0

    # numpy functions
    assert_equal(x.any(), x.x.any())
    assert_array_equal(x.any(dim), x.x.any(axis))
    assert_array_equal(x.any(dims), x.x.any(axes))
    assert_array_equal(x.any(idx0), [x_[idx0.x].any() for x_ in x.x])
    assert_array_equal(x.any(idx), [x_[i].any() for x_, i in izip(x.x, idx.x)])
    assert_array_equal(x0.any(idx0), x0.x[idx0.x].any())
    assert_array_equal(x.any(idxsub), xsub.any(idxsub))
    assert_array_equal(x.any(idx1d), x.x[:, idx1d.x].any(1))

    assert_equal(x.max(), x.x.max())
    assert_array_equal(x.max(dim), x.x.max(axis))
    assert_array_equal(x.max(dims), x.x.max(axes))
    assert_array_equal(x.max(idx0), [x_[idx0.x].max() for x_ in x.x])
    assert_array_equal(x.max(idx), [x_[i].max() for x_, i in izip(x.x, idx.x)])
    assert_array_equal(x0.max(idx0), x0.x[idx0.x].max())
    assert_array_equal(x.max(idxsub), xsub.max(idxsub))
    assert_array_equal(x.max(idx1d), x.x[:, idx1d.x].max(1))

    assert_equal(x.mean(), x.x.mean())
    assert_array_equal(x.mean(dim), x.x.mean(axis))
    assert_array_equal(x.mean(dims), x.x.mean(axes))
    assert_array_equal(x.mean(idx0), [x_[idx0.x].mean() for x_ in x.x])
    assert_array_equal(x.mean(idx), [x_[i].mean() for x_, i in izip(x.x, idx.x)])
    assert_array_equal(x0.mean(idx0), x0.x[idx0.x].mean())
    assert_array_equal(x.mean(idxsub), xsub.mean(idxsub))
    assert_array_equal(x.mean(idx1d), x.x[:, idx1d.x].mean(1))

    assert_equal(x.min(), x.x.min())
    assert_array_equal(x.min(dim), x.x.min(axis))
    assert_array_equal(x.min(dims), x.x.min(axes))
    assert_array_equal(x.min(idx0), [x_[idx0.x].min() for x_ in x.x])
    assert_array_equal(x.min(idx), [x_[i].min() for x_, i in izip(x.x, idx.x)])
    assert_array_equal(x0.min(idx0), x0.x[idx0.x].min())
    assert_array_equal(x.min(idxsub), xsub.min(idxsub))
    assert_array_equal(x.min(idx1d), x.x[:, idx1d.x].min(1))

    assert_equal(x.std(), x.x.std())
    assert_array_equal(x.std(dim), x.x.std(axis))
    assert_array_equal(x.std(dims), x.x.std(axes))
    assert_array_equal(x.std(idx0), [x_[idx0.x].std() for x_ in x.x])
    assert_array_equal(x.std(idx), [x_[i].std() for x_, i in izip(x.x, idx.x)])
    assert_array_equal(x0.std(idx0), x0.x[idx0.x].std())
    assert_array_equal(x.std(idxsub), xsub.std(idxsub))
    assert_array_equal(x.std(idx1d), x.x[:, idx1d.x].std(1))

    # non-numpy
    assert_equal(x.rms(), rms(x.x))
    assert_array_equal(x.rms(dim), rms(x.x, axis))
    assert_array_equal(x.rms(dims), rms(x.x, axes))
    assert_array_equal(x.rms(idx0), [rms(x_[idx0.x]) for x_ in x.x])
    assert_array_equal(x.rms(idx), [rms(x_[i]) for x_, i in izip(x.x, idx.x)])
    assert_array_equal(x0.rms(idx0), rms(x0.x[idx0.x]))
    assert_array_equal(x.rms(idxsub), xsub.rms(idxsub))
    assert_array_equal(x.rms(idx1d), rms(x.x[:, idx1d.x], 1))


def test_ols():
    "Test NDVar.ols() method"
    from rpy2.robjects import r

    # simulate data
    ds = datasets.get_uts(True)
    n_times = len(ds['uts'].time)
    x = np.zeros(n_times)
    x[20:40] = np.hanning(20)
    utsc = ds.eval("uts.copy()")
    utsc.x += ds['Y'].x[:, None] * x[None, :]
    ds_ = Dataset()
    ds_['x'] = Var(ds['Y'].x)
    ds_['x2'] = ds_['x'] + np.random.normal(0, 1, ds.n_cases)

    # ols regression
    m1 = ds_['x']
    b1 = utsc.ols(m1)
    res1 = utsc.residuals(m1)
    t1 = utsc.ols_t(m1)
    m2 = ds_.eval("x + x2")
    b2 = utsc.ols(m2)
    res2 = utsc.residuals(m2)
    t2 = utsc.ols_t(m2)
    # compare with R
    for i in xrange(n_times):
        ds_['y'] = Var(utsc.x[:, i])
        ds_.to_r('ds')
        # 1 predictor
        r('lm1 <- lm(y ~ x, ds)')
        beta = r('coef(lm1)')[1]
        assert_almost_equal(b1.x[0, i], beta)
        res = r('residuals(lm1)')
        assert_array_almost_equal(res1.x[:, i], res)
        t = r('coef(summary(lm1))')[5]
        assert_almost_equal(t1.x[0, i], t)
        # 2 predictors
        r('lm2 <- lm(y ~ x + x2, ds)')
        beta = r('coef(lm2)')[1:]
        assert_array_almost_equal(b2.x[:, i], beta)
        res = r('residuals(lm2)')
        assert_array_almost_equal(res2.x[:, i], res)
        lm2_coefs = r('coef(summary(lm2))')
        t = [lm2_coefs[7], lm2_coefs[8]]
        assert_array_almost_equal(t2.x[:, i], t)

    # 3d
    utsnd = ds['utsnd']
    ds_['utsnd'] = utsnd
    b1 = ds_.eval("utsnd.ols(x)")
    res1 = ds_.eval("utsnd.residuals(x)")
    t1 = ds_.eval("utsnd.ols_t(x)")
    for i in xrange(len(b1.time)):
        ds_['y'] = Var(utsnd.x[:, 1, i])
        ds_.to_r('ds')
        # 1 predictor
        r('lm1 <- lm(y ~ x, ds)')
        beta = r('coef(lm1)')[1]
        assert_almost_equal(b1.x[0, 1, i], beta)
        res = r('residuals(lm1)')
        assert_array_almost_equal(res1.x[:, 1, i], res)
        t = r('coef(summary(lm1))')[5]
        assert_almost_equal(t1.x[0, 1, i], t)


def test_io_pickle():
    "Test io by pickling"
    ds = datasets.get_uts()
    ds.info['info'] = "Some very useful information about the Dataset"
    tempdir = tempfile.mkdtemp()
    try:
        dest = os.path.join(tempdir, 'test.pickled')
        with open(dest, 'wb') as fid:
            pickle.dump(ds, fid, protocol=pickle.HIGHEST_PROTOCOL)
        with open(dest, 'rb') as fid:
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


def test_r():
    "Test interaction with R thorugh rpy2"
    from rpy2.robjects import r

    r("data(sleep)")
    ds = Dataset.from_r("sleep")
    assert_equal(ds.name, 'sleep')
    extra = (0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0, 1.9, 0.8,
             1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4)
    assert_array_equal(ds.eval('extra'), extra)
    assert_array_equal(ds.eval('ID'), map(str, xrange(1, 11)) * 2)
    assert_array_equal(ds.eval('group'), ['1'] * 10 + ['2'] * 10)

    # test putting
    ds.to_r('sleep_copy')
    ds_copy = Dataset.from_r('sleep_copy')
    assert_dataset_equal(ds_copy, ds)


def test_source_space():
    "Test SourceSpace Dimension"
    subject = 'fsaverage'
    data_path = mne.datasets.sample.data_path()
    mri_sdir = os.path.join(data_path, 'subjects')
    mri_dir = os.path.join(mri_sdir, subject)
    src_path = os.path.join(mri_dir, 'bem', subject + '-ico-5-src.fif')
    label_dir = os.path.join(mri_dir, 'label')
    label_ba1 = mne.read_label(os.path.join(label_dir, 'lh.BA1.label'))
    label_v1 = mne.read_label(os.path.join(label_dir, 'lh.V1.label'))
    label_mt = mne.read_label(os.path.join(label_dir, 'lh.MT.label'))
    label_ba1_v1 = label_ba1 + label_v1
    label_v1_mt = label_v1 + label_mt

    src = mne.read_source_spaces(src_path)
    source = SourceSpace((src[0]['vertno'], src[1]['vertno']), subject,
                         'ico-5', mri_sdir)
    index = source.dimindex(label_v1)
    source_v1 = source[index]
    index = source.dimindex(label_ba1_v1)
    source_ba1_v1 = source[index]
    index = source.dimindex(label_v1_mt)
    source_v1_mt = source[index]
    index = source_ba1_v1.dimindex(source_v1_mt)
    source_v1_intersection = source_ba1_v1[index]
    assert_source_space_equal(source_v1, source_v1_intersection)

    # index from label
    index = source.index_for_label(label_v1)
    assert_array_equal(index.source[index.x].vertno[0],
                       np.intersect1d(source.lh_vertno, label_v1.vertices, 1))

    # parcellation and cluster localization
    if mne.__version__ < '0.8':
        return
    parc = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=mri_sdir)
    indexes = [source.index_for_label(label) for label in parc
               if len(label) > 10]
    x = np.vstack([index.x for index in indexes])
    ds = source._cluster_properties(x)
    for i in xrange(ds.n_cases):
        assert_equal(ds[i, 'location'], parc[i].name)


def test_var():
    "Test Var objects"
    base = Factor('aabbcde')
    Y = Var.from_dict(base, {'a': 5, 'e': 8}, default=0)
    assert_array_equal(Y.x, [5, 5, 0, 0, 0, 0, 8])

    # .split()
    y = Var(np.arange(16))
    for i in xrange(1, 9):
        split = y.split(i)
        assert_equal(len(split.cells), i)

    # .as_factor()
    v = Var(np.arange(4))
    assert_array_equal(v.as_factor(), Factor('0123'))
    assert_array_equal(v.as_factor({0: 'a'}), Factor('a123'))
    assert_array_equal(v.as_factor({(0, 1): 'a', (2, 3): 'b'}), Factor('aabb'))
