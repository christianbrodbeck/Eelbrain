'''
Created on Dec 2, 2012

@author: christian
'''
from eelbrain import datasets, table, combine


def test_difference():
    "Test table.difference"
    ds = datasets.get_uv()
    print table.difference('fltvar', 'A', 'a1', 'a2', 'rm', ds=ds)
    print table.difference('fltvar', 'A', 'a1', 'a2', 'rm', by='B', ds=ds)
    print table.difference('fltvar', 'A%B', ('a1', 'b1'), ('a2', 'b2'), 'rm',
                           ds=ds)

    # create bigger dataset
    ds['C', :] = 'c1'
    ds2 = datasets.get_uv()
    ds2['C', :] = 'c2'
    ds = combine((ds, ds2))
    print table.difference('fltvar', 'A', 'a1', 'a2', 'rm', 'B%C', ds=ds)
    print table.difference('fltvar', 'A%B', ('a1', 'b1'), ('a2', 'b2'), 'rm',
                           'C', ds=ds)


def test_frequencies():
    "test table.frequencies"
    ds = datasets.get_uts()
    A = ds['A']
    B = ds['B']
    Cat = ds['YCat']
    print table.frequencies(Cat, A)
    print table.frequencies(Cat, A % B)
    print table.frequencies(Cat % A, B)


def test_repmeas():
    "Test table.repmeas (repeated measures table)"
    ds = datasets.get_uv()
    print table.repmeas('fltvar', 'A', 'rm', ds=ds)
    print table.repmeas('fltvar', 'A%B', 'rm', ds=ds)
    print table.repmeas('fltvar', 'A', 'B%rm', ds=ds)
