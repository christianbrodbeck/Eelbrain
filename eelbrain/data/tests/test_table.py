'''
Created on Dec 2, 2012

@author: christian
'''
from eelbrain.data import datasets, table


def test_frequencies():
    "test table.frequencies"
    ds = datasets.get_rand()
    A = ds['A']
    B = ds['B']
    Cat = ds['YCat']
    print table.frequencies(Cat, A)
    print table.frequencies(Cat, A % B)
    print table.frequencies(Cat % A, B)
