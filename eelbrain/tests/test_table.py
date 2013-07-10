'''
Created on Dec 2, 2012

@author: christian
'''
from eelbrain.vessels import datasets
from eelbrain import table


def test_frequencies():
    "test table.frequencies"
    ds = datasets.get_basic()
    A = ds['A']
    B = ds['B']
    Cat = ds['Cat']
    print table.frequencies(Cat, A)
    print table.frequencies(Cat, A % B)
