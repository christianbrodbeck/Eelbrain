'''
Created on Dec 2, 2012

@author: christian
'''
from ...vessels.tests.test_data import ds
from .. import table


def test_frequencies():
    "test table.frequencies"
    A = ds['A']
    B = ds['B']
    Cat = ds['Cat']
    print table.frequencies(Cat, A)
    print table.frequencies(Cat, A % B)
