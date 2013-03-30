from nose.tools import ok_, eq_

import numpy as np

from eelbrain.vessels.design import (get_permutated_dataset, Variable,
                                     random_factor)

def test_random_factor():
    """Test the design module for creating an experiemnt design"""
    ds = get_permutated_dataset((
                                 Variable('A', '123456'),
                                 Variable('Bin', '01'),
                                 Variable('B', 'abcdef'),
                                 ))
    n = ds.n_cases

    rand = random_factor(('1', '2', '3'), n, 'rand')
    nv = (rand == '1').sum()
    eq_(nv, (rand == '2').sum(), "overall balancing")
    eq_(nv, (rand == '3').sum(), "overall balancing")

    # test urn kwarg
    randu = random_factor(('1', '2', '3'), n, urn=[rand])
    ok_((rand == randu).sum() == 0, "`urn` arg failed")
    nv = (randu == '1').sum()
    eq_(nv, 24, "random value assignment")
    eq_(nv, (randu == '2').sum(), "overall balancing")
    eq_(nv, (randu == '3').sum(), "overall balancing")

    # test sub kwarg
    sub = ds['Bin'] == '1'
    subrand = random_factor(('1', '2', '3'), n, urn=[rand], sub=sub)
    ok_((rand == randu).sum() == 0, "`urn` arg failed with `sub` arg")
    subc = (sub == False)
    ok_(np.all(subrand[subc] == ''), "values outside of sub are not ''")
    nv = (subrand == '1').sum()
    eq_(nv, 12, "random value assignment with `sub` arg")
    eq_(nv, (subrand == '2').sum(), "sub balancing")
    eq_(nv, (subrand == '3').sum(), "sub balancing")
