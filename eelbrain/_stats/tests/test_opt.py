# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np
from numpy.testing import assert_array_almost_equal
from eelbrain import datasets
from eelbrain._stats import opt, glm
from eelbrain._stats.permutation import permute_order



def test_perm():
    "Test permutation argument"
    ds = datasets.get_uts()
    y = ds['uts'].x
    y_perm = np.empty_like(y)
    n_cases, n_tests = y.shape

    # lm_res_ss
    r1 = np.empty(n_tests)
    r2 = np.empty(n_tests)
    m = ds.eval('A*B')
    for perm in permute_order(n_cases, 2):
        opt.lm_res_ss(y, m.full, m.xsinv, r1, perm)
        opt.lm_res_ss(y[perm], m.full, m.xsinv, r2)
        assert_array_equal(r1, r2)
    # repeated measures
    for perm in permute_order(n_cases, 2, unit=ds['rm']):
        opt.lm_res_ss(y, m.full, m.xsinv, r1, perm)
        opt.lm_res_ss(y[perm], m.full, m.xsinv, r2)
        assert_array_equal(r1, r2)

    # balanced anova
    aov = glm._BalancedFixedNDANOVA(ds.eval('A*B'))
    r1 = aov.preallocate(y.shape)
    for perm in permute_order(n_cases, 2):
        aov.map(y, perm)
        r2 = r1.copy()
        y_perm[perm] = y
        aov.map(y_perm)
        assert_array_almost_equal(r2, r1, 12)

    # full repeated measures anova
    aov = glm._FullNDANOVA(ds.eval('A*B*rm'))
    r1 = aov.preallocate(y.shape)
    for perm in permute_order(n_cases, 2):
        aov.map(y, perm)
        r2 = r1.copy()
        y_perm[perm] = y
        aov.map(y_perm)
        assert_array_almost_equal(r2, r1, 12)
