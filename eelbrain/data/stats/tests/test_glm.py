from nose.tools import assert_raises

from eelbrain.data import datasets
from eelbrain.data.test import anova


def test_anova():
    "Test univariate ANOVA"
    ds = datasets.get_rand()
    aov = anova('Y', 'A*B*rm', ds=ds)
    print aov

    # not fully specified model with random effects
    assert_raises(NotImplementedError, anova, 'Y', 'A*rm', ds=ds)
