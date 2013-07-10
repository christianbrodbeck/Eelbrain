from nose.tools import assert_raises

from eelbrain.vessels import datasets
from eelbrain.test.glm import anova


def test_anova():
    "Test univariate ANOVA"
    ds = datasets.get_rm()
    aov = anova('Y', 'A*B*random', ds=ds)
    print aov

    # not fully specified model with random effects
    assert_raises(NotImplementedError, anova, 'Y', 'A*random', ds=ds)
