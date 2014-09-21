# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from nose.tools import assert_equal

from eelbrain import datasets
from eelbrain._stats import stats


def test_confidence_interval():
    "Test confidence_interval()"
    from rpy2.robjects import r

    ds = datasets.get_loftus_masson_1994()
    y = ds['n_recalled'].x[:, None]
    ds.to_r('ds')

    # simple confidence interval of the mean
    ci = stats.confidence_interval(y)[0]
    r("s <- sd(ds$n_recalled)")
    r("n <- length(ds$n_recalled)")
    ci_r = r("qt(0.975, df=n-1) * s / sqrt(n)")[0]
    assert_equal(ci, ci_r)
