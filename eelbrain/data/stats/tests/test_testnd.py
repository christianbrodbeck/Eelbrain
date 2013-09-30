# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

from eelbrain.data import datasets, testnd, plot


def test_anova():
    "Test testnd.anova()"
    plot.configure_backend(False, False)
    ds = datasets.get_rand(True)

    testnd.anova('utsnd', 'A*B', ds=ds)

    res = testnd.anova('utsnd', 'A*B*rm', ds=ds)
    p = plot.Array(res)
    p.close()

    res = testnd.anova('utsnd', 'A*B*rm', ds=ds, samples=2)
    p = plot.Array(res)
    p.close()


def test_corr():
    "Test testnd.corr()"
    plot.configure_backend(False, False)
    ds = datasets.get_rand(True)

    # add correlation
    Y = ds['Y']
    utsnd = ds['utsnd']
    utsnd.x.shape
    utsnd.x[:, 3:5, 50:65] += Y.x[:, None, None]

    res = testnd.corr('utsnd', 'Y', 'rm', ds=ds)
    p = plot.Array(res)
    p.close()

    res = testnd.corr('utsnd', 'Y', 'rm', ds=ds, samples=2)
    p = plot.Array(res)
    p.close()
