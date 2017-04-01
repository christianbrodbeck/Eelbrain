from eelbrain import datasets, regression


def test_regression():
    ds = datasets._get_continuous()
    y, x1, x2 = ds['y'], ds['x1'], ds['x2']

    res = regression(y, x1, 0, 1)
    res = regression(y, x2, 0, 1)
    res = regression(y, (x1, x2), 0, 1)
