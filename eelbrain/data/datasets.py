'''
Defines some basic example datasets that are used in testing.
'''
import numpy as np

from .data_obj import Dataset, Factor, Var, NDVar, Sensor, UTS
from .design import permute


def get_rand(utsnd=False):
    """Create a sample Dataset with 60 cases and random data.

    Parameters
    ----------
    utsnd : bool
        Add a sensor by time NDVar (called 'utsnd').

    Returns
    -------
    ds : Dataset
        Datasets with data from random distributions.
    """
    np.random.seed(0)

    ds = Dataset()

    # add a model
    ds['A'] = Factor(['a0', 'a1'], rep=30)
    ds['B'] = Factor(['b0', 'b1'], rep=15, tile=2)
    ds['rm'] = Factor(('R%.2i' % i for i in xrange(15)), tile=4, random=True)

    # add dependent variables
    Y = np.hstack((np.random.normal(size=45), np.random.normal(1, size=15)))
    ds['Y'] = Var(Y)
    ybin = np.random.randint(0, 2, size=60)
    ds['YBin'] = Factor(ybin, labels={0:'c1', 1:'c2'})
    ycat = np.random.randint(0, 3, size=60)
    ds['YCat'] = Factor(ycat, labels={0:'c1', 1:'c2', 2:'c3'})

    # add a uts NDVar
    time = UTS(-.2, .01, 100)

    y = np.random.normal(0, .5, (60, len(time)))
    y[:15, 20:60] += np.hanning(40) * 1  # interaction
    y[:30, 50:80] += np.hanning(30) * 1  # main effect
    ds['uts'] = NDVar(y, dims=('case', time))

    # add sensor NDVar
    if utsnd:
        locs = np.array([[-1.0, 0.0, 0.0],
                         [ 0.0, 1.0, 0.0],
                         [ 1.0, 0.0, 0.0],
                         [ 0.0, -1.0, 0.0],
                         [ 0.0, 0.0, 1.0]])
        sensor = Sensor(locs, sysname='test_sens')

        Y = np.random.normal(0, 1, (60, 5, len(time)))
        for i in xrange(15):
            phi = np.random.uniform(0, 2 * np.pi, 1)
            x = np.sin(10 * 2 * np.pi * (time.times + phi))
            x *= np.hanning(len(time))
            Y[i, 0] += x

        dims = ('case', sensor, time)
        ds['utsnd'] = NDVar(Y, dims=dims)

    return ds


def get_uv():
    ds = permute([('A', ('a1', 'a2')),
                  ('B', ('b1', 'b2')),
                  ('rm', ['s%03i' % i for i in xrange(20)])])
    ds['rm'].random = True
    ds['intvar'] = Var(np.random.randint(5, 15, 80))
    ds['intvar'][:20] += 3
    ds['fltvar'] = Var(np.random.normal(0, 1, 80))
    ds['fltvar'][:40] += 1.
    ds['index'] = Var(np.repeat([True, False], 40))
    return ds
