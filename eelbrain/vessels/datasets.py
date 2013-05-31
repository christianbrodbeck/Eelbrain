'''
Defines some basic example datasets that are used in testing.
'''
import numpy as np

from .data import dataset, factor, var, ndvar
from .dimensions import Sensor, UTS


def get_basic():
    """
    Create a sample dataset with 60 cases and random data.
    """
    np.random.seed(0)

    ds = dataset()

    # add a model
    ds['A'] = factor(['a0', 'a1'], rep=30)
    ds['B'] = factor(['b0', 'b1'], rep=15, tile=2)
    ds['Cat'] = factor(np.random.randint(0, 2, size=60), labels={0:'c1', 1:'c2'})

    # add a dependent variable
    Y = np.hstack((np.random.normal(size=45), np.random.normal(1, size=15)))
    ds['Y'] = var(Y)

    # add an ndvar
    time = UTS(-.2, .01, 100)

    y = np.random.normal(0, .5, (60, len(time)))
    y[:15, 20:60] += np.hanning(40) * 1  # interaction
    y[:30, 50:80] += np.hanning(30) * 1  # main effect
    ds['Ynd'] = ndvar(y, dims=('case', time))

    # add sensor ndvar
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
    EEG = ndvar(Y, dims=dims, name='EEG')
    ds.add(EEG)

    return ds


def get_rm():
    """
    Create a sample dataset with 2 orthogonal factors 'A' and 'B', a random
    factor 'random', and dependent variable 'Y'.
    """
    np.random.seed(0)

    ds = dataset()

    # add a model
    ds['A'] = factor(['a0', 'a1'], rep=30)
    ds['B'] = factor(['b0', 'b1'], rep=15, tile=2)
    ds['random'] = factor(('R%.2i' % i for i in xrange(15)), tile=4,
                          random=True)

    # add a dependent variable
    Y = np.hstack((np.random.normal(size=45), np.random.normal(1, size=15)))
    ds['Y'] = var(Y)

    return ds
