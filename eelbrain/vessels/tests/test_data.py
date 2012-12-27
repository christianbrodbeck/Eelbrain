'''
Created on Dec 2, 2012

@author: christian
'''
import numpy as np

from ..data import dataset, factor, var, ndvar


ds = dataset()

# add a model
ds['A'] = factor(['a0', 'a1'], rep=30)
ds['B'] = factor(['b0', 'b1'], rep=15, tile=2)
ds['Cat'] = factor(np.random.randint(0, 2, size=60), labels={0:'c1', 1:'c2'})

# add a dpendent variable
Y = np.hstack((np.random.normal(size=45), np.random.normal(1, size=15)))
ds['Y'] = var(Y)

# add an ndvar
T = var(np.arange(-.2, .8, .01), name='time')

y = np.random.normal(0, .5, (60, len(T)))
y[:15, 20:60] += np.hanning(40) * 1  # interaction
y[:30, 50:80] += np.hanning(30) * 1  # main effect
ds['Ynd'] = ndvar(y, dims=('case', T))


def test_print():
    "Run the string representation methods"
    A = ds['A']
    print repr(A)
    print A
    Y = ds['Y']
    print repr(Y)
    print y
    print repr(ds)
    print ds
