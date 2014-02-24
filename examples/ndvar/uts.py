import numpy as np
from eelbrain.lab import *
# the time dimension object
from eelbrain.data.data_obj import UTS

T = UTS(-.2, .01, 100)

# create simulated data:
# 4 conditions, 15 subjects
y = np.random.normal(0, .5, (60, len(T)))

# add an interaction effect
y[:15,20:60] += np.hanning(40) * 1
# add a main effect
y[:30,50:80] += np.hanning(30) * 1


Y = NDVar(y, dims=('case', T), name='Y')
A = Factor(['a0', 'a1'], rep=30, name='A')
B = Factor(['b0', 'b1'], rep=15, tile=2, name='B')

# plot Y
plot.UTSStat(Y, B)
# make the next plot bigger by specifying the width
plot.UTSStat(Y, A%B, w=6)
